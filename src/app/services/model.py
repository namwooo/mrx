import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from app.problems import conflict, not_found, not_implemented
from app.schemas.model import Model, ModelApply, ModelCreate, ModelExport, ModelExportRequest, ModelMetadata
from core import jobs as job_repo
from core.jobs import Job, JobType, jobs
from core.orm import ModelRow, ModelType, new_model_id
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

REGISTRY_DIR = Path(os.environ.get("REGISTRY_DIR", "./data/registry"))
WEIGHTS_FILE = "best.pt"


def create_model(session: Session, body: ModelCreate) -> Model:
    if body.type == ModelType.LINE:
        # TODO: register from the succeeded init training or training once those APIs exist.
        raise not_implemented("registering a 'line' model needs the init-trainings and trainings APIs")
    return _register_init_model(session, body)


def _register_init_model(session: Session, body: ModelCreate) -> Model:
    meta = body.metadata
    source = Path(body.weights_path)
    if not source.is_file():
        raise not_found(f"weights_path '{body.weights_path}' not found")
    if _find_model(session, meta):
        raise _duplicate(meta)

    model_id = new_model_id()
    target = REGISTRY_DIR.resolve() / meta.model_name / meta.model_version / WEIGHTS_FILE
    partial = target.with_name(f"{target.name}.{model_id}.part")
    target.parent.mkdir(parents=True, exist_ok=True)

    record = ModelRow(
        model_id=model_id,
        type=body.type.value,
        model_name=meta.model_name,
        model_version=meta.model_version,
        task=meta.task,
        line=meta.line,
        weights_path=str(target),
    )
    try:
        # Copied outside the transaction, and moved in place only by the request that got the name and version.
        shutil.copyfile(source, partial)
        session.add(record)
        session.flush()
        partial.replace(target)
        session.commit()
    except IntegrityError:
        session.rollback()
        raise _duplicate(meta) from None
    finally:
        partial.unlink(missing_ok=True)

    return _to_model(record, [])


def _find_model(session: Session, meta: ModelMetadata) -> Optional[ModelRow]:
    return session.scalar(
        select(ModelRow).where(ModelRow.model_name == meta.model_name, ModelRow.model_version == meta.model_version)
    )


def _duplicate(meta: ModelMetadata):
    return conflict(f"model '{meta.model_name}' version '{meta.model_version}' already exists")


def list_models(
        session: Session,
        model_type: Optional[ModelType] = None,
        model_name: Optional[str] = None,
        line: Optional[str] = None,
        task: Optional[str] = None,
        applied: Optional[bool] = None,
        limit: int = 50,
        offset: int = 0,
) -> List[Model]:
    stmt = select(ModelRow)
    if model_type:
        stmt = stmt.where(ModelRow.type == model_type.value)
    if model_name:
        stmt = stmt.where(ModelRow.model_name == model_name)
    if line:
        stmt = stmt.where(ModelRow.line == line)
    if task:
        stmt = stmt.where(ModelRow.task == task)
    if applied is not None:
        stmt = stmt.where(ModelRow.applied.is_(applied))
    stmt = stmt.order_by(ModelRow.created_at.desc(), ModelRow.model_id.desc()).limit(limit).offset(offset)

    records = list(session.scalars(stmt))
    exports = _export_jobs(session, [r.model_id for r in records])
    return [_to_model(r, exports.get(r.model_id, [])) for r in records]


def get_model(session: Session, model_id: str) -> Model:
    record = _get_record(session, model_id)
    return _to_model(record, _export_jobs(session, [model_id]).get(model_id, []))


def apply_model(session: Session, model_id: str, body: Optional[ModelApply]) -> Model:
    record = _get_record(session, model_id)

    # Plain UPDATEs, not attribute changes: the loaded row may be stale, and an unchanged attribute writes nothing.
    session.execute(
        update(ModelRow)
        .where(
            ModelRow.line.is_not_distinct_from(record.line),
            ModelRow.task == record.task,
            ModelRow.model_id != model_id,
            ModelRow.applied.is_(True),
        )
        .values(applied=False)
    )
    session.execute(
        update(ModelRow)
        .where(ModelRow.model_id == model_id)
        .values(
            applied=True,
            applied_at=datetime.now(timezone.utc),
            apply_comment=body.comment if body else None,
        )
    )
    session.commit()
    session.refresh(record)
    return get_model(session, model_id)


def export_model(session: Session, model_id: str, body: ModelExportRequest) -> ModelExport:
    _get_record(session, model_id)
    job = job_repo.enqueue_job(
        job_type=JobType.MODEL_EXPORT,
        payload={"model_id": model_id, "format": body.format.value},
    )
    return _to_export(job)


def _get_record(session: Session, model_id: str) -> ModelRow:
    record = session.get(ModelRow, model_id)
    if record is None:
        raise not_found(f"model '{model_id}' not found")
    return record


def _export_jobs(session: Session, model_ids: Sequence[str]) -> Dict[str, List[Job]]:
    """Export history is the job queue itself: each export job, newest first, grouped by model."""
    if not model_ids:
        return {}

    rows = session.execute(
        select(jobs)
        .where(
            jobs.c.job_type == JobType.MODEL_EXPORT,
            jobs.c.payload["model_id"].as_string().in_(model_ids),
        )
        .order_by(jobs.c.id.desc())
    ).mappings().all()

    grouped: Dict[str, List[Job]] = {}
    for row in rows:
        job = Job.from_row(row)
        grouped.setdefault(job.payload["model_id"], []).append(job)
    return grouped


def _to_model(record: ModelRow, export_jobs: List[Job]) -> Model:
    return Model(
        model_id=record.model_id,
        type=record.type,
        metadata=ModelMetadata(
            model_name=record.model_name,
            model_version=record.model_version,
            task=record.task,
            line=record.line,
        ),
        training_id=record.training_id,
        init_training_id=record.init_training_id,
        dataset_id=record.dataset_id,
        metrics=record.metrics,
        weights_path=record.weights_path,
        applied=record.applied,
        applied_at=_utc(record.applied_at),
        exports=[_to_export(job) for job in export_jobs],
        created_at=_utc(record.created_at),
    )


def _to_export(job: Job) -> ModelExport:
    result = job.result or {}
    return ModelExport(
        format=job.payload["format"],
        status=job.status,
        file_path=result.get("file_path"),
        file_hash=result.get("file_hash"),
        requested_at=_utc(job.created_at),
        error=job.error,
    )


def _utc(value: Optional[datetime]) -> Optional[datetime]:
    """SQLite drops the time zone. Every stored time is UTC."""
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=timezone.utc)
