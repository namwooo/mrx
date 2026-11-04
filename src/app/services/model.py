import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

from app.problems import conflict, not_found
from app.schemas.model import ModelApply, ModelCreate, ModelExportRequest, ModelMetadata
from app.services import init_training as init_training_service
from app.services import training as training_service
from core import jobs as job_repo
from core.jobs import Job, JobType, jobs
from core.orm import JobRow, JobStatus, LabelingRow, ModelRow, ModelType
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

REGISTRY_DIR = Path(os.environ.get("REGISTRY_DIR", "./data/registry"))
WEIGHTS_FILE = "best.pt"
PARTIAL_DIR = ".partial"  # same disk as the registry, so moving a finished copy in place is atomic

ModelWithExports = Tuple[ModelRow, List[Job]]


def create_model(session: Session, body: ModelCreate) -> ModelWithExports:
    if body.type == ModelType.INIT:
        return _store_model(session, body, Path(body.weights_path))
    if body.init_training_id is not None:
        return _register_from_init_training(session, body)
    return _register_from_training(session, body)


def _register_from_init_training(session: Session, body: ModelCreate) -> ModelWithExports:
    init_training_id = body.init_training_id
    training = init_training_service.get_init_training(session, init_training_id)
    source = f"init training '{init_training_id}'"

    _check_succeeded(training.job, source)
    if _find_by_init_training(session, init_training_id):
        raise _already_registered(source)
    _check_task(body, session.get(ModelRow, training.model_id).task, source)

    result = training.job.result
    return _store_model(
        session,
        body,
        Path(result["weights_path"]),
        init_training_id=init_training_id,
        metrics=result["metrics"],
    )


def _register_from_training(session: Session, body: ModelCreate) -> ModelWithExports:
    training_id = body.training_id
    training = training_service.get_training(session, training_id)
    source = f"training '{training_id}'"

    _check_succeeded(training.job, source)
    if _find_by_training(session, training_id):
        raise _already_registered(source)
    labeling = session.get(LabelingRow, training.labeling_ids[0])
    _check_task(body, session.get(ModelRow, labeling.model_id).task, source)

    result = training.job.result
    return _store_model(
        session,
        body,
        Path(result["weights_path"]),
        training_id=training_id,
        metrics=result["metrics"],
    )


def _check_succeeded(job: JobRow, source: str) -> None:
    if job.status != JobStatus.SUCCEEDED.value:
        raise conflict(f"{source} is '{job.status}', not 'succeeded'")


def _check_task(body: ModelCreate, task: str, source: str) -> None:
    if body.metadata.task != task:
        raise conflict(f"task '{body.metadata.task}' does not match the task '{task}' of {source}")


def _store_model(session: Session, body: ModelCreate, source: Path, **origin: Any) -> ModelWithExports:
    meta = body.metadata
    if not source.is_file():
        raise not_found(f"weights_path '{source}' not found")
    if _find_model(session, meta):
        raise _duplicate(meta)

    registry = REGISTRY_DIR.resolve()
    target = registry / meta.model_name / meta.model_version / WEIGHTS_FILE
    partial = registry / PARTIAL_DIR / f"{uuid4().hex}.part"
    partial.parent.mkdir(parents=True, exist_ok=True)

    record = ModelRow(
        type=body.type.value,
        model_name=meta.model_name,
        model_version=meta.model_version,
        task=meta.task,
        line=meta.line,
        weights_path=str(target),
        **origin,
    )
    try:
        shutil.copyfile(source, partial)
        session.add(record)
        session.flush()
        target.parent.mkdir(parents=True, exist_ok=True)
        partial.replace(target)
        session.commit()
    except IntegrityError:
        session.rollback()
        init_training_id, training_id = origin.get("init_training_id"), origin.get("training_id")
        if init_training_id is not None and _find_by_init_training(session, init_training_id):
            raise _already_registered(f"init training '{init_training_id}'") from None
        if training_id is not None and _find_by_training(session, training_id):
            raise _already_registered(f"training '{training_id}'") from None
        raise _duplicate(meta) from None
    finally:
        partial.unlink(missing_ok=True)

    return record, []


def _find_model(session: Session, meta: ModelMetadata) -> Optional[ModelRow]:
    return session.scalar(
        select(ModelRow).where(ModelRow.model_name == meta.model_name, ModelRow.model_version == meta.model_version)
    )


def _find_by_init_training(session: Session, init_training_id: int) -> Optional[ModelRow]:
    return session.scalar(select(ModelRow).where(ModelRow.init_training_id == init_training_id))


def _find_by_training(session: Session, training_id: int) -> Optional[ModelRow]:
    return session.scalar(select(ModelRow).where(ModelRow.training_id == training_id))


def _already_registered(source: str):
    return conflict(f"{source} is already registered as a model")


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
) -> List[ModelWithExports]:
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
    stmt = stmt.order_by(ModelRow.created_at.desc(), ModelRow.id.desc()).limit(limit).offset(offset)

    records = list(session.scalars(stmt))
    exports = _export_jobs(session, [r.id for r in records])
    return [(r, exports.get(r.id, [])) for r in records]


def get_model(session: Session, model_id: int) -> ModelWithExports:
    record = _get_record(session, model_id)
    return record, _export_jobs(session, [model_id]).get(model_id, [])


def apply_model(session: Session, model_id: int, body: Optional[ModelApply]) -> ModelWithExports:
    record = _get_record(session, model_id)

    # Plain UPDATEs, not attribute changes: the loaded row may be stale, and an unchanged attribute writes nothing.
    session.execute(
        update(ModelRow)
        .where(
            ModelRow.line.is_not_distinct_from(record.line),
            ModelRow.task == record.task,
            ModelRow.id != model_id,
            ModelRow.applied.is_(True),
        )
        .values(applied=False)
    )
    session.execute(
        update(ModelRow)
        .where(ModelRow.id == model_id)
        .values(
            applied=True,
            applied_at=datetime.now(timezone.utc),
            apply_comment=body.comment if body else None,
        )
    )
    session.commit()
    session.refresh(record)
    return get_model(session, model_id)


def export_model(session: Session, model_id: int, body: ModelExportRequest) -> Job:
    _get_record(session, model_id)
    return job_repo.enqueue_job(
        job_type=JobType.MODEL_EXPORT,
        payload={"model_id": model_id, "format": body.format.value},
    )


def _get_record(session: Session, model_id: int) -> ModelRow:
    record = session.get(ModelRow, model_id)
    if record is None:
        raise not_found(f"model '{model_id}' not found")
    return record


def _export_jobs(session: Session, model_ids: Sequence[int]) -> Dict[int, List[Job]]:
    """Export history is the job queue itself: each export job, newest first, grouped by model."""
    if not model_ids:
        return {}

    rows = session.execute(
        select(jobs)
        .where(
            jobs.c.job_type == JobType.MODEL_EXPORT,
            jobs.c.payload["model_id"].as_integer().in_(model_ids),
        )
        .order_by(jobs.c.id.desc())
    ).mappings().all()

    grouped: Dict[int, List[Job]] = {}
    for row in rows:
        job = Job.from_row(row)
        grouped.setdefault(job.payload["model_id"], []).append(job)
    return grouped
