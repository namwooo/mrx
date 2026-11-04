import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

from app.problems import already_finished, conflict, not_found
from app.schemas.training import TrainingCreate
from core.orm import JobRow, JobStatus, JobType, LabelingRow, ModelRow, TrainingMetricRow, TrainingRow
from core.training_data import select_images
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session, contains_eager, joinedload

TRAINING_DIR = Path(os.environ.get("TRAINING_DIR", "./data/trainings"))
UNFINISHED = (JobStatus.QUEUED.value, JobStatus.RUNNING.value)


def create_training(session: Session, body: TrainingCreate) -> TrainingRow:
    if body.base_model_id is not None and session.get(ModelRow, body.base_model_id) is None:
        raise not_found(f"model '{body.base_model_id}' not found")

    labelings = list(session.scalars(
        select(LabelingRow).options(joinedload(LabelingRow.job)).where(LabelingRow.id.in_(body.labeling_ids))
    ))
    missing = sorted(set(body.labeling_ids) - {labeling.id for labeling in labelings})
    if missing:
        raise not_found(f"labelings {missing} not found")

    unfinished = sorted(x.id for x in labelings if x.job.status != JobStatus.SUCCEEDED.value)
    if unfinished:
        raise conflict(f"labelings {unfinished} are not finished")
    if any(labeling.class_names != labelings[0].class_names for labeling in labelings):
        raise conflict("the labelings do not share the same classes")

    split = body.split.model_dump()
    counts = Counter(image.split for image in select_images(session, body.labeling_ids, split))
    if not counts["train"] or not counts["val"]:
        raise conflict(f"not enough usable images: {counts['train']} for train, {counts['val']} for val")

    now = datetime.now(timezone.utc)
    job = JobRow(
        job_type=JobType.TRAINING.value,
        status=JobStatus.QUEUED.value,
        payload={},
        created_at=now,
        updated_at=now,
    )
    record = TrainingRow(
        job=job,
        base_model_id=body.base_model_id,
        labeling_ids=body.labeling_ids,
        split=split,
        params=body.params.model_dump(exclude_none=True),
        output_dir="",
    )
    session.add(record)
    session.flush()
    record.output_dir = str(TRAINING_DIR.resolve() / str(record.id))
    job.payload = {"training_id": record.id}
    session.commit()  # one commit for both: a worker can never pick up a job without its record
    return record


def list_trainings(
        session: Session,
        labeling_id: Optional[int] = None,
        status: Optional[Sequence[JobStatus]] = None,
        limit: int = 50,
        offset: int = 0,
) -> List[TrainingRow]:
    stmt = select(TrainingRow).join(TrainingRow.job).options(contains_eager(TrainingRow.job))
    if labeling_id is not None:
        used = func.json_each(TrainingRow.labeling_ids).table_valued("value")
        stmt = stmt.where(select(used.c.value).where(used.c.value == labeling_id).exists())
    if status:
        stmt = stmt.where(JobRow.status.in_([s.value for s in status]))
    return list(session.scalars(stmt.order_by(TrainingRow.id.desc()).limit(limit).offset(offset)))


def get_training(session: Session, training_id: int) -> TrainingRow:
    record = session.scalar(
        select(TrainingRow).options(joinedload(TrainingRow.job)).where(TrainingRow.id == training_id)
    )
    if record is None:
        raise not_found(f"training '{training_id}' not found")
    return record


def list_metrics(session: Session, training_id: int) -> List[TrainingMetricRow]:
    get_training(session, training_id)
    return list(session.scalars(
        select(TrainingMetricRow)
        .where(TrainingMetricRow.training_id == training_id)
        .order_by(TrainingMetricRow.epoch, TrainingMetricRow.id)
    ))


def get_result(session: Session, training_id: int) -> TrainingRow:
    record = get_training(session, training_id)
    if record.job.status != JobStatus.SUCCEEDED.value:
        raise conflict(f"training '{training_id}' is '{record.job.status}', not 'succeeded'")
    return record


def read_logs(session: Session, training_id: int, tail: Optional[int] = None) -> str:
    log_path = Path(get_training(session, training_id).output_dir) / "train.log"
    if not log_path.is_file():
        return ""  # the training has not started yet
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    return "".join(lines[-tail:] if tail else lines)


def cancel_training(session: Session, training_id: int) -> TrainingRow:
    record = get_training(session, training_id)
    now = datetime.now(timezone.utc)
    # The condition decides against a worker finishing at the same moment. The worker sees the new status and stops.
    canceled = session.execute(
        update(JobRow)
        .where(JobRow.id == record.job_id, JobRow.status.in_(UNFINISHED))
        .values(status=JobStatus.CANCELED.value, updated_at=now, finished_at=now)
        .execution_options(synchronize_session=False)
    ).rowcount
    session.commit()
    session.refresh(record.job)
    if canceled == 0:
        raise already_finished(f"training '{training_id}' is already {record.job.status}")
    return record
