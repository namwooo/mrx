from datetime import datetime, timezone
from pathlib import Path

from app.problems import conflict, not_found
from app.schemas.init_training import InitTrainingCreate
from core.orm import InitTrainingRow, JobRow, JobStatus, JobType, ModelRow, ModelType
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload


def create_init_training(session: Session, body: InitTrainingCreate) -> InitTrainingRow:
    model = session.get(ModelRow, body.model_id)
    if model is None:
        raise not_found(f"model '{body.model_id}' not found")
    if model.type != ModelType.INIT.value:
        raise conflict(f"model '{body.model_id}' is not an 'init' model")
    for name in ("image_path", "label_path"):
        path = getattr(body, name)
        if not Path(path).is_dir():
            raise not_found(f"{name} '{path}' not found")

    now = datetime.now(timezone.utc)
    job = JobRow(
        job_type=JobType.INIT_TRAINING.value,
        status=JobStatus.QUEUED.value,
        payload={},
        created_at=now,
        updated_at=now,
    )
    record = InitTrainingRow(
        job=job,
        model_id=body.model_id,
        image_path=body.image_path,
        label_path=body.label_path,
        params=body.params.model_dump(exclude_none=True),
    )
    session.add(record)
    session.flush()
    job.payload = {"init_training_id": record.id}
    session.commit()  # one commit for both: a worker can never pick up a job without its record
    return record


def get_init_training(session: Session, init_training_id: int) -> InitTrainingRow:
    record = session.scalar(
        select(InitTrainingRow)
        .options(joinedload(InitTrainingRow.job))
        .where(InitTrainingRow.id == init_training_id)
    )
    if record is None:
        raise not_found(f"init training '{init_training_id}' not found")
    return record
