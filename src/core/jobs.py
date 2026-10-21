from datetime import datetime, timezone
from typing import Dict, Any, Optional

from core.db import engine
from sqlalchemy import Table, Column, Integer, MetaData, String, Enum, JSON, Text, DateTime, Index, select, insert
from sqlalchemy.exc import IntegrityError


class JobType(str, Enum):
    TRAINING = "training"
    BATCH = "batch"

class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"

metadata = MetaData()

jobs = Table(
    "jobs",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("job_type", String(32), nullable=False),
    Column("status", String(16), nullable=False),
    Column("payload", JSON, nullable=False),
    Column("result", JSON),
    Column("error", Text),
    Column("worker_id", String(64)),
    Column("idempotency_key", String(128), unique=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("started_at", DateTime(timezone=True)),
    Column("finished_at", DateTime(timezone=True)),
    Index("idx_jobs_dequeue", "job_type", "status", "created_at"),
    sqlite_autoincrement=True,
)

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _get_by_key(idempotency_key: str) -> Optional[Dict[Any, Any]]:
    with engine.connect() as conn:
        row = conn.execute(
            select(jobs).where(jobs.c.idempotency_key == idempotency_key)
        ).mappings().first()
    return dict(row) if row else None


def enqueue_job(
        job_type: JobType,
        payload: Dict[str, Any],
        idempotency_key: Optional[str] = None,
) -> Dict[Any, Any] | None:
    job_type = JobType(job_type)

    if idempotency_key:
        existing = _get_by_key(idempotency_key)
        if existing:
            return existing

    now = _now()

    try:
        with engine.begin() as conn:
            result = conn.execute(
                insert(jobs).values(
                    job_type=job_type,
                    status=JobStatus.QUEUED,
                    payload=payload,
                    idempotency_key=idempotency_key,
                    created_at=now,
                    updated_at=now,
                )
            )
            job_id = result.inserted_primary_key[0]  # DB가 부여한 id
    except IntegrityError:
        if not idempotency_key:
            raise
        return _get_by_key(idempotency_key)

    return get_job(job_id)


def get_job(job_id: int) -> Optional[Dict[Any, Any]]:
    with engine.connect() as conn:
        row = conn.execute(select(jobs).where(jobs.c.id == job_id)).mappings().first()
    return dict(row) if row else None