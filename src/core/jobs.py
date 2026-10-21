from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, Optional, List

from core.db import engine
from sqlalchemy import Table, Column, Integer, MetaData, String, JSON, Text, DateTime, Index, select, insert, update
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


def list_jobs(
        job_type: Optional[JobType] = None,
        status: Optional[JobStatus] = None,
        limit: int = 50,
        offset: int = 0,
) -> List[Dict[Any, Any]]:
    stmt = select(jobs)
    if job_type:
        stmt = stmt.where(jobs.c.job_type == JobType(job_type))
    if status:
        stmt = stmt.where(jobs.c.status == JobStatus(status))

    stmt = stmt.order_by(jobs.c.id.desc()).limit(limit).offset(offset)

    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()
    return [dict(r) for r in rows]


def cancel_job(job_id: int, job_type: Optional[JobType] = None) -> Optional[Dict[str, Any]]:
    now = _now()
    conditions = [jobs.c.id == job_id, jobs.c.status == JobStatus.QUEUED]
    if job_type:
        conditions.append(jobs.c.job_type == JobType(job_type))

    with engine.begin() as conn:
        conn.execute(
            update(jobs)
            .where(*conditions)
            .values(status=JobStatus.CANCELED, updated_at=now, finished_at=now)
        )

    job = get_job(job_id)
    if job is None or (job_type and job["job_type"] != JobType(job_type)):
        return None
    return job


def claim_next_job(job_type: JobType, worker_id: str) -> Optional[Dict[str, Any]]:
    job_type = JobType(job_type)

    while True:
        with engine.begin() as conn:
            job_id = conn.execute(
                select(jobs.c.id)
                .where(jobs.c.job_type == job_type, jobs.c.status == JobStatus.QUEUED)
                .order_by(jobs.c.id)
                .limit(1)
            ).scalar()
            if job_id is None:
                return None

            now = _now()
            claimed = conn.execute(
                update(jobs)
                .where(jobs.c.id == job_id, jobs.c.status == JobStatus.QUEUED)
                .values(
                    status=JobStatus.RUNNING,
                    worker_id=worker_id,
                    started_at=now,
                    updated_at=now,
                )
            ).rowcount

        if claimed == 1:
            return get_job(job_id)


def _finish_job(job_id: int, worker_id: str, status: JobStatus, **values: Any) -> bool:
    now = _now()
    with engine.begin() as conn:
        updated = conn.execute(
            update(jobs)
            .where(
                jobs.c.id == job_id,
                jobs.c.worker_id == worker_id,
                jobs.c.status == JobStatus.RUNNING,
            )
            .values(status=status, updated_at=now, finished_at=now, **values)
        ).rowcount
    return updated == 1


def complete_job(job_id: int, worker_id: str, result: Optional[Dict[str, Any]] = None) -> bool:
    return _finish_job(job_id, worker_id, JobStatus.SUCCEEDED, result=result)


def fail_job(job_id: int, worker_id: str, error: str) -> bool:
    return _finish_job(job_id, worker_id, JobStatus.FAILED, error=error)