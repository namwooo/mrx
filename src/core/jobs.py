from sqlalchemy import Table, Column, Integer, MetaData, String, Enum, JSON, Text, DateTime, Index


class JobType(str, Enum):
    TRAINING = "training"
    INFERENCE = "inference"

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