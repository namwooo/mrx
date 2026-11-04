"""Every table of the database. A new table is added here, so that Base.metadata and Alembic see it."""
from core.orm.base import Base
from core.orm.init_training import InitTrainingRow
from core.orm.job import JobRow, JobStatus, JobType
from core.orm.model import ModelRow, ModelType

metadata = Base.metadata

__all__ = [
    "InitTrainingRow",
    "JobRow",
    "JobStatus",
    "JobType",
    "ModelRow",
    "ModelType",
    "metadata",
]
