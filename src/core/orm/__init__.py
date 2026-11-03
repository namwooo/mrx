"""Every table of the database. A new table is added here, so that Base.metadata and Alembic see it."""
from core.db import Base
from core.orm.job import JobRow, JobStatus, JobType
from core.orm.model import ModelRow, ModelType, new_model_id

metadata = Base.metadata

__all__ = ["JobRow", "JobStatus", "JobType", "ModelRow", "ModelType", "metadata", "new_model_id"]
