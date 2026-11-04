from typing import Any, Dict

from core.orm.base import Base
from core.orm.job import JobRow
from sqlalchemy import ForeignKey, JSON, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship


class InitTrainingRow(Base):
    __tablename__ = "init_trainings"
    __table_args__ = {"sqlite_autoincrement": True}

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"), unique=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("models.id"))
    image_path: Mapped[str] = mapped_column(Text)
    label_path: Mapped[str] = mapped_column(Text)
    params: Mapped[Dict[str, Any]] = mapped_column(JSON)
    epochs_done: Mapped[int] = mapped_column(default=0)

    job: Mapped[JobRow] = relationship(lazy="raise")
