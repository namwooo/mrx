from datetime import datetime
from typing import Any, Dict, List, Optional

from core.orm.base import Base
from core.orm.job import JobRow
from sqlalchemy import DateTime, Float, ForeignKey, Index, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship


class TrainingRow(Base):
    """What was requested and how far it got. Status, times, error and outcome are on the job."""
    __tablename__ = "trainings"
    __table_args__ = {"sqlite_autoincrement": True}

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"), unique=True)
    base_model_id: Mapped[Optional[int]] = mapped_column(ForeignKey("models.id"))
    labeling_ids: Mapped[List[int]] = mapped_column(JSON)
    split: Mapped[Dict[str, float]] = mapped_column(JSON)
    params: Mapped[Dict[str, Any]] = mapped_column(JSON)
    output_dir: Mapped[str] = mapped_column(Text)
    epochs_done: Mapped[int] = mapped_column(default=0)

    job: Mapped[JobRow] = relationship(lazy="raise")


class TrainingMetricRow(Base):
    __tablename__ = "training_metrics"
    __table_args__ = (
        Index("idx_training_metrics_training", "training_id", "epoch"),
        {"sqlite_autoincrement": True},
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    training_id: Mapped[int] = mapped_column(ForeignKey("trainings.id"))
    epoch: Mapped[int]
    name: Mapped[str] = mapped_column(String(64))
    value: Mapped[float] = mapped_column(Float)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
