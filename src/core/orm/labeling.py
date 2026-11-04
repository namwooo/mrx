from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from core.orm.base import Base
from core.orm.job import JobRow
from sqlalchemy import DateTime, ForeignKey, Index, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship


class Suggestion(str, Enum):
    APPROVED = "approved"
    REVIEW = "review"


class ReviewResult(str, Enum):
    OK = "OK"
    MODIFY = "MODIFY"
    DISCARD = "DISCARD"


class LabelingRow(Base):
    """What was requested and how far it got. Status, times, error and summary are on the job."""
    __tablename__ = "labelings"
    __table_args__ = {"sqlite_autoincrement": True}

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"), unique=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("models.id"))
    source_path: Mapped[str] = mapped_column(Text)
    params: Mapped[Dict[str, Any]] = mapped_column(JSON)
    class_names: Mapped[Optional[Dict[str, str]]] = mapped_column(JSON(none_as_null=True))
    images_done: Mapped[int] = mapped_column(default=0)
    images_total: Mapped[int] = mapped_column(default=0)

    job: Mapped[JobRow] = relationship(lazy="raise")


class LabelingImageRow(Base):
    __tablename__ = "labeling_images"
    __table_args__ = (
        Index("idx_labeling_images_suggested", "labeling_id", "suggested"),
        Index("idx_labeling_images_result", "labeling_id", "result"),
        {"sqlite_autoincrement": True},
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    labeling_id: Mapped[int] = mapped_column(ForeignKey("labelings.id"))
    image_path: Mapped[str] = mapped_column(Text)
    width: Mapped[int]
    height: Mapped[int]
    labels: Mapped[List[Dict[str, Any]]] = mapped_column(JSON)
    num_labels: Mapped[int]
    suggested: Mapped[str] = mapped_column(String(16))
    result: Mapped[Optional[str]] = mapped_column(String(16))
    reviewed_labels: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON(none_as_null=True))
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    predictions: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON(none_as_null=True))
