from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Optional

from core.orm.base import Base
from sqlalchemy import DateTime, Index, JSON, String, Text, UniqueConstraint, func, text
from sqlalchemy.orm import Mapped, mapped_column


class ModelType(str, Enum):
    LINE = "line"
    INIT = "init"


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ModelRow(Base):
    __tablename__ = "models"
    __table_args__ = (
        UniqueConstraint("model_name", "model_version", name="uq_models_name_version"),
        Index("uq_models_init_training", "init_training_id", unique=True),
        {"sqlite_autoincrement": True},
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    type: Mapped[str] = mapped_column(String(16))
    model_name: Mapped[str] = mapped_column(String(128))
    model_version: Mapped[str] = mapped_column(String(64))
    task: Mapped[str] = mapped_column(String(64))
    line: Mapped[Optional[str]] = mapped_column(String(64))
    weights_path: Mapped[Optional[str]] = mapped_column(Text)
    training_id: Mapped[Optional[int]]
    init_training_id: Mapped[Optional[int]]
    dataset_id: Mapped[Optional[int]]
    metrics: Mapped[Optional[Dict[str, float]]] = mapped_column(JSON(none_as_null=True))
    applied: Mapped[bool] = mapped_column(default=False)
    applied_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    apply_comment: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


Index(
    "uq_models_applied_scope",
    func.coalesce(ModelRow.__table__.c.line, ""),
    ModelRow.__table__.c.task,
    unique=True,
    sqlite_where=text("applied = 1"),
)
