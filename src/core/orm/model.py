import secrets
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Optional

from core.db import Base
from sqlalchemy import DateTime, Index, JSON, String, Text, UniqueConstraint, func, text
from sqlalchemy.orm import Mapped, mapped_column

ID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
ID_LENGTH = 8


class ModelType(str, Enum):
    LINE = "line"
    INIT = "init"


def new_model_id() -> str:
    return "mdl_" + "".join(secrets.choice(ID_ALPHABET) for _ in range(ID_LENGTH))


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ModelRow(Base):
    __tablename__ = "models"
    __table_args__ = (
        UniqueConstraint("model_name", "model_version", name="uq_models_name_version"),
    )

    model_id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_model_id)
    type: Mapped[str] = mapped_column(String(16))
    model_name: Mapped[str] = mapped_column(String(128))
    model_version: Mapped[str] = mapped_column(String(64))
    task: Mapped[str] = mapped_column(String(64))
    line: Mapped[Optional[str]] = mapped_column(String(64))
    weights_path: Mapped[Optional[str]] = mapped_column(Text)
    training_id: Mapped[Optional[str]] = mapped_column(String(32))
    init_training_id: Mapped[Optional[str]] = mapped_column(String(32))
    dataset_id: Mapped[Optional[str]] = mapped_column(String(32))
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
