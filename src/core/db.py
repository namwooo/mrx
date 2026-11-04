import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker


def create_db_engine(path: str) -> Engine:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(
        f"sqlite:///{path}",
        json_serializer=lambda obj: json.dumps(obj, ensure_ascii=False),
    )

    return engine

engine = create_db_engine(os.environ.get("DB_PATH", "./data/local.db"))

SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def get_session() -> Iterator[Session]:
    with SessionLocal() as session:
        yield session


def as_utc(value: Optional[datetime]) -> Optional[datetime]:
    """SQLite drops the time zone. Every stored time is UTC."""
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=timezone.utc)
