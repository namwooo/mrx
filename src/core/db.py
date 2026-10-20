import json
import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


def create_db_engine(path: str) -> Engine:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(
        f"sqlite:///{path}",
        json_serializer=lambda obj: json.dumps(obj, ensure_ascii=False),
    )

    return engine

engine = create_db_engine(os.environ.get("DB_PATH", "./data/local.db"))