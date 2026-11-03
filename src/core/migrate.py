from pathlib import Path
from typing import Optional

from alembic import command
from alembic.config import Config
from sqlalchemy.engine import Connection

MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "migrations"


def upgrade_db(connection: Optional[Connection] = None) -> None:
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    config.attributes["connection"] = connection
    command.upgrade(config, "head")
