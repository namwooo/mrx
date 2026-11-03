from alembic import context
from sqlalchemy.engine import Connection

from core.db import engine
from core.orm import metadata


def run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=metadata,
        # SQLite rewrites the table for most ALTER operations. A rewrite of `models` loses uq_models_applied_scope,
        # because expression indexes cannot be reflected: create it again in that migration.
        # tests/core/test_migrations.py fails if a migration forgets it.
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


connection = context.config.attributes.get("connection")
if connection is not None:
    run_migrations(connection)
else:
    with engine.connect() as connection:
        run_migrations(connection)
