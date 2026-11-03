import warnings

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import inspect, text

from core.db import create_db_engine
from core.db import engine as global_engine
from core.migrate import upgrade_db
from core.orm import metadata


@pytest.fixture
def empty_engine(tmp_path):
    eng = create_db_engine(str(tmp_path / "migrated.db"))
    yield eng
    eng.dispose()


def _upgrade(engine):
    with engine.connect() as conn:
        upgrade_db(conn)
        conn.commit()


def _schema(engine):
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT name, sql FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' AND name != 'alembic_version' AND sql IS NOT NULL"
        )).all()
    return {name: " ".join(sql.split()) for name, sql in rows}


def _version(engine):
    with engine.connect() as conn:
        return conn.execute(text("SELECT version_num FROM alembic_version")).scalar()


def test_tests_never_touch_the_local_database():
    assert "local.db" not in str(global_engine.url)


def test_upgrade_creates_the_same_schema_as_the_orm(empty_engine, tmp_path):
    _upgrade(empty_engine)

    orm_engine = create_db_engine(str(tmp_path / "orm.db"))
    metadata.create_all(orm_engine)
    assert _schema(empty_engine) == _schema(orm_engine)
    orm_engine.dispose()


def test_orm_has_no_changes_missing_from_the_migrations(empty_engine):
    _upgrade(empty_engine)

    with empty_engine.connect() as conn, warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Alembic cannot compare the expression index, the schema test above does
        assert compare_metadata(MigrationContext.configure(conn), metadata) == []


def test_upgrade_adopts_a_database_from_before_alembic(empty_engine):
    metadata.tables["jobs"].create(empty_engine)
    with empty_engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO jobs (id, job_type, status, payload, created_at, updated_at) "
            "VALUES (7, 'batch', 'queued', '{}', '2026-10-30 00:00:00', '2026-10-30 00:00:00')"
        ))

    _upgrade(empty_engine)

    assert {"jobs", "models", "alembic_version"} <= set(inspect(empty_engine).get_table_names())
    assert _version(empty_engine)
    with empty_engine.connect() as conn:
        assert conn.execute(text("SELECT id FROM jobs")).scalars().all() == [7]


def test_upgrade_twice_changes_nothing(empty_engine):
    _upgrade(empty_engine)
    before = _schema(empty_engine)

    _upgrade(empty_engine)

    assert _schema(empty_engine) == before
