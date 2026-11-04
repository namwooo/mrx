import json
import warnings

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import inspect, text

from core.db import create_db_engine
from core.db import engine as global_engine
from core.migrate import MIGRATIONS_DIR, upgrade_db
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


def _alembic(conn):
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    config.attributes["connection"] = conn
    return config


def _seed_string_ids(conn):
    for model_id, version, created_at, applied in (
            ("mdl_ZZZZZZZZ", "1", "2026-11-01 00:00:00", 0),
            ("mdl_AAAAAAAA", "2", "2026-11-02 00:00:00", 1),
    ):
        conn.execute(text(
            "INSERT INTO models (model_id, type, model_name, model_version, task, applied, created_at) "
            "VALUES (:id, 'init', 'm', :version, 'detection', :applied, :created_at)"
        ), {"id": model_id, "version": version, "applied": applied, "created_at": created_at})
    conn.execute(text(
        "INSERT INTO jobs (id, job_type, status, payload, created_at, updated_at) VALUES "
        "(1, 'model_export', 'queued', :export, '2026-11-02 00:00:00', '2026-11-02 00:00:00'), "
        "(2, 'batch', 'queued', :batch, '2026-11-02 00:00:00', '2026-11-02 00:00:00')"
    ), {
        "export": json.dumps({"model_id": "mdl_AAAAAAAA", "format": "onnx"}),
        "batch": json.dumps({"model_id": "keep-me"}),
    })


def test_0002_keeps_models_and_their_export_jobs(empty_engine):
    with empty_engine.connect() as conn:
        command.upgrade(_alembic(conn), "0001")
        _seed_string_ids(conn)
        conn.commit()
        command.upgrade(_alembic(conn), "0002")
        conn.commit()

        models_after = conn.execute(text("SELECT id, model_version, applied FROM models ORDER BY id")).all()
        payloads = dict(conn.execute(text("SELECT id, payload FROM jobs")).all())

    assert [tuple(row) for row in models_after] == [(1, "1", 0), (2, "2", 1)]  # the oldest gets the first id
    assert json.loads(payloads[1]) == {"model_id": 2, "format": "onnx"}
    assert json.loads(payloads[2]) == {"model_id": "keep-me"}


def test_0002_can_be_undone(empty_engine):
    with empty_engine.connect() as conn:
        command.upgrade(_alembic(conn), "0001")
        _seed_string_ids(conn)
        conn.commit()
        before = _schema(empty_engine)
        command.upgrade(_alembic(conn), "0002")
        conn.commit()
        command.downgrade(_alembic(conn), "0001")
        conn.commit()

        ids = conn.execute(text("SELECT model_id FROM models ORDER BY created_at")).scalars().all()
        payload = conn.execute(text("SELECT payload FROM jobs WHERE id = 1")).scalar()

    assert _schema(empty_engine) == before
    assert ids == ["mdl_1", "mdl_2"]
    assert json.loads(payload) == {"model_id": "mdl_2", "format": "onnx"}


def test_upgrade_twice_changes_nothing(empty_engine):
    _upgrade(empty_engine)
    before = _schema(empty_engine)

    _upgrade(empty_engine)

    assert _schema(empty_engine) == before


def _model_columns(conn):
    return [row.name for row in conn.execute(text("PRAGMA table_info(models)"))]


def test_0006_keeps_models_and_can_be_undone(empty_engine):
    with empty_engine.connect() as conn:
        command.upgrade(_alembic(conn), "0005")
        conn.execute(text(
            "INSERT INTO models (id, type, model_name, model_version, task, dataset_id, applied, created_at) "
            "VALUES (3, 'init', 'm', '1', 'detection', 8, 1, '2026-11-01 00:00:00')"
        ))
        conn.commit()

        command.upgrade(_alembic(conn), "0006")
        conn.commit()
        model = conn.execute(text("SELECT id, model_name, applied FROM models")).one()
        columns_after = _model_columns(conn)
        schema_after = _schema(empty_engine)

        command.downgrade(_alembic(conn), "0005")
        conn.commit()
        columns_undone = _model_columns(conn)

    assert tuple(model) == (3, "m", 1)
    assert "dataset_id" not in columns_after
    assert "uq_models_applied_scope" in schema_after
    assert "dataset_id" in columns_undone


def test_0007_keeps_labeling_images_and_can_be_undone(empty_engine):
    now = "2026-11-01 00:00:00"
    with empty_engine.connect() as conn:
        command.upgrade(_alembic(conn), "0006")
        conn.execute(text(
            "INSERT INTO jobs (id, job_type, status, payload, created_at, updated_at) "
            "VALUES (1, 'labeling', 'succeeded', '{}', :now, :now)"
        ), {"now": now})
        conn.execute(text(
            "INSERT INTO models (id, type, model_name, model_version, task, applied, created_at) "
            "VALUES (1, 'line', 'm', '1', 'detection', 0, :now)"
        ), {"now": now})
        conn.execute(text(
            "INSERT INTO labelings (id, job_id, model_id, source_path, params, images_done, images_total) "
            "VALUES (1, 1, 1, '/images', '{}', 1, 1)"
        ))
        conn.execute(text(
            "INSERT INTO labeling_images (id, labeling_id, image_path, width, height, labels, num_labels, suggested) "
            "VALUES (5, 1, 'a.jpg', 10, 10, '[]', 0, 'review')"
        ))
        conn.commit()

        command.upgrade(_alembic(conn), "0007")
        conn.commit()
        image = conn.execute(text("SELECT id, image_path, predictions FROM labeling_images")).one()

        command.downgrade(_alembic(conn), "0006")
        conn.commit()
        kept = conn.execute(text("SELECT id FROM labeling_images")).scalars().all()
        columns = [row.name for row in conn.execute(text("PRAGMA table_info(labeling_images)"))]

    assert tuple(image) == (5, "a.jpg", None)
    assert kept == [5]
    assert "predictions" not in columns
