from sqlalchemy import inspect, text


def test_table_exists(engine):
    assert "jobs" in inspect(engine).get_table_names()

def test_columns(engine):
    cols = {c["name"]: c for c in inspect(engine).get_columns("jobs")}

    assert set(cols) == {
        "id", "job_type", "status", "payload", "result", "error", "worker_id",
        "idempotency_key", "created_at", "updated_at", "started_at", "finished_at",
    }

    not_null = {"id", "job_type", "status", "payload", "created_at", "updated_at"}
    for name, col in cols.items():
        assert col["nullable"] is (name not in not_null), f"{name} nullable 설정이 다름"

def test_primary_key_is_autoincrement(engine):
    pk = inspect(engine).get_pk_constraint("jobs")
    assert pk["constrained_columns"] == ["id"]

    ddl = engine.connect().execute(
        text("SELECT sql FROM sqlite_master WHERE type='table' AND name='jobs'")
    ).scalar()
    assert "AUTOINCREMENT" in ddl
