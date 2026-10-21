import pytest
from core.db import create_db_engine
from core.db import engine as global_engine
from core.jobs import metadata
from fastapi import FastAPI
from fastapi.testclient import TestClient

@pytest.fixture
def engine(tmp_path):
    eng = create_db_engine(str(tmp_path / "test.db"))
    metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def app_db():
    metadata.drop_all(global_engine)
    metadata.create_all(global_engine)
    yield global_engine
    metadata.drop_all(global_engine)


@pytest.fixture
def client(app_db):
    from app.routers import batches

    app = FastAPI()
    app.include_router(batches.router)
    with TestClient(app) as c:
        yield c