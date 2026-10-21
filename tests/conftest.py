import pytest
from core.db import create_db_engine
from core.jobs import metadata


@pytest.fixture
def engine(tmp_path):
    eng = create_db_engine(str(tmp_path / "test.db"))
    metadata.create_all(eng)
    yield eng
    eng.dispose()