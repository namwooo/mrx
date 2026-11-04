import os
import tempfile

# Before core.db is imported: app_db drops every table, which must never hit the local data/local.db.
os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(prefix="mrx-test-"), "test.db")

import pytest
from core.db import create_db_engine
from core.db import engine as global_engine
from core.orm import metadata

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
