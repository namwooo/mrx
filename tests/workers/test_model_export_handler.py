import hashlib
import sys
import types
from datetime import datetime, timezone

import pytest

from core import jobs as job_repo
from core.db import SessionLocal
from core.jobs import Job, JobStatus, JobType
from core.orm import ModelRow, ModelType
from workers.handlers import HANDLERS
from workers.handlers.model_export import handle_model_export_job
from workers.worker import Worker

WEIGHTS = b"weights"


def _store(model):
    with SessionLocal() as session:
        session.add(model)
        session.commit()
    return model


@pytest.fixture
def model(app_db, tmp_path):
    weights = tmp_path / "defect-detector" / "1.4.0" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(WEIGHTS)
    return _store(ModelRow(
        type=ModelType.INIT.value,
        model_name="defect-detector",
        model_version="1.4.0",
        task="detection",
        weights_path=str(weights),
    ))


@pytest.fixture
def fake_yolo(monkeypatch):
    class FakeYolo:
        def __init__(self, path):
            self.path = path

        def export(self, format):
            suffix = {"onnx": ".onnx", "torchscript": ".torchscript"}[format]
            produced = self.path.removesuffix(".pt") + suffix
            with open(produced, "wb") as f:
                f.write(format.encode())
            return produced

    module = types.ModuleType("ultralytics")
    module.YOLO = FakeYolo
    monkeypatch.setitem(sys.modules, "ultralytics", module)


def _job(payload):
    now = datetime.now(timezone.utc)
    return Job(
        id=1, job_type=JobType.MODEL_EXPORT, status=JobStatus.RUNNING, payload=payload, result=None, error=None,
        worker_id="w1", idempotency_key=None, created_at=now, updated_at=now, started_at=now, finished_at=None,
    )


def test_handler_is_registered():
    assert HANDLERS[JobType.MODEL_EXPORT] is handle_model_export_job


def test_pt_export_is_the_registered_weights(model):
    result = handle_model_export_job(_job({"model_id": model.model_id, "format": "pt"}))

    assert result == {
        "file_path": model.weights_path,
        "file_hash": f"sha256:{hashlib.sha256(WEIGHTS).hexdigest()}",
    }


@pytest.mark.parametrize("export_format, file_name", [("onnx", "model.onnx"), ("torchscript", "model.torchscript")])
def test_converted_export_lands_next_to_the_weights(model, fake_yolo, tmp_path, export_format, file_name):
    result = handle_model_export_job(_job({"model_id": model.model_id, "format": export_format}))

    folder = tmp_path / "defect-detector" / "1.4.0"
    assert result["file_path"] == str(folder / file_name)
    assert result["file_hash"] == f"sha256:{hashlib.sha256(export_format.encode()).hexdigest()}"
    assert sorted(p.name for p in folder.iterdir()) == ["best.pt", file_name]


def test_engine_format_is_not_supported_yet(model):
    with pytest.raises(ValueError, match="engine"):
        handle_model_export_job(_job({"model_id": model.model_id, "format": "engine"}))


def test_missing_model_fails(app_db):
    with pytest.raises(ValueError, match="not found"):
        handle_model_export_job(_job({"model_id": "mdl_NOPE", "format": "pt"}))


def test_model_without_weights_fails(app_db):
    model = _store(ModelRow(
        type=ModelType.LINE.value, model_name="m", model_version="1", task="detection", line="LINE3",
    ))
    with pytest.raises(ValueError, match="no weights"):
        handle_model_export_job(_job({"model_id": model.model_id, "format": "pt"}))


def test_worker_stores_the_export(model):
    job = job_repo.enqueue_job(JobType.MODEL_EXPORT, {"model_id": model.model_id, "format": "pt"})

    assert Worker(JobType.MODEL_EXPORT, handle_model_export_job).run_once() is True

    done = job_repo.get_job(job.id)
    assert done.status == JobStatus.SUCCEEDED
    assert done.result["file_path"] == model.weights_path


def test_worker_records_an_unsupported_format_as_failed(model):
    job = job_repo.enqueue_job(JobType.MODEL_EXPORT, {"model_id": model.model_id, "format": "engine"})

    Worker(JobType.MODEL_EXPORT, handle_model_export_job).run_once()

    failed = job_repo.get_job(job.id)
    assert failed.status == JobStatus.FAILED
    assert "engine" in failed.error
