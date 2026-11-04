from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.services import model as model_service
from core.db import SessionLocal
from core.jobs import JobType
from core.orm import InitTrainingRow, JobRow, JobStatus, LabelingRow, ModelRow, ModelType, TrainingRow

BASE = "/v1/registry/models"
TRAINED = b"trained-weights"
METRICS = {"mAP50": 0.7, "mAP50-95": 0.5, "precision": 0.8, "recall": 0.6}
METADATA = {"model_name": "defect-detector", "model_version": "1.0.0", "task": "detection", "line": "LINE3"}


@pytest.fixture
def client(app_db, tmp_path, monkeypatch):
    from app.routers import models as models_router

    monkeypatch.setattr(model_service, "REGISTRY_DIR", tmp_path / "registry")
    app = FastAPI()
    app.include_router(models_router.router)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def best_pt(tmp_path):
    path = tmp_path / "runs" / "1" / "weights" / "best.pt"
    path.parent.mkdir(parents=True)
    path.write_bytes(TRAINED)
    return path


def _init_training(best_pt, status=JobStatus.SUCCEEDED, task="detection"):
    now = datetime.now(timezone.utc)
    start = ModelRow(type=ModelType.INIT.value, model_name=f"start-{now.timestamp()}", model_version="1", task=task)
    with SessionLocal() as session:
        session.add(start)
        session.flush()
        record = InitTrainingRow(
            job=JobRow(
                job_type=JobType.INIT_TRAINING.value,
                status=status.value,
                payload={},
                result={"metrics": METRICS, "weights_path": str(best_pt)} if status == JobStatus.SUCCEEDED else None,
                created_at=now,
                updated_at=now,
            ),
            model_id=start.id,
            image_path="/data/images",
            label_path="/data/labels",
            params={"epochs": 1},
        )
        session.add(record)
        session.commit()
    return record.id


def _body(init_training_id, **metadata):
    return {"type": "line", "metadata": {**METADATA, **metadata}, "init_training_id": init_training_id}


def _assert_problem(res, status, code):
    assert res.status_code == status, res.text
    assert res.json()["type"] == f"https://mrx.internal/problems/{code}"
    assert res.json()["detail"]


def test_register_from_a_succeeded_init_training(client, best_pt, tmp_path):
    init_training_id = _init_training(best_pt)

    res = client.post(BASE, json=_body(init_training_id))

    assert res.status_code == 201, res.text
    model = res.json()
    stored = tmp_path / "registry" / "defect-detector" / "1.0.0" / "best.pt"
    assert model["type"] == "line"
    assert model["metadata"] == METADATA
    assert model["init_training_id"] == init_training_id
    assert model["metrics"] == METRICS
    assert model["weights_path"] == str(stored)
    assert stored.read_bytes() == TRAINED
    assert best_pt.read_bytes() == TRAINED


def test_registered_model_can_be_read_applied_and_exported(client, best_pt):
    model = client.post(BASE, json=_body(_init_training(best_pt))).json()
    url = f"{BASE}/{model['model_id']}"

    assert client.get(url).json() == model
    assert client.post(f"{url}/apply").json()["applied"] is True
    assert client.post(f"{url}/export", json={"format": "pt"}).status_code == 202


def test_missing_init_training_returns_404(client):
    res = client.post(BASE, json=_body(99999))
    _assert_problem(res, 404, "not-found")
    assert "99999" in res.json()["detail"]


@pytest.mark.parametrize("status", [JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.FAILED, JobStatus.CANCELED])
def test_unfinished_init_training_returns_409(client, best_pt, status):
    res = client.post(BASE, json=_body(_init_training(best_pt, status)))
    _assert_problem(res, 409, "conflict")
    assert status.value in res.json()["detail"]


def test_missing_weights_file_returns_404(client, best_pt):
    init_training_id = _init_training(best_pt)
    best_pt.unlink()

    _assert_problem(client.post(BASE, json=_body(init_training_id)), 404, "not-found")


def test_same_init_training_cannot_be_registered_twice(client, best_pt):
    init_training_id = _init_training(best_pt)
    client.post(BASE, json=_body(init_training_id))

    res = client.post(BASE, json=_body(init_training_id, model_version="2.0.0"))

    _assert_problem(res, 409, "conflict")
    assert "already registered" in res.json()["detail"]


def test_losing_a_race_for_the_same_init_training_returns_409(client, best_pt, tmp_path, monkeypatch):
    init_training_id = _init_training(best_pt)
    client.post(BASE, json=_body(init_training_id))

    checks = iter([None])  # the first check passes, as if the winner had not committed yet
    real = model_service._find_by_init_training
    monkeypatch.setattr(model_service, "_find_by_init_training", lambda *a: next(checks, real(*a)))

    res = client.post(BASE, json=_body(init_training_id, model_version="2.0.0"))

    _assert_problem(res, 409, "conflict")
    assert "already registered" in res.json()["detail"]
    assert not (tmp_path / "registry" / "defect-detector" / "2.0.0").exists()
    assert list((tmp_path / "registry" / ".partial").iterdir()) == []


def test_task_must_match_the_starting_model(client, best_pt):
    res = client.post(BASE, json=_body(_init_training(best_pt), task="classification"))

    _assert_problem(res, 409, "conflict")
    assert "classification" in res.json()["detail"]


def test_name_and_version_must_be_free(client, best_pt):
    client.post(BASE, json=_body(_init_training(best_pt)))

    res = client.post(BASE, json=_body(_init_training(best_pt)))

    _assert_problem(res, 409, "conflict")
    assert "already exists" in res.json()["detail"]


def test_failed_register_can_be_retried_after_fixing_the_name(client, best_pt):
    first, second = _init_training(best_pt), _init_training(best_pt)
    client.post(BASE, json=_body(first))
    assert client.post(BASE, json=_body(second)).status_code == 409

    assert client.post(BASE, json=_body(second, model_version="2.0.0")).status_code == 201


def test_listing_filters_by_line(client, best_pt):
    model = client.post(BASE, json=_body(_init_training(best_pt))).json()

    listed = client.get(BASE, params={"type": "line", "line": "LINE3"}).json()

    assert [m["model_id"] for m in listed] == [model["model_id"]]
    assert listed[0]["metrics"] == METRICS


def test_init_models_stay_free_of_the_one_model_rule(client, tmp_path):
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"x")
    for version in ("1", "2"):
        body = {
            "type": "init",
            "metadata": {"model_name": "base", "model_version": version, "task": "detection"},
            "weights_path": str(weights),
        }
        assert client.post(BASE, json=body).status_code == 201


def _training(best_pt, status=JobStatus.SUCCEEDED, task="detection"):
    now = datetime.now(timezone.utc)
    start = ModelRow(type=ModelType.LINE.value, model_name=f"used-{now.timestamp()}", model_version="1", task=task,
                     line="LINE3")
    with SessionLocal() as session:
        session.add(start)
        session.flush()
        labeling = LabelingRow(
            job=JobRow(job_type=JobType.LABELING.value, status="succeeded", payload={}, created_at=now, updated_at=now),
            model_id=start.id, source_path="/data", params={},
        )
        session.add(labeling)
        session.flush()
        record = TrainingRow(
            job=JobRow(job_type=JobType.TRAINING.value, status=status.value, payload={},
                       result={"metrics": METRICS, "weights_path": str(best_pt)} if status == JobStatus.SUCCEEDED else None,
                       created_at=now, updated_at=now),
            base_model_id=start.id, labeling_ids=[labeling.id], split={}, params={"epochs": 1}, output_dir="/runs",
        )
        session.add(record)
        session.commit()
    return record.id


def _body_from_training(training_id, **metadata):
    return {"type": "line", "metadata": {**METADATA, **metadata}, "training_id": training_id}


def test_register_from_a_succeeded_training(client, best_pt, tmp_path):
    training_id = _training(best_pt)

    res = client.post(BASE, json=_body_from_training(training_id))

    assert res.status_code == 201, res.text
    model = res.json()
    assert model["training_id"] == training_id
    assert model["metrics"] == METRICS
    assert (tmp_path / "registry" / "defect-detector" / "1.0.0" / "best.pt").read_bytes() == TRAINED


@pytest.mark.parametrize("status", [JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.FAILED, JobStatus.CANCELED])
def test_unfinished_training_returns_409(client, best_pt, status):
    _assert_problem(client.post(BASE, json=_body_from_training(_training(best_pt, status))), 409, "conflict")


def test_same_training_cannot_be_registered_twice(client, best_pt):
    training_id = _training(best_pt)
    client.post(BASE, json=_body_from_training(training_id))

    res = client.post(BASE, json=_body_from_training(training_id, model_version="2.0.0"))

    _assert_problem(res, 409, "conflict")
    assert "already registered" in res.json()["detail"]


def test_losing_a_race_for_the_same_training_returns_409(client, best_pt, monkeypatch):
    training_id = _training(best_pt)
    client.post(BASE, json=_body_from_training(training_id))
    checks = iter([None])
    real = model_service._find_by_training
    monkeypatch.setattr(model_service, "_find_by_training", lambda *a: next(checks, real(*a)))

    res = client.post(BASE, json=_body_from_training(training_id, model_version="2.0.0"))

    _assert_problem(res, 409, "conflict")
    assert "already registered" in res.json()["detail"]


def test_task_must_match_the_labeling_model(client, best_pt):
    res = client.post(BASE, json=_body_from_training(_training(best_pt), task="classification"))
    _assert_problem(res, 409, "conflict")
