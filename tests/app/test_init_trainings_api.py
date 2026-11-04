from datetime import datetime, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import update

from core import jobs as job_repo
from core.db import SessionLocal
from core.jobs import JobType, jobs
from core.orm import InitTrainingRow, ModelRow, ModelType

BASE = "/v1/init-trainings"
DEFAULT_PARAMS = {"epochs": 100, "batch_size": 16, "learning_rate": 0.001, "image_size": 640}


@pytest.fixture
def client(app_db):
    from app.routers import init_trainings

    app = FastAPI()
    app.include_router(init_trainings.router)
    with TestClient(app) as c:
        yield c


def _model(model_type=ModelType.INIT):
    model = ModelRow(
        type=model_type.value,
        model_name="yolo11m",
        model_version=str(model_type.value),
        task="detection",
        line="LINE3" if model_type == ModelType.LINE else None,
        weights_path="/registry/best.pt",
    )
    with SessionLocal() as session:
        session.add(model)
        session.commit()
    return model


@pytest.fixture
def body(app_db, tmp_path):
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    return {"model_id": _model().id, "image_path": str(images), "label_path": str(labels)}


def _create(client, body, **overrides):
    res = client.post(BASE, json={**body, **overrides})
    assert res.status_code == 202, res.text
    return res.json()


def _assert_problem(res, status, code):
    assert res.status_code == status, res.text
    assert res.json()["type"] == f"https://mrx.internal/problems/{code}"
    assert res.json()["status"] == status
    assert res.json()["detail"]
    assert res.json()["instance"] == res.request.url.path


def test_create_is_queued(client, body):
    training = _create(client, body)

    assert training["init_training_id"] >= 1
    assert training["status"] == "queued"
    assert training["model_id"] == body["model_id"]
    assert training["image_path"] == body["image_path"]
    assert training["label_path"] == body["label_path"]
    assert training["params"] == DEFAULT_PARAMS
    assert training["progress"] == {"done": 0, "total": 100, "percent": 0.0}
    assert datetime.fromisoformat(training["created_at"]).utcoffset() == timedelta(0)
    assert not {"metrics", "weights_path", "error", "started_at", "finished_at"} & set(training)


def test_create_queues_exactly_one_job_for_its_record(client, body):
    training = _create(client, body)

    queued = job_repo.list_jobs(job_type=JobType.INIT_TRAINING)
    assert [j.payload for j in queued] == [{"init_training_id": training["init_training_id"]}]
    assert queued[0].status == "queued"


def test_create_keeps_given_params_and_fills_the_rest(client, body):
    classes = [{"class_id": 0, "class_name": "scratch"}, {"class_id": 1, "class_name": "dent"}]
    params = {"epochs": 5, "model_arch": "yolo11m", "classes": classes}

    training = _create(client, body, params=params)

    assert training["params"] == {**DEFAULT_PARAMS, **params}
    assert training["progress"]["total"] == 5


@pytest.mark.parametrize("change", [
    {"model_id": None},
    {"image_path": None},
    {"label_path": None},
    {"image_path": ""},
    {"params": {"epochs": 0}},
    {"params": {"batch_size": 0}},
    {"params": {"learning_rate": 0}},
    {"params": {"image_size": 16}},
    {"params": {"classes": [{"class_id": -1, "class_name": "a"}]}},
    {"params": {"classes": [{"class_id": 0}]}},
    {"params": {"classes": [{"class_id": 0, "class_name": "a"}, {"class_id": 0, "class_name": "b"}]}},
])
def test_create_rejects_invalid_body_with_400(client, body, change):
    payload = {k: v for k, v in {**body, **change}.items() if v is not None}
    _assert_problem(client.post(BASE, json=payload), 400, "invalid-request")


def test_create_with_missing_model_returns_404(client, body):
    _assert_problem(client.post(BASE, json={**body, "model_id": 99999}), 404, "not-found")


def test_create_with_a_model_id_that_is_no_number_returns_400(client, body):
    _assert_problem(client.post(BASE, json={**body, "model_id": "mdl_1"}), 400, "invalid-request")


@pytest.mark.parametrize("field", ["image_path", "label_path"])
def test_create_with_missing_folder_returns_404(client, body, tmp_path, field):
    res = client.post(BASE, json={**body, field: str(tmp_path / "nope")})
    _assert_problem(res, 404, "not-found")
    assert field in res.json()["detail"]


def test_create_with_a_file_as_folder_returns_404(client, body, tmp_path):
    file = tmp_path / "file.txt"
    file.write_text("x")
    _assert_problem(client.post(BASE, json={**body, "image_path": str(file)}), 404, "not-found")


def test_create_with_line_model_returns_409(client, body):
    line_model = _model(ModelType.LINE)
    _assert_problem(client.post(BASE, json={**body, "model_id": line_model.id}), 409, "conflict")
    assert job_repo.list_jobs(job_type=JobType.INIT_TRAINING) == []


def test_rejected_request_leaves_no_record(client, body, tmp_path):
    client.post(BASE, json={**body, "image_path": str(tmp_path / "nope")})

    with SessionLocal() as session:
        assert session.query(InitTrainingRow).count() == 0


def test_get_returns_the_created_training(client, body):
    training = _create(client, body)
    assert client.get(f"{BASE}/{training['init_training_id']}").json() == training


def test_get_missing_returns_404(client):
    res = client.get(f"{BASE}/99999")
    _assert_problem(res, 404, "not-found")
    assert res.json()["detail"] == "init training '99999' not found"


@pytest.mark.parametrize("init_training_id", ["abc", "0", "itr_1"])
def test_get_with_invalid_id_returns_400(client, init_training_id):
    _assert_problem(client.get(f"{BASE}/{init_training_id}"), 400, "invalid-request")


def test_get_shows_running_progress(client, body):
    training = _create(client, body, params={"epochs": 8})
    job = job_repo.claim_next_job(JobType.INIT_TRAINING, "w1")
    with SessionLocal() as session:
        session.execute(update(InitTrainingRow).values(epochs_done=3))
        session.commit()

    got = client.get(f"{BASE}/{training['init_training_id']}").json()

    assert got["status"] == "running"
    assert got["progress"] == {"done": 3, "total": 8, "percent": 37.5}
    assert datetime.fromisoformat(got["started_at"]).utcoffset() == timedelta(0)
    assert job.id


def test_get_shows_metrics_and_weights_when_succeeded(client, body):
    training = _create(client, body)
    job = job_repo.claim_next_job(JobType.INIT_TRAINING, "w1")
    job_repo.complete_job(job.id, "w1", {"metrics": {"mAP50": 0.7}, "weights_path": "/runs/best.pt"})

    got = client.get(f"{BASE}/{training['init_training_id']}").json()

    assert got["status"] == "succeeded"
    assert got["metrics"] == {"mAP50": 0.7}
    assert got["weights_path"] == "/runs/best.pt"
    assert "error" not in got
    assert "finished_at" in got


def test_get_shows_the_reason_when_failed(client, body):
    training = _create(client, body)
    job = job_repo.claim_next_job(JobType.INIT_TRAINING, "w1")
    job_repo.fail_job(job.id, "w1", "RuntimeError: no classes")

    got = client.get(f"{BASE}/{training['init_training_id']}").json()

    assert got["status"] == "failed"
    assert got["error"] == "RuntimeError: no classes"
    assert not {"metrics", "weights_path"} & set(got)


def test_openapi_documents_problems_instead_of_422():
    from app.routers import init_trainings

    app = FastAPI()
    app.include_router(init_trainings.router)

    for path in app.openapi()["paths"].values():
        for operation in path.values():
            assert "422" not in operation["responses"]
            assert "4XX" in operation["responses"]
