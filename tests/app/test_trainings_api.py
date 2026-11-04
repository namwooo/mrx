from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.services import training as training_service
from core import jobs as job_repo
from core.db import SessionLocal
from core.jobs import JobType
from core.orm import (
    JobRow, JobStatus, LabelingImageRow, LabelingRow, ModelRow, ModelType, TrainingMetricRow, TrainingRow,
)

BASE = "/v1/trainings"
NAMES = {"0": "scratch"}
LABELS = [{"class_id": 0, "confidence": 0.95, "bbox_xywhn": [0.5, 0.5, 0.2, 0.2]}]
PARAMS = {"epochs": 100, "batch_size": 16, "learning_rate": 0.001, "image_size": 640}
SPLIT = {"train": 0.7, "val": 0.2, "test": 0.1}


@pytest.fixture
def client(app_db, tmp_path, monkeypatch):
    from app.routers import trainings

    monkeypatch.setattr(training_service, "TRAINING_DIR", tmp_path / "runs")
    app = FastAPI()
    app.include_router(trainings.router)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def model(app_db):
    row = ModelRow(type=ModelType.LINE.value, model_name="m", model_version="1", task="detection", line="L",
                   weights_path="/registry/best.pt")
    with SessionLocal() as session:
        session.add(row)
        session.commit()
    return row


def _labeling(model, status=JobStatus.SUCCEEDED, names=NAMES, count=60, suggested="approved"):
    now = datetime.now(timezone.utc)
    record = LabelingRow(
        job=JobRow(job_type=JobType.LABELING.value, status=status.value, payload={}, created_at=now, updated_at=now),
        model_id=model.id, source_path="/data", params={}, class_names=names,
    )
    with SessionLocal() as session:
        session.add(record)
        session.flush()
        session.add_all([
            LabelingImageRow(labeling_id=record.id, image_path=f"/data/{record.id}/{i}.png", width=10, height=10,
                             labels=LABELS, num_labels=1, suggested=suggested)
            for i in range(count)
        ])
        session.commit()
    return record.id


def _body(labeling_ids, model, **extra):
    return {"labeling_ids": labeling_ids, "base_model_id": model.id, **extra}


def _create(client, model, labeling_id=None, **extra):
    res = client.post(BASE, json=_body([labeling_id or _labeling(model)], model, **extra))
    assert res.status_code == 202, res.text
    return res.json()


def _finish(result=None, error=None):
    job = job_repo.claim_next_job(JobType.TRAINING, "w1")
    if error:
        job_repo.fail_job(job.id, "w1", error)
    else:
        job_repo.complete_job(job.id, "w1", result or {"metrics": {"mAP50": 0.7}, "weights_path": "/runs/best.pt"})
    return job


def _assert_problem(res, status, code):
    assert res.status_code == status, res.text
    assert res.json()["type"] == f"https://mrx.internal/problems/{code}"
    assert res.json()["detail"]


def test_create_is_queued(client, model):
    labeling = _labeling(model)

    res = client.post(BASE, json=_body([labeling], model))

    assert res.status_code == 202, res.text
    training = res.json()
    assert training["status"] == "queued"
    assert training["labeling_ids"] == [labeling]
    assert training["base_model_id"] == model.id
    assert training["split"] == SPLIT
    assert training["params"] == PARAMS
    assert training["progress"] == {"done": 0, "total": 100, "percent": 0.0}
    job = job_repo.list_jobs(job_type=JobType.TRAINING)[0]
    assert job.payload == {"training_id": training["training_id"]}


def test_create_from_scratch_needs_only_an_architecture(client, model):
    body = {"labeling_ids": [_labeling(model)], "params": {"model_arch": "yolo11n"}}

    training = client.post(BASE, json=body).json()

    assert "base_model_id" not in training
    assert training["params"]["model_arch"] == "yolo11n"


def test_create_keeps_the_given_split_and_params(client, model):
    split = {"train": 0.8, "val": 0.1, "test": 0.1}

    training = _create(client, model, split=split, params={"epochs": 5})

    assert training["split"] == split
    assert training["params"] == {**PARAMS, "epochs": 5}


def test_create_with_several_labelings(client, model):
    ids = [_labeling(model), _labeling(model)]
    assert client.post(BASE, json=_body(ids, model)).json()["labeling_ids"] == ids


@pytest.mark.parametrize("body", [
    {},
    {"labeling_ids": [], "base_model_id": 1},
    {"labeling_ids": [1]},                                                      # no base model, no architecture
    {"labeling_ids": [1, 1], "base_model_id": 1},
    {"labeling_ids": ["lbl_1"], "base_model_id": 1},
    {"labeling_ids": [1], "base_model_id": 1, "split": {"train": 0.5, "val": 0.2, "test": 0.1}},  # not 1
    {"labeling_ids": [1], "base_model_id": 1, "split": {"train": 1, "val": 0, "test": 0}},
    {"labeling_ids": [1], "base_model_id": 1, "split": {"train": 0, "val": 0.9, "test": 0.1}},
    {"labeling_ids": [1], "base_model_id": 1, "split": {"train": 1.2, "val": -0.3, "test": 0.1}},
    {"labeling_ids": [1], "base_model_id": 1, "params": {"epochs": 0}},
])
def test_create_rejects_invalid_body_with_400(client, body):
    _assert_problem(client.post(BASE, json=body), 400, "invalid-request")


def test_missing_labeling_or_base_model_returns_404(client, model):
    labeling = _labeling(model)
    _assert_problem(client.post(BASE, json=_body([labeling, 99999], model)), 404, "not-found")
    _assert_problem(client.post(BASE, json={"labeling_ids": [labeling], "base_model_id": 99999}), 404, "not-found")


@pytest.mark.parametrize("status", [JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.FAILED, JobStatus.CANCELED])
def test_unfinished_labeling_returns_409(client, model, status):
    res = client.post(BASE, json=_body([_labeling(model), _labeling(model, status)], model))
    _assert_problem(res, 409, "conflict")


def test_labelings_with_other_classes_return_409(client, model):
    ids = [_labeling(model), _labeling(model, names={"0": "scratch", "1": "dent"})]
    res = client.post(BASE, json=_body(ids, model))
    _assert_problem(res, 409, "conflict")
    assert "classes" in res.json()["detail"]


def test_no_usable_images_return_409_and_queue_nothing(client, model):
    res = client.post(BASE, json=_body([_labeling(model, suggested="review")], model))

    _assert_problem(res, 409, "conflict")
    assert job_repo.list_jobs(job_type=JobType.TRAINING) == []


def test_too_few_images_for_a_split_return_409(client, model):
    res = client.post(BASE, json=_body([_labeling(model, count=1)], model))
    _assert_problem(res, 409, "conflict")


def test_get_and_list(client, model):
    first = _labeling(model)
    a = _create(client, model, first)
    b = _create(client, model)
    _finish()

    assert client.get(f"{BASE}/{a['training_id']}").json()["status"] == "succeeded"
    assert [t["training_id"] for t in client.get(BASE).json()] == [b["training_id"], a["training_id"]]
    assert [t["training_id"] for t in client.get(BASE, params={"status": "queued"}).json()] == [b["training_id"]]
    assert [t["training_id"] for t in client.get(BASE, params={"limit": 1, "offset": 1}).json()] == [a["training_id"]]
    assert [t["training_id"] for t in client.get(BASE, params={"labeling_id": first}).json()] == [a["training_id"]]
    assert client.get(BASE, params={"labeling_id": 99999}).json() == []


@pytest.mark.parametrize("path", ["", "/metrics", "/result", "/logs"])
def test_reads_of_a_missing_training_return_404(client, path):
    _assert_problem(client.get(f"{BASE}/99999{path}"), 404, "not-found")


def test_invalid_id_returns_400(client):
    _assert_problem(client.get(f"{BASE}/abc"), 400, "invalid-request")
    _assert_problem(client.post(f"{BASE}/0/cancel"), 400, "invalid-request")


def test_metrics_are_listed_in_order(client, model):
    training = _create(client, model)
    now = datetime.now(timezone.utc)
    with SessionLocal() as session:
        session.add_all([
            TrainingMetricRow(training_id=training["training_id"], epoch=epoch, name=name, value=value, recorded_at=now)
            for epoch, name, value in ((2, "val/mAP50", 0.7), (1, "val/mAP50", 0.5), (1, "val/recall", 0.4))
        ])
        session.commit()

    points = client.get(f"{BASE}/{training['training_id']}/metrics").json()

    assert [(p["epoch"], p["name"], p["value"]) for p in points] == [
        (1, "val/mAP50", 0.5), (1, "val/recall", 0.4), (2, "val/mAP50", 0.7),
    ]
    assert points[0]["recorded_at"].endswith("Z")


def test_result_after_success(client, model):
    training = _create(client, model)
    _finish()

    assert client.get(f"{BASE}/{training['training_id']}/result").json() == {
        "training_id": training["training_id"], "metrics": {"mAP50": 0.7}, "weights_path": "/runs/best.pt",
    }


def test_result_before_success_returns_409(client, model):
    training = _create(client, model)
    _assert_problem(client.get(f"{BASE}/{training['training_id']}/result"), 409, "conflict")
    _finish(error="boom")
    _assert_problem(client.get(f"{BASE}/{training['training_id']}/result"), 409, "conflict")


def test_logs(client, model):
    training = _create(client, model)
    url = f"{BASE}/{training['training_id']}/logs"
    assert client.get(url).text == ""

    with SessionLocal() as session:
        log = session.get(TrainingRow, training["training_id"]).output_dir + "/train.log"
    import pathlib
    pathlib.Path(log).parent.mkdir(parents=True)
    pathlib.Path(log).write_text("one\ntwo\nthree\n")

    res = client.get(url)
    assert res.headers["content-type"].startswith("text/plain")
    assert res.text == "one\ntwo\nthree\n"
    assert client.get(url, params={"tail": 2}).text == "two\nthree\n"
    assert client.get(url, params={"tail": 0}).status_code == 400


def test_cancel_a_queued_training(client, model):
    training = _create(client, model)

    res = client.post(f"{BASE}/{training['training_id']}/cancel")

    assert res.status_code == 200, res.text
    assert res.json()["status"] == "canceled"
    assert "finished_at" in res.json()
    assert job_repo.claim_next_job(JobType.TRAINING, "w1") is None


def test_cancel_a_running_training(client, model):
    training = _create(client, model)
    job_repo.claim_next_job(JobType.TRAINING, "w1")

    assert client.post(f"{BASE}/{training['training_id']}/cancel").json()["status"] == "canceled"


def test_a_worker_cannot_overwrite_a_cancel(client, model):
    training = _create(client, model)
    job = job_repo.claim_next_job(JobType.TRAINING, "w1")
    client.post(f"{BASE}/{training['training_id']}/cancel")

    assert job_repo.complete_job(job.id, "w1", {"x": 1}) is False
    assert client.get(f"{BASE}/{training['training_id']}").json()["status"] == "canceled"


@pytest.mark.parametrize("finish", ["success", "failure", "cancel"])
def test_cancel_a_finished_training_returns_409(client, model, finish):
    training = _create(client, model)
    url = f"{BASE}/{training['training_id']}/cancel"
    if finish == "cancel":
        client.post(url)
    else:
        _finish(error="boom" if finish == "failure" else None)

    res = client.post(url)

    _assert_problem(res, 409, "conflict")
    assert res.json()["title"] == "이미 끝난 작업"


def test_cancel_a_missing_training_returns_404(client):
    _assert_problem(client.post(f"{BASE}/99999/cancel"), 404, "not-found")
