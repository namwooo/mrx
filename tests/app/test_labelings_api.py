import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import update

from core import jobs as job_repo
from core.jobs import JobType, jobs

LABELING = {"model": "/models/yolo11/yolo11m.pt", "input_path": "/data/raw/2026-10"}
DEFAULT_PARAMS = {"conf": 0.25, "iou": 0.7, "max_det": 300}


@pytest.fixture
def client(app_db):
    from app.routers import labelings

    app = FastAPI()
    app.include_router(labelings.router)
    with TestClient(app) as c:
        yield c


def _create(client, **overrides):
    res = client.post("/labelings/", json={**LABELING, **overrides})
    assert res.status_code == 201, res.text
    return res.json()


def _set_status(engine, job_id, status):
    with engine.begin() as conn:
        conn.execute(update(jobs).where(jobs.c.id == job_id).values(status=status))


def test_create_labeling_job(client):
    job = _create(client)
    assert job["job_type"] == "labeling"
    assert job["status"] == "queued"
    assert job["payload"] == {**LABELING, "params": DEFAULT_PARAMS}


def test_create_keeps_given_params_and_fills_the_rest(client):
    job = _create(client, params={"conf": 0.1})
    assert job["payload"]["params"] == {**DEFAULT_PARAMS, "conf": 0.1}


@pytest.mark.parametrize("body", [
    {"input_path": "/data/raw"},                          # model missing
    {"model": "/models/m.pt"},                            # input_path missing
    {**LABELING, "params": {"conf": 1.5}},                # conf out of range
    {**LABELING, "params": {"max_det": 0}},               # max_det below 1
    {**LABELING, "idempotency_key": "k" * 129},           # key too long
])
def test_create_rejects_invalid_body(client, body):
    assert client.post("/labelings/", json=body).status_code == 422


def test_same_idempotency_key_returns_same_job(client):
    first = _create(client, idempotency_key="k1")
    second = _create(client, idempotency_key="k1")
    assert first["id"] == second["id"]


def test_list_shows_only_labeling_jobs(client):
    labeling = _create(client)
    job_repo.enqueue_job(JobType.BATCH, {"model": "m.pt", "input_path": "/x", "params": {}})

    assert [j["id"] for j in client.get("/labelings/").json()] == [labeling["id"]]


def test_list_filters_by_status_newest_first(client, app_db):
    a, b, c = (_create(client) for _ in range(3))
    _set_status(app_db, b["id"], "running")

    assert [j["id"] for j in client.get("/labelings/").json()] == [c["id"], b["id"], a["id"]]
    assert [j["id"] for j in client.get("/labelings/", params={"status": "queued"}).json()] == [c["id"], a["id"]]
    assert client.get("/labelings/", params={"status": "done"}).status_code == 422


def test_list_pagination(client):
    ids = [_create(client)["id"] for _ in range(5)]
    newest_first = list(reversed(ids))

    page2 = client.get("/labelings/", params={"limit": 2, "offset": 2}).json()
    assert [j["id"] for j in page2] == newest_first[2:4]


def test_get_labeling_job(client, app_db):
    job = _create(client)
    _set_status(app_db, job["id"], "running")

    got = client.get(f"/labelings/{job['id']}").json()
    assert got["id"] == job["id"]
    assert got["status"] == "running"


def test_get_returns_404_for_missing_or_other_job_type(client):
    batch = job_repo.enqueue_job(JobType.BATCH, {"model": "m.pt", "input_path": "/x", "params": {}})

    assert client.get("/labelings/99999").status_code == 404
    assert client.get(f"/labelings/{batch.id}").status_code == 404
    assert client.get("/labelings/abc").status_code == 422


def test_cancel_queued_labeling_job(client):
    job = _create(client)
    res = client.delete(f"/labelings/{job['id']}")
    assert res.status_code == 200
    assert res.json()["status"] == "canceled"
    assert res.json()["finished_at"] is not None

    # canceling again keeps the same answer
    assert client.delete(f"/labelings/{job['id']}").json()["status"] == "canceled"


@pytest.mark.parametrize("status", ["running", "succeeded", "failed"])
def test_cancel_started_or_finished_job_returns_409(client, app_db, status):
    job = _create(client)
    _set_status(app_db, job["id"], status)

    res = client.delete(f"/labelings/{job['id']}")
    assert res.status_code == 409
    assert status in res.json()["detail"]


def test_cancel_missing_or_invalid_id(client):
    batch = job_repo.enqueue_job(JobType.BATCH, {"model": "m.pt", "input_path": "/x", "params": {}})

    assert client.delete("/labelings/99999").status_code == 404
    assert client.delete(f"/labelings/{batch.id}").status_code == 404
    assert client.delete("/labelings/0").status_code == 422
