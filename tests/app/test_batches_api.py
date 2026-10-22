from sqlalchemy import update

from core.jobs import jobs

BATCH = {"model": "yolov11m.pt", "input_path": "/data/imgs", "params": {"conf": 0.25}}

def _create(client, **overrides):
    res = client.post("/batches/", json={**BATCH, **overrides})
    assert res.status_code == 201
    return res.json()


def _set_status(engine, job_id, status):
    with engine.begin() as conn:
        conn.execute(update(jobs).where(jobs.c.id == job_id).values(status=status))


def test_create_batch_job(client):
    job = _create(client)
    assert job["status"] == "queued"
    assert job["payload"] == BATCH


def test_same_idempotency_key_returns_same_job(client):
    first = _create(client, idempotency_key="k1")
    second = _create(client, idempotency_key="k1")
    assert first["id"] == second["id"]


def test_list_filters_by_status(client, app_db):
    a = _create(client)
    _create(client)
    _set_status(app_db, a["id"], "running")

    assert len(client.get("/batches/").json()) == 2
    assert [j["id"] for j in client.get("/batches/", params={"status": "running"}).json()] == [a["id"]]
    assert client.get("/batches/", params={"status": "done"}).status_code == 422


def test_list_is_newest_first(client):
    ids = [_create(client)["id"] for _ in range(3)]
    assert [j["id"] for j in client.get("/batches/").json()] == list(reversed(ids))


def test_list_pagination(client):
    ids = [_create(client)["id"] for _ in range(5)]
    newest_first = list(reversed(ids))

    page1 = client.get("/batches/", params={"limit": 2, "offset": 0}).json()
    page2 = client.get("/batches/", params={"limit": 2, "offset": 2}).json()
    page3 = client.get("/batches/", params={"limit": 2, "offset": 4}).json()

    assert [j["id"] for j in page1] == newest_first[0:2]
    assert [j["id"] for j in page2] == newest_first[2:4]
    assert [j["id"] for j in page3] == newest_first[4:5]


def test_get_batch(client):
    job = _create(client)
    print(job)
    assert client.get(f"/batches/{job["id"]}").json()["id"] == job["id"]


def test_get_missing_batch_returns_404(client):
    assert client.get("/batches/99999").status_code == 404


def test_get_non_integer_id_returns_422(client):
    assert client.get("/batches/abc").status_code == 422


def test_get_returns_current_status(client, app_db):
    job = _create(client)
    _set_status(app_db, job["id"], "running")
    assert client.get(f"/batches/{job['id']}").json()["status"] == "running"


def test_cancel_queued_batch(client):
    job = _create(client)
    res = client.delete(f"/batches/{job['id']}")
    assert res.status_code == 200
    assert res.json()["status"] == "canceled"
    assert res.json()["finished_at"] is not None


def test_cancel_twice_returns_canceled_both_times(client):
    job = _create(client)
    assert client.delete(f"/batches/{job['id']}").json()["status"] == "canceled"
    second = client.delete(f"/batches/{job['id']}")
    assert second.status_code == 200
    assert second.json()["status"] == "canceled"


def test_canceled_batch_is_not_picked_as_queued(client):
    job = _create(client)
    client.delete(f"/batches/{job['id']}")
    assert client.get("/batches/", params={"status": "queued"}).json() == []


def test_cancel_finished_batch_returns_409(client, app_db):
    for status in ("succeeded", "failed"):
        job = _create(client)
        _set_status(app_db, job["id"], status)
        res = client.delete(f"/batches/{job['id']}")
        assert res.status_code == 409
        assert status in res.json()["detail"]


def test_cancel_missing_or_invalid_id(client):
    assert client.delete("/batches/99999").status_code == 404
    assert client.delete("/batches/0").status_code == 422


def test_cancel_running_batch_returns_409(client, app_db):
    job = _create(client)
    _set_status(app_db, job["id"], "running")
    assert client.delete(f"/batches/{job['id']}").status_code == 409
