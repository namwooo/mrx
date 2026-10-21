BATCH = {"model": "yolov11m.pt", "input_path": "/data/imgs", "params": {"conf": 0.25}}

def _create(client, **overrides):
    res = client.post("/batches/", json={**BATCH, **overrides})
    assert res.status_code == 201
    return res.json()


def test_create_batch_job(client):
    job = _create(client)
    assert job["status"] == "queued"
    assert job["payload"] == BATCH


def test_same_idempotency_key_returns_same_job(client):
    first = _create(client, idempotency_key="k1")
    second = _create(client, idempotency_key="k1")
    assert first["id"] == second["id"]