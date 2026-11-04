from datetime import datetime
from itertools import count

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from app.services import model as model_service
from core import jobs as job_repo
from core.db import SessionLocal
from core.jobs import JobType
from core.orm import ModelRow, ModelType

BASE = "/v1/registry/models"
WEIGHTS = b"weights-v1"
METADATA = {"model_name": "defect-detector", "model_version": "1.4.0", "task": "detection"}
MISSING_ID = 99999
versions = count(1)


@pytest.fixture
def client(app_db, tmp_path, monkeypatch):
    from app.routers import models as models_router

    monkeypatch.setattr(model_service, "REGISTRY_DIR", tmp_path / "registry")
    app = FastAPI()
    app.include_router(models_router.router)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def weights(tmp_path):
    path = tmp_path / "pretrained" / "yolo11m.pt"
    path.parent.mkdir()
    path.write_bytes(WEIGHTS)
    return path


def _body(weights, **metadata):
    return {"type": "init", "metadata": {**METADATA, **metadata}, "weights_path": str(weights)}


def _create(client, weights, **metadata):
    res = client.post(BASE, json=_body(weights, **metadata))
    assert res.status_code == 201, res.text
    return res.json()


def _line_model(name="line-model", line="LINE3", task="detection", training_id=None):
    model = ModelRow(
        type=ModelType.LINE.value,
        model_name=name,
        model_version=str(next(versions)),
        task=task,
        line=line,
        training_id=training_id,
        dataset_id=8,
        metrics={"mAP50": 0.742},
    )
    with SessionLocal() as session:
        session.add(model)
        session.commit()
    return model


def _stored_models():
    with SessionLocal() as session:
        return list(session.scalars(select(ModelRow)))


def _assert_problem(res, status, code):
    assert res.status_code == status, res.text
    problem = res.json()
    assert problem["type"] == f"https://mrx.internal/problems/{code}"
    assert problem["status"] == status
    assert problem["title"]
    assert problem["detail"]
    assert problem["instance"] == res.request.url.path


def test_register_init_model(client, weights, tmp_path):
    model = _create(client, weights)

    assert model["model_id"] >= 1
    assert model["type"] == "init"
    assert model["metadata"] == METADATA
    assert model["applied"] is False
    assert model["exports"] == []
    assert datetime.fromisoformat(model["created_at"]).utcoffset().total_seconds() == 0
    assert not {"line", "training_id", "applied_at", "metrics"} & (set(model) | set(model["metadata"]))


def test_register_copies_weights_into_the_registry(client, weights, tmp_path):
    model = _create(client, weights)

    stored = tmp_path / "registry" / "defect-detector" / "1.4.0" / "best.pt"
    assert model["weights_path"] == str(stored)
    assert stored.read_bytes() == WEIGHTS
    assert weights.read_bytes() == WEIGHTS
    assert [p.name for p in stored.parent.iterdir()] == ["best.pt"]


def test_register_keeps_line_of_init_model(client, weights):
    assert _create(client, weights, line="LINE3")["metadata"]["line"] == "LINE3"


@pytest.mark.parametrize("change", [
    {"type": None},                                                    # type missing
    {"type": "other"},                                                 # unknown type
    {"metadata": None},                                                # metadata missing
    {"weights_path": None},                                            # init needs weights
    {"weights_path": ""},                                              # empty path
    {"training_id": 1},                                                # init cannot have a training
    {"init_training_id": 1},                                           # init cannot have an init training
    {"metadata": {**METADATA, "task": ""}},                            # empty task
    {"metadata": {**METADATA, "model_name": "../../etc"}},             # would leave the registry
    {"metadata": {**METADATA, "model_name": ".."}},
    {"metadata": {**METADATA, "model_version": "1/2"}},
    {"type": "line", "weights_path": None, "training_id": 1},          # line needs metadata.line
    {"type": "line", "weights_path": None,
     "metadata": {**METADATA, "line": "LINE3"}},                       # line needs a source
    {"type": "line", "weights_path": None, "metadata": {**METADATA, "line": "LINE3"},
     "training_id": 1, "init_training_id": 2},                         # line takes only one source
    {"type": "line", "training_id": 1,
     "metadata": {**METADATA, "line": "LINE3"}},                       # line takes no weights_path
    {"type": "line", "weights_path": None, "training_id": "trn_1",
     "metadata": {**METADATA, "line": "LINE3"}},                       # ids are integers
])
def test_register_rejects_invalid_body_with_400(client, weights, change):
    body = {k: v for k, v in {**_body(weights), **change}.items() if v is not None}
    _assert_problem(client.post(BASE, json=body), 400, "invalid-request")


def test_register_rejects_broken_json_with_400(client):
    res = client.post(BASE, content="{", headers={"content-type": "application/json"})
    _assert_problem(res, 400, "invalid-request")


def test_validation_detail_names_the_field(client):
    res = client.post(BASE, json={"type": "init"})
    assert "metadata" in res.json()["detail"]


def test_register_missing_weights_returns_404(client, tmp_path, app_db):
    res = client.post(BASE, json=_body(tmp_path / "nope.pt"))
    _assert_problem(res, 404, "not-found")
    assert _stored_models() == []


def test_register_folder_as_weights_returns_404(client, tmp_path):
    _assert_problem(client.post(BASE, json=_body(tmp_path)), 404, "not-found")


def test_register_duplicate_returns_409_and_keeps_first_file(client, weights, tmp_path):
    first = _create(client, weights)
    weights.write_bytes(b"other-weights")

    _assert_problem(client.post(BASE, json=_body(weights)), 409, "conflict")

    stored = tmp_path / "registry" / "defect-detector" / "1.4.0" / "best.pt"
    assert stored.read_bytes() == WEIGHTS
    assert [m.id for m in _stored_models()] == [first["model_id"]]


def test_register_losing_a_race_returns_409_and_keeps_first_file(client, weights, tmp_path, monkeypatch):
    _create(client, weights)
    weights.write_bytes(b"other-weights")
    monkeypatch.setattr(model_service, "_find_model", lambda *_: None)  # passed the check before the winner committed

    _assert_problem(client.post(BASE, json=_body(weights)), 409, "conflict")

    stored = tmp_path / "registry" / "defect-detector" / "1.4.0"
    assert (stored / "best.pt").read_bytes() == WEIGHTS
    assert [p.name for p in stored.iterdir()] == ["best.pt"]


def test_same_name_with_new_version_is_allowed(client, weights):
    _create(client, weights)
    assert _create(client, weights, model_version="1.5.0")["metadata"]["model_version"] == "1.5.0"


def test_register_from_a_missing_retraining_returns_404(client):
    body = {"type": "line", "metadata": {**METADATA, "line": "LINE3"}, "training_id": 99999}
    _assert_problem(client.post(BASE, json=body), 404, "not-found")


def test_get_model(client, weights):
    model = _create(client, weights)
    assert client.get(f"{BASE}/{model['model_id']}").json() == model


def test_get_line_model_shows_its_sources(client):
    record = _line_model(training_id=7)
    model = client.get(f"{BASE}/{record.id}").json()

    assert model["training_id"] == 7
    assert model["dataset_id"] == 8
    assert model["metrics"] == {"mAP50": 0.742}
    assert model["metadata"]["line"] == "LINE3"


def test_get_missing_model_returns_404(client):
    res = client.get(f"{BASE}/{MISSING_ID}")
    _assert_problem(res, 404, "not-found")
    assert res.json()["detail"] == f"model '{MISSING_ID}' not found"


@pytest.mark.parametrize("model_id", ["abc", "0", "mdl_1"])
def test_invalid_model_id_returns_400(client, model_id):
    _assert_problem(client.get(f"{BASE}/{model_id}"), 400, "invalid-request")
    _assert_problem(client.post(f"{BASE}/{model_id}/apply"), 400, "invalid-request")
    _assert_problem(client.post(f"{BASE}/{model_id}/export", json={"format": "pt"}), 400, "invalid-request")


def test_ids_count_up_and_are_never_reused(client, weights):
    first = _create(client, weights)["model_id"]
    second = _create(client, weights, model_version="1.5.0")["model_id"]
    assert second == first + 1

    with SessionLocal() as session:
        session.delete(session.get(ModelRow, second))
        session.commit()

    assert _create(client, weights, model_version="1.6.0")["model_id"] == second + 1


def test_list_newest_first_with_pagination(client, weights):
    ids = [_create(client, weights, model_version=f"1.{i}.0")["model_id"] for i in range(5)]
    newest_first = list(reversed(ids))

    assert [m["model_id"] for m in client.get(BASE).json()] == newest_first
    page2 = client.get(BASE, params={"limit": 2, "offset": 2}).json()
    assert [m["model_id"] for m in page2] == newest_first[2:4]


def test_list_filters(client, weights):
    init = _create(client, weights)
    other = _line_model(name="other", line="LINE1", task="classification")
    applied = _line_model(name="applied", line="LINE3")
    client.post(f"{BASE}/{applied.id}/apply")

    def ids(**params):
        return {m["model_id"] for m in client.get(BASE, params=params).json()}

    assert ids(type="init") == {init["model_id"]}
    assert ids(type="line") == {other.id, applied.id}
    assert ids(model_name="other") == {other.id}
    assert ids(line="LINE3") == {applied.id}
    assert ids(task="classification") == {other.id}
    assert ids(applied="true") == {applied.id}
    assert ids(applied="false") == {init["model_id"], other.id}
    assert ids(line="LINE3", task="classification") == set()


@pytest.mark.parametrize("params", [
    {"type": "other"}, {"applied": "maybe"}, {"limit": 0}, {"limit": 501}, {"offset": -1},
])
def test_list_rejects_invalid_query_with_400(client, params):
    _assert_problem(client.get(BASE, params=params), 400, "invalid-request")


def test_apply_model(client):
    record = _line_model()
    res = client.post(f"{BASE}/{record.id}/apply", json={"comment": "mAP 3% 향상"})

    assert res.status_code == 200
    assert res.json()["applied"] is True
    assert datetime.fromisoformat(res.json()["applied_at"]).utcoffset().total_seconds() == 0
    assert "comment" not in res.json()
    assert _stored_models()[0].apply_comment == "mAP 3% 향상"


def test_apply_without_body(client):
    record = _line_model()
    assert client.post(f"{BASE}/{record.id}/apply").json()["applied"] is True


def test_apply_replaces_model_of_the_same_line_and_task_only(client):
    old = _line_model(name="old")
    other_line = _line_model(name="other-line", line="LINE1")
    other_task = _line_model(name="other-task", task="classification")
    new = _line_model(name="new")
    for record in (old, other_line, other_task, new):
        client.post(f"{BASE}/{record.id}/apply")

    applied = {m["model_id"] for m in client.get(BASE, params={"applied": "true"}).json()}
    assert applied == {new.id, other_line.id, other_task.id}


def test_models_without_a_line_share_one_scope(client, weights):
    first = _create(client, weights)
    second = _create(client, weights, model_version="1.5.0")
    client.post(f"{BASE}/{first['model_id']}/apply")
    client.post(f"{BASE}/{second['model_id']}/apply")

    applied = client.get(BASE, params={"applied": "true"}).json()
    assert [m["model_id"] for m in applied] == [second["model_id"]]


def test_apply_twice_keeps_it_applied(client):
    record = _line_model()
    client.post(f"{BASE}/{record.id}/apply")
    assert client.post(f"{BASE}/{record.id}/apply").json()["applied"] is True


def test_apply_writes_even_when_the_loaded_row_is_stale(client):
    model = _line_model(name="stale")
    other = _line_model(name="other")
    client.post(f"{BASE}/{model.id}/apply")

    with SessionLocal() as session:
        assert session.get(ModelRow, model.id).applied is True
        client.post(f"{BASE}/{other.id}/apply")  # another request takes over while the row is loaded
        model_service.apply_model(session, model.id, None)

    applied = client.get(BASE, params={"applied": "true"}).json()
    assert [m["model_id"] for m in applied] == [model.id]


def test_apply_missing_model_returns_404(client):
    _assert_problem(client.post(f"{BASE}/{MISSING_ID}/apply"), 404, "not-found")


def test_db_refuses_two_applied_models_in_one_scope(client, app_db):
    a, b = _line_model(name="a"), _line_model(name="b")
    with app_db.begin() as conn:
        conn.execute(update(ModelRow).where(ModelRow.id == a.id).values(applied=True))
    with pytest.raises(IntegrityError), app_db.begin() as conn:
        conn.execute(update(ModelRow).where(ModelRow.id == b.id).values(applied=True))


def test_export_queues_a_job(client, weights):
    model = _create(client, weights)
    res = client.post(f"{BASE}/{model['model_id']}/export", json={"format": "onnx"})

    assert res.status_code == 202
    assert res.json()["format"] == "onnx"
    assert res.json()["status"] == "queued"
    assert "file_path" not in res.json()

    job = job_repo.list_jobs(job_type=JobType.MODEL_EXPORT)[0]
    assert job.payload == {"model_id": model["model_id"], "format": "onnx"}


def test_export_rejects_unknown_format_with_400(client, weights):
    model = _create(client, weights)
    res = client.post(f"{BASE}/{model['model_id']}/export", json={"format": "tflite"})
    _assert_problem(res, 400, "invalid-request")


def test_export_missing_model_returns_404(client):
    res = client.post(f"{BASE}/{MISSING_ID}/export", json={"format": "onnx"})
    _assert_problem(res, 404, "not-found")
    assert job_repo.list_jobs(job_type=JobType.MODEL_EXPORT) == []


def test_exports_show_job_outcome_newest_first(client, weights):
    model = _create(client, weights)
    other = _create(client, weights, model_version="1.5.0")
    url = f"{BASE}/{model['model_id']}/export"
    for export_format in ("pt", "onnx", "torchscript"):
        client.post(url, json={"format": export_format})
    client.post(f"{BASE}/{other['model_id']}/export", json={"format": "pt"})

    pt, onnx, _ = reversed(job_repo.list_jobs(job_type=JobType.MODEL_EXPORT)[1:])
    job_repo.claim_next_job(JobType.MODEL_EXPORT, "w1")
    job_repo.complete_job(pt.id, "w1", {"file_path": "/registry/best.pt", "file_hash": "sha256:abc"})
    job_repo.claim_next_job(JobType.MODEL_EXPORT, "w1")
    job_repo.fail_job(onnx.id, "w1", "ValueError: broken")

    exports = client.get(f"{BASE}/{model['model_id']}").json()["exports"]
    assert [e["format"] for e in exports] == ["torchscript", "onnx", "pt"]
    assert [e["status"] for e in exports] == ["queued", "failed", "succeeded"]
    assert exports[1]["error"] == "ValueError: broken"
    assert exports[2]["file_path"] == "/registry/best.pt"
    assert exports[2]["file_hash"] == "sha256:abc"

    by_id = {m["model_id"]: m for m in client.get(BASE).json()}
    assert by_id[model["model_id"]]["exports"] == exports
    assert [e["format"] for e in by_id[other["model_id"]]["exports"]] == ["pt"]


def test_openapi_documents_problems_instead_of_422():
    from app.routers import models as models_router

    app = FastAPI()
    app.include_router(models_router.router)

    for path in app.openapi()["paths"].values():
        for operation in path.values():
            assert "422" not in operation["responses"]
            assert "4XX" in operation["responses"]
