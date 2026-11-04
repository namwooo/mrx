from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core import jobs as job_repo
from core.db import SessionLocal
from core.jobs import JobType
from core.orm import JobRow, JobStatus, LabelingImageRow, LabelingRow, ModelRow, ModelType

BASE = "/v1/labelings"
PARAMS = {"conf": 0.25, "iou": 0.7, "max_det": 300}
NAMES = {"0": "scratch", "1": "dent"}
HIGH = {"class_id": 0, "class_name": "scratch", "confidence": 0.95,
        "bbox_xyxy": [20.0, 10.0, 60.0, 50.0], "bbox_xywhn": [0.2, 0.15, 0.2, 0.2]}
LOW = {"class_id": 1, "class_name": "dent", "confidence": 0.42,
       "bbox_xyxy": [100.0, 0.0, 200.0, 100.0], "bbox_xywhn": [0.75, 0.25, 0.5, 0.5]}
SUMMARY = {"num_images": 3, "num_images_without_labels": 1, "num_unreadable_images": 0, "num_labels": 3,
           "class_counts": {"scratch": 2, "dent": 1}, "confidence_histogram": {"0.9-1.0": 2}}


@pytest.fixture
def client(app_db):
    from app.routers import labelings

    app = FastAPI()
    app.include_router(labelings.router)
    with TestClient(app) as c:
        yield c


def _model(model_type=ModelType.LINE):
    model = ModelRow(type=model_type.value, model_name=model_type.value, model_version="1", task="detection",
                     line="LINE3" if model_type == ModelType.LINE else None, weights_path="/registry/best.pt")
    with SessionLocal() as session:
        session.add(model)
        session.commit()
    return model


@pytest.fixture
def model(app_db):
    return _model()


def _seed(model, status=JobStatus.SUCCEEDED):
    """A labeling with three images: approved, review, and review without labels."""
    now = datetime.now(timezone.utc)
    done = status == JobStatus.SUCCEEDED
    record = LabelingRow(
        job=JobRow(job_type=JobType.LABELING.value, status=status.value, payload={},
                   result=SUMMARY if done else None, created_at=now, updated_at=now),
        model_id=model.id, source_path="/data/raw", params=PARAMS, class_names=NAMES,
        images_done=3 if done else 1, images_total=3,
    )
    with SessionLocal() as session:
        session.add(record)
        session.flush()
        images = [
            LabelingImageRow(labeling_id=record.id, image_path=f"/data/raw/{name}.png", width=200, height=100,
                             labels=labels, num_labels=len(labels), suggested=suggested)
            for name, labels, suggested in (
                ("a", [HIGH], "approved"), ("b", [HIGH, LOW], "review"), ("c", [], "review"),
            )
        ]
        session.add_all(images)
        session.commit()
    return record.id, [image.id for image in images]


def _assert_problem(res, status, code):
    assert res.status_code == status, res.text
    assert res.json()["type"] == f"https://mrx.internal/problems/{code}"
    assert res.json()["detail"]
    assert res.json()["instance"] == res.request.url.path


def _ok(image_id):
    return {"image_id": image_id, "result": "OK"}


def _modify(image_id, *labels):
    return {"image_id": image_id, "result": "MODIFY", "labels": list(labels)}


def _review(client, labeling_id, *items, method="post"):
    return getattr(client, method)(f"{BASE}/{labeling_id}/review", json={"items": list(items)})


def test_create_is_queued(client, model, tmp_path):
    res = client.post(BASE, json={"model_id": model.id, "source_path": str(tmp_path)})

    assert res.status_code == 202, res.text
    labeling = res.json()
    assert labeling["status"] == "queued"
    assert labeling["params"] == PARAMS
    assert labeling["progress"] == {"done": 0, "total": 0, "percent": 0.0}
    assert "summary" not in labeling
    job = job_repo.list_jobs(job_type=JobType.LABELING)[0]
    assert job.payload == {"labeling_id": labeling["labeling_id"]}


def test_create_keeps_given_params_and_fills_the_rest(client, model, tmp_path):
    body = {"model_id": model.id, "source_path": str(tmp_path), "params": {"conf": 0.1}}
    assert client.post(BASE, json=body).json()["params"] == {**PARAMS, "conf": 0.1}


@pytest.mark.parametrize("change", [
    {"model_id": None}, {"model_id": "mdl_1"}, {"source_path": None}, {"source_path": ""},
    {"params": {"conf": 1.5}}, {"params": {"iou": -1}}, {"params": {"max_det": 0}},
])
def test_create_rejects_invalid_body_with_400(client, model, tmp_path, change):
    body = {"model_id": model.id, "source_path": str(tmp_path), **change}
    body = {k: v for k, v in body.items() if v is not None}
    _assert_problem(client.post(BASE, json=body), 400, "invalid-request")


def test_create_with_missing_model_returns_404(client, tmp_path):
    _assert_problem(client.post(BASE, json={"model_id": 99999, "source_path": str(tmp_path)}), 404, "not-found")


def test_create_with_missing_source_returns_404(client, model, tmp_path):
    res = client.post(BASE, json={"model_id": model.id, "source_path": str(tmp_path / "nope")})
    _assert_problem(res, 404, "not-found")
    assert job_repo.list_jobs(job_type=JobType.LABELING) == []


def test_create_with_init_model_returns_409(client, tmp_path):
    init = _model(ModelType.INIT)
    _assert_problem(client.post(BASE, json={"model_id": init.id, "source_path": str(tmp_path)}), 409, "conflict")


def test_get_shows_progress_while_running(client, model):
    labeling_id, _ = _seed(model, JobStatus.RUNNING)

    labeling = client.get(f"{BASE}/{labeling_id}").json()

    assert labeling["status"] == "running"
    assert labeling["progress"] == {"done": 1, "total": 3, "percent": 33.3}
    assert "summary" not in labeling


def test_get_shows_summary_with_live_review_count(client, model):
    labeling_id, ids = _seed(model)
    _review(client, labeling_id, _ok(ids[1]))

    summary = client.get(f"{BASE}/{labeling_id}").json()["summary"]

    assert summary == {**SUMMARY, "num_reviewed_images": 1}


@pytest.mark.parametrize("labeling_id, status, code", [(99999, 404, "not-found"), (0, 400, "invalid-request")])
def test_get_missing_or_invalid(client, labeling_id, status, code):
    _assert_problem(client.get(f"{BASE}/{labeling_id}"), status, code)


def test_list_is_newest_first_filtered_and_paged(client, model):
    first, _ = _seed(model)
    second, _ = _seed(model, JobStatus.RUNNING)
    third, _ = _seed(model, JobStatus.FAILED)

    assert [x["labeling_id"] for x in client.get(BASE).json()] == [third, second, first]
    filtered = client.get(BASE, params={"status": ["running", "failed"]}).json()
    assert [x["labeling_id"] for x in filtered] == [third, second]
    assert [x["labeling_id"] for x in client.get(BASE, params={"limit": 1, "offset": 1}).json()] == [second]
    _assert_problem(client.get(BASE, params={"status": "done"}), 400, "invalid-request")


def test_images_list_and_filters(client, model):
    labeling_id, (a, b, c) = _seed(model)
    _review(client, labeling_id, _ok(b))

    def ids(**params):
        return [i["image_id"] for i in client.get(f"{BASE}/{labeling_id}/images", params=params).json()]

    assert ids() == [a, b, c]
    assert ids(suggested="approved") == [a]
    assert ids(suggested="review") == [b, c]
    assert ids(reviewed="true") == [b]
    assert ids(reviewed="false") == [a, c]
    assert ids(result="OK") == [b]
    assert ids(result=["MODIFY", "DISCARD"]) == []
    assert ids(limit=1, offset=1) == [b]


def test_images_shape(client, model):
    labeling_id, (a, _, _) = _seed(model)

    image = client.get(f"{BASE}/{labeling_id}/images").json()[0]

    assert image == {"image_id": a, "image_path": "/data/raw/a.png", "width": 200, "height": 100,
                     "num_labels": 1, "review": {"suggested": "approved"}}


@pytest.mark.parametrize("path", ["images", "result"])
def test_lists_of_a_missing_labeling_return_404(client, path):
    _assert_problem(client.get(f"{BASE}/99999/{path}"), 404, "not-found")


def test_images_rejects_unknown_filter(client, model):
    labeling_id, _ = _seed(model)
    _assert_problem(client.get(f"{BASE}/{labeling_id}/images", params={"suggested": "OK"}), 400, "invalid-request")


def test_result_list_and_min_confidence(client, model):
    labeling_id, _ = _seed(model)

    results = client.get(f"{BASE}/{labeling_id}/result").json()
    assert [len(r["labels"]) for r in results] == [1, 2, 0]
    assert results[1]["labels"] == [HIGH, LOW]

    filtered = client.get(f"{BASE}/{labeling_id}/result", params={"min_confidence": 0.5}).json()
    assert [len(r["labels"]) for r in filtered] == [1, 1, 0]
    assert client.get(f"{BASE}/{labeling_id}/result", params={"min_confidence": 2}).status_code == 400


def test_result_of_one_image(client, model):
    labeling_id, (_, b, _) = _seed(model)

    result = client.get(f"{BASE}/{labeling_id}/result/{b}").json()

    assert result["image_id"] == b
    assert result["labels"] == [HIGH, LOW]
    assert result["review"] == {"suggested": "review"}


def test_result_of_an_image_from_another_labeling_returns_404(client, model):
    labeling_id, _ = _seed(model)
    _, other = _seed(model)
    _assert_problem(client.get(f"{BASE}/{labeling_id}/result/{other[0]}"), 404, "not-found")
    _assert_problem(client.get(f"{BASE}/{labeling_id}/result/99999"), 404, "not-found")
    _assert_problem(client.get(f"{BASE}/{labeling_id}/result/abc"), 400, "invalid-request")
