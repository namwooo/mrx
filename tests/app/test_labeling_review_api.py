from datetime import datetime, timedelta

import pytest

from core.db import SessionLocal
from core.orm import JobStatus, LabelingImageRow
from tests.app.test_labelings_api import (
    BASE, HIGH, LOW, _assert_problem, _modify, _ok, _review, _seed, client, model,  # noqa: F401
)

BOX = {"class_id": 1, "bbox_xyxy": [10.0, 20.0, 50.0, 60.0]}


def _result(client, labeling_id, image_id):
    return client.get(f"{BASE}/{labeling_id}/result/{image_id}").json()


def _stored(image_id):
    with SessionLocal() as session:
        return session.get(LabelingImageRow, image_id)


def test_register_judgments(client, model):
    labeling_id, (a, b, c) = _seed(model)

    res = _review(client, labeling_id, _ok(a), _modify(b, BOX), {"image_id": c, "result": "DISCARD"})

    assert res.status_code == 201, res.text
    body = res.json()
    assert body["labeling_id"] == labeling_id
    assert body["count"] == 3
    assert body["result_counts"] == {"OK": 1, "MODIFY": 1, "DISCARD": 1}
    assert datetime.fromisoformat(body["updated_at"]).utcoffset() == timedelta(0)


def test_modify_stores_corrected_labels_next_to_the_automatic_ones(client, model):
    labeling_id, (_, b, _) = _seed(model)
    _review(client, labeling_id, _modify(b, BOX))

    result = _result(client, labeling_id, b)

    assert result["labels"] == [HIGH, LOW]
    assert result["review"]["suggested"] == "review"
    assert result["review"]["result"] == "MODIFY"
    assert result["review"]["labels"] == [{
        "class_id": 1, "class_name": "dent", "bbox_xyxy": [10.0, 20.0, 50.0, 60.0],
        "bbox_xywhn": [0.15, 0.4, 0.2, 0.4],
    }]
    assert "reviewed_at" in result["review"]


def test_modify_with_empty_labels_means_no_defect(client, model):
    labeling_id, (_, b, _) = _seed(model)
    assert _review(client, labeling_id, _modify(b)).status_code == 201
    assert _result(client, labeling_id, b)["review"]["labels"] == []


def test_ok_keeps_the_automatic_labels(client, model):
    labeling_id, (_, b, _) = _seed(model)
    _review(client, labeling_id, _ok(b))

    review = _result(client, labeling_id, b)["review"]

    assert review["result"] == "OK"
    assert "labels" not in review


def test_post_refuses_images_with_a_judgment_and_saves_nothing(client, model):
    labeling_id, (a, b, _) = _seed(model)
    _review(client, labeling_id, _ok(a))

    res = _review(client, labeling_id, _ok(b), _modify(a, BOX))

    _assert_problem(res, 409, "conflict")
    assert _stored(b).result is None
    assert _stored(a).result == "OK"


def test_put_changes_a_judgment(client, model):
    labeling_id, (_, b, _) = _seed(model)
    _review(client, labeling_id, _modify(b, BOX))

    res = _review(client, labeling_id, _ok(b), method="put")

    assert res.status_code == 200, res.text
    assert res.json()["result_counts"] == {"OK": 1, "MODIFY": 0, "DISCARD": 0}
    assert "labels" not in _result(client, labeling_id, b)["review"]


def test_put_registers_images_without_a_judgment(client, model):
    labeling_id, (_, b, c) = _seed(model)
    _review(client, labeling_id, _ok(b))

    res = _review(client, labeling_id, _modify(b, BOX), _ok(c), method="put")

    assert res.json()["count"] == 2
    assert res.json()["result_counts"] == {"OK": 1, "MODIFY": 1, "DISCARD": 0}


@pytest.mark.parametrize("method", ["post", "put"])
@pytest.mark.parametrize("status", [JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.FAILED, JobStatus.CANCELED])
def test_unfinished_labeling_returns_409(client, model, method, status):
    labeling_id, (a, _, _) = _seed(model, status)
    _assert_problem(_review(client, labeling_id, _ok(a), method=method), 409, "conflict")


@pytest.mark.parametrize("method", ["post", "put"])
def test_missing_labeling_or_foreign_image_returns_404(client, model, method):
    labeling_id, (a, _, _) = _seed(model)
    _, (other, _, _) = _seed(model)

    _assert_problem(_review(client, 99999, _ok(a), method=method), 404, "not-found")
    res = _review(client, labeling_id, _ok(a), _ok(other), method=method)
    _assert_problem(res, 404, "not-found")
    assert _stored(a).result is None


@pytest.mark.parametrize("items", [
    [],
    [{"image_id": 1}],
    [{"image_id": 1, "result": "NOPE"}],
    [{"image_id": 1, "result": "approved"}],
    [{"image_id": 1, "result": "MODIFY"}],
    [{"image_id": 1, "result": "OK", "labels": []}],
    [{"image_id": 1, "result": "DISCARD", "labels": [BOX]}],
    [{"image_id": 1, "result": "OK"}, {"image_id": 1, "result": "DISCARD"}],
    [{"image_id": 1, "result": "MODIFY", "labels": [{"class_id": 0}]}],
    [{"image_id": 1, "result": "MODIFY", "labels": [{"class_id": 0, "bbox_xyxy": [1, 2, 3]}]}],
])
def test_invalid_body_returns_400(client, model, items):
    labeling_id, _ = _seed(model)
    res = client.post(f"{BASE}/{labeling_id}/review", json={"items": items})
    _assert_problem(res, 400, "invalid-request")


@pytest.mark.parametrize("label", [
    {"class_id": 9, "bbox_xyxy": [10, 20, 50, 60]},    # the model has no such class
    {"class_id": 0, "bbox_xyxy": [50, 20, 10, 60]},    # x1 > x2
    {"class_id": 0, "bbox_xyxy": [10, 60, 50, 20]},    # y1 > y2
    {"class_id": 0, "bbox_xyxy": [10, 20, 10, 60]},    # no width
    {"class_id": 0, "bbox_xyxy": [-1, 20, 50, 60]},    # outside on the left
    {"class_id": 0, "bbox_xyxy": [10, 20, 201, 60]},   # outside: the image is 200 wide
    {"class_id": 0, "bbox_xyxy": [10, 20, 50, 101]},   # outside: the image is 100 high
])
def test_impossible_labels_return_400_and_save_nothing(client, model, label):
    labeling_id, (a, b, _) = _seed(model)

    res = _review(client, labeling_id, _ok(a), _modify(b, label))

    _assert_problem(res, 400, "invalid-request")
    assert _stored(a).result is None
