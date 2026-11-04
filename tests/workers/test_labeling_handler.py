from datetime import datetime, timezone

import numpy as np
import pytest

from core import jobs as job_repo
from core.db import SessionLocal
from core.jobs import JobStatus, JobType
from core.orm import JobRow, LabelingImageRow, LabelingRow, ModelRow, ModelType
from sqlalchemy import select
from workers import handlers
from workers.handlers import labeling
from workers.worker import Worker

PARAMS = {"conf": 0.25, "iou": 0.7, "max_det": 300}
HIGH = {"class_id": 0, "confidence": 0.95, "xyxy": [20.0, 10.0, 60.0, 50.0]}
LOW = {"class_id": 1, "confidence": 0.42, "xyxy": [100.0, 0.0, 200.0, 100.0]}


class FakeModel:
    """Stands in for inference.yolo.YoloPredictor: same predict()/model.names surface."""

    def __init__(self, detections_by_width):
        self.detections_by_width = detections_by_width
        self.model = type("Inner", (), {"names": {0: "scratch", 1: "dent"}})()
        self.calls = []

    def predict(self, images, *, conf, iou, max_det):
        self.calls.append({"n": len(images), "conf": conf, "iou": iou, "max_det": max_det})
        return [self.detections_by_width.get(img.shape[1], []) for img in images]


@pytest.fixture
def source(tmp_path):
    folder = tmp_path / "images"
    folder.mkdir()
    for name in ("a.jpg", "b.png", "c.png", "empty.png", "broken.jpg", "notes.txt"):
        (folder / name).write_bytes(b"")
    return folder


@pytest.fixture
def fake(monkeypatch):
    # (height, width); the width picks what the fake model finds
    sizes = {"a.jpg": (100, 200), "b.png": (100, 300), "c.png": (100, 400), "empty.png": (100, 500)}

    def read_image(path):
        size = sizes.get(path.name)
        return None if size is None else np.zeros((*size, 3), dtype=np.uint8)

    model = FakeModel({200: [HIGH], 300: [HIGH, LOW], 400: [LOW]})
    monkeypatch.setattr(labeling, "read_image", read_image)
    monkeypatch.setattr(labeling, "_load_model", lambda path: model)
    return model


def _labeling(source, params=None):
    now = datetime.now(timezone.utc)
    model = ModelRow(type=ModelType.LINE.value, model_name="m", model_version="1", task="detection", line="L",
                     weights_path="/registry/best.pt")
    job = JobRow(job_type=JobType.LABELING.value, status=JobStatus.QUEUED.value, payload={},
                 created_at=now, updated_at=now)
    with SessionLocal() as session:
        session.add(model)
        session.flush()
        record = LabelingRow(job=job, model_id=model.id, source_path=str(source), params=params or PARAMS)
        session.add(record)
        session.flush()
        job.payload = {"labeling_id": record.id}
        session.commit()
    return record


def _run():
    assert Worker(JobType.LABELING, labeling.handle_labeling_job).run_once() is True
    return job_repo.list_jobs(job_type=JobType.LABELING)[0]


def _images(record):
    with SessionLocal() as session:
        rows = session.scalars(
            select(LabelingImageRow).where(LabelingImageRow.labeling_id == record.id).order_by(LabelingImageRow.id)
        )
        return {row.image_path.rsplit("/", 1)[1]: row for row in rows}


def _record(record):
    with SessionLocal() as session:
        return session.get(LabelingRow, record.id)


def test_handler_is_registered():
    assert handlers.HANDLERS[JobType.LABELING] is labeling.handle_labeling_job


def test_stores_every_readable_image(app_db, source, fake):
    record = _labeling(source)

    job = _run()

    assert job.status == JobStatus.SUCCEEDED
    assert sorted(_images(record)) == ["a.jpg", "b.png", "c.png", "empty.png"]


def test_image_row_holds_labels_in_pixels_and_yolo_format(app_db, source, fake):
    record = _labeling(source)
    _run()

    image = _images(record)["a.jpg"]

    assert (image.width, image.height, image.num_labels) == (200, 100, 1)
    assert image.labels == [{
        "class_id": 0, "class_name": "scratch", "confidence": 0.95,
        "bbox_xyxy": [20.0, 10.0, 60.0, 50.0], "bbox_xywhn": [0.2, 0.3, 0.2, 0.4],
    }]
    assert image.result is None


def test_suggestion_needs_every_box_confident(app_db, source, fake):
    record = _labeling(source)
    _run()

    images = _images(record)

    assert images["a.jpg"].suggested == "approved"
    assert images["b.png"].suggested == "review"    # one box below the threshold
    assert images["c.png"].suggested == "review"
    assert images["empty.png"].suggested == "review"  # no labels: possibly a missed defect


def test_approve_threshold_is_inclusive(monkeypatch):
    monkeypatch.setattr(labeling, "APPROVE_CONF", 0.9)
    assert labeling.suggest([{"confidence": 0.9}]).value == "approved"
    assert labeling.suggest([{"confidence": 0.8999}]).value == "review"
    assert labeling.suggest([]).value == "review"


def test_summary_is_the_job_result(app_db, source, fake):
    _labeling(source)

    result = _run().result

    assert result["num_images"] == 4
    assert result["num_images_without_labels"] == 1
    assert result["num_unreadable_images"] == 1
    assert result["num_labels"] == 4
    assert result["class_counts"] == {"scratch": 2, "dent": 2}
    assert result["confidence_histogram"]["0.9-1.0"] == 2


def test_progress_and_class_names_are_saved(app_db, source, fake, monkeypatch):
    monkeypatch.setattr(labeling, "CHUNK_SIZE", 2)
    seen = []
    real = labeling._save_chunk

    def spy(labeling_id, rows, done):
        real(labeling_id, rows, done)
        seen.append(done)

    monkeypatch.setattr(labeling, "_save_chunk", spy)
    record = _labeling(source)
    _run()

    stored = _record(record)
    assert seen == [2, 4, 5]
    assert (stored.images_done, stored.images_total) == (5, 5)
    assert stored.class_names == {"0": "scratch", "1": "dent"}


def test_params_reach_the_model(app_db, source, fake):
    _labeling(source, {"conf": 0.1, "iou": 0.5, "max_det": 7})
    _run()

    assert {(c["conf"], c["iou"], c["max_det"]) for c in fake.calls} == {(0.1, 0.5, 7)}


def test_folder_without_images_fails(app_db, tmp_path, fake):
    _labeling(tmp_path)
    job = _run()

    assert job.status == JobStatus.FAILED
    assert "no images" in job.error


def test_fails_when_no_image_can_be_read(app_db, tmp_path, fake):
    folder = tmp_path / "only_broken"
    folder.mkdir()
    (folder / "broken.jpg").write_bytes(b"")
    _labeling(folder)

    job = _run()

    assert job.status == JobStatus.FAILED
    assert "could be read" in job.error


def test_single_image_file_works(app_db, source, fake):
    record = _labeling(source / "a.jpg")
    _run()

    assert list(_images(record)) == ["a.jpg"]


def test_missing_record_fails(app_db):
    job = job_repo.enqueue_job(JobType.LABELING, {"labeling_id": 99999})
    with pytest.raises(ValueError, match="not found"):
        labeling.handle_labeling_job(job)
