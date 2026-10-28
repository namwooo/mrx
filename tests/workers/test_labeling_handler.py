import json
from datetime import datetime, timezone

import numpy as np
import pytest

from core.jobs import Job, JobStatus, JobType
from workers.handlers import labeling


class FakeModel:
    """Stands in for inference.yolo.YOLO: same predict()/release()/model.names surface."""

    def __init__(self, detections_by_width):
        self.detections_by_width = detections_by_width
        self.model = type("Inner", (), {"names": {0: "person", 1: "car"}})()
        self.calls = []

    def predict(self, images, *, conf, iou, max_det):
        self.calls.append({"n": len(images), "conf": conf, "iou": iou, "max_det": max_det})
        return [self.detections_by_width.get(img.shape[1], []) for img in images]

    def release(self):
        pass


def _job(input_path, params=None):
    now = datetime.now(timezone.utc)
    return Job(
        id=7, job_type=JobType.LABELING, status=JobStatus.RUNNING,
        payload={"model": "fake.pt", "input_path": str(input_path), "params": params or {}},
        result=None, error=None, worker_id="w1", idempotency_key=None,
        created_at=now, updated_at=now, started_at=now, finished_at=None,
    )


@pytest.fixture
def images(tmp_path):
    """Three image files: a.jpg (200x100), b.png (400x200, no boxes), broken.jpg (unreadable)."""
    folder = tmp_path / "images"
    folder.mkdir()
    for name in ("a.jpg", "b.png", "broken.jpg", "notes.txt"):
        (folder / name).write_bytes(b"")
    return folder


@pytest.fixture
def fake(monkeypatch, tmp_path):
    sizes = {"a.jpg": (100, 200), "b.png": (200, 400)}  # (height, width)

    def read_image(path):
        size = sizes.get(path.name)
        return None if size is None else np.zeros((*size, 3), dtype=np.uint8)

    model = FakeModel({
        200: [
            {"class_id": 0, "confidence": 0.95, "xyxy": [20.0, 10.0, 60.0, 50.0]},
            {"class_id": 1, "confidence": 0.42, "xyxy": [100.0, 0.0, 200.0, 100.0]},
        ],
    })
    monkeypatch.setattr(labeling, "read_image", read_image)
    monkeypatch.setattr(labeling, "_load_model", lambda path: model)
    monkeypatch.setattr(labeling, "OUTPUT_DIR", tmp_path / "out")
    return model


def _read_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def test_writes_one_record_per_readable_image(images, fake):
    result = labeling.handle_labeling_job(_job(images))

    records = _read_jsonl(result["output_path"])
    assert [r["image"].rsplit("/", 1)[-1] for r in records] == ["a.jpg", "b.png"]
    assert result["output_path"].endswith("/7/labels.jsonl")


def test_record_has_size_confidence_and_normalized_box(images, fake):
    result = labeling.handle_labeling_job(_job(images))
    a = _read_jsonl(result["output_path"])[0]

    assert (a["width"], a["height"]) == (200, 100)
    assert a["labels"][0] == {
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.95,
        "bbox_xyxy": [20.0, 10.0, 60.0, 50.0],
        "bbox_xywhn": [0.2, 0.3, 0.2, 0.4],
    }


def test_summary_counts_and_histogram(images, fake):
    result = labeling.handle_labeling_job(_job(images))

    assert result["num_images"] == 2
    assert result["num_images_without_labels"] == 1
    assert result["num_unreadable_images"] == 1
    assert result["num_labels"] == 2
    assert result["class_counts"] == {"person": 1, "car": 1}
    assert result["confidence_histogram"]["0.4-0.5"] == 1
    assert result["confidence_histogram"]["0.9-1.0"] == 1
    assert sum(result["confidence_histogram"].values()) == 2


def test_passes_params_to_model(images, fake):
    labeling.handle_labeling_job(_job(images, params={"conf": 0.1, "iou": 0.5, "max_det": 50}))
    assert fake.calls == [{"n": 2, "conf": 0.1, "iou": 0.5, "max_det": 50}]


def test_confidence_of_one_lands_in_last_bin(images, fake):
    fake.detections_by_width[200] = [{"class_id": 0, "confidence": 1.0, "xyxy": [0, 0, 10, 10]}]
    result = labeling.handle_labeling_job(_job(images))
    assert result["confidence_histogram"]["0.9-1.0"] == 1


def test_no_partial_file_left_behind(images, fake):
    result = labeling.handle_labeling_job(_job(images))
    out_dir = labeling.OUTPUT_DIR / "7"
    assert sorted(p.name for p in out_dir.iterdir()) == ["labels.jsonl"]
    assert result["output_path"] == str(out_dir / "labels.jsonl")


def test_empty_folder_fails(tmp_path, fake):
    with pytest.raises(ValueError, match="no images"):
        labeling.handle_labeling_job(_job(tmp_path))


def test_missing_path_fails(tmp_path, fake):
    with pytest.raises(FileNotFoundError):
        labeling.handle_labeling_job(_job(tmp_path / "nope"))


def test_fails_when_no_image_can_be_read(tmp_path, fake):
    folder = tmp_path / "only_broken"
    folder.mkdir()
    (folder / "broken.jpg").write_bytes(b"")

    with pytest.raises(ValueError, match="could be read"):
        labeling.handle_labeling_job(_job(folder))
    assert not (labeling.OUTPUT_DIR / "7" / "labels.jsonl.part").exists()


def test_worker_runs_labeling_job_through_the_queue(app_db, images, fake):
    from core import jobs as job_repo
    from workers.worker import Worker

    job = job_repo.enqueue_job(JobType.LABELING, {"model": "fake.pt", "input_path": str(images), "params": {}})
    assert Worker(JobType.LABELING, labeling.handle_labeling_job, worker_id="w1").run_once() is True

    done = job_repo.get_job(job.id)
    assert done.status == JobStatus.SUCCEEDED
    assert done.result["num_images"] == 2
    assert done.result["output_path"].endswith(f"/{job.id}/labels.jsonl")
