from datetime import datetime, timezone

import pytest
import yaml
from PIL import Image

from core.db import SessionLocal
from core.jobs import JobType
from core.orm import JobRow, JobStatus, LabelingImageRow, LabelingRow, ModelRow, ModelType
from core.training_data import pick_split, select_images, write_dataset

SPLIT = {"train": 0.7, "val": 0.2, "test": 0.1}
AUTO = [{"class_id": 0, "bbox_xywhn": [0.5, 0.5, 0.2, 0.2]}]
FIXED = [{"class_id": 1, "bbox_xywhn": [0.1, 0.2, 0.3, 0.4]}]


@pytest.fixture
def model(app_db):
    row = ModelRow(type=ModelType.LINE.value, model_name="m", model_version="1", task="detection", line="L")
    with SessionLocal() as session:
        session.add(row)
        session.commit()
    return row


def _labeling(model, *images):
    now = datetime.now(timezone.utc)
    record = LabelingRow(
        job=JobRow(job_type=JobType.LABELING.value, status=JobStatus.SUCCEEDED.value, payload={},
                   created_at=now, updated_at=now),
        model_id=model.id, source_path="/data", params={},
    )
    with SessionLocal() as session:
        session.add(record)
        session.flush()
        session.add_all([
            LabelingImageRow(labeling_id=record.id, image_path=path, width=10, height=10, labels=AUTO, num_labels=1,
                             suggested=suggested, result=result, reviewed_labels=FIXED if result == "MODIFY" else None)
            for path, suggested, result in images
        ])
        session.commit()
    return record.id


def _select(ids, split=SPLIT):
    with SessionLocal() as session:
        return {image.image_path: image for image in select_images(session, ids, split)}


def test_which_images_are_used_and_with_which_labels(model):
    labeling = _labeling(
        model,
        ("/approved.png", "approved", None),
        ("/ok.png", "review", "OK"),
        ("/modified.png", "review", "MODIFY"),
        ("/discarded.png", "approved", "DISCARD"),
        ("/unreviewed.png", "review", None),
    )

    images = _select([labeling])

    assert sorted(images) == ["/approved.png", "/modified.png", "/ok.png"]
    assert images["/approved.png"].labels == AUTO
    assert images["/ok.png"].labels == AUTO
    assert images["/modified.png"].labels == FIXED


def test_the_newest_labeling_decides_for_a_repeated_image(model):
    older = _labeling(model, ("/same.png", "review", "MODIFY"))
    newer = _labeling(model, ("/same.png", "approved", None))

    assert _select([older, newer])["/same.png"].labels == AUTO


def test_a_newer_discard_removes_the_image_even_if_an_older_labeling_used_it(model):
    older = _labeling(model, ("/same.png", "approved", None))
    newer = _labeling(model, ("/same.png", "approved", "DISCARD"))

    assert _select([older, newer]) == {}


def test_only_the_given_labelings_count(model):
    first = _labeling(model, ("/a.png", "approved", None))
    _labeling(model, ("/b.png", "approved", None))

    assert list(_select([first])) == ["/a.png"]


def _paths(count=20000):
    return [f"/data/{i}.png" for i in range(count)]


def test_split_follows_the_ratios():
    picks = [pick_split(path, SPLIT) for path in _paths()]

    assert picks.count("train") / len(picks) == pytest.approx(0.7, abs=0.02)
    assert picks.count("val") / len(picks) == pytest.approx(0.2, abs=0.02)
    assert picks.count("test") / len(picks) == pytest.approx(0.1, abs=0.02)


def test_split_of_an_image_never_changes():
    assert pick_split("/data/a.png", SPLIT) == pick_split("/data/a.png", dict(SPLIT))


def test_changing_train_and_val_keeps_the_test_images():
    other = {"train": 0.85, "val": 0.05, "test": 0.1}

    assert [p for p in _paths() if pick_split(p, SPLIT) == "test"] == [p for p in _paths() if pick_split(p, other) == "test"]


def test_no_test_ratio_means_no_test_images():
    assert {pick_split(p, {"train": 0.8, "val": 0.2, "test": 0.0}) for p in _paths(500)} == {"train", "val"}


def _image(tmp_path, name):
    folder = tmp_path / "source"
    folder.mkdir(exist_ok=True)
    Image.new("RGB", (8, 8)).save(folder / name)
    return str(folder / name)


def test_dataset_layout(tmp_path):
    from core.training_data import TrainingImage

    images = [
        TrainingImage(1, _image(tmp_path, "a.png"), AUTO, "train"),
        TrainingImage(2, _image(tmp_path, "b.png"), [], "val"),
        TrainingImage(3, _image(tmp_path, "c.png"), FIXED + AUTO, "test"),
    ]
    root = tmp_path / "dataset"

    data_yaml = write_dataset(images, {0: "scratch", 1: "dent"}, root)

    assert (root / "images" / "train" / "1.png").is_symlink()
    assert (root / "labels" / "train" / "1.txt").read_text() == "0 0.5 0.5 0.2 0.2"
    assert (root / "labels" / "val" / "2.txt").read_text() == ""
    assert (root / "labels" / "test" / "3.txt").read_text() == "1 0.1 0.2 0.3 0.4\n0 0.5 0.5 0.2 0.2"
    assert yaml.safe_load(data_yaml.read_text()) == {
        "path": str(root.resolve()), "train": "images/train", "val": "images/val", "test": "images/test",
        "names": {0: "scratch", 1: "dent"},
    }


def test_dataset_without_test_images_leaves_test_out(tmp_path):
    from core.training_data import TrainingImage

    images = [TrainingImage(1, _image(tmp_path, "a.png"), AUTO, "train"),
              TrainingImage(2, _image(tmp_path, "b.png"), AUTO, "val")]

    assert "test" not in yaml.safe_load(write_dataset(images, {0: "a"}, tmp_path / "d").read_text())


def test_same_file_name_from_two_folders_does_not_collide(tmp_path):
    from core.training_data import TrainingImage

    (tmp_path / "x").mkdir()
    (tmp_path / "y").mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "x" / "a.png")
    Image.new("RGB", (8, 8)).save(tmp_path / "y" / "a.png")

    write_dataset([TrainingImage(1, str(tmp_path / "x" / "a.png"), AUTO, "train"),
                   TrainingImage(2, str(tmp_path / "y" / "a.png"), AUTO, "train")], {0: "a"}, tmp_path / "d")

    assert sorted(p.name for p in (tmp_path / "d" / "images" / "train").iterdir()) == ["1.png", "2.png"]
