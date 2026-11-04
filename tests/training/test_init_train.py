import json
import sys
import types

import pytest
import yaml

from training import init_train

PNG = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def folders(tmp_path):
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    return images, labels


def _dataset(images, labels, count=10, line="0 0.5 0.5 0.1 0.1\n"):
    from PIL import Image

    for i in range(count):
        Image.new("RGB", (8, 8)).save(images / f"img{i}.png")
        (labels / f"img{i}.txt").write_text(line)


def _names(root, split, kind):
    return sorted(p.name for p in (root / kind / split).iterdir())


def test_classes_from_the_request_win(folders):
    _, labels = folders
    (labels / "classes.txt").write_text("from_file\n")
    classes = [{"class_id": 0, "class_name": "scratch"}, {"class_id": 1, "class_name": "dent"}]

    assert init_train.resolve_classes(classes, labels) == {0: "scratch", 1: "dent"}


def test_classes_from_classes_txt(folders):
    _, labels = folders
    (labels / "classes.txt").write_text("scratch\ndent\n")

    assert init_train.resolve_classes(None, labels) == {0: "scratch", 1: "dent"}


def test_classes_from_label_files(folders):
    _, labels = folders
    (labels / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n\n2 0.5 0.5 0.1 0.1\n")

    assert init_train.resolve_classes(None, labels) == {0: "class_0", 1: "class_1", 2: "class_2"}


def test_gaps_in_given_classes_are_filled(folders):
    _, labels = folders
    classes = [{"class_id": 2, "class_name": "dent"}]

    assert init_train.resolve_classes(classes, labels) == {0: "class_0", 1: "class_1", 2: "dent"}


def test_no_classes_anywhere_fails(folders):
    _, labels = folders
    (labels / "a.txt").write_text("")

    with pytest.raises(ValueError, match="no classes"):
        init_train.resolve_classes(None, labels)


def test_dataset_is_split_without_overlap(folders, tmp_path):
    images, labels = folders
    _dataset(images, labels)
    root = tmp_path / "out" / "dataset"

    data_yaml = init_train.prepare_dataset(images, labels, {0: "scratch"}, root)

    train, val = _names(root, "train", "images"), _names(root, "val", "images")
    assert len(val) == 2 and len(train) == 8
    assert not set(train) & set(val)
    assert [n.replace(".txt", ".png") for n in _names(root, "train", "labels")] == train
    assert yaml.safe_load(data_yaml.read_text()) == {
        "path": str(root.resolve()), "train": "images/train", "val": "images/val", "names": {0: "scratch"},
    }


def test_split_is_the_same_on_every_run(folders, tmp_path):
    images, labels = folders
    _dataset(images, labels)

    init_train.prepare_dataset(images, labels, {0: "a"}, tmp_path / "one")
    init_train.prepare_dataset(images, labels, {0: "a"}, tmp_path / "two")

    assert _names(tmp_path / "one", "val", "images") == _names(tmp_path / "two", "val", "images")


def test_dataset_links_to_the_originals(folders, tmp_path):
    images, labels = folders
    _dataset(images, labels, count=2)
    root = tmp_path / "dataset"

    init_train.prepare_dataset(images, labels, {0: "a"}, root)

    linked = next((root / "images" / "train").iterdir())
    assert linked.is_symlink()
    assert linked.resolve().parent == images.resolve()


@pytest.mark.parametrize("count", [0, 1])
def test_too_few_images_fail(folders, tmp_path, count):
    images, labels = folders
    _dataset(images, labels, count=count)

    with pytest.raises(ValueError, match="at least 2"):
        init_train.prepare_dataset(images, labels, {0: "a"}, tmp_path / "dataset")


def test_files_that_are_no_images_are_ignored(folders, tmp_path):
    images, labels = folders
    _dataset(images, labels, count=2)
    (images / "notes.txt").write_text("x")
    root = tmp_path / "dataset"

    init_train.prepare_dataset(images, labels, {0: "a"}, root)

    assert "notes.txt" not in _names(root, "train", "images") + _names(root, "val", "images")


@pytest.fixture
def fake_yolo(monkeypatch):
    calls = {}

    class Results:
        def __init__(self, save_dir):
            self.save_dir = save_dir
            self.results_dict = {
                "metrics/precision(B)": 0.8, "metrics/recall(B)": 0.6, "metrics/mAP50(B)": 0.7,
                "metrics/mAP50-95(B)": 0.5, "fitness": 0.52,
            }

    class FakeYolo:
        def __init__(self, path):
            calls["weights"] = path
            self.callbacks = {}

        def add_callback(self, event, callback):
            self.callbacks[event] = callback

        def train(self, **kwargs):
            calls.update(kwargs)
            for epoch in (0, 1, 2, 3):  # 3: the extra call of the final evaluation
                self.callbacks["on_fit_epoch_end"](types.SimpleNamespace(epoch=epoch, epochs=kwargs["epochs"]))
            return Results(f"{kwargs['project']}/{kwargs['name']}")

    ultralytics = types.ModuleType("ultralytics")
    ultralytics.YOLO = FakeYolo
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics)
    monkeypatch.setitem(sys.modules, "torch", torch)
    return calls


def _request(images, labels, output_dir, **params):
    return {
        "weights_path": "/registry/best.pt",
        "image_path": str(images),
        "label_path": str(labels),
        "output_dir": str(output_dir),
        "params": {"epochs": 3, "batch_size": 2, "learning_rate": 0.01, "image_size": 64, **params},
    }


def test_run_trains_with_the_requested_settings(folders, tmp_path, fake_yolo, capsys):
    images, labels = folders
    _dataset(images, labels)
    output = tmp_path / "runs" / "itr_1"

    result = init_train.run(_request(images, labels, output))

    assert fake_yolo["weights"] == "/registry/best.pt"
    assert (fake_yolo["epochs"], fake_yolo["batch"], fake_yolo["imgsz"]) == (3, 2, 64)
    assert fake_yolo["lr0"] == 0.01
    assert fake_yolo["optimizer"] != "auto"
    assert fake_yolo["device"] == "cpu"
    assert (fake_yolo["project"], fake_yolo["name"]) == (str(output.parent), "itr_1")
    assert fake_yolo["exist_ok"] is True
    assert result == {
        "metrics": {"mAP50": 0.7, "mAP50-95": 0.5, "precision": 0.8, "recall": 0.6},
        "weights_path": str(output.resolve() / "weights" / "best.pt"),
    }
    progress = [line for line in capsys.readouterr().out.splitlines() if line.startswith("MRX_PROGRESS")]
    assert [json.loads(line.split(" ", 1)[1])["epochs_done"] for line in progress] == [1, 2, 3, 3]


def test_run_stops_before_training_on_bad_labels(folders, tmp_path, fake_yolo):
    images, labels = folders
    _dataset(images, labels, line="9 0.5 0.5 0.1 0.1\n")

    with pytest.raises(ValueError, match="dataset check failed"):
        init_train.run(_request(images, labels, tmp_path / "out", classes=[{"class_id": 0, "class_name": "a"}]))

    assert "epochs" not in fake_yolo


def test_run_stops_on_a_missing_label_file(folders, tmp_path, fake_yolo):
    images, labels = folders
    _dataset(images, labels)
    (labels / "img0.txt").unlink()

    with pytest.raises(ValueError, match="dataset check failed"):
        init_train.run(_request(images, labels, tmp_path / "out"))


def test_main_reports_the_result(folders, tmp_path, fake_yolo, capsys):
    images, labels = folders
    _dataset(images, labels)
    request = tmp_path / "request.json"
    request.write_text(json.dumps(_request(images, labels, tmp_path / "out")))

    init_train.main(str(request))

    assert "MRX_RESULT " in capsys.readouterr().out


def test_main_reports_the_reason_and_exits_with_an_error(folders, tmp_path, capsys):
    images, labels = folders
    request = tmp_path / "request.json"
    request.write_text(json.dumps(_request(images, labels, tmp_path / "out")))

    with pytest.raises(ValueError):
        init_train.main(str(request))

    assert 'MRX_ERROR {"message": "ValueError: no classes found' in capsys.readouterr().out
