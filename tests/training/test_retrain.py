import json
import sys
import types

import pytest

from training import retrain

TRAIN_METRICS = {"metrics/precision(B)": 0.8, "metrics/recall(B)": 0.6, "metrics/mAP50(B)": 0.7, "metrics/mAP50-95(B)": 0.5}
TEST_METRICS = {"metrics/precision(B)": 0.4, "metrics/recall(B)": 0.3, "metrics/mAP50(B)": 0.2, "metrics/mAP50-95(B)": 0.1}


@pytest.fixture
def fake_yolo(monkeypatch):
    calls = {}

    class FakeYolo:
        def __init__(self, path):
            calls.setdefault("loaded", []).append(path)
            self.callbacks = {}

        def add_callback(self, event, callback):
            self.callbacks[event] = callback

        def train(self, **kwargs):
            calls["train"] = kwargs
            return types.SimpleNamespace(save_dir=f"{kwargs['project']}/{kwargs['name']}", results_dict=TRAIN_METRICS)

        def val(self, **kwargs):
            calls["val"] = kwargs
            return types.SimpleNamespace(results_dict=TEST_METRICS)

    ultralytics = types.ModuleType("ultralytics")
    ultralytics.YOLO = FakeYolo
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics)
    monkeypatch.setitem(sys.modules, "torch", torch)
    return calls


def _request(tmp_path, has_test=True):
    return {
        "weights_path": "/registry/best.pt",
        "data_yaml": str(tmp_path / "dataset.yaml"),
        "output_dir": str(tmp_path / "runs" / "1"),
        "has_test": has_test,
        "params": {"epochs": 3, "batch_size": 2, "learning_rate": 0.01, "image_size": 64},
    }


def test_final_metrics_come_from_the_test_images(tmp_path, fake_yolo):
    result = retrain.run(_request(tmp_path))

    assert result["metrics"] == {"mAP50": 0.2, "mAP50-95": 0.1, "precision": 0.4, "recall": 0.3}
    assert result["weights_path"].endswith("/runs/1/weights/best.pt")
    assert fake_yolo["loaded"] == ["/registry/best.pt", result["weights_path"]]
    assert fake_yolo["val"]["split"] == "test"
    assert fake_yolo["val"]["data"] == str(tmp_path / "dataset.yaml")


def test_without_test_images_the_validation_metrics_stay(tmp_path, fake_yolo):
    result = retrain.run(_request(tmp_path, has_test=False))

    assert result["metrics"]["mAP50"] == 0.7
    assert "val" not in fake_yolo


def test_main_reports_the_result(tmp_path, fake_yolo, capsys):
    path = tmp_path / "request.json"
    path.write_text(json.dumps(_request(tmp_path)))

    retrain.main(str(path))

    assert "MRX_RESULT " in capsys.readouterr().out


def test_main_reports_an_error_and_fails(tmp_path, monkeypatch, capsys):
    path = tmp_path / "request.json"
    path.write_text(json.dumps(_request(tmp_path)))

    def boom(*_):
        raise ValueError("bad")

    monkeypatch.setattr(retrain, "train", boom)
    with pytest.raises(ValueError):
        retrain.main(str(path))

    assert 'MRX_ERROR {"message": "ValueError: bad"}' in capsys.readouterr().out
