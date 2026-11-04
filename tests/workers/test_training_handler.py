import json
import sys
import threading
from datetime import datetime, timezone

import pytest
from PIL import Image

from core import jobs as job_repo
from core.db import SessionLocal
from core.jobs import JobStatus, JobType
from core.orm import (
    JobRow, LabelingImageRow, LabelingRow, ModelRow, ModelType, TrainingMetricRow, TrainingRow,
)
from sqlalchemy import select
from workers import handlers
from workers.handlers import process, training as training_handler
from workers.worker import Worker

PARAMS = {"epochs": 3, "batch_size": 2, "learning_rate": 0.001, "image_size": 64}
SPLIT = {"train": 0.7, "val": 0.2, "test": 0.1}
LABELS = [{"class_id": 0, "bbox_xywhn": [0.5, 0.5, 0.2, 0.2]}]
RESULT = 'print(\'MRX_RESULT {"metrics": {"mAP50": 0.7}, "weights_path": "/w"}\', flush=True)\n'


@pytest.fixture
def source(tmp_path):
    folder = tmp_path / "images"
    folder.mkdir()
    for i in range(30):
        Image.new("RGB", (8, 8)).save(folder / f"{i}.png")
    return folder


def _store(*rows):
    with SessionLocal() as session:
        session.add_all(rows)
        session.commit()


def _training(source, tmp_path, base=True, params=PARAMS):
    now = datetime.now(timezone.utc)
    model = ModelRow(type=ModelType.LINE.value, model_name="m", model_version="1", task="detection", line="L",
                     weights_path="/registry/best.pt")
    _store(model)
    labeling = LabelingRow(
        job=JobRow(job_type=JobType.LABELING.value, status=JobStatus.SUCCEEDED.value, payload={},
                   created_at=now, updated_at=now),
        model_id=model.id, source_path=str(source), params={}, class_names={"0": "scratch"},
    )
    _store(labeling)
    _store(*[
        LabelingImageRow(labeling_id=labeling.id, image_path=str(path), width=8, height=8, labels=LABELS, num_labels=1,
                         suggested="approved")
        for path in sorted(source.iterdir())
    ])
    job = JobRow(job_type=JobType.TRAINING.value, status=JobStatus.QUEUED.value, payload={},
                 created_at=now, updated_at=now)
    record = TrainingRow(job=job, base_model_id=model.id if base else None, labeling_ids=[labeling.id], split=SPLIT,
                         params=params, output_dir=str(tmp_path / "runs" / "1"))
    with SessionLocal() as session:
        session.add(record)
        session.flush()
        job.payload = {"training_id": record.id}
        session.commit()
    return record


def _use_script(monkeypatch, tmp_path, body):
    script = tmp_path / "fake_train.py"
    script.write_text(body)
    monkeypatch.setattr(training_handler, "_train_command", lambda request: [sys.executable, str(script), str(request)])


def _run():
    assert Worker(JobType.TRAINING, training_handler.handle_training_job).run_once() is True


def _job(record):
    return job_repo.get_job(record.job_id)


def test_handler_is_registered():
    assert handlers.HANDLERS[JobType.TRAINING] is training_handler.handle_training_job


def test_success_stores_the_outcome(app_db, source, tmp_path, monkeypatch):
    record = _training(source, tmp_path)
    _use_script(monkeypatch, tmp_path, RESULT)

    _run()

    assert _job(record).status == JobStatus.SUCCEEDED
    assert _job(record).result == {"metrics": {"mAP50": 0.7}, "weights_path": "/w"}


def test_dataset_and_request_are_ready_before_the_training_starts(app_db, source, tmp_path, monkeypatch):
    record = _training(source, tmp_path)
    _use_script(monkeypatch, tmp_path, '''
import json, sys
from pathlib import Path
request = json.load(open(sys.argv[1]))
root = Path(request["data_yaml"]).parent
print("MRX_RESULT " + json.dumps({"request": request, "train": len(list((root / "images" / "train").iterdir()))}), flush=True)
''')

    _run()

    result = _job(record).result
    request = result["request"]
    assert request["weights_path"] == "/registry/best.pt"
    assert request["params"] == PARAMS
    assert request["output_dir"] == str(tmp_path / "runs" / "1")
    assert result["train"] > 0
    assert (tmp_path / "runs" / "1" / "train.log").is_file()


def test_from_scratch_starts_from_the_architecture(app_db, source, tmp_path, monkeypatch):
    record = _training(source, tmp_path, base=False, params={**PARAMS, "model_arch": "yolo11n"})
    _use_script(monkeypatch, tmp_path, '''
import json, sys
print("MRX_RESULT " + json.dumps({"weights": json.load(open(sys.argv[1]))["weights_path"]}), flush=True)
''')

    _run()

    assert _job(record).result == {"weights": "yolo11n.yaml"}


def test_progress_and_metrics_are_saved_per_epoch(app_db, source, tmp_path, monkeypatch):
    record = _training(source, tmp_path)
    _use_script(monkeypatch, tmp_path, '''
print('MRX_PROGRESS {"epochs_done": 1, "metrics": {"val/mAP50": 0.5, "train/box_loss": 1.5}}', flush=True)
print('MRX_PROGRESS {"epochs_done": 2, "metrics": {"val/mAP50": 0.6}}', flush=True)
print('MRX_PROGRESS {"epochs_done": 2}', flush=True)
''' + RESULT)

    _run()

    with SessionLocal() as session:
        assert session.get(TrainingRow, record.id).epochs_done == 2
        rows = session.scalars(select(TrainingMetricRow).order_by(TrainingMetricRow.id)).all()
    assert [(r.epoch, r.name, r.value) for r in rows] == [
        (1, "val/mAP50", 0.5), (1, "train/box_loss", 1.5), (2, "val/mAP50", 0.6),
    ]


def test_failure_keeps_the_reason(app_db, source, tmp_path, monkeypatch):
    record = _training(source, tmp_path)
    _use_script(monkeypatch, tmp_path, '''
import sys
print('MRX_ERROR {"message": "ValueError: bad data"}', flush=True)
sys.exit(1)
''')

    _run()

    assert _job(record).status == JobStatus.FAILED
    assert "ValueError: bad data" in _job(record).error


def test_cancel_stops_the_running_process(app_db, source, tmp_path, monkeypatch):
    monkeypatch.setattr(process, "STOP_CHECK_SECONDS", 0.05)
    record = _training(source, tmp_path)
    marker = tmp_path / "started"
    _use_script(monkeypatch, tmp_path, f'''
import time
open({str(marker)!r}, "w").write("x")
time.sleep(60)
''')

    def cancel():
        while not marker.exists():
            pass
        with SessionLocal() as session:
            session.get(JobRow, record.job_id).status = JobStatus.CANCELED.value
            session.commit()

    threading.Thread(target=cancel, daemon=True).start()
    worker = threading.Thread(target=_run)
    worker.start()
    worker.join(20)

    assert not worker.is_alive()
    assert _job(record).status == JobStatus.CANCELED
    assert _job(record).result is None


def test_a_failing_stop_check_does_not_end_the_training(app_db, source, tmp_path, monkeypatch):
    monkeypatch.setattr(process, "STOP_CHECK_SECONDS", 0.01)
    monkeypatch.setattr(training_handler, "_is_canceled", lambda _: 1 / 0)
    record = _training(source, tmp_path)
    _use_script(monkeypatch, tmp_path, 'import time\ntime.sleep(0.2)\n' + RESULT)

    _run()

    assert _job(record).status == JobStatus.SUCCEEDED


def test_missing_record_fails(app_db):
    job = job_repo.enqueue_job(JobType.TRAINING, {"training_id": 99999})
    with pytest.raises(ValueError, match="not found"):
        training_handler.handle_training_job(job)


def test_real_command_runs_the_retraining_module(tmp_path):
    command = training_handler._train_command(tmp_path / "request.json")
    assert command[1:3] == ["-m", "training.retrain"]
