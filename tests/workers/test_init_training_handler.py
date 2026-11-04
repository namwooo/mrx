import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from core import jobs as job_repo
from core.db import SessionLocal
from core.jobs import JobStatus, JobType
from core.orm import InitTrainingRow, JobRow, ModelRow, ModelType
from workers import handlers
from workers.handlers import init_training
from workers.worker import Worker

PARAMS = {"epochs": 3, "batch_size": 2, "learning_rate": 0.001, "image_size": 64}


@pytest.fixture
def training(app_db, tmp_path, monkeypatch):
    monkeypatch.setattr(init_training, "OUTPUT_DIR", tmp_path / "runs")
    model = ModelRow(
        type=ModelType.INIT.value, model_name="m", model_version="1", task="detection", weights_path="/registry/best.pt",
    )
    now = datetime.now(timezone.utc)
    with SessionLocal() as session:
        session.add(model)
        session.flush()
        job = JobRow(
            job_type=JobType.INIT_TRAINING.value,
            status=JobStatus.QUEUED.value,
            payload={},
            created_at=now,
            updated_at=now,
        )
        record = InitTrainingRow(
            job=job,
            model_id=model.id,
            image_path="/data/images",
            label_path="/data/labels",
            params=PARAMS,
        )
        session.add(record)
        session.flush()
        job.payload = {"init_training_id": record.id}
        session.commit()
    return record


def _use_script(monkeypatch, tmp_path, body):
    """Replaces the real training with a small script that speaks the same stdout lines."""
    script = tmp_path / "fake_train.py"
    script.write_text(body)
    monkeypatch.setattr(init_training, "_train_command", lambda request: [sys.executable, str(script), str(request)])


def _run_worker():
    return Worker(JobType.INIT_TRAINING, init_training.handle_init_training_job).run_once()


def _job(record):
    return job_repo.get_job(record.job_id)


def _epochs_done(record):
    with SessionLocal() as session:
        return session.get(InitTrainingRow, record.id).epochs_done


def test_handler_is_registered():
    assert handlers.HANDLERS[JobType.INIT_TRAINING] is init_training.handle_init_training_job


def test_success_stores_outcome_and_progress(training, monkeypatch, tmp_path):
    _use_script(monkeypatch, tmp_path, '''
print("some ultralytics output")
print('MRX_PROGRESS {"epochs_done": 1}', flush=True)
print('MRX_PROGRESS {"epochs_done": 3}', flush=True)
print('MRX_RESULT {"metrics": {"mAP50": 0.7}, "weights_path": "/runs/best.pt"}', flush=True)
''')

    assert _run_worker() is True

    job = _job(training)
    assert job.status == JobStatus.SUCCEEDED
    assert job.result == {"metrics": {"mAP50": 0.7}, "weights_path": "/runs/best.pt"}
    assert _epochs_done(training) == 3


def test_progress_is_visible_while_training_runs(training, monkeypatch, tmp_path):
    marker = tmp_path / "seen.txt"
    monkeypatch.setattr(init_training, "_save_progress", lambda _, done: marker.write_text(str(done)))
    _use_script(monkeypatch, tmp_path, f'''
import pathlib, time
print('MRX_PROGRESS {{"epochs_done": 1}}', flush=True)
for _ in range(200):
    if pathlib.Path({str(marker)!r}).exists():
        break
    time.sleep(0.01)
else:
    raise SystemExit(3)
print('MRX_RESULT {{"metrics": {{}}, "weights_path": "/w"}}', flush=True)
''')

    _run_worker()

    assert _job(training).status == JobStatus.SUCCEEDED


def test_request_file_carries_what_the_training_needs(training, monkeypatch, tmp_path):
    _use_script(monkeypatch, tmp_path, '''
import json, sys
print("MRX_RESULT " + json.dumps(json.load(open(sys.argv[1]))), flush=True)
''')

    _run_worker()

    assert _job(training).result == {
        "weights_path": "/registry/best.pt",
        "image_path": "/data/images",
        "label_path": "/data/labels",
        "params": PARAMS,
        "output_dir": str(tmp_path / "runs" / str(training.id)),
    }


def test_reported_error_fails_the_job_with_its_reason_and_log(training, monkeypatch, tmp_path):
    _use_script(monkeypatch, tmp_path, '''
import sys
print('MRX_ERROR {"message": "ValueError: no classes found"}', flush=True)
sys.exit(1)
''')

    _run_worker()

    job = _job(training)
    assert job.status == JobStatus.FAILED
    assert "ValueError: no classes found" in job.error
    assert str(tmp_path / "runs" / str(training.id) / "train.log") in job.error


def test_crash_without_a_message_fails_the_job(training, monkeypatch, tmp_path):
    _use_script(monkeypatch, tmp_path, 'import os; os._exit(7)')

    _run_worker()

    job = _job(training)
    assert job.status == JobStatus.FAILED
    assert "code 7" in job.error


def test_clean_exit_without_a_result_fails_the_job(training, monkeypatch, tmp_path):
    _use_script(monkeypatch, tmp_path, 'print("done?")')

    _run_worker()

    assert _job(training).status == JobStatus.FAILED


def test_all_output_lands_in_the_log(training, monkeypatch, tmp_path):
    _use_script(monkeypatch, tmp_path, '''
import sys
print("to stdout", flush=True)
print("to stderr", file=sys.stderr, flush=True)
print('MRX_RESULT {"metrics": {}, "weights_path": "/w"}', flush=True)
''')

    _run_worker()

    log = (tmp_path / "runs" / str(training.id) / "train.log").read_text()
    assert "to stdout" in log and "to stderr" in log


def test_training_process_is_killed_when_the_worker_stops(training, monkeypatch, tmp_path):
    pid_file = tmp_path / "pid"
    _use_script(monkeypatch, tmp_path, f'''
import os, time
open({str(pid_file)!r}, "w").write(str(os.getpid()))
print('MRX_PROGRESS {{"epochs_done": 1}}', flush=True)
time.sleep(60)
''')

    def stop(*_):
        raise KeyboardInterrupt
    monkeypatch.setattr(init_training, "_save_progress", stop)

    with pytest.raises(KeyboardInterrupt):
        init_training.handle_init_training_job(_job(training))

    pid = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_missing_record_fails(app_db):
    job = job_repo.enqueue_job(JobType.INIT_TRAINING, {"init_training_id": 99999})
    with pytest.raises(ValueError, match="not found"):
        init_training.handle_init_training_job(job)


def test_real_command_runs_the_training_module():
    command = init_training._train_command(Path("/runs/request.json"))
    assert command[1:] == ["-m", "training.init_train", "/runs/request.json"]


def test_child_process_finds_the_training_package(training, monkeypatch, tmp_path):
    _use_script(monkeypatch, tmp_path, '''
import json, training.validation
print("MRX_RESULT " + json.dumps({"metrics": {}, "weights_path": training.validation.__file__}), flush=True)
''')

    _run_worker()

    assert _job(training).result["weights_path"].endswith("training/validation.py")
