"""Runs a real worker process, because signal handling only exists there."""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from core import jobs as job_repo
from core.jobs import JobStatus, JobType

SRC = Path(__file__).resolve().parents[2] / "src"

SLOW_WORKER = """
import sys, time
from pathlib import Path
import workers.__main__ as cli
from core.jobs import JobType

def slow(job):
    Path(sys.argv[1]).write_text("started")
    time.sleep(float(sys.argv[2]))
    return {"done": True}

cli.HANDLERS[JobType.MODEL_EXPORT] = slow
cli.main(["model_export", "--poll-interval", "0.05"])
"""


def _start(*args):
    return subprocess.Popen(
        [sys.executable, *args],
        env={**os.environ, "PYTHONPATH": str(SRC)},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _wait_for(path, process, timeout=30):
    deadline = time.monotonic() + timeout
    while not path.exists():
        assert process.poll() is None, process.stdout.read()
        assert time.monotonic() < deadline, "the job never started"
        time.sleep(0.02)


def _slow_job(tmp_path, seconds):
    job = job_repo.enqueue_job(JobType.MODEL_EXPORT, {})
    marker = tmp_path / "started"
    return job, marker, _start("-c", SLOW_WORKER, str(marker), str(seconds))


def test_keeps_running_without_jobs_and_stops_on_sigterm(app_db):
    process = _start("-m", "workers", "model_export", "--poll-interval", "0.05")
    time.sleep(3)
    assert process.poll() is None

    process.send_signal(signal.SIGTERM)

    assert process.wait(10) == 0


def test_sigterm_lets_the_current_job_finish(app_db, tmp_path):
    job, marker, process = _slow_job(tmp_path, 1.5)
    _wait_for(marker, process)

    process.send_signal(signal.SIGTERM)

    assert process.wait(30) == 0
    assert job_repo.get_job(job.id).status == JobStatus.SUCCEEDED


def test_second_signal_gives_up_and_records_the_failure(app_db, tmp_path):
    job, marker, process = _slow_job(tmp_path, 60)
    _wait_for(marker, process)

    process.send_signal(signal.SIGTERM)
    time.sleep(0.3)
    process.send_signal(signal.SIGTERM)

    assert process.wait(10) == 130
    stored = job_repo.get_job(job.id)
    assert stored.status == JobStatus.FAILED
    assert "stopped before" in stored.error


def test_ctrl_c_stops_the_same_way(app_db):
    process = _start("-m", "workers", "model_export", "--poll-interval", "0.05")
    time.sleep(3)

    process.send_signal(signal.SIGINT)

    assert process.wait(10) == 0


@pytest.mark.parametrize("job_type", ["init_training", "labeling", "model_export", "training"])
def test_every_job_type_has_a_worker(job_type):
    from workers.handlers import HANDLERS

    assert JobType(job_type) in HANDLERS
