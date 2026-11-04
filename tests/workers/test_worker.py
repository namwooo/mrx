import threading
import time

import pytest

from core import jobs as job_repo
from core.jobs import JobStatus, JobType
from workers import worker as worker_module
from workers.worker import Worker


def _enqueue(count=1):
    return [job_repo.enqueue_job(JobType.MODEL_EXPORT, {"n": i}) for i in range(count)]


def _status(job):
    return job_repo.get_job(job.id).status


def _in_thread(worker):
    thread = threading.Thread(target=worker.run_forever, daemon=True)
    thread.start()
    return thread


def _wait_for(condition, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.01)
    raise AssertionError("condition not met in time")


def test_runs_every_queued_job_in_order(app_db):
    jobs = _enqueue(3)
    seen = []

    def handler(job):
        seen.append(job.id)
        if len(seen) == 3:
            worker.stop()

    worker = Worker(JobType.MODEL_EXPORT, handler, poll_interval=0.01)
    worker.run_forever()

    assert seen == [j.id for j in jobs]
    assert all(_status(j) == JobStatus.SUCCEEDED for j in jobs)


def test_keeps_waiting_for_new_jobs(app_db):
    seen = []
    worker = Worker(JobType.MODEL_EXPORT, lambda job: seen.append(job.id), poll_interval=0.01)
    thread = _in_thread(worker)

    time.sleep(0.05)  # the queue is empty for a while
    job = _enqueue()[0]
    _wait_for(lambda: seen == [job.id])

    worker.stop()
    thread.join(5)
    assert not thread.is_alive()


def test_only_takes_its_own_job_type(app_db):
    other = job_repo.enqueue_job(JobType.LABELING, {})
    worker = Worker(JobType.MODEL_EXPORT, lambda job: None, poll_interval=0.01)
    thread = _in_thread(worker)
    time.sleep(0.05)
    worker.stop()
    thread.join(5)

    assert _status(other) == JobStatus.QUEUED


def test_stop_ends_the_wait_between_polls_at_once(app_db):
    worker = Worker(JobType.MODEL_EXPORT, lambda job: None, poll_interval=60)
    thread = _in_thread(worker)
    time.sleep(0.05)

    worker.stop()
    thread.join(2)

    assert not thread.is_alive()


def test_stop_finishes_the_current_job_and_leaves_the_rest(app_db):
    first, second = _enqueue(2)

    def handler(job):
        worker.stop()

    worker = Worker(JobType.MODEL_EXPORT, handler, poll_interval=0.01)
    worker.run_forever()

    assert _status(first) == JobStatus.SUCCEEDED
    assert _status(second) == JobStatus.QUEUED


def test_a_failing_job_does_not_end_the_worker(app_db):
    first, second = _enqueue(2)

    def handler(job):
        if job.id == first.id:
            raise ValueError("bad job")
        worker.stop()

    worker = Worker(JobType.MODEL_EXPORT, handler, poll_interval=0.01)
    worker.run_forever()

    assert _status(first) == JobStatus.FAILED
    assert _status(second) == JobStatus.SUCCEEDED


def test_a_busy_database_does_not_end_the_worker(app_db, monkeypatch):
    job = _enqueue()[0]
    real = worker_module.claim_next_job
    calls = []

    def flaky(*args):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("database is locked")
        return real(*args)

    monkeypatch.setattr(worker_module, "claim_next_job", flaky)
    worker = Worker(JobType.MODEL_EXPORT, lambda _: worker.stop(), poll_interval=0.01)
    worker.run_forever()

    assert _status(job) == JobStatus.SUCCEEDED


def test_forced_stop_records_the_job_as_failed(app_db):
    job = _enqueue()[0]

    def handler(_):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        Worker(JobType.MODEL_EXPORT, handler, poll_interval=0.01).run_forever()

    stored = job_repo.get_job(job.id)
    assert stored.status == JobStatus.FAILED
    assert "stopped before" in stored.error
