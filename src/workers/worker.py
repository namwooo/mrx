import logging
import os
import socket
from typing import Callable, Dict, Optional, Any

from core.jobs import JobType, claim_next_job, fail_job, complete_job

logger = logging.getLogger(__name__)

JobHandler = Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]

def default_worker_id() -> str:
    worker_id = f"{socket.gethostname()}:{os.getpid()}"
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES")
    if gpus:
        worker_id += f":gpu{gpus}"
    return worker_id


class Worker:
    def __init__(
        self,
        job_type: JobType,
        handler: JobHandler,
        worker_id: Optional[str] = None,
        poll_interval: float = 2.0,
    ) -> None:
        self.job_type = job_type
        self.handler = handler
        self.worker_id = worker_id  or default_worker_id()
        self.poll_interval = poll_interval

    def run_once(self):
        job = claim_next_job(self.job_type, self.worker_id)
        if job is None:
            return False

        job_id = job["id"]
        logger.info("job %s starts (worker=%s)", job_id, self.worker_id)

        try:
            result = self.handler(job)
        except Exception as exc:
            logger.exception("job %s failed", job_id)
            fail_job(job_id, self.worker_id, f"{type(exc).__name__}: {exc}")
            return True

        try:
            done = complete_job(job_id, self.worker_id, result)
        except Exception as exc:
            logger.exception("job %s failed to store result", job_id)
            fail_job(job_id, self.worker_id, f"result could not be saved: {type(exc).__name__}: {exc}")
            return True

        if done:
            logger.info("job %s completed", job_id)
        else:
            logger.warning("job %s could not be completed", job_id)
        return True

    def run_forever(self):
        pass

    def stop(self):
        pass