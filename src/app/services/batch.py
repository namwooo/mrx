from typing import Dict, Any

from app.schemas.batch import BatchJobCreate
from core import jobs as job_repo
from core.jobs import JobType

JOB_TYPE = JobType.BATCH

def create_batch_job(body: BatchJobCreate) -> Dict[Any, Any] | None:
    payload = {
        "model": body.model,
        "input_path": body.input_path,
        "params": body.params,
    }

    return job_repo.enqueue_job(
        job_type=JobType(JOB_TYPE),
        payload=payload,
        idempotency_key=body.idempotency_key,
    )