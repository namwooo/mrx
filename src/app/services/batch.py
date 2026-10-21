from typing import Dict, Any, Optional, List

from app.schemas.batch import BatchJobCreate
from core import jobs as job_repo
from core.jobs import JobType
from fastapi import HTTPException

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


def list_batch_jobs(
        status: Optional[job_repo.JobStatus] = None,
        limit: int = 50,
        offset: int = 0,
) -> List[Dict[str, Any]]:
    return job_repo.list_jobs(job_type=JobType(JOB_TYPE), status=status, limit=limit, offset=offset)


def get_batch_job(batch_id: int) -> Dict[Any, Any]:
    job = job_repo.get_job(batch_id)
    if job is None or job["job_type"] != JOB_TYPE:
        raise HTTPException(status_code=404, detail="batch job not found")
    return job