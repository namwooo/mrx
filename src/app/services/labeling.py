from typing import List, Optional

from app.schemas.labeling import LabelingCreate
from core import jobs as job_repo
from core.jobs import JobType
from fastapi import HTTPException

JOB_TYPE = JobType.LABELING


def create_labeling_job(body: LabelingCreate) -> job_repo.Job | None:
    payload = {
        "model": body.model,
        "input_path": body.input_path,
        "params": body.params.model_dump(),
    }

    return job_repo.enqueue_job(
        job_type=JOB_TYPE,
        payload=payload,
        idempotency_key=body.idempotency_key,
    )


def list_labeling_jobs(
        status: Optional[job_repo.JobStatus] = None,
        limit: int = 50,
        offset: int = 0,
) -> List[job_repo.Job]:
    return job_repo.list_jobs(job_type=JOB_TYPE, status=status, limit=limit, offset=offset)


def get_labeling_job(labeling_id: int) -> job_repo.Job:
    job = job_repo.get_job(labeling_id)
    if job is None or job.job_type != JOB_TYPE:
        raise HTTPException(status_code=404, detail="labeling job not found")
    return job


def cancel_labeling_job(labeling_id: int) -> job_repo.Job:
    job = job_repo.cancel_job(labeling_id, job_type=JOB_TYPE)
    if job is None:
        raise HTTPException(status_code=404, detail="labeling job not found")
    if job.status != job_repo.JobStatus.CANCELED:
        raise HTTPException(
            status_code=409,
            detail=f"cannot cancel labeling job in '{job.status}' status",
        )
    return job
