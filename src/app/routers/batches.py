from typing import Optional

from core import jobs as job_repo
from fastapi import APIRouter, Query, Path
from app.schemas.batch import BatchJobCreate
from app.services import batch as batch_service

router = APIRouter(
    prefix="/batches",
    tags=["Batch Job Inference"]
)


@router.post("/", summary="Create batch inference job", status_code=201)
def create_batch_job(body: BatchJobCreate):
    return batch_service.create_batch_job(body)


@router.get("/", summary="List all batch inference jobs", status_code=200)
def list_batch_jobs(
        status: Optional[job_repo.JobStatus] = Query(None),
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0)
):
    return batch_service.list_batch_jobs(status=status, limit=limit, offset=offset)


@router.get("/{batch_id}", summary="Get batch job status and result")
def get_batch_job(batch_id: int = Path(ge=1)):
    return batch_service.get_batch_job(batch_id)


@router.delete("/{batch_id}", summary="Revoke or cancel batch job")
def revoke_batch_job(batch_id: int = Path(ge=1)):
    return batch_service.revoke_batch_job(batch_id)