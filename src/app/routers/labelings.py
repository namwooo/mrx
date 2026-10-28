from typing import Optional

from core import jobs as job_repo
from fastapi import APIRouter, Query, Path
from app.schemas.labeling import LabelingCreate
from app.services import labeling as labeling_service

router = APIRouter(
    prefix="/labelings",
    tags=["Auto Labeling"]
)


@router.post("/", summary="Create auto labeling job", status_code=201)
def create_labeling_job(body: LabelingCreate):
    return labeling_service.create_labeling_job(body)


@router.get("/", summary="List all auto labeling jobs", status_code=200)
def list_labeling_jobs(
        status: Optional[job_repo.JobStatus] = Query(None),
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0)
):
    return labeling_service.list_labeling_jobs(status=status, limit=limit, offset=offset)


@router.get("/{labeling_id}", summary="Get labeling job status and result")
def get_labeling_job(labeling_id: int = Path(ge=1)):
    return labeling_service.get_labeling_job(labeling_id)


@router.delete("/{labeling_id}", summary="Cancel queued labeling job")
def cancel_labeling_job(labeling_id: int = Path(ge=1)):
    return labeling_service.cancel_labeling_job(labeling_id)
