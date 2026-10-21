from fastapi import APIRouter
from app.schemas.batch import BatchJobCreate
from app.services import batch as batch_service

router = APIRouter(
    prefix="/batches",
    tags=["Batch Job Inference"]
)

@router.post("/", summary="Create batch inference job", status_code=201)
def create_batch_job(body: BatchJobCreate):
    return batch_service.create_batch_job(body)

@router.get("/", summary="List all batch inference jobs")
def list_batch_jobs():
    pass

@router.get("/{batch_id}", summary="Get batch job status and result")
def get_batch_job():
    pass

@router.delete("/{batch_id}", summary="Revoke or cancel batch job")
def revoke_batch_job():
    pass