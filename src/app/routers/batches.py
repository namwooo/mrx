from fastapi import APIRouter

router = APIRouter(
    prefix="/batches",
    tags=["Batch Job Inference"]
)

@router.post("/", summary="Create batch inference job")
def create_batch_job():
    pass

@router.get("/", summary="List all batch inference jobs")
def list_batch_jobs():
    pass

@router.get("/{batch_id}", summary="Get batch job status and result")
def get_batch_job():
    pass

@router.delete("/{batch_id}", summary="Revoke or cancel batch job")
def revoke_batch_job():
    pass