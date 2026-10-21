from fastapi import APIRouter

router = APIRouter(
    prefix="/trainings",
    tags=["Training Job"]
)

@router.post("/", summary="Create training job")
def create_training_job():
    pass

@router.get("/", summary="List all training jobs")
def list_training_jobs():
    pass

@router.get("/{training_id}", summary="Get training job status and result")
def get_training_job():
    pass

@router.delete("/{training_id}", summary="Revoke or cancel training job")
def revoke_training_job():
    pass

@router.get("/{training_id}/artifacts", summary="Get training job artifacts")
def get_training_artifacts():
    pass