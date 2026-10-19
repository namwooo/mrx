from fastapi import APIRouter

router = APIRouter(
    prefix="/models",
    tags=["Model Registry"]
)

@router.post("/", summary="List all models")
def list_models():
    pass

@router.post("/{model_name}", summary="Get model")
def get_model():
    pass

@router.get("/{model_name}/versions", summary="List versions of a model")
def list_model_versions():
    pass

@router.put("/{model_name}/versions/{version}/stage", summary="Update model version stage")
def patch_model_version_stage():
    pass