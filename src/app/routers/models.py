from typing import List, Optional

from app.problems import Problem, ProblemRoute
from app.schemas.model import Model, ModelApply, ModelCreate, ModelExport, ModelExportRequest
from app.services import model as model_service
from core.db import get_session
from core.orm import ModelType
from fastapi import APIRouter, Depends, Path, Query
from sqlalchemy.orm import Session

router = APIRouter(
    prefix="/v1/registry/models",
    tags=["Model Registry"],
    route_class=ProblemRoute,
    responses={"4XX": {"model": Problem, "description": "Problem Details (RFC 9457)"}},
)


@router.post(
    "",
    summary="Register model",
    status_code=201,
    response_model=Model,
    response_model_exclude_none=True,
)
def create_model(body: ModelCreate, session: Session = Depends(get_session)):
    return Model.from_row(*model_service.create_model(session, body))


@router.get("", summary="List models", response_model=List[Model], response_model_exclude_none=True)
def list_models(
        model_type: Optional[ModelType] = Query(None, alias="type"),
        model_name: Optional[str] = Query(None),
        line: Optional[str] = Query(None),
        task: Optional[str] = Query(None),
        applied: Optional[bool] = Query(None, description="true: only the models currently applied"),
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
        session: Session = Depends(get_session),
):
    models = model_service.list_models(
        session,
        model_type=model_type,
        model_name=model_name,
        line=line,
        task=task,
        applied=applied,
        limit=limit,
        offset=offset,
    )
    return [Model.from_row(*model) for model in models]


@router.get("/{model_id}", summary="Get model", response_model=Model, response_model_exclude_none=True)
def get_model(model_id: int = Path(ge=1), session: Session = Depends(get_session)):
    return Model.from_row(*model_service.get_model(session, model_id))


@router.post("/{model_id}/apply", summary="Apply model", response_model=Model, response_model_exclude_none=True)
def apply_model(
        model_id: int = Path(ge=1),
        body: Optional[ModelApply] = None,
        session: Session = Depends(get_session),
):
    return Model.from_row(*model_service.apply_model(session, model_id, body))


@router.post(
    "/{model_id}/export",
    summary="Export model",
    status_code=202,
    response_model=ModelExport,
    response_model_exclude_none=True,
)
def export_model(
        body: ModelExportRequest,
        model_id: int = Path(ge=1),
        session: Session = Depends(get_session),
):
    return ModelExport.from_job(model_service.export_model(session, model_id, body))
