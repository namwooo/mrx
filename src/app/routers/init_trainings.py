from app.problems import Problem, ProblemRoute
from app.schemas.init_training import InitTraining, InitTrainingCreate
from app.services import init_training as init_training_service
from core.db import get_session
from fastapi import APIRouter, Depends, Path
from sqlalchemy.orm import Session

router = APIRouter(
    prefix="/v1/init-trainings",
    tags=["Init Training"],
    route_class=ProblemRoute,
    responses={"4XX": {"model": Problem, "description": "Problem Details (RFC 9457)"}},
)


@router.post(
    "",
    summary="Create init training",
    status_code=202,
    response_model=InitTraining,
    response_model_exclude_none=True,
)
def create_init_training(body: InitTrainingCreate, session: Session = Depends(get_session)):
    return InitTraining.from_row(init_training_service.create_init_training(session, body))


@router.get(
    "/{init_training_id}",
    summary="Get init training",
    response_model=InitTraining,
    response_model_exclude_none=True,
)
def get_init_training(init_training_id: int = Path(ge=1), session: Session = Depends(get_session)):
    return InitTraining.from_row(init_training_service.get_init_training(session, init_training_id))
