from typing import List, Optional

from app.problems import Problem, ProblemRoute
from app.schemas.training import MetricPoint, Training, TrainingCreate, TrainingResult
from app.services import training as training_service
from core.db import get_session
from core.orm import JobStatus
from fastapi import APIRouter, Depends, Path, Query
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session

router = APIRouter(
    prefix="/v1/trainings",
    tags=["Training"],
    route_class=ProblemRoute,
    responses={"4XX": {"model": Problem, "description": "Problem Details (RFC 9457)"}},
)


@router.post(
    "",
    summary="Create training",
    status_code=202,
    response_model=Training,
    response_model_exclude_none=True,
)
def create_training(body: TrainingCreate, session: Session = Depends(get_session)):
    return Training.from_row(training_service.create_training(session, body))


@router.get("", summary="List trainings", response_model=List[Training], response_model_exclude_none=True)
def list_trainings(
        labeling_id: Optional[int] = Query(None, description="Only trainings that learned from this labeling"),
        status: Optional[List[JobStatus]] = Query(None),
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
        session: Session = Depends(get_session),
):
    records = training_service.list_trainings(
        session,
        labeling_id=labeling_id,
        status=status,
        limit=limit,
        offset=offset,
    )
    return [Training.from_row(record) for record in records]


@router.get("/{training_id}", summary="Get training", response_model=Training, response_model_exclude_none=True)
def get_training(training_id: int = Path(ge=1), session: Session = Depends(get_session)):
    return Training.from_row(training_service.get_training(session, training_id))


@router.get("/{training_id}/metrics", summary="List training metrics", response_model=List[MetricPoint])
def list_training_metrics(training_id: int = Path(ge=1), session: Session = Depends(get_session)):
    return [MetricPoint.from_row(row) for row in training_service.list_metrics(session, training_id)]


@router.get("/{training_id}/result", summary="Get training result", response_model=TrainingResult)
def get_training_result(training_id: int = Path(ge=1), session: Session = Depends(get_session)):
    return TrainingResult.from_row(training_service.get_result(session, training_id))


@router.get("/{training_id}/logs", summary="Get training logs", response_class=PlainTextResponse)
def get_training_logs(
        training_id: int = Path(ge=1),
        tail: Optional[int] = Query(None, ge=1, description="Only the last N lines"),
        session: Session = Depends(get_session),
):
    return training_service.read_logs(session, training_id, tail)


@router.post(
    "/{training_id}/cancel",
    summary="Cancel training",
    response_model=Training,
    response_model_exclude_none=True,
)
def cancel_training(training_id: int = Path(ge=1), session: Session = Depends(get_session)):
    return Training.from_row(training_service.cancel_training(session, training_id))
