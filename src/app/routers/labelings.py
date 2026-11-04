from typing import List, Optional

from app.problems import Problem, ProblemRoute
from app.schemas.labeling import (
    ImageLabelResult,
    Labeling,
    LabelingCreate,
    LabelingImage,
    ReviewRequest,
    ReviewResponse,
)
from app.services import labeling as labeling_service
from core.db import get_session
from core.orm import JobStatus, ReviewResult, Suggestion
from fastapi import APIRouter, Depends, Path, Query
from sqlalchemy.orm import Session

router = APIRouter(
    prefix="/v1/labelings",
    tags=["Auto Labeling"],
    route_class=ProblemRoute,
    responses={"4XX": {"model": Problem, "description": "Problem Details (RFC 9457)"}},
)


@router.post(
    "",
    summary="Create auto labeling",
    status_code=202,
    response_model=Labeling,
    response_model_exclude_none=True,
)
def create_labeling(body: LabelingCreate, session: Session = Depends(get_session)):
    return Labeling.from_row(*labeling_service.create_labeling(session, body))


@router.get("", summary="List labelings", response_model=List[Labeling], response_model_exclude_none=True)
def list_labelings(
        status: Optional[List[JobStatus]] = Query(None),
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
        session: Session = Depends(get_session),
):
    labelings = labeling_service.list_labelings(session, status=status, limit=limit, offset=offset)
    return [Labeling.from_row(*labeling) for labeling in labelings]


@router.get("/{labeling_id}", summary="Get labeling", response_model=Labeling, response_model_exclude_none=True)
def get_labeling(labeling_id: int = Path(ge=1), session: Session = Depends(get_session)):
    return Labeling.from_row(*labeling_service.get_labeling(session, labeling_id))


@router.get(
    "/{labeling_id}/images",
    summary="List labeling images",
    response_model=List[LabelingImage],
    response_model_exclude_none=True,
)
def list_labeling_images(
        labeling_id: int = Path(ge=1),
        reviewed: Optional[bool] = Query(None, description="true: reviewed by a person, false: not reviewed yet"),
        suggested: Optional[List[Suggestion]] = Query(None),
        result: Optional[List[ReviewResult]] = Query(None),
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
        session: Session = Depends(get_session),
):
    images = labeling_service.list_images(
        session,
        labeling_id,
        reviewed=reviewed,
        suggested=suggested,
        result=result,
        limit=limit,
        offset=offset,
    )
    return [LabelingImage.from_row(image) for image in images]


@router.get(
    "/{labeling_id}/result",
    summary="List labeling results",
    response_model=List[ImageLabelResult],
    response_model_exclude_none=True,
)
def list_labeling_results(
        labeling_id: int = Path(ge=1),
        min_confidence: Optional[float] = Query(None, ge=0, le=1, description="Keep labels at or above this confidence"),
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
        session: Session = Depends(get_session),
):
    images = labeling_service.list_images(session, labeling_id, limit=limit, offset=offset)
    return [ImageLabelResult.from_row(image, min_confidence) for image in images]


@router.get(
    "/{labeling_id}/result/{image_id}",
    summary="Get the labeling result of one image",
    response_model=ImageLabelResult,
    response_model_exclude_none=True,
)
def get_labeling_result(
        labeling_id: int = Path(ge=1),
        image_id: int = Path(ge=1),
        session: Session = Depends(get_session),
):
    return ImageLabelResult.from_row(labeling_service.get_image(session, labeling_id, image_id))


@router.post(
    "/{labeling_id}/review",
    summary="Register review results",
    status_code=201,
    response_model=ReviewResponse,
)
def create_review(body: ReviewRequest, labeling_id: int = Path(ge=1), session: Session = Depends(get_session)):
    return ReviewResponse(**labeling_service.create_review(session, labeling_id, body)._asdict())


@router.put(
    "/{labeling_id}/review",
    summary="Register or change review results",
    response_model=ReviewResponse,
)
def update_review(body: ReviewRequest, labeling_id: int = Path(ge=1), session: Session = Depends(get_session)):
    return ReviewResponse(**labeling_service.update_review(session, labeling_id, body)._asdict())
