from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

from app.problems import bad_request, conflict, not_found
from app.schemas.labeling import LabelingCreate, ReviewItem, ReviewRequest
from core.boxes import xywhn
from core.orm import (
    JobRow,
    JobStatus,
    JobType,
    LabelingImageRow,
    LabelingRow,
    ModelRow,
    ModelType,
    ReviewResult,
    Suggestion,
)
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session, contains_eager, joinedload

LabelingWithReviews = Tuple[LabelingRow, int]


class ReviewOutcome(NamedTuple):
    labeling_id: int
    count: int
    result_counts: Dict[str, int]
    updated_at: datetime


def create_labeling(session: Session, body: LabelingCreate) -> LabelingWithReviews:
    model = session.get(ModelRow, body.model_id)
    if model is None:
        raise not_found(f"model '{body.model_id}' not found")
    if model.type != ModelType.LINE.value:
        raise conflict(f"model '{body.model_id}' is not a 'line' model")
    if not Path(body.source_path).exists():
        raise not_found(f"source_path '{body.source_path}' not found")

    now = datetime.now(timezone.utc)
    job = JobRow(
        job_type=JobType.LABELING.value,
        status=JobStatus.QUEUED.value,
        payload={},
        created_at=now,
        updated_at=now,
    )
    record = LabelingRow(
        job=job,
        model_id=body.model_id,
        source_path=body.source_path,
        params=body.params.model_dump(),
    )
    session.add(record)
    session.flush()
    job.payload = {"labeling_id": record.id}
    session.commit()  # one commit for both: a worker can never pick up a job without its record
    return record, 0


def list_labelings(
        session: Session,
        status: Optional[Sequence[JobStatus]] = None,
        limit: int = 50,
        offset: int = 0,
) -> List[LabelingWithReviews]:
    stmt = select(LabelingRow).join(LabelingRow.job).options(contains_eager(LabelingRow.job))
    if status:
        stmt = stmt.where(JobRow.status.in_([s.value for s in status]))
    stmt = stmt.order_by(LabelingRow.id.desc()).limit(limit).offset(offset)

    records = list(session.scalars(stmt))
    reviewed = _reviewed_counts(session, [r.id for r in records])
    return [(r, reviewed.get(r.id, 0)) for r in records]


def get_labeling(session: Session, labeling_id: int) -> LabelingWithReviews:
    record = _get_record(session, labeling_id)
    return record, _reviewed_counts(session, [labeling_id]).get(labeling_id, 0)


def list_images(
        session: Session,
        labeling_id: int,
        reviewed: Optional[bool] = None,
        suggested: Optional[Sequence[Suggestion]] = None,
        result: Optional[Sequence[ReviewResult]] = None,
        limit: int = 50,
        offset: int = 0,
) -> List[LabelingImageRow]:
    _get_record(session, labeling_id)

    stmt = select(LabelingImageRow).where(LabelingImageRow.labeling_id == labeling_id)
    if reviewed is not None:
        stmt = stmt.where(LabelingImageRow.result.is_not(None) if reviewed else LabelingImageRow.result.is_(None))
    if suggested:
        stmt = stmt.where(LabelingImageRow.suggested.in_([s.value for s in suggested]))
    if result:
        stmt = stmt.where(LabelingImageRow.result.in_([r.value for r in result]))
    return list(session.scalars(stmt.order_by(LabelingImageRow.id).limit(limit).offset(offset)))


def get_image(session: Session, labeling_id: int, image_id: int) -> LabelingImageRow:
    _get_record(session, labeling_id)
    image = session.get(LabelingImageRow, image_id)
    if image is None or image.labeling_id != labeling_id:
        raise not_found(f"image '{image_id}' not found in labeling '{labeling_id}'")
    return image


def create_review(session: Session, labeling_id: int, body: ReviewRequest) -> ReviewOutcome:
    return _save_review(session, labeling_id, body, overwrite=False)


def update_review(session: Session, labeling_id: int, body: ReviewRequest) -> ReviewOutcome:
    return _save_review(session, labeling_id, body, overwrite=True)


def _save_review(session: Session, labeling_id: int, body: ReviewRequest, overwrite: bool) -> ReviewOutcome:
    record = _get_record(session, labeling_id)
    if record.job.status != JobStatus.SUCCEEDED.value:
        raise conflict(f"labeling '{labeling_id}' is '{record.job.status}', not 'succeeded'")

    image_ids = [item.image_id for item in body.items]
    images = {
        image.id: image
        for image in session.scalars(
            select(LabelingImageRow).where(
                LabelingImageRow.labeling_id == labeling_id,
                LabelingImageRow.id.in_(image_ids),
            )
        )
    }
    missing = sorted(set(image_ids) - set(images))
    if missing:
        raise not_found(f"images {missing} not found in labeling '{labeling_id}'")

    now = datetime.now(timezone.utc)
    values = [
        {
            "result": item.result.value,
            "reviewed_labels": _reviewed_labels(item, images[item.image_id], record.class_names or {}),
            "reviewed_at": now,
        }
        for item in body.items
    ]

    # Each UPDATE carries the condition, so two reviewers can never both register the same image.
    for item, value in zip(body.items, values):
        stmt = update(LabelingImageRow).where(LabelingImageRow.id == item.image_id)
        if not overwrite:
            stmt = stmt.where(LabelingImageRow.result.is_(None))
        if session.execute(stmt.values(**value)).rowcount == 0:
            session.rollback()
            raise conflict(f"image '{item.image_id}' already has a review result")
    session.commit()

    return ReviewOutcome(labeling_id, len(body.items), _result_counts(session, labeling_id), now)


def _reviewed_labels(item: ReviewItem, image: LabelingImageRow, class_names: Dict[str, str]) -> Optional[List[dict]]:
    if item.result != ReviewResult.MODIFY:
        return None

    labels = []
    for label in item.labels:
        class_name = class_names.get(str(label.class_id))
        if class_name is None:
            raise bad_request(f"image '{item.image_id}': unknown class_id {label.class_id}")
        x1, y1, x2, y2 = label.bbox_xyxy
        if not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
            raise bad_request(f"image '{item.image_id}': bbox_xyxy {label.bbox_xyxy} is outside the image")
        labels.append({
            "class_id": label.class_id,
            "class_name": class_name,
            "bbox_xyxy": label.bbox_xyxy,
            "bbox_xywhn": xywhn(label.bbox_xyxy, image.width, image.height),
        })
    return labels


def _get_record(session: Session, labeling_id: int) -> LabelingRow:
    record = session.scalar(
        select(LabelingRow).options(joinedload(LabelingRow.job)).where(LabelingRow.id == labeling_id)
    )
    if record is None:
        raise not_found(f"labeling '{labeling_id}' not found")
    return record


def _reviewed_counts(session: Session, labeling_ids: Sequence[int]) -> Dict[int, int]:
    if not labeling_ids:
        return {}
    rows = session.execute(
        select(LabelingImageRow.labeling_id, func.count())
        .where(LabelingImageRow.labeling_id.in_(labeling_ids), LabelingImageRow.result.is_not(None))
        .group_by(LabelingImageRow.labeling_id)
    )
    return dict(rows.all())


def _result_counts(session: Session, labeling_id: int) -> Dict[str, int]:
    rows = session.execute(
        select(LabelingImageRow.result, func.count())
        .where(LabelingImageRow.labeling_id == labeling_id, LabelingImageRow.result.is_not(None))
        .group_by(LabelingImageRow.result)
    )
    counts = dict(rows.all())
    return {result.value: counts.get(result.value, 0) for result in ReviewResult}
