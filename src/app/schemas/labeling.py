from datetime import datetime
from typing import Dict, List, Optional

from app.schemas.init_training import Progress
from core.db import as_utc
from core.orm import JobStatus, LabelingImageRow, LabelingRow, ReviewResult, Suggestion
from pydantic import BaseModel, Field, model_validator

REQUIRED = ...


class LabelingParams(BaseModel):
    conf: float = Field(0.25, ge=0, le=1, description="Keep boxes at or above this confidence")
    iou: float = Field(0.7, ge=0, le=1, description="IoU threshold for NMS")
    max_det: int = Field(300, ge=1, description="Maximum number of boxes per image")


class LabelingCreate(BaseModel):
    model_id: int = Field(REQUIRED, description="Model used for labeling. Must be registered with type 'line'")
    source_path: str = Field(REQUIRED, min_length=1, description="Image file or folder to label")
    params: LabelingParams = Field(default_factory=LabelingParams)


class LabelingSummary(BaseModel):
    num_images: int
    num_images_without_labels: int
    num_unreadable_images: int
    num_labels: int
    num_reviewed_images: int
    class_counts: Dict[str, int]
    confidence_histogram: Dict[str, int]


class Labeling(BaseModel):
    labeling_id: int
    model_id: int
    source_path: str
    params: LabelingParams
    status: JobStatus
    progress: Progress
    summary: Optional[LabelingSummary] = None
    error: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    @classmethod
    def from_row(cls, record: LabelingRow, num_reviewed: int) -> "Labeling":
        job = record.job
        total = record.images_total
        return cls(
            labeling_id=record.id,
            model_id=record.model_id,
            source_path=record.source_path,
            params=record.params,
            status=job.status,
            progress=Progress(
                done=record.images_done,
                total=total,
                percent=round(record.images_done / total * 100, 1) if total else 0.0,
            ),
            summary={**job.result, "num_reviewed_images": num_reviewed} if job.result else None,
            error=job.error,
            created_at=as_utc(job.created_at),
            started_at=as_utc(job.started_at),
            finished_at=as_utc(job.finished_at),
        )


class ReviewLabel(BaseModel):
    class_id: int = Field(REQUIRED, ge=0)
    class_name: Optional[str] = Field(None, description="Filled in by the server")
    bbox_xyxy: List[float] = Field(REQUIRED, min_length=4, max_length=4, description="Pixels: x1, y1, x2, y2")
    bbox_xywhn: Optional[List[float]] = Field(None, description="Filled in by the server")


class ImageReview(BaseModel):
    suggested: Suggestion
    result: Optional[ReviewResult] = None
    labels: Optional[List[ReviewLabel]] = None
    reviewed_at: Optional[datetime] = None

    @classmethod
    def from_row(cls, image: LabelingImageRow) -> "ImageReview":
        return cls(
            suggested=image.suggested,
            result=image.result,
            labels=image.reviewed_labels,
            reviewed_at=as_utc(image.reviewed_at),
        )


class LabelingImage(BaseModel):
    image_id: int
    image_path: str
    width: int
    height: int
    num_labels: int
    review: ImageReview

    @classmethod
    def from_row(cls, image: LabelingImageRow) -> "LabelingImage":
        return cls(
            image_id=image.id,
            image_path=image.image_path,
            width=image.width,
            height=image.height,
            num_labels=image.num_labels,
            review=ImageReview.from_row(image),
        )


class AutoLabel(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: List[float]
    bbox_xywhn: List[float]


class ImageLabelResult(BaseModel):
    image_id: int
    image_path: str
    width: int
    height: int
    labels: List[AutoLabel]
    review: ImageReview

    @classmethod
    def from_row(cls, image: LabelingImageRow, min_confidence: Optional[float] = None) -> "ImageLabelResult":
        threshold = min_confidence or 0
        return cls(
            image_id=image.id,
            image_path=image.image_path,
            width=image.width,
            height=image.height,
            labels=[label for label in image.labels if label["confidence"] >= threshold],
            review=ImageReview.from_row(image),
        )


class ReviewItem(BaseModel):
    image_id: int = Field(REQUIRED, ge=1)
    result: ReviewResult
    labels: Optional[List[ReviewLabel]] = Field(
        None,
        description="Required for MODIFY, not allowed otherwise. Replaces all labels of the image",
    )

    @model_validator(mode="after")
    def check_labels(self) -> "ReviewItem":
        if self.result == ReviewResult.MODIFY and self.labels is None:
            raise ValueError("labels is required for MODIFY")
        if self.result != ReviewResult.MODIFY and self.labels is not None:
            raise ValueError(f"labels is not allowed for {self.result.value}")
        return self


class ReviewRequest(BaseModel):
    items: List[ReviewItem] = Field(REQUIRED, min_length=1)

    @model_validator(mode="after")
    def check_unique_images(self) -> "ReviewRequest":
        ids = [item.image_id for item in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("image_id must be unique in items")
        return self


class ReviewResponse(BaseModel):
    labeling_id: int
    count: int
    result_counts: Dict[str, int]
    updated_at: datetime
