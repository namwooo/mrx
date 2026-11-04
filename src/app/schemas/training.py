from datetime import datetime
from typing import Dict, List, Optional

from app.schemas.init_training import Progress, TrainParams
from core.db import as_utc
from core.orm import JobRow, JobStatus, TrainingMetricRow, TrainingRow
from pydantic import BaseModel, Field, model_validator

REQUIRED = ...
SUM_TOLERANCE = 1e-6


class Split(BaseModel):
    train: float = Field(0.7, ge=0, le=1)
    val: float = Field(0.2, ge=0, le=1)
    test: float = Field(0.1, ge=0, le=1)

    @model_validator(mode="after")
    def check_ratios(self) -> "Split":
        if abs(self.train + self.val + self.test - 1) > SUM_TOLERANCE:
            raise ValueError("train, val and test must add up to 1")
        if self.train == 0 or self.val == 0:
            raise ValueError("train and val must be above 0")
        return self


class TrainingCreate(BaseModel):
    labeling_ids: List[int] = Field(REQUIRED, min_length=1, description="Labelings to learn from")
    split: Split = Field(default_factory=Split)
    base_model_id: Optional[int] = Field(None, description="Model to continue from. Learns from scratch when omitted")
    params: TrainParams = Field(default_factory=TrainParams)

    @model_validator(mode="after")
    def check_starting_point(self) -> "TrainingCreate":
        if self.base_model_id is None and self.params.model_arch is None:
            raise ValueError("base_model_id or params.model_arch is required")
        if len(self.labeling_ids) != len(set(self.labeling_ids)):
            raise ValueError("labeling_ids must be unique")
        return self


class Training(BaseModel):
    training_id: int
    labeling_ids: List[int]
    split: Split
    base_model_id: Optional[int] = None
    params: TrainParams
    status: JobStatus
    progress: Progress
    error: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    @classmethod
    def from_row(cls, record: TrainingRow) -> "Training":
        job = record.job
        total = record.params["epochs"]
        return cls(
            training_id=record.id,
            labeling_ids=record.labeling_ids,
            split=record.split,
            base_model_id=record.base_model_id,
            params=record.params,
            status=job.status,
            progress=Progress(
                done=record.epochs_done,
                total=total,
                percent=round(record.epochs_done / total * 100, 1),
            ),
            error=job.error,
            created_at=as_utc(job.created_at),
            started_at=as_utc(job.started_at),
            finished_at=as_utc(job.finished_at),
        )


class MetricPoint(BaseModel):
    epoch: int
    name: str
    value: float
    recorded_at: datetime

    @classmethod
    def from_row(cls, row: TrainingMetricRow) -> "MetricPoint":
        return cls(epoch=row.epoch, name=row.name, value=row.value, recorded_at=as_utc(row.recorded_at))


class TrainingResult(BaseModel):
    training_id: int
    metrics: Dict[str, float]
    weights_path: str

    @classmethod
    def from_row(cls, record: TrainingRow) -> "TrainingResult":
        job: JobRow = record.job
        return cls(training_id=record.id, metrics=job.result["metrics"], weights_path=job.result["weights_path"])
