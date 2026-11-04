from datetime import datetime
from typing import Dict, List, Optional

from core.db import as_utc
from core.orm import InitTrainingRow, JobStatus
from pydantic import BaseModel, Field, model_validator

REQUIRED = ...


class ClassInfo(BaseModel):
    class_id: int = Field(REQUIRED, ge=0)
    class_name: str = Field(REQUIRED, min_length=1)


class TrainParams(BaseModel):
    model_arch: Optional[str] = Field(None, description="Model architecture. The starting model's is used when it has one")
    epochs: int = Field(100, ge=1)
    batch_size: int = Field(16, ge=1)
    learning_rate: float = Field(0.001, gt=0)
    image_size: int = Field(640, ge=32)


class InitTrainingParams(TrainParams):
    classes: Optional[List[ClassInfo]] = Field(
        None,
        description="Class definitions. Read from label_path when omitted",
    )

    @model_validator(mode="after")
    def check_class_ids(self) -> "InitTrainingParams":
        ids = [c.class_id for c in self.classes or []]
        if len(ids) != len(set(ids)):
            raise ValueError("class_id must be unique in classes")
        return self


class InitTrainingCreate(BaseModel):
    model_id: int = Field(REQUIRED, description="Starting model. Must be registered with type 'init'")
    image_path: str = Field(REQUIRED, min_length=1, description="Folder of training images")
    label_path: str = Field(REQUIRED, min_length=1, description="Folder of label data")
    params: InitTrainingParams = Field(default_factory=InitTrainingParams)


class Progress(BaseModel):
    done: int
    total: int
    percent: float


class InitTraining(BaseModel):
    init_training_id: int
    model_id: int
    image_path: str
    label_path: str
    params: InitTrainingParams
    status: JobStatus
    progress: Progress
    metrics: Optional[Dict[str, float]] = None
    weights_path: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    @classmethod
    def from_row(cls, record: InitTrainingRow) -> "InitTraining":
        job = record.job
        result = job.result or {}
        total = record.params["epochs"]
        return cls(
            init_training_id=record.id,
            model_id=record.model_id,
            image_path=record.image_path,
            label_path=record.label_path,
            params=record.params,
            status=job.status,
            progress=Progress(
                done=record.epochs_done,
                total=total,
                percent=round(record.epochs_done / total * 100, 1),
            ),
            metrics=result.get("metrics"),
            weights_path=result.get("weights_path"),
            error=job.error,
            created_at=as_utc(job.created_at),
            started_at=as_utc(job.started_at),
            finished_at=as_utc(job.finished_at),
        )
