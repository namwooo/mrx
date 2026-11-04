from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from core.db import as_utc
from core.jobs import Job
from core.orm import JobStatus, ModelRow, ModelType
from pydantic import BaseModel, Field, model_validator

REQUIRED = ...

# model_name and model_version become folder names under the registry.
PATH_SAFE = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"


class ExportFormat(str, Enum):
    PT = "pt"
    ONNX = "onnx"
    TORCHSCRIPT = "torchscript"
    ENGINE = "engine"


class ModelMetadata(BaseModel):
    model_name: str = Field(REQUIRED, max_length=128, pattern=PATH_SAFE)
    model_version: str = Field(REQUIRED, max_length=64, pattern=PATH_SAFE, description="Unique within model_name")
    task: str = Field(REQUIRED, min_length=1, max_length=64)
    line: Optional[str] = Field(None, min_length=1, max_length=64, description="Required for line models")


class ModelCreate(BaseModel):
    type: ModelType = Field(REQUIRED, description="init = starting point of an init training, line = process model")
    metadata: ModelMetadata
    weights_path: Optional[str] = Field(None, min_length=1, description="Weights file. Required for init models")
    init_training_id: Optional[int] = Field(None, description="Line model produced by an init training")
    training_id: Optional[int] = Field(None, description="Line model produced by a retraining")

    @model_validator(mode="after")
    def check_type_rules(self) -> "ModelCreate":
        if self.type == ModelType.INIT:
            if self.weights_path is None:
                raise ValueError("weights_path is required for type 'init'")
            if self.init_training_id is not None or self.training_id is not None:
                raise ValueError("init_training_id and training_id are only for type 'line'")
        else:
            if self.metadata.line is None:
                raise ValueError("metadata.line is required for type 'line'")
            if self.weights_path is not None:
                raise ValueError("weights_path is only for type 'init'")
            if (self.init_training_id is None) == (self.training_id is None):
                raise ValueError("type 'line' needs exactly one of init_training_id and training_id")
        return self


class ModelApply(BaseModel):
    comment: Optional[str] = Field(None, description="Reason for applying")


class ModelExportRequest(BaseModel):
    format: ExportFormat = Field(REQUIRED)


class ModelExport(BaseModel):
    format: ExportFormat
    status: JobStatus
    file_path: Optional[str] = None
    file_hash: Optional[str] = None
    requested_at: datetime
    error: Optional[str] = None

    @classmethod
    def from_job(cls, job: Job) -> "ModelExport":
        result = job.result or {}
        return cls(
            format=job.payload["format"],
            status=job.status,
            file_path=result.get("file_path"),
            file_hash=result.get("file_hash"),
            requested_at=as_utc(job.created_at),
            error=job.error,
        )


class Model(BaseModel):
    model_id: int
    type: ModelType
    metadata: ModelMetadata
    training_id: Optional[int] = None
    init_training_id: Optional[int] = None
    dataset_id: Optional[int] = None
    metrics: Optional[Dict[str, float]] = None
    weights_path: Optional[str] = None
    applied: bool
    applied_at: Optional[datetime] = None
    exports: List[ModelExport] = Field(default_factory=list)
    created_at: datetime

    @classmethod
    def from_row(cls, record: ModelRow, export_jobs: List[Job]) -> "Model":
        return cls(
            model_id=record.id,
            type=record.type,
            metadata=ModelMetadata(
                model_name=record.model_name,
                model_version=record.model_version,
                task=record.task,
                line=record.line,
            ),
            training_id=record.training_id,
            init_training_id=record.init_training_id,
            dataset_id=record.dataset_id,
            metrics=record.metrics,
            weights_path=record.weights_path,
            applied=record.applied,
            applied_at=as_utc(record.applied_at),
            exports=[ModelExport.from_job(job) for job in export_jobs],
            created_at=as_utc(record.created_at),
        )
