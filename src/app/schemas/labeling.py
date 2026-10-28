from typing import Optional

from pydantic import BaseModel, Field

REQUIRED = ...


class LabelingParams(BaseModel):
    conf: float = Field(0.25, ge=0, le=1, description="Keep boxes at or above this confidence")
    iou: float = Field(0.7, ge=0, le=1, description="IoU threshold for NMS")
    max_det: int = Field(300, ge=1, description="Maximum number of boxes per image")


class LabelingCreate(BaseModel):
    model: str = Field(REQUIRED, description="Path of the model weights used for labeling")
    input_path: str = Field(REQUIRED, description="Image file or folder to label")
    params: LabelingParams = Field(default_factory=LabelingParams)
    idempotency_key: Optional[str] = Field(None, max_length=128)
