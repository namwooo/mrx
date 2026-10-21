from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

REQUIRED = ...

class BatchJobCreate(BaseModel):
    model: str = Field(REQUIRED, description="Model name")
    input_path: str = Field(REQUIRED, description="Input path")
    params: Dict[str, Any] = Field(default_factory=dict, description="conf, iou ...")
    idempotency_key: Optional[str] = Field(None, max_length=128)