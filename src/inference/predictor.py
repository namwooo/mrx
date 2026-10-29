from abc import ABC, abstractmethod
from typing import Any, TypedDict

import numpy as np

class RawDetection(TypedDict):
    """One box exactly as the model produced it."""
    class_id: int
    confidence: float
    xyxy: list[float]


ImageDetection = list[RawDetection]


class Predictor(ABC):
    @abstractmethod
    def predict(self, images: list[np.ndarray], **options: Any) -> list[ImageDetection]:
        """Return one list of detections per image."""