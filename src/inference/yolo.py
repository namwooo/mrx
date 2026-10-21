import numpy as np
from ultralytics import YOLO as UYOLO

class YOLO:
    def __init__(self, model: str) -> None:
        self.model = UYOLO(model)

    def predict(self, source: list[np.ndarray]):
        results = self.model.predict(
            source=source
        )

        return results