import logging
from typing import Any

import numpy as np
from inference.predictor import Predictor, ImageDetections, RawDetection
from inference.yolo_features import INPUT_SIZE, YoloFeatureExtractor
from torch import nn
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class YoloPredictor(Predictor):
    def __init__(
            self,
            model_path: str,
            feature_extractor: YoloFeatureExtractor | None = None,
    ) -> None:
        logger.info("Loading YOLO model: %s", model_path)
        self.model = YOLO(model_path)
        self.feature_extractor = feature_extractor
        if feature_extractor is not None:
            feature_extractor.attach(_layers_of(self.model))

    def predict(self, images: list[np.ndarray], **options: Any) -> list[ImageDetections]:
        return self._predict(images, **options)

    def _predict(
            self,
            images: list[np.ndarray],
            *,
            conf: float = 0.25,
            iou: float = 0.7,
            max_det: int = 300,
            classes = None,
            image_size= INPUT_SIZE,
    ) -> list[ImageDetections]:
        options = dict(conf=conf, iou=iou, max_det=max_det, classes=classes, image_size=image_size)

        if self.feature_extractor is None:
            results = self._run(images, **options)
        elif image_size == INPUT_SIZE:
            with self.feature_extractor.capture(images):
                results = self._run(images, **options)
        else:
            self.feature_extractor.release()
            results = self._run(images, **options)

        return [_to_raw_detections(result) for result in results]

    def _run(
            self,
            images: list[np.ndarray],
            *,
            conf: float,
            iou: float,
            max_det: int,
            classes: list[int] | None,
            image_size: int
    ) -> list[Any]:
        return list(self.model.predict(
            source=images,
            conf=conf,
            iou=iou,
            imgsz=image_size,
            rect=False,
            max_det=max_det,
            classes=classes,
            batch=len(images),
            save=False,
            verbose=False,
        ))


def _to_raw_detections(result: Any) -> ImageDetections:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.cpu().tolist()  # ultralytics already maps boxes back to original pixels
    confidence = boxes.conf.cpu().tolist()
    class_ids = boxes.cls.cpu().tolist()  # floats in ultralytics
    return [
        RawDetection(class_id=int(c), confidence=float(s), xyxy=[float(v) for v in b])
        for b, s, c in zip(xyxy, confidence, class_ids)
    ]


def _layers_of(model: YOLO) -> list[nn.Module]:
    return model.model.model