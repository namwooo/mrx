from typing import Optional, TypedDict, Any

import numpy as np
from inference.features import NeckCapture, CapturedBatch
from torch import nn
from ultralytics import YOLO as UYOLO

DEFAULT_NECK_LAYERS = [16, 19, 22]
INPUT_SIZE = 640


class Detection(TypedDict):
    class_id: int
    confidence: float
    xyxy: list[float]


class YOLO:
    def __init__(
            self,
            model_path: str,
            embed_layers: Optional[list[int]] = None,
            feature_layers: Optional[list[int]] = None
    ) -> None:
        print("model_path", model_path)
        self.model = UYOLO(model_path)

        self.embed_layers = embed_layers or DEFAULT_NECK_LAYERS
        self.feature_layers = feature_layers or DEFAULT_NECK_LAYERS

        layers = _layers_of(self.model)
        hooked = sorted(set(self.embed_layers) | set(self.feature_layers))
        _check_layer_range(hooked, len(layers))

        self._capture = NeckCapture([layers[i] for i in hooked], hooked)
        self._last_batch: Optional[CapturedBatch] = None

    def predict(
            self,
            images: list[np.ndarray],
            *,
            conf: float = 0.25,
            iou: float = 0.7,
            max_det: int = 300,
            classes = None,
            image_size= INPUT_SIZE):
        keep_features = image_size == INPUT_SIZE
        self._last_batch = None

        if keep_features:
            self._capture.start()
        try:
            results = self.model.predict(
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
            )
        finally:
            feature_maps = self._capture.stop()

        if keep_features:
            self._remember_batch(images, feature_maps)

        return [_to_detections(result) for result in results]

    def embed(self, images):
        if not images:
            return np.zeros((0, 0), np.float32)
        return self._features_for(images).embed(self.embed_layers)

    def box_features(self, images, boxes, levels="all"):
        if not images:
            return []
        return self._features_for(images).box_features(self.feature_layers, boxes, levels)

    def release(self) -> None:
        self._last_batch = None

    def _features_for(self, images) -> CapturedBatch:
        batch = self._last_batch
        if batch is None or not batch.matches(images):
            self.predict(images, conf=1.0, iou=0.7, max_det=1, classes=None)
            batch = self._last_batch
            if batch is None:
                raise RuntimeError
        return batch

    def _remember_batch(self, images, feature_maps) -> None:
        if set(feature_maps) != set(self._capture.indices):
            raise RuntimeError
        self._last_batch = CapturedBatch(images, feature_maps)


def _to_detections(result: Any) -> list[Detection]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.cpu().tolist()
    confidence = boxes.conf.cpu().tolist()
    class_ids = boxes.cls.cpu().tolist()
    return [
        {"class_id": int(c), "confidence": float(s), "xyxy": [float(v) for v in b]}
        for b, s, c in zip(xyxy, confidence, class_ids)
    ]


def _check_layer_range(layers: list[int], num_layers: int) -> None:
    if max(layers) >= num_layers:
        raise RuntimeError(f"{layers} are out of range ({num_layers})")


def _layers_of(model) -> nn.Sequential:
    return model.model.model