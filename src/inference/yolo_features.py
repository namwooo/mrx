from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Sequence, Iterator, Union

import numpy as np


INPUT_SIZE = 640
YOLO11_NECK_LAYER_INDICES = [16, 19, 22]
SMALL, MEDIUM = 32.0, 96.0
MIN_BOX_SIDE = 1.0
LEVEL_POLICIES = ("all", "size_aware")
ROI_SIZE = 3


class YoloFeatureExtractor:
    def __init__(
            self,
            embed_layer_indices: list[int],
            box_layer_indices: list[int]
    ) -> None:
        self.embed_layer_indices = embed_layer_indices
        self.box_layer_indices = box_layer_indices
        self._capture: NeckCapture | None = None
        self._batch: CapturedBatch | None = None

    @property
    def attached(self) -> bool:
        return self._capture is not None

    def attach(self, layers: Sequence[Any]) -> None:
        if self.attached:
            raise RuntimeError("Attached feature extractor already attached")

        hooked_indices = sorted(
            set(self.embed_layer_indices) |
            set(self.box_layer_indices)
        )

        if max(hooked_indices) >= len(layers):
            raise RuntimeError(f"Layers {hooked_indices} are out of range (model has {len(layers)} layers)")

        self._capture = NeckCapture([layers[i] for i in hooked_indices], hooked_indices)

    @contextmanager
    def capture(self, images: list[np.ndarray]) -> Iterator[None]:
        capture = self._require_attached()
        self._batch = None
        capture.start()
        try:
            yield
        finally:
            maps = capture.stop()
        if set(maps) != set(capture.indices):
            raise RuntimeError(f"Captured layers {sorted(maps)} differ from hooked layers {capture.indices}")
        self._batch = CapturedBatch(images, maps)

    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        batch = self._captured_batch(images)
        pooled = [batch.maps[i].float().mean((2, 3)).cpu().numpy() for i in self.embed_layer_indices]
        return np.concatenate(pooled, 1).astype(np.float32)

    def box_features(
            self,
            images: list[np.ndarray],
            boxes: list[np.ndarray],
            level_policy: str = "all"
    ) -> list[np.ndarray]:
        batch = self._captured_batch(images)
        rois = to_letterbox_rois(boxes, [image.shape[:2] for image in batch.images])
        maps = [batch.maps[i] for i in self.box_layer_indices]
        features = pool_boxes(maps, batch.strides(self.box_layer_indices), rois, level_policy)
        return _split_per_image(features, boxes)

    def release(self) -> None:
        self._batch = None

    def _require_attached(self) -> NeckCapture:
        if self._capture is None:
            raise RuntimeError("FeatureExtractor is not attached to a model; pass it to a predictor first")
        return self._capture

    def _captured_batch(self, images: list[np.ndarray]) -> CapturedBatch:
        if not images:
            raise ValueError("images is empty")
        if self._batch is None or not self._batch.matches(images):
            raise RuntimeError("No feature maps for these images; call predict() on the same batch first")
        return self._batch


class NeckCapture:
    def __init__(self, layers: list[Any], indices: list[int]) -> None:
        self.indices = list(indices)
        self.maps: dict[int, Any] = {}
        self.enabled = False
        for index, layer in zip(indices, layers):
            layer.register_forward_hook(self._make_hook(index))

    def _make_hook(self, index: int):
        def hook(_module, _inputs, output):
            if self.enabled:
                self.maps[index] = output.detach()

        return hook

    def start(self) -> None:
        self.maps = {}
        self.enabled = True

    def stop(self) -> dict[int, Any]:
        self.enabled = False
        maps, self.maps = self.maps, {}
        return maps


class CapturedBatch:
    def __init__(self, images: list[np.ndarray], maps: dict[int, Any]) -> None:
        self.images = images
        self.maps = maps

    def matches(self, images: list[np.ndarray]) -> bool:
        return len(images) == len(self.images) and all(a is b for a, b in zip(images, self.images))

    def strides(self, indices: list[int], size: int = INPUT_SIZE) -> list[int]:
        return [round(size / int(self.maps[i].shape[2])) for i in indices]


def _split_per_image(features: np.ndarray, boxes: list[np.ndarray]) -> list[np.ndarray]:
    per_image, start = [], 0
    for image_boxes in boxes:
        n = len(np.asarray(image_boxes).reshape(-1, 4))
        per_image.append(features[start:start + n])
        start += n
    return per_image


def letterbox(height: int, width: int, size: int = INPUT_SIZE) -> tuple[float, float, float]:
    gain = min(size / height, size / width)
    pad_x = round((size - round(width * gain)) / 2 - 0.1)
    pad_y = round((size - round(height * gain)) / 2 - 0.1)
    return gain, pad_x, pad_y


def to_letterbox_rois(boxes: list[np.ndarray], shapes: list[tuple[int, int]], size: int = INPUT_SIZE) -> np.ndarray:
    rows = []
    for image_index, (image_boxes, (height, width)) in enumerate(zip(boxes, shapes)):
        xyxy = np.asarray(image_boxes, np.float64).reshape(-1, 4)
        if not len(xyxy):
            continue

        gain, pad_x, pad_y = letterbox(height, width, size)
        xyxy = xyxy * gain + np.array([pad_x, pad_y, pad_x, pad_y])
        _widen_thin_sides(xyxy)

        image_column = np.full((len(xyxy), 1), image_index, np.float64)
        rows.append(np.concatenate([image_column, xyxy], 1))

    return np.concatenate(rows) if rows else np.zeros((0, 5))


def _widen_thin_sides(xyxy: np.ndarray) -> None:
    half = MIN_BOX_SIDE / 2
    for lo, hi in ((0, 2), (1, 3)):  # x side, then y side
        thin = (xyxy[:, hi] - xyxy[:, lo]) < MIN_BOX_SIDE
        center = (xyxy[thin, hi] + xyxy[thin, lo]) / 2
        xyxy[thin, lo] = center - half
        xyxy[thin, hi] = center + half


def box_size(xyxy: Union[list[float], np.ndarray], height: int, width: int) -> float:
    gain = letterbox(height, width)[0]
    x1, y1, x2, y2 = xyxy
    return float(math.sqrt(max(x2 - x1, 0.0) * max(y2 - y1, 0.0)) * gain)


def size_bucket(size: float) -> str:
    if size < SMALL:
        return "small"
    if size < MEDIUM:
        return "medium"
    return "large"


def pool_boxes(
        maps: list[Any],
        strides: list[int],
        rois: np.ndarray,
        policy: str = "all",
        roi_size: int = ROI_SIZE,
) -> np.ndarray:
    import torch
    from torchvision.ops import roi_align

    channels = sum(int(m.shape[1]) for m in maps)
    if not len(rois):
        return np.zeros((0, channels), np.float32)

    weights = level_weights(rois, strides, policy)
    blocks = []
    for level, (feature_map, stride) in enumerate(zip(maps, strides)):
        level_rois = torch.as_tensor(rois, dtype=feature_map.dtype, device=feature_map.device)
        pooled = roi_align(
            feature_map,
            level_rois,
            roi_size,
            spatial_scale=1.0 / stride,  # input pixels -> feature map cells
            sampling_ratio=2,
            aligned=True,
        )
        per_box = pooled.mean((2, 3))  # average the roi_size x roi_size cells -> (boxes, C_l)
        blocks.append(per_box.float().cpu().numpy() * weights[:, level:level + 1])
    return np.concatenate(blocks, 1).astype(np.float32)


def level_weights(rois: np.ndarray, strides: list[int], policy: str) -> np.ndarray:
    if policy not in LEVEL_POLICIES:
        raise ValueError(f"unknown level policy {policy!r}. allowed: {LEVEL_POLICIES}")
    if policy == "all" or not len(rois):
        return np.ones((len(rois), len(strides)))

    widths = rois[:, 3] - rois[:, 1]
    heights = rois[:, 4] - rois[:, 2]
    short_side = np.minimum(widths, heights)
    return np.minimum(1.0, short_side[:, None] / np.asarray(strides, np.float64)[None, :])
