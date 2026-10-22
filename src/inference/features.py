"""
Image embeddings and box features from YOLO neck feature maps.

Ported from the validation script work/yolo11_feature_extract.py (verified 10/02) and extended.

Forward hooks on the neck outputs (YOLO11: layers 16, 19, 22 = P3, P4, P5) let one inference give three things:
    detections        ultralytics predict (640x640 letterbox, rect=False)
    image embedding   global average pooling (GAP) of each captured feature map, concatenated.
                      Same values as ultralytics model.embed, which also GAPs the same layers.
    box features      box in original pixels -> letterboxed input pixels -> RoIAlign (3x3, aligned) on each level
                      -> average over the 3x3 cells -> concatenated across levels

Details
- Batching: the boxes of all images in a batch go through a single RoIAlign call. Each RoI carries the
  index of its image (first column), and each image uses its own letterbox scale and padding.
  With rect=False every input in the batch is 640x640.
- Shared inference: feature maps captured during the main inference (640) are reused for that batch's
  embeddings and box features. A different batch or input size (TTA) needs a new inference.
- Small objects: sides shorter than 1px in the letterboxed input are widened to 1px, so a clipped box
  never has zero area. box_size() gives the box size (sqrt(w*h) in letterboxed pixels), which evaluation
  uses to look at AUROC per size bucket.
- Level policy (how levels are combined), chosen by configuration:
    all         every level gets the same weight (default, same as the validation script)
    size_aware  if the box's short side is smaller than a level's grid cell (stride), that level's block
                is weighted by (short side / stride). For a box that does not fill even one P5 cell
                (32px), the P5 value describes the background around it more than the object.
"""

from __future__ import annotations

import math
from typing import Any, Union

import numpy as np

INPUT_SIZE = 640
# Each box is pooled into ROI_SIZE x ROI_SIZE cells with RoIAlign, then the cells are averaged.
ROI_SIZE = 3

# Box size buckets, by sqrt(w*h) in letterboxed input pixels.
# Same boundaries as COCO's small/medium/large area ranges (32^2, 96^2).
SMALL, MEDIUM = 32.0, 96.0

# Smallest box side in letterboxed input pixels. Thinner sides are widened to this around their center.
MIN_BOX_SIDE = 1.0

LEVEL_POLICIES = ("all", "size_aware")


# ---------------------------------------------------------------------------
# Letterbox geometry and box sizes
# ---------------------------------------------------------------------------

def letterbox(height: int, width: int, size: int = INPUT_SIZE) -> tuple[float, float, float]:
    """Scale and padding ultralytics uses to fit an image into a size x size square.

    Returns (gain, pad_x, pad_y), so that input = original * gain + pad.
    Padding is rounded to whole pixels exactly like ultralytics LetterBox / scale_boxes
    (computing it as a float would be off by up to 0.5px).
    """
    gain = min(size / height, size / width)
    pad_x = round((size - round(width * gain)) / 2 - 0.1)
    pad_y = round((size - round(height * gain)) / 2 - 0.1)
    return gain, pad_x, pad_y


def box_size(xyxy: Union[list[float], np.ndarray], height: int, width: int) -> float:
    """Box size sqrt(w*h) in letterboxed input pixels. height/width are of the original image."""
    gain = letterbox(height, width)[0]
    x1, y1, x2, y2 = xyxy
    return float(math.sqrt(max(x2 - x1, 0.0) * max(y2 - y1, 0.0)) * gain)


def size_bucket(size: float) -> str:
    """"small", "medium" or "large" for a box_size() value."""
    if size < SMALL:
        return "small"
    if size < MEDIUM:
        return "medium"
    return "large"


def as_matrix(features: Any, rows: int) -> np.ndarray:
    """Box features as a (rows, dim) float32 matrix. An image with no boxes may give shape (0, 0)."""
    array = np.asarray(features, np.float32)
    if rows == 0:
        return array.reshape(0, array.shape[-1] if array.ndim == 2 else 0)
    return array.reshape(rows, -1)


# ---------------------------------------------------------------------------
# Box features
# ---------------------------------------------------------------------------

def to_input_rois(boxes: list[np.ndarray], shapes: list[tuple[int, int]], size: int = INPUT_SIZE) -> np.ndarray:
    """Convert per-image boxes in original pixels into RoIAlign input for the whole batch.

    boxes   one (n_i, 4) xyxy array per image, in original image pixels
    shapes  (height, width) of each original image
    Returns shape (sum of n_i, 5): [image index in the batch, x1, y1, x2, y2] in letterboxed input pixels.
    """
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
    """In place: widen every side shorter than MIN_BOX_SIDE to MIN_BOX_SIDE, keeping its center."""
    half = MIN_BOX_SIDE / 2
    for lo, hi in ((0, 2), (1, 3)):  # x side, then y side
        thin = (xyxy[:, hi] - xyxy[:, lo]) < MIN_BOX_SIDE
        center = (xyxy[thin, hi] + xyxy[thin, lo]) / 2
        xyxy[thin, lo] = center - half
        xyxy[thin, hi] = center + half


def level_weights(rois: np.ndarray, strides: list[int], policy: str) -> np.ndarray:
    """Weight of each level's block for each box, shape (number of boxes, number of levels).

    all         every weight is 1
    size_aware  min(1, short side / stride)
    """
    if policy not in LEVEL_POLICIES:
        raise ValueError(f"unknown level policy {policy!r}. allowed: {LEVEL_POLICIES}")
    if policy == "all" or not len(rois):
        return np.ones((len(rois), len(strides)))

    widths = rois[:, 3] - rois[:, 1]
    heights = rois[:, 4] - rois[:, 2]
    short_side = np.minimum(widths, heights)
    return np.minimum(1.0, short_side[:, None] / np.asarray(strides, np.float64)[None, :])


def pool_boxes(
        maps: list[Any],
        strides: list[int],
        rois: np.ndarray,
        policy: str = "all",
        roi_size: int = ROI_SIZE,
) -> np.ndarray:
    """Pool box features from each level's feature map and concatenate them.

    maps     one feature map per level, shape (B, C_l, H_l, W_l)
    strides  input pixels per feature map cell, for each level
    rois     output of to_input_rois
    For each level: RoIAlign into roi_size x roi_size cells -> average the cells -> multiply by the level weight.
    Returns shape (number of boxes, sum of C_l), float32.
    """
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


# ---------------------------------------------------------------------------
# Capturing feature maps
# ---------------------------------------------------------------------------

class NeckCapture:
    """Forward hooks that record the outputs of the given layers.

    Records only between start() and stop(). If a layer runs more than once in that time
    (e.g. the ultralytics warmup pass), the last output wins.
    """

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
        """Start recording, dropping anything recorded before."""
        self.maps = {}
        self.enabled = True

    def stop(self) -> dict[int, Any]:
        """Stop recording and hand over what was recorded: {layer index: feature map}."""
        self.enabled = False
        maps, self.maps = self.maps, {}
        return maps


class CapturedBatch:
    """Feature maps captured for one batch (a list of image arrays).

    matches() compares the arrays by identity. This object holds references to the arrays,
    so their ids cannot be reused by new arrays while it exists.
    """

    def __init__(self, images: list[np.ndarray], maps: dict[int, Any]) -> None:
        self.images = list(images)
        self.maps = maps

    def matches(self, images: list[np.ndarray]) -> bool:
        return len(images) == len(self.images) and all(a is b for a, b in zip(images, self.images))

    def strides(self, indices: list[int], size: int = INPUT_SIZE) -> list[int]:
        """Input pixels per feature map cell for each layer, e.g. [8, 16, 32] for P3, P4, P5."""
        return [round(size / int(self.maps[i].shape[2])) for i in indices]

    def embed(self, indices: list[int]) -> np.ndarray:
        """GAP each layer and concatenate: shape (number of images, sum of C), float32.

        Layers are concatenated in ascending index order: the same computation as ultralytics model.embed.
        """
        import torch

        pooled = [self.maps[i].float().mean((2, 3)) for i in sorted(indices)]
        return torch.cat(pooled, 1).cpu().numpy().astype(np.float32)

    def box_features(
            self,
            indices: list[int],
            boxes: list[np.ndarray],
            policy: str,
            roi_size: int = ROI_SIZE,
    ) -> list[np.ndarray]:
        """Box features for each image: (n_i, sum of C) float32, levels in the order of indices."""
        shapes = [image.shape[:2] for image in self.images]
        rois = to_input_rois(boxes, shapes)
        features = pool_boxes([self.maps[i] for i in indices], self.strides(indices), rois, policy, roi_size)

        # Split the batch result back into one array per image.
        per_image, start = [], 0
        for image_boxes in boxes:
            n = len(np.asarray(image_boxes).reshape(-1, 4))
            per_image.append(features[start:start + n])
            start += n
        return per_image