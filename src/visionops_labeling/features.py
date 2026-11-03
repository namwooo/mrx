"""검출 모델(YOLO11) 이미지 임베딩(③)·박스 특징(④) 추출. work/yolo11_feature_extract.py(10/02 검증 스크립트)를 옮기고 보완했다.
torch·torchvision 은 이 모듈의 특징맵 함수를 부를 때만 필요하다(좌표 변환·크기 구간은 numpy 만 쓴다).

neck 출력(YOLO11: 16·19·22 = P3·P4·P5)에 forward hook 을 걸어 추론 한 번으로 세 가지를 얻는다.
    탐지 결과          ultralytics predict (640×640 letterbox, rect=False)
    이미지 임베딩(③)   hook 으로 잡은 특징맵의 GAP 를 이어붙인다. ultralytics model.embed 와 같은 값이다(그쪽도 같은 레이어 GAP)
    박스 특징(④)       원본 좌표 박스 → letterbox 입력 좌표 → 레벨마다 RoIAlign(3×3, aligned) → 칸 평균 → 이어붙임

보완한 점
- 배치: 묶음 안 여러 이미지의 박스를 한 번의 RoIAlign 으로 처리한다. 박스마다 몇 번째 이미지인지(RoI 첫 열)와 이미지별
  letterbox 배율·패딩을 따로 넘긴다(검증 스크립트는 한 장씩만 처리했다). rect=False 라 묶음의 모든 입력이 640×640 이다.
- 추론 공유: 원본 추론(640) 때 잡은 특징맵을 같은 이미지 묶음의 임베딩·박스 특징에 다시 쓴다(묶음당 한 번). 예전에는
  model.embed 로 한 번 더 추론했다. 다른 묶음·다른 입력 크기(TTA)이면 특징맵을 새로 계산한다.
- 작은 객체: letterbox 입력에서 1px 보다 작은 박스는 1px 로 넓힌다(잘린 박스가 넓이 0 이 되지 않게). 박스 크기
  (letterbox 입력 기준 √(w·h))를 함께 돌려줘 평가(LABEL_EVAL)에서 크기 구간별 AUROC 를 본다. 레벨 사용 방식은 설정으로 고른다.
    all         모든 레벨을 같은 비중으로 이어붙인다(검증 스크립트 방식, 기본)
    size_aware  박스 짧은 변이 그 레벨의 격자 한 칸(stride)보다 작으면 그 레벨 블록에 (짧은 변 / stride) 비중을 준다.
                P5(32px 격자)에서 한 칸도 채우지 못하는 작은 박스는 P5 값이 물체보다 주변 배경을 담기 때문이다

회전 박스(obb, OBB 라벨링 지원 계획 O5): 박스 (n, 5) xywhr 는 pooling 방식으로 특징을 뽑는다(거리 통계에 기록해 같은 방식을 쓴다).
    rotated     회전 샘플링(기본). RoIAlign 과 같은 표본 점(3×3 칸 × 칸마다 2×2 = 박스 안 6×6 균등 격자)을 박스 각도로 돌려
                grid_sample 로 읽고 평균한다. 각도 0 이면 RoIAlign(aligned) 과 같은 값이다(테스트). 칸을 평균하므로 칸 순서·각도
                표현(w·h 바뀜, π 주기)과 무관하다
    enclosing   감싸는 사각형으로 RoIAlign. 비교용. 길고 기운 박스는 특징 대부분이 배경이고, 각도만 다른 박스(거울상)를 구분하지 못한다
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .constants import INPUT_SIZE

# 박스 하나를 3×3 칸으로 나눠 특징을 뽑은 뒤 칸 평균을 낸다(RoIAlign 출력 크기).
ROI_SIZE = 3
SMALL, MEDIUM = 32.0, 96.0  # 박스 크기 구간(letterbox 입력 기준 √(w·h), COCO 의 32²·96² 넓이 구간과 같은 경계)


OBB_POOLING = ("rotated", "enclosing")


# 원본 좌표 → letterbox 입력 좌표 변환: x' = x·gain + padw, y' = y·gain + padh
def letterbox(height: int, width: int, size: int = INPUT_SIZE) -> tuple[float, float, float]:
    """(배율, x 패딩, y 패딩). ultralytics LetterBox·scale_boxes 와 같은 정수 패딩(소수로 계산하면 최대 0.5px 어긋난다)."""
    gain = min(size / height, size / width)
    padw = round((size - round(width * gain)) / 2 - 0.1)
    padh = round((size - round(height * gain)) / 2 - 0.1)
    return gain, padw, padh


def box_size(xyxy: list[float] | np.ndarray, height: int, width: int) -> float:
    """박스의 letterbox 입력 기준 크기 √(w·h) (px)."""
    gain = letterbox(height, width)[0]
    x1, y1, x2, y2 = xyxy
    return float(math.sqrt(max(x2 - x1, 0.0) * max(y2 - y1, 0.0)) * gain)


# 박스 크기 구간 이름: small(<32) · medium(<96) · large. 평가에서 크기별 AUROC 를 따로 본다.
def size_bucket(size: float) -> str:
    return "small" if size < SMALL else "medium" if size < MEDIUM else "large"


def as_matrix(features, rows: int) -> np.ndarray:
    """박스 특징을 (rows, 차원) float32 로. 박스가 없는 이미지는 (0, 0) 이 될 수 있다."""
    array = np.asarray(features, np.float32)
    if rows == 0:
        return array.reshape(0, array.shape[-1] if array.ndim == 2 else 0)
    return array.reshape(rows, -1)


def to_input_rois(boxes: list[np.ndarray], shapes: list[tuple[int, int]], size: int = INPUT_SIZE) -> np.ndarray:
    """이미지별 원본 좌표 박스 (n_i, 4) → RoIAlign 입력 (Σn_i, 5) [묶음 번호, x1, y1, x2, y2] (letterbox 입력 좌표).
    1px 보다 작은 변은 중심을 유지한 채 1px 로 넓힌다."""
    rows = []
    for index, (b, (height, width)) in enumerate(zip(boxes, shapes)):
        b = np.asarray(b, np.float64).reshape(-1, 4)
        if not len(b):
            continue
        gain, padw, padh = letterbox(height, width, size)
        r = b * gain + np.array([padw, padh, padw, padh])
        for lo, hi in ((0, 2), (1, 3)):
            short = (r[:, hi] - r[:, lo]) < 1.0
            center = (r[short, hi] + r[short, lo]) / 2
            r[short, lo], r[short, hi] = center - 0.5, center + 0.5
        rows.append(np.concatenate([np.full((len(r), 1), index, np.float64), r], 1))
    return np.concatenate(rows) if rows else np.zeros((0, 5))


def level_weights(rois: np.ndarray, strides: list[int], policy: str) -> np.ndarray:
    """(박스 수, 레벨 수) 레벨 블록 비중. all 은 모두 1, size_aware 는 min(1, 짧은 변 / stride).
    rois: RoIAlign 입력 (n, 5) [묶음, x1, y1, x2, y2] 또는 회전 박스 (n, 6) [묶음, cx, cy, w, h, θ]."""
    if policy == "all" or not len(rois):
        return np.ones((len(rois), len(strides)))
    if rois.shape[1] == 6:
        short = np.minimum(rois[:, 3], rois[:, 4])
    else:
        short = np.minimum(rois[:, 3] - rois[:, 1], rois[:, 4] - rois[:, 2])
    return np.minimum(1.0, short[:, None] / np.asarray(strides, np.float64)[None, :])


def to_input_rotated(boxes: list[np.ndarray], shapes: list[tuple[int, int]], size: int = INPUT_SIZE) -> np.ndarray:
    """이미지별 원본 좌표 회전 박스 (n_i, 5) xywhr → (Σn_i, 6) [묶음 번호, cx, cy, w, h, θ] (letterbox 입력 좌표).
    letterbox 는 같은 배율이라 각도가 그대로다. 1px 보다 작은 변은 1px 로 넓힌다."""
    rows = []
    for index, (b, (height, width)) in enumerate(zip(boxes, shapes)):
        b = np.asarray(b, np.float64).reshape(-1, 5)
        if not len(b):
            continue
        gain, padw, padh = letterbox(height, width, size)
        r = np.column_stack([b[:, 0] * gain + padw, b[:, 1] * gain + padh, np.maximum(b[:, 2] * gain, 1.0),
                             np.maximum(b[:, 3] * gain, 1.0), b[:, 4]])
        rows.append(np.concatenate([np.full((len(r), 1), index, np.float64), r], 1))
    return np.concatenate(rows) if rows else np.zeros((0, 6))


def pool_rotated(maps: list[Any], strides: list[int], rois: np.ndarray, policy: str = "all", roi_size: int = ROI_SIZE,
                 sampling: int = 2) -> np.ndarray:
    """레벨별 특징맵에서 회전 박스 (n, 6) 안의 (roi_size·sampling)² 균등 표본 점을 bilinear 로 읽어 평균 → 이어붙임.
    RoIAlign(aligned, sampling_ratio=sampling) 의 칸 평균과 같은 점·같은 보간이다(박스가 맵 안에 있을 때). (박스 수, ΣC_l) float32"""
    import torch
    import torch.nn.functional as F

    channels = sum(int(m.shape[1]) for m in maps)
    if not len(rois):
        return np.zeros((0, channels), np.float32)
    weights = level_weights(rois, strides, policy)
    g = roi_size * sampling
    u = (np.arange(g) + 0.5) / g - 0.5                       # 박스 좌표계 표본 위치 (-0.5 ~ 0.5)
    ux, uy = np.meshgrid(u, u)
    ux, uy = ux.reshape(1, -1), uy.reshape(1, -1)
    cx, cy, w, h, theta = (rois[:, k:k + 1] for k in range(1, 6))
    c, s = np.cos(theta), np.sin(theta)
    x = cx + ux * w * c - uy * h * s                          # geometry.corners 와 같은 축: w 축 (cos, sin), h 축 (−sin, cos)
    y = cy + ux * w * s + uy * h * c
    batch = rois[:, 0].astype(int)
    blocks = []
    for level, (fmap, stride) in enumerate(zip(maps, strides)):
        height, width = int(fmap.shape[2]), int(fmap.shape[3])
        # 입력 좌표 → grid_sample 정규화 좌표(align_corners=False): 맵 칸 중심이 x/stride − 0.5 (RoIAlign aligned 와 같다)
        grid = np.stack([x / stride * 2 / width - 1, y / stride * 2 / height - 1], -1)
        pooled = np.zeros((len(rois), int(fmap.shape[1])), np.float32)
        for b in np.unique(batch):
            rows = np.nonzero(batch == b)[0]
            sample = torch.as_tensor(grid[rows][None], dtype=fmap.dtype, device=fmap.device)  # (1, n_b, g², 2)
            # 맵 밖 표본은 가장자리 값(RoIAlign 은 맵 밖 한 칸까지 가장자리로 자르고 그 밖은 0 — 박스가 맵 밖으로 한 칸 넘게 나갈 때만 다르다)
            values = F.grid_sample(fmap[b:b + 1], sample, mode="bilinear", padding_mode="border", align_corners=False)
            pooled[rows] = values[0].mean(2).transpose(0, 1).float().cpu().numpy()       # (C, n_b, g²) → (n_b, C)
        blocks.append(pooled * weights[:, level:level + 1])
    return np.concatenate(blocks, 1).astype(np.float32)


# spatial_scale=1/stride: 입력 픽셀 좌표를 그 레벨 특징맵 좌표로 줄인다(P3=8, P4=16, P5=32).
def pool_boxes(maps: list[Any], strides: list[int], rois: np.ndarray, policy: str = "all",
               roi_size: int = ROI_SIZE) -> np.ndarray:
    """레벨별 특징맵 (B, C_l, H_l, W_l) 에서 RoIAlign → 칸 평균 → 이어붙임. (박스 수, ΣC_l) float32"""
    import torch
    from torchvision.ops import roi_align

    channels = sum(int(m.shape[1]) for m in maps)
    if not len(rois):
        return np.zeros((0, channels), np.float32)
    weights = level_weights(rois, strides, policy)
    blocks = []
    for level, (fmap, stride) in enumerate(zip(maps, strides)):
        r = torch.as_tensor(rois, dtype=fmap.dtype, device=fmap.device)
        pooled = roi_align(fmap, r, roi_size, spatial_scale=1.0 / stride, sampling_ratio=2, aligned=True).mean((2, 3))
        blocks.append(pooled.float().cpu().numpy() * weights[:, level:level + 1])
    return np.concatenate(blocks, 1).astype(np.float32)


# hook 은 모델에 한 번 걸어 두고 enabled 플래그로 켜고 끈다(TTA 추론 등 필요 없을 때는 잡지 않음).
class NeckCapture:
    """지정 레이어 출력을 잡는 forward hook. capture 가 켜진 동안의 마지막 forward 출력만 남긴다(워밍업 출력은 덮어쓴다)."""

    def __init__(self, layers: list[Any], indices: list[int]):
        self.indices, self.maps, self.enabled = list(indices), {}, False
        for index, layer in zip(indices, layers):
            layer.register_forward_hook(self._hook(index))

    def _hook(self, index: int):
        def fn(_module, _inputs, output):
            if self.enabled:
                self.maps[index] = output.detach()
        return fn

    def start(self) -> None:
        self.maps, self.enabled = {}, True

    def stop(self) -> dict[int, Any]:
        self.enabled = False
        maps, self.maps = self.maps, {}
        return maps


class CapturedBatch:
    """한 묶음(이미지 배열 목록)의 특징맵. 같은 배열 객체 목록이면 다시 쓴다(배열 참조를 쥐고 있어 id 재사용이 없다)."""

    def __init__(self, images: list[np.ndarray], maps: dict[int, Any]):
        self.images, self.maps = list(images), maps

    def matches(self, images: list[np.ndarray]) -> bool:
        return len(images) == len(self.images) and all(a is b for a, b in zip(images, self.images))

    # 특징맵 높이로 stride 를 역산한다. 예: 640 입력에서 80×80 맵이면 stride 8.
    def strides(self, indices: list[int], size: int = INPUT_SIZE) -> list[int]:
        return [round(size / int(self.maps[i].shape[2])) for i in indices]

    def embed(self, indices: list[int]) -> np.ndarray:
        """레이어 GAP 이어붙임 (이미지 수, ΣC). ultralytics model.embed 와 같은 계산(레이어 번호 순)."""
        import torch

        pooled = [self.maps[i].float().mean((2, 3)) for i in sorted(indices)]
        return torch.cat(pooled, 1).cpu().numpy().astype(np.float32)

    def box_features(self, indices: list[int], boxes: list[np.ndarray], policy: str, roi_size: int = ROI_SIZE,
                     pooling: str = "rotated") -> list[np.ndarray]:
        """이미지마다 박스 특징. 박스가 (n, 4) xyxy 면 RoIAlign, (n, 5) xywhr 면 pooling(rotated·enclosing)."""
        shapes = [image.shape[:2] for image in self.images]
        columns = next((np.asarray(b).shape[1] for b in boxes if np.asarray(b).ndim == 2 and len(b)), 4)
        maps, strides = [self.maps[i] for i in indices], self.strides(indices)
        if columns == 5 and pooling == "rotated":
            feats = pool_rotated(maps, strides, to_input_rotated(boxes, shapes), policy, roi_size)
        else:
            if columns == 5:  # enclosing: 감싸는 사각형
                from .geometry import enclosing
                boxes = [enclosing(np.asarray(b, np.float64).reshape(-1, 5)) for b in boxes]
            feats = pool_boxes(maps, strides, to_input_rois(boxes, shapes), policy, roi_size)
        out, start = [], 0
        for b in boxes:
            n = len(np.asarray(b).reshape(-1, columns if columns == 5 and pooling == "rotated" else 4))
            out.append(feats[start:start + n])
            start += n
        return out
