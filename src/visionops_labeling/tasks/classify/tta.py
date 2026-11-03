"""분류 ② TTA 변환 (분류 어댑터 규격 1.0 4.4절의 변환 이름을 플랫폼이 구현).

어댑터는 정답(클래스)을 바꾸지 않는 변환 이름만 선언하고(Adapter.tta), 변환은 여기서 원본 이미지에 적용한 뒤 predict 로 점수를 받는다.
이름 하나가 변형 하나 또는 둘(± 방향)로 펼쳐진다. 결과 자료형·채널 수는 입력과 같다(uint8·uint16 컬러·회색조).

    hflip · vflip              좌우 · 상하 뒤집기
    rot90 · rot180 · rot270    시계 방향 회전 (정사각이 아니면 가로·세로가 바뀐다)
    brightness                 밝기 ×1.1 · ×0.9 (자료형 범위로 자름)                      변형 brightness+10 · brightness-10
    contrast                   채널 평균 기준 대비 ×1.1 · ×0.9                            변형 contrast+10 · contrast-10
    blur                       가우시안 블러 σ 0.8
    scale                      크기 ×1.1 · ×0.9 후 가운데를 원래 크기로 자르기 · 가장자리 복제로 채우기  변형 scale+10 · scale-10
"""
from __future__ import annotations

import numpy as np

TRANSFORMS = ("hflip", "vflip", "rot90", "rot180", "rot270", "brightness", "contrast", "blur", "scale")
VARIANTS = {"hflip": ("hflip",), "vflip": ("vflip",), "rot90": ("rot90",), "rot180": ("rot180",), "rot270": ("rot270",),
            "brightness": ("brightness+10", "brightness-10"), "contrast": ("contrast+10", "contrast-10"),
            "blur": ("blur",), "scale": ("scale+10", "scale-10")}
BLUR_SIGMA = 0.8
STEP = 0.1  # 밝기·대비·크기 변화 비율


def expand(names: list[str] | tuple[str, ...]) -> list[str]:
    """변환 이름 → 변형 이름 목록(선언 순서, 중복 제거). 규격 밖 이름은 ValueError."""
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        raise ValueError(f"규격에 없는 TTA 변환입니다: {unknown} (가능: {list(TRANSFORMS)})")
    out: list[str] = []
    for name in names:
        out += [v for v in VARIANTS[name] if v not in out]
    return out


def _clip(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.rint(values), info.min, info.max).astype(dtype)
    return values.astype(dtype)


def apply(image: np.ndarray, variant: str) -> np.ndarray:
    """변형 하나를 적용한 새 배열(입력은 바꾸지 않는다)."""
    import cv2

    if variant == "hflip":
        return np.ascontiguousarray(image[:, ::-1])
    if variant == "vflip":
        return np.ascontiguousarray(image[::-1])
    if variant in ("rot90", "rot180", "rot270"):
        k = {"rot90": -1, "rot180": 2, "rot270": 1}[variant]  # np.rot90 은 반시계 방향이 양수
        return np.ascontiguousarray(np.rot90(image, k, axes=(0, 1)))
    sign = 1.0 if variant.endswith("+10") else -1.0
    if variant.startswith("brightness"):
        return _clip(image.astype(np.float32) * (1.0 + sign * STEP), image.dtype)
    if variant.startswith("contrast"):
        values = image.astype(np.float32)
        mean = values.mean(axis=(0, 1), keepdims=True)
        return _clip(mean + (values - mean) * (1.0 + sign * STEP), image.dtype)
    if variant == "blur":
        return cv2.GaussianBlur(image, (0, 0), sigmaX=BLUR_SIGMA)
    if variant.startswith("scale"):
        height, width = image.shape[:2]
        factor = 1.0 + sign * STEP
        resized = cv2.resize(image, (max(1, round(width * factor)), max(1, round(height * factor))),
                             interpolation=cv2.INTER_LINEAR)
        rh, rw = resized.shape[:2]
        if factor > 1:  # 가운데를 원래 크기로 자른다
            top, left = (rh - height) // 2, (rw - width) // 2
            return np.ascontiguousarray(resized[top:top + height, left:left + width])
        top, left = (height - rh) // 2, (width - rw) // 2
        return cv2.copyMakeBorder(resized, top, height - rh - top, left, width - rw - left, cv2.BORDER_REPLICATE)
    raise ValueError(f"알 수 없는 TTA 변형입니다: {variant}")
