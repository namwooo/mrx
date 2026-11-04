"""분류 모델 경계: 분류 어댑터 규격 1.0 의 LoadedModel 과 어댑터 속성 중 라벨링이 쓰는 것.

라벨링은 어댑터가 준 점수(predict)·임베딩(embed)·허용 TTA 변환(tta)만 쓴다. 판정·임계값은 플랫폼(이 라이브러리)이 정한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

BINARY_NAMES = ["OK", "NG"]  # 이진 분류 class_names (0 = OK, 1 = NG). 규격 1.0 4.2절


class ClassifierModel(Protocol):
    """= 분류 어댑터 규격 1.0 의 LoadedModel"""
    class_names: list[str]

    def predict(self, images: list[np.ndarray]) -> np.ndarray:
        """(이미지 수, 클래스 수) float32 0~1. 열 순서 = class_names 순서"""
        ...

    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        """(이미지 수, D) float32"""
        ...


@dataclass(frozen=True)
class ModelInfo:
    """어댑터 속성 중 라벨링이 쓰는 것 (어댑터 규격 3장)."""
    name: str                       # 어댑터 이름 (ymap · dotting …). 신호 플러그인을 고르는 데 쓴다
    tta: tuple[str, ...] = ()       # 정답을 바꾸지 않는 TTA 변환 이름(tasks.classify.tta.TRANSFORMS 중)
    max_batch: int = 16             # predict·embed 한 번에 넣을 수 있는 최대 이미지 수
    threshold: float | None = None  # 이진 분류 NG 기준(어댑터 defaults.threshold)
    decode: str = "color"           # 이미지 디코딩 방식 (color · unchanged)

    @classmethod
    def from_adapter(cls, adapter: Any) -> ModelInfo:
        """어댑터 객체(규격 1.0 Adapter)에서 만든다."""
        defaults = getattr(adapter, "defaults", None) or {}
        return cls(name=str(adapter.name), tta=tuple(getattr(adapter, "tta", ()) or ()),
                   max_batch=int(getattr(adapter, "max_batch", 16) or 16), threshold=defaults.get("threshold"),
                   decode=str((getattr(adapter, "input", None) or {}).get("decode", "color")))


def class_map(model_names: list[str], class_names: list[str]) -> list[int]:
    """클래스 구성 순서 → 모델 열 번호. 분류는 클래스 구성과 모델의 클래스가 이름으로 같아야 한다(순서는 달라도 된다).
    모델에만 있는 클래스가 있으면 점수 합이 맞지 않으므로 받지 않는다(ValueError)."""
    if sorted(model_names) != sorted(class_names) or len(set(model_names)) != len(model_names):
        raise ValueError(f"분류 모델의 클래스 {model_names} 가 클래스 구성 {class_names} 과 같지 않습니다")
    index = {name: i for i, name in enumerate(model_names)}
    return [index[name] for name in class_names]


def batched(fn, images: list[np.ndarray], max_batch: int) -> np.ndarray:
    """max_batch 장씩 나눠 불러 이어붙인다. 결과 모양·자료형을 확인한다(2차원 float32, 이미지 수 일치)."""
    parts = []
    for start in range(0, len(images), max(1, max_batch)):
        chunk = images[start:start + max(1, max_batch)]
        out = np.asarray(fn(chunk))
        if out.ndim != 2 or out.shape[0] != len(chunk):
            raise ValueError(f"모델 출력 모양이 (이미지 수 {len(chunk)}, ·) 가 아닙니다: {out.shape}")
        parts.append(out.astype(np.float32, copy=False))
    if not parts:
        return np.zeros((0, 0), np.float32)
    return np.concatenate(parts)
