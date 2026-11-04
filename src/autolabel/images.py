"""이미지 읽기 (모든 작업 종류 공통).

decode="color" 는 BGR 3채널 8비트(검출 모델·분류 어댑터 기본), "unchanged" 는 OpenCV IMREAD_UNCHANGED 결과(회색조·16비트 그대로).
분류 어댑터 규격 1.0 의 input.decode 와 같은 값이다.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

Decode = Literal["color", "unchanged"]


# 실패하면 ValueError 를 내고, 호출한 쪽이 그 이미지에만 오류를 기록한다.
def read_image(path: str, max_pixels: int, decode: Decode = "color") -> np.ndarray:
    import cv2

    data = np.fromfile(path, dtype=np.uint8)  # Windows 한글 경로도 읽도록 바이트로 읽는다
    flag = cv2.IMREAD_COLOR if decode == "color" else cv2.IMREAD_UNCHANGED
    image = cv2.imdecode(data, flag) if data.size else None
    if image is None:
        raise ValueError("이미지를 읽을 수 없습니다")
    if image.shape[0] * image.shape[1] > max_pixels:
        raise ValueError(f"이미지가 너무 큽니다: {image.shape[1]}x{image.shape[0]}")
    return image
