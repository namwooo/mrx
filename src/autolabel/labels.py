"""작업 종류별 확정 라벨 형식 (확정 데이터·평가셋·검수 결과·학습 요청이 같은 형식을 쓴다).

이미지 한 장의 라벨은 {"boxes": [항목, …]} 이고(검출 계열), 이미지 단위 작업(분류·이상탐지)은 항목 하나다.

    detect    박스 {class_id, bbox_yolo: [cx, cy, w, h]}                0~1 정규화
    obb       박스 {class_id, polygon_norm: [x1, y1, …, x4, y4]}        0~1 정규화 네 꼭짓점
    classify  이미지 {class_id}                                         이진 분류는 class_names ["OK", "NG"]
    segment   영역 {class_id, polygons_norm: [[x1, y1, x2, y2, …], …]}  0~1 정규화 다각형(점 3개 이상), 영역 하나에 다각형 여러 개
    anomaly   이미지 {judge: "OK" | "NG"}                               이상 점수(score)는 예측에만 있다

segment·anomaly 는 형식만 정해 두었고 라벨링 모듈은 아직 없다(tasks.available_tasks).
"""
from __future__ import annotations

import math
from typing import Any

from .geometry import box_format_error

# task → 라벨 항목의 값 필드
LABEL_KEY = {"detect": "bbox_yolo", "obb": "polygon_norm", "classify": "class_id", "segment": "polygons_norm",
             "anomaly": "judge"}
IMAGE_LEVEL_TASKS = ("classify", "anomaly")  # 이미지 한 장 = 라벨 항목 하나


def _unit(values: list[Any]) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) and 0.0 <= v <= 1.0 for v in values)


def label_error(task: str, item: dict[str, Any], num_classes: int | None = None) -> str | None:
    """라벨 항목이 task 형식에 맞지 않으면 이유(사람이 읽을 문장), 맞으면 None."""
    if task not in LABEL_KEY:
        return f"알 수 없는 작업 종류입니다: {task}"
    if task == "anomaly":
        return None if item.get("judge") in ("OK", "NG") else "anomaly 라벨은 judge 가 OK 또는 NG 여야 합니다"
    class_id = item.get("class_id")
    if not isinstance(class_id, int) or class_id < 0 or (num_classes is not None and class_id >= num_classes):
        return f"class_id 가 클래스 범위를 벗어났습니다: {class_id}"
    if task in ("detect", "obb"):
        error = box_format_error(item, task)
        if error:
            return error
        values = item[LABEL_KEY[task]]
        size = 4 if task == "detect" else 8
        if len(values) != size or not _unit(values):
            return f"{LABEL_KEY[task]} 는 0~1 값 {size}개여야 합니다"
        return None
    if task == "segment":
        polygons = item.get("polygons_norm")
        if not polygons or not all(isinstance(p, list) and len(p) >= 6 and len(p) % 2 == 0 and _unit(p) for p in polygons):
            return "polygons_norm 은 0~1 좌표 (x, y) 3점 이상 다각형의 목록이어야 합니다"
        return None
    return None  # classify: class_id 만
