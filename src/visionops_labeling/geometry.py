"""박스 기하: 사각형 박스(detect)와 회전 박스(obb) 표현 변환.

클래스 구성의 task 로 박스 형식이 정해진다.
    detect  라벨 {class_id, bbox_yolo: [cx, cy, w, h]}                 (0~1 정규화, 학습 API DetectionBox)
    obb     라벨 {class_id, polygon_norm: [x1, y1, …, x4, y4]}           (0~1 정규화 네 꼭짓점, 학습 API ObbBox·YOLO OBB txt)
자동라벨링 박스(predictions.boxes)는 두 형식 모두 bbox(원본 픽셀 xyxy, 회전 박스는 감싸는 사각형)를 갖고, obb 는 xywhr(원본 픽셀,
아래 규칙으로 맞춘 값)·polygon(원본 픽셀 꼭짓점)·polygon_norm 을 더 갖는다.

회전 박스 각도 규칙(ultralytics regularize_rboxes 와 같음): θ ≥ π/2 쪽이면 w·h 를 바꾸고 θ 를 [0, π/2) 로 맞춘다.
같은 박스의 표현이 하나가 되어 저장·비교가 흔들리지 않는다.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

TASKS = ("detect", "obb")
# task → 라벨 박스의 좌표 필드 이름. 확정 데이터·평가셋·검수 결과가 모두 이 규칙을 따른다.
LABEL_KEY = {"detect": "bbox_yolo", "obb": "polygon_norm"}


def regularize(cx: float, cy: float, w: float, h: float, theta: float) -> tuple[float, float, float, float, float]:
    """회전 박스 각도를 [0, π/2) 로 맞춘다(필요하면 w·h 를 바꾼다)."""
    if theta % math.pi >= math.pi / 2:
        w, h = h, w
    return cx, cy, w, h, theta % (math.pi / 2)


# vx: 너비 방향 반 벡터, vy: 높이 방향 반 벡터. 중심 ± vx ± vy 로 네 꼭짓점을 시계 방향으로 만든다.
def corners(cx: float, cy: float, w: float, h: float, theta: float) -> np.ndarray:
    """회전 박스 → 네 꼭짓점 (4, 2). 순서대로 이어지는 꼭짓점(ultralytics xywhr2xyxyxyxy 와 같은 순서)."""
    c, s = math.cos(theta), math.sin(theta)
    vx, vy = np.array([w / 2 * c, w / 2 * s]), np.array([-h / 2 * s, h / 2 * c])
    center = np.array([cx, cy])
    return np.stack([center + vx + vy, center + vx - vy, center - vx - vy, center - vx + vy])


# 신발끈 공식: 꼭짓점 순서대로 x_i·y_(i+1) − y_i·x_(i+1) 합의 절반.
def polygon_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def label_box(box: dict[str, Any], task: str) -> dict[str, Any]:
    """자동라벨링 박스·검수 결과 → 확정 라벨 형식 (class_id 와 task 의 좌표 하나)."""
    key = LABEL_KEY[task]
    return {"class_id": int(box["class_id"]), key: [float(v) for v in box[key]]}


def box_format_error(box: dict[str, Any], task: str) -> str | None:
    """라벨 박스가 클래스 구성 task 의 형식이 아니면 이유. detect 는 bbox_yolo, obb 는 polygon_norm 만 받는다."""
    key, other = LABEL_KEY[task], LABEL_KEY["obb" if task == "detect" else "detect"]
    if box.get(key) is None:
        return f"{task} 클래스 구성의 박스는 {key} 가 필요합니다" + (f" ({other} 는 받지 않습니다)" if box.get(other) is not None else "")
    return None


def obb_detection(class_id: int, confidence: float, cx: float, cy: float, w: float, h: float, theta: float) -> dict[str, Any]:
    """회전 박스 예측 결과 하나: {"class_id", "confidence", "xywhr"(각도 규칙), "polygon"(꼭짓점), "xyxy"(감싸는 사각형)}"""
    xywhr = regularize(float(cx), float(cy), float(w), float(h), float(theta))
    points = corners(*xywhr)
    return {"class_id": class_id, "confidence": confidence, "xywhr": list(xywhr), "polygon": points.tolist(),
            "xyxy": [*points.min(0).tolist(), *points.max(0).tolist()]}


def flip_xywhr(xywhr: list[float], width: float) -> tuple[float, float, float, float, float]:
    """좌우 뒤집은 이미지의 회전 박스를 원본 좌표로: cx → width − cx, θ → π − θ (거울상), 각도 규칙으로 다시 맞춘다."""
    cx, cy, w, h, theta = (float(v) for v in xywhr)
    return regularize(width - cx, cy, w, h, math.pi - theta)


# 회전 박스를 감싸는 축 정렬 사각형의 반폭·반높이 = (w·|cos| + h·|sin|)/2, (w·|sin| + h·|cos|)/2
def enclosing(xywhr: np.ndarray) -> np.ndarray:
    """회전 박스 (n, 5) → 감싸는 사각형 (n, 4) xyxy"""
    xywhr = np.asarray(xywhr, np.float64).reshape(-1, 5)
    c, s = np.abs(np.cos(xywhr[:, 4])), np.abs(np.sin(xywhr[:, 4]))
    half_w = (xywhr[:, 2] * c + xywhr[:, 3] * s) / 2
    half_h = (xywhr[:, 2] * s + xywhr[:, 3] * c) / 2
    return np.column_stack([xywhr[:, 0] - half_w, xywhr[:, 1] - half_h, xywhr[:, 0] + half_w, xywhr[:, 1] + half_h])


def rotated_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """회전 박스 (N, 5) × (M, 5) xywhr(픽셀·라디안) → (N, M) 정확한 IoU (교집합 다각형 넓이, OpenCV).

    ProbIoU(가우시안 근사)는 정사각형을 돌려도 1 이라 각도 오류를 못 잡으므로 쓰지 않는다(OBB 계획 2.1절).
    감싸는 사각형이 겹치지 않는 쌍은 계산하지 않는다(0)."""
    import cv2

    a, b = np.asarray(a, np.float64).reshape(-1, 5), np.asarray(b, np.float64).reshape(-1, 5)
    out = np.zeros((len(a), len(b)))
    if not len(a) or not len(b):
        return out
    ea, eb = enclosing(a), enclosing(b)
    # 빠른 사전 거르기: 감싸는 사각형끼리 겹치는 쌍만 OpenCV 교집합 계산을 한다(대부분의 쌍은 겹치지 않는다).
    overlap = ((np.minimum(ea[:, None, 2], eb[None, :, 2]) > np.maximum(ea[:, None, 0], eb[None, :, 0]))
               & (np.minimum(ea[:, None, 3], eb[None, :, 3]) > np.maximum(ea[:, None, 1], eb[None, :, 1])))
    area_a, area_b = a[:, 2] * a[:, 3], b[:, 2] * b[:, 3]
    for i, j in zip(*np.nonzero(overlap)):
        # 두 회전 사각형의 교집합 꼭짓점 → 볼록 껍질 넓이 = 교집합 넓이.
        _, points = cv2.rotatedRectangleIntersection(_cv_rect(a[i]), _cv_rect(b[j]))
        if points is None or len(points) < 3:
            continue
        inter = cv2.contourArea(cv2.convexHull(points))
        union = area_a[i] + area_b[j] - inter
        if union > 0:
            out[i, j] = min(1.0, max(0.0, inter / union))
    return out


def _cv_rect(r: np.ndarray) -> tuple:
    """xywhr → OpenCV RotatedRect ((cx, cy), (w, h), 각도°). 같은 이미지 좌표(y 아래)·같은 회전 방향(ultralytics 도 minAreaRect 로 바꾼다)"""
    return (float(r[0]), float(r[1])), (float(r[2]), float(r[3])), math.degrees(float(r[4]))


def polygon_to_xywhr(polygon_norm: list[float], width: int, height: int) -> tuple[float, float, float, float, float]:
    """라벨 polygon_norm(0~1) → 원본 픽셀 xywhr(각도 규칙). 네 꼭짓점의 최소 외접 회전 사각형(ultralytics 학습 라벨 변환과 같음)."""
    import cv2

    points = (np.asarray(polygon_norm, np.float64).reshape(4, 2) * [width, height]).astype(np.float32)
    (cx, cy), (w, h), angle = cv2.minAreaRect(points)
    return regularize(cx, cy, w, h, math.radians(angle))


RECTIFY_TOLERANCE_PX = 1.0  # 검수 결과 꼭짓점이 최소 외접 회전 사각형과 이만큼 넘게 다르면 '직사각형 아님'으로 보고 바꾼다


# 검수 도구에서 꼭짓점 순서를 잘못 찍으면 나비(⋈) 모양이 된다. 이런 박스는 반영하지 않고 오류로 알린다.
def quad_self_intersects(points: np.ndarray) -> bool:
    """네 꼭짓점 다각형이 꼬였는지(마주 보는 변이 교차, 나비 모양). 정규화·픽셀 어느 좌표에서나 같다(축별 확대는 교차를 바꾸지 않는다)."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def intersects(p1, p2, q1, q2):
        return cross(p1, p2, q1) * cross(p1, p2, q2) < 0 and cross(q1, q2, p1) * cross(q1, q2, p2) < 0

    p = np.asarray(points, np.float64).reshape(4, 2)
    return intersects(p[0], p[1], p[2], p[3]) or intersects(p[1], p[2], p[3], p[0])


def rectify_polygon(polygon_norm: list[float], width: int, height: int) -> tuple[list[float], bool]:
    """검수 결과 네 꼭짓점 → 회전 사각형 꼭짓점. 직사각형이 아니면(꼭짓점이 최소 외접 회전 사각형과 1px 넘게 다름) 최소 외접 회전
    사각형으로 바꾼다(ultralytics 학습도 같은 변환을 한다). 픽셀 좌표에서 계산한다(정규화 좌표의 직사각형은 픽셀에서 평행사변형일 수 있다).
    (polygon_norm, 바꿨는지). 꼬인 도형·넓이 0 은 ValueError."""
    import cv2

    scale = np.array([width, height], np.float64)
    points = np.asarray(polygon_norm, np.float64).reshape(4, 2) * scale
    if quad_self_intersects(points):
        raise ValueError("꼭짓점 순서가 꼬였습니다(변이 서로 교차)")
    if polygon_area(points) <= 1e-6:
        raise ValueError("넓이가 0 입니다")
    (cx, cy), (w, h), angle = cv2.minAreaRect(points.astype(np.float32))
    rect = corners(*regularize(cx, cy, w, h, math.radians(angle)))
    # 같은 사각형이면 꼭짓점 순서만 다를 수 있다: 돌리고 뒤집은 순서 중 가장 가까운 것으로 비교
    deviation = min(np.abs(np.roll(order, k, 0) - points).max() for order in (rect, rect[::-1]) for k in range(4))
    if deviation <= RECTIFY_TOLERANCE_PX:
        return [float(v) for v in polygon_norm], False
    return [round(float(v), 6) for v in np.clip(rect / scale, 0.0, 1.0).reshape(-1)], True


def yolo_to_xyxy(boxes: list[dict[str, Any]], height: int, width: int) -> np.ndarray:
    """라벨(bbox_yolo 정규화 cx cy w h) → 원본 픽셀 xyxy (n, 4)"""
    out = []
    for box in boxes:
        cx, cy, w, h = box["bbox_yolo"]
        out.append([(cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height])
    return np.clip(np.array(out, np.float64).reshape(-1, 4), 0, [width, height, width, height])


def truth_boxes(boxes: list[dict[str, Any]], height: int, width: int) -> np.ndarray:
    """라벨(확정·평가셋) → 원본 픽셀 좌표. 사각형 박스(bbox_yolo)는 xyxy (m, 4), 회전 박스(polygon_norm)는 xywhr (m, 5).
    거리 통계의 확정 박스 특징(box_features 입력)과 평가 정답 대조(match_golden)에 쓴다."""
    if boxes and "polygon_norm" in boxes[0]:
        return np.array([polygon_to_xywhr(b["polygon_norm"], width, height) for b in boxes], np.float64).reshape(-1, 5)
    return yolo_to_xyxy(boxes, height, width)


def size_box(box: dict[str, Any]) -> list[float]:
    """박스 크기(√(w·h)) 계산용 xyxy. 회전 박스는 감싸는 사각형이 아니라 같은 w·h 의 사각형(기울수록 커지지 않게)."""
    if box.get("xywhr"):
        cx, cy, w, h, _ = box["xywhr"]
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
    b = box["bbox"]
    return [b["x1"], b["y1"], b["x2"], b["y2"]]
