"""검출 판정 신호 ①②③④⑤ 계산. 판정(자동 승인·검수·보류)은 rules 가 이 값으로 정한다.

    ① confidence      박스 신뢰도 ≥ 클래스별 임계값(신호 상태, 초기값 VISIONOPS_LABELING_CONF_THRESHOLD)
    ② tta             좌우 뒤집기·크기 변경(입력 크기 imgsz) 추론 결과를 원본 좌표로 되돌려 원본 박스와 Hungarian 매칭
                      비용 = (1 − IoU) + 클래스 불일치 0.5. 매칭 IoU 가 match_iou 미만이면 그 변형에서 '사라진 박스'
                      회전 박스(obb)는 뒤집기를 cx → W − cx, θ → π − θ 로 되돌리고 IoU 는 정확한 회전 IoU(geometry)
    ③ image_distance  이미지 임베딩 거리와 임계값(95분위). 거리 통계가 있을 때만(distance.image)
    ④ box_distance    박스 특징의 kNN·MDS·RMDS 점수(distance.box). 통계에 그 클래스가 있을 때만 세 가지를 모두 남기고,
                      B단계로 켜진 클래스(같은 모델 버전)는 고른 방식의 임계값 초과 여부를 함께 남긴다
    ⑤ model_agreement 운영 모델의 원본 추론 결과와 ② 와 같은 Hungarian 매칭. 매칭 IoU 가 match_iou 미만이면
                      '운영 모델에 없는 박스'. 운영 모델이 없으면 계산하지 않는다(null)

predictions.signals (박스 순서는 predictions.boxes 와 같다)
    {"version": 4, "stats_version": 통계 ID 또는 null,
     "tta": {"variants": [{"name": "flip", "flip": true, "imgsz": 640}, ...],
             "missing_boxes": 한 변형에서라도 사라진 원본 박스 수, "extra_boxes": {변형: 원본에 없던 TTA 박스 수},
             "extra_top": {변형: 원본에 없던 TTA 박스 중 신뢰도 높은 20개의 [클래스 ID, 신뢰도]}},
     "image_distance": {"value", "threshold", "exceeded", "method"} 또는 null (통계 없음),
     "boxes": [{"conf_threshold", "confidence_ok",
                "tta": {변형: {"iou", "class_id", "class_match"} 또는 null(사라짐)},
                "tta_min_iou": 변형 중 가장 낮은 매칭 IoU(사라지면 0), "tta_class_match": 모든 변형에서 매칭·클래스 일치,
                "tta_missing": [사라진 변형],
                "box_distance": {"knn", "knn_k", "mds", "rmds", "margin": {방식: 가장 가까운 다른 클래스 − 예측 클래스 (음수 = 혼동)},
                                 "size": letterbox 입력 기준 √(w·h), "active", "method", "threshold", "exceeded"} 또는 null,
                "agreement": {"iou", "class_id", "class_match"} 또는 null(운영 모델에 없음),
                "agreement_iou": 매칭 IoU(없으면 0), "agreement_class_match", "agreement_missing"   (⑤ 를 계산하지 않으면 모두 null)}],
     "agreement": {"model_version", "missing_boxes", "extra_boxes", "extra_top": 운영 모델에만 있는 박스 중 신뢰도 높은 20개의
                   [클래스 ID, 신뢰도]} 또는 null (운영 모델 없음)}
    version 1(10/15) 은 box_distance 가, version 2(10/16) 는 tta.extra_top 이, version 3(10/21 이전) 은 ⑤ 가 없다.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ...config import TtaConfig
from ...constants import INPUT_SIZE
from ...geometry import flip_xywhr, obb_detection, rotated_iou

SIGNALS_VERSION = 4
CLASS_MISMATCH_COST = 0.5  # IoU 가 0.5 이상 차이 나지 않으면 같은 클래스 쪽을 먼저 짝짓는다
EXTRA_TOP = 20             # 변형에서만 나온 박스는 신뢰도 높은 순으로 이만큼만 [클래스, 신뢰도] 를 남긴다(판정 게이트용)


# TTA 변형 하나: 이름(flip, scale512 …), 좌우 뒤집기 여부, 추론 입력 크기.
@dataclass(frozen=True)
class TtaVariant:
    name: str
    flip: bool
    imgsz: int


# 설정으로 TTA 변형 목록을 만든다. 예: flip=True, scales=(0.8,) → [flip(640), scale512(512)]
def tta_variants(cfg: TtaConfig, input_size: int = INPUT_SIZE) -> list[TtaVariant]:
    variants = [TtaVariant("flip", True, input_size)] if cfg.flip else []
    for scale in cfg.scales:
        imgsz = max(64, round(input_size * scale / 32) * 32)  # 모델 stride(32) 배수
        if imgsz != input_size and f"scale{imgsz}" not in {v.name for v in variants}:
            variants.append(TtaVariant(f"scale{imgsz}", False, imgsz))
    return variants


def variant_params(variants: list[TtaVariant]) -> list[dict[str, Any]]:
    return [asdict(v) for v in variants]


# 이미지 쪽 변형은 좌우 뒤집기만 한다. 크기 변경은 추론 입력 크기(imgsz)로 처리한다.
def transform(image: np.ndarray, variant: TtaVariant) -> np.ndarray:
    import cv2

    return cv2.flip(image, 1) if variant.flip else image


def restore(detections: list[dict[str, Any]], variant: TtaVariant, width: int) -> list[dict[str, Any]]:
    """변형 이미지의 결과를 원본 이미지 좌표로 되돌린다(좌우 뒤집기: x → width − x, 회전 박스는 각도도 거울상).
    크기 변경 추론은 ultralytics 가 원본 좌표로 돌려주므로 그대로 둔다."""
    if not variant.flip:
        return detections
    out = []
    for det in detections:
        if "xywhr" in det:
            out.append({**det, **obb_detection(det["class_id"], det["confidence"], *flip_xywhr(det["xywhr"], width))})
            continue
        x1, y1, x2, y2 = det["xyxy"]
        out.append({**det, "xyxy": [width - x2, y1, width - x1, y2]})
    return out


# 넘파이 브로드캐스팅으로 N×M 쌍의 IoU 를 한 번에 계산한다.
def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N, 4) × (M, 4) xyxy → (N, M) IoU"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area_a = (a[:, 2:] - a[:, :2]).prod(1)
    area_b = (b[:, 2:] - b[:, :2]).prod(1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def _xyxy(boxes: list[dict[str, Any]]) -> np.ndarray:
    return np.array([[b["bbox"]["x1"], b["bbox"]["y1"], b["bbox"]["x2"], b["bbox"]["y2"]] for b in boxes],
                    np.float64).reshape(-1, 4)


def pairwise_iou(base: list[dict[str, Any]], other: list[dict[str, Any]]) -> np.ndarray:
    """박스 목록 사이 IoU. 양쪽이 회전 박스(xywhr)면 정확한 회전 IoU, 아니면 사각형 IoU."""
    if base and other and "xywhr" in base[0] and "xywhr" in other[0]:
        return rotated_iou(np.array([b["xywhr"] for b in base]), np.array([b["xywhr"] for b in other]))
    return box_iou(_xyxy(base), _xyxy(other))


def match_boxes(base: list[dict[str, Any]], other: list[dict[str, Any]],
                min_iou: float) -> tuple[list[dict[str, Any] | None], int]:
    """원본 박스마다 TTA 박스 하나를 Hungarian 매칭한다. ([원본 박스별 {"iou", "class_id", "class_match"} 또는 None], 남은 TTA 박스 수)"""
    matches, unmatched = match_detail(base, other, min_iou)
    return matches, len(unmatched)


# [Hungarian 매칭] 비용 행렬 = (1 − IoU) + 0.5 × 클래스 다름. scipy linear_sum_assignment 로 총비용 최소 1:1 짝을 찾고,
# IoU 가 min_iou 미만인 짝은 '매칭 안 됨'으로 버린다.
def match_detail(base: list[dict[str, Any]], other: list[dict[str, Any]],
                 min_iou: float) -> tuple[list[dict[str, Any] | None], list[int]]:
    """match_boxes 와 같고, 남은 TTA 박스는 수 대신 other 의 번호 목록으로 돌려준다."""
    from scipy.optimize import linear_sum_assignment

    matches: list[dict[str, Any] | None] = [None] * len(base)
    if not base or not other:
        return matches, list(range(len(other)))
    iou = pairwise_iou(base, other)
    differ = np.array([[a["class_id"] != b["class_id"] for b in other] for a in base], np.float64)
    rows, cols = linear_sum_assignment((1.0 - iou) + CLASS_MISMATCH_COST * differ)
    used = set()
    for i, j in zip(rows, cols):
        if iou[i, j] >= min_iou:
            matches[i] = {"iou": round(float(iou[i, j]), 4), "class_id": other[j]["class_id"],
                          "class_match": other[j]["class_id"] == base[i]["class_id"]}
            used.add(int(j))
    return matches, [j for j in range(len(other)) if j not in used]


# [신호 계산] 박스별 기록(per_box)을 만들고 ⑤ → ② 순서로 매칭 결과를 채운 뒤, ② 요약 값(tta_min_iou·tta_class_match)을 정한다.
def compute(boxes: list[dict[str, Any]], tta: dict[str, list[dict[str, Any]]], variants: list[TtaVariant],
            conf_thresholds: dict[int, float], match_iou: float, distance: dict[str, Any] | None,
            stats_version: str | None, box_distance: list[dict[str, Any] | None] | None = None,
            agreement: tuple[str, list[dict[str, Any]]] | None = None) -> dict[str, Any]:
    """이미지 한 장의 신호. boxes·tta 박스는 analysis.to_boxes 형식(클래스 구성 ID, 원본 픽셀 bbox).
    box_distance: 박스 순서의 ④ 기록(box_distance_record) 또는 None(통계 없음).
    agreement: ⑤ (운영 모델 버전, 운영 모델 박스) 또는 None(운영 모델 없음)."""
    per_box = [{"conf_threshold": conf_thresholds.get(b["class_id"]),
                "confidence_ok": b["confidence"] >= conf_thresholds.get(b["class_id"], 0.0),
                "tta": {}, "tta_missing": [],
                "box_distance": box_distance[i] if box_distance else None,
                "agreement": None, "agreement_iou": None, "agreement_class_match": None, "agreement_missing": None}
               for i, b in enumerate(boxes)]
    agreement_summary = None
    if agreement is not None:
        version, found = agreement
        matches, unmatched = match_detail(boxes, found, match_iou)
        for record, match in zip(per_box, matches):
            record.update(agreement=match, agreement_iou=match["iou"] if match else 0.0,
                          agreement_class_match=bool(match and match["class_match"]), agreement_missing=match is None)
        top = sorted(unmatched, key=lambda j: -found[j]["confidence"])[:EXTRA_TOP]
        agreement_summary = {"model_version": version, "missing_boxes": sum(1 for m in matches if m is None),
                             "extra_boxes": len(unmatched),
                             "extra_top": [[found[j]["class_id"], round(float(found[j]["confidence"]), 4)] for j in top]}
    extra, extra_top = {}, {}
    for variant in variants:
        found = tta.get(variant.name, [])
        matches, unmatched = match_detail(boxes, found, match_iou)
        extra[variant.name] = len(unmatched)
        top = sorted(unmatched, key=lambda j: -found[j]["confidence"])[:EXTRA_TOP]
        extra_top[variant.name] = [[found[j]["class_id"], round(float(found[j]["confidence"]), 4)] for j in top]
        for record, match in zip(per_box, matches):
            record["tta"][variant.name] = match
            if match is None:
                record["tta_missing"].append(variant.name)
    # ② 요약: 한 변형에서라도 사라지면 최소 IoU 0, 모든 변형에서 같은 클래스로 매칭돼야 class_match.
    for record in per_box:
        found = [m for m in record["tta"].values() if m is not None]
        if variants:
            record["tta_min_iou"] = 0.0 if record["tta_missing"] else min(m["iou"] for m in found)
            record["tta_class_match"] = not record["tta_missing"] and all(m["class_match"] for m in found)
        else:
            record["tta_min_iou"] = record["tta_class_match"] = None
    return {"version": SIGNALS_VERSION, "stats_version": stats_version,
            "tta": {"variants": variant_params(variants), "missing_boxes": sum(1 for r in per_box if r["tta_missing"]),
                    "extra_boxes": extra, "extra_top": extra_top},
            "image_distance": distance, "agreement": agreement_summary, "boxes": per_box}


def box_distance_record(scores: dict[str, Any] | None, size: float, active: tuple[str, float | None] | None) -> dict[str, Any] | None:
    """④ 박스 하나의 기록. scores: BoxDistance.scores 의 항목(None 이면 그 클래스 통계 없음), active: (방식, 임계값) 또는 None."""
    if scores is None:
        return None
    record = {"knn": round(scores["knn"], 4), "knn_k": scores["knn_k"], "mds": round(scores["mds"], 4),
              "rmds": round(scores["rmds"], 4),
              "margin": {m: (round(v, 4) if v is not None else None) for m, v in scores["margin"].items()},
              "size": round(float(size), 1), "active": active is not None, "method": None, "threshold": None,
              "exceeded": None}
    if active is not None:
        method, threshold = active
        record.update(method=method, threshold=round(float(threshold), 4) if threshold is not None else None,
                      exceeded=bool(threshold is not None and scores[method] > threshold))
    return record


# 아래 score_* 함수: 신호를 '클수록 틀렸을 가능성이 높은' 점수 하나로 바꾼다(평가 AUROC 계산용).
def score_tta(record: dict[str, Any]) -> float | None:
    """② 를 '틀렸을 가능성' 점수로: 1 − 최소 매칭 IoU, 클래스가 바뀌면 +1 (평가 AUROC 용)."""
    if record.get("tta_min_iou") is None:
        return None
    return 1.0 - record["tta_min_iou"] + (0.0 if record["tta_class_match"] else 1.0)


def score_agreement(record: dict[str, Any]) -> float | None:
    """⑤ 를 '틀렸을 가능성' 점수로: 1 − 운영 모델과의 매칭 IoU(없으면 0), 클래스가 다르거나 없으면 +1 (평가 AUROC 용)."""
    if record.get("agreement_iou") is None:
        return None
    return 1.0 - record["agreement_iou"] + (0.0 if record["agreement_class_match"] else 1.0)
