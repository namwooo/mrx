"""검출 이미지 묶음 분석: 원본 추론 → (신호를 켰으면) ③④ → TTA ② → (운영 모델) ⑤ → 신호. 판정은 rules.judge.

서버의 자동라벨링(AUTO_LABEL)·평가(LABEL_EVAL)·통계(LABEL_STATS)가 같은 analyze 를 쓰므로 같은 조건으로 점수가 계산된다.
판정 조건(TTA 변형·클래스별 임계값·거리 통계·④ B단계 상태·⑤ 운영 모델)은 SignalSetup 에 담아 넘긴다.
서버는 workers.handlers.labeling 에서 SignalSetup 을 만든다.

박스 dict 형식
    detect  class_id, class_name, confidence, bbox{x1,y1,x2,y2 원본 픽셀}, bbox_yolo[cx,cy,w,h 0~1]
    obb     class_id, class_name, confidence, bbox(감싸는 사각형), xywhr(원본 픽셀, 각도 규칙), polygon(픽셀), polygon_norm(0~1 8개)
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ...distance.box import BoxDistance
from ...distance.image import ImageDistance, distance_record
from ...features import as_matrix, box_size
from ...geometry import polygon_area, size_box
from . import signals as sig
from .predictor import Predictor
from .rules import RuleParams

logger = logging.getLogger("autolabel")


# [박스 정리] 클래스 구성에 없는 클래스는 버리고, 이미지 밖으로 나간 좌표는 자르고, 면적이 0 이 된 박스는 버린다.
def to_boxes(detections: list[dict[str, Any]], width: int, height: int, cmap: dict[int, int],
             names: list[str]) -> list[dict[str, Any]]:
    """모델 결과를 클래스 구성 ID·원본 픽셀 박스·YOLO 정규화 좌표로 바꾼다. 신뢰도 내림차순. 회전 박스는 obb_box."""
    boxes = []
    for det in detections:
        if det["class_id"] not in cmap:
            continue
        if "xywhr" in det:
            box = obb_box(det, width, height, cmap[det["class_id"]], names)
            if box is not None:
                boxes.append(box)
            continue
        x1, y1, x2, y2 = det["xyxy"]
        x1, x2 = min(max(x1, 0.0), width), min(max(x2, 0.0), width)
        y1, y2 = min(max(y1, 0.0), height), min(max(y2, 0.0), height)
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            continue
        class_id = cmap[det["class_id"]]
        boxes.append({"class_id": class_id, "class_name": names[class_id], "confidence": round(float(det["confidence"]), 6),
                      "bbox": {"x1": round(x1, 2), "y1": round(y1, 2), "x2": round(x2, 2), "y2": round(y2, 2)},
                      "bbox_yolo": [round((x1 + x2) / 2 / width, 6), round((y1 + y2) / 2 / height, 6),
                                    round((x2 - x1) / width, 6), round((y2 - y1) / height, 6)]})
    boxes.sort(key=lambda b: -b["confidence"])
    return boxes


def obb_box(det: dict[str, Any], width: int, height: int, class_id: int, names: list[str]) -> dict[str, Any] | None:
    """회전 박스 결과 → 박스. 꼭짓점은 이미지 안으로 자르고(학습 라벨은 0~1), 잘라서 면적이 없어지면 버린다.
    xywhr 는 모델 결과 그대로(원본 픽셀, 각도 규칙), bbox 는 자른 꼭짓점을 감싸는 사각형."""
    points = np.clip(np.asarray(det["polygon"], np.float64).reshape(4, 2), 0.0, [width, height])
    if polygon_area(points) <= 1e-6:
        return None
    (x1, y1), (x2, y2) = points.min(0), points.max(0)
    return {"class_id": class_id, "class_name": names[class_id], "confidence": round(float(det["confidence"]), 6),
            "bbox": {"x1": round(float(x1), 2), "y1": round(float(y1), 2), "x2": round(float(x2), 2), "y2": round(float(y2), 2)},
            "xywhr": [round(float(v), 4) for v in det["xywhr"][:4]] + [round(float(det["xywhr"][4]), 6)],
            "polygon": [[round(float(x), 2), round(float(y), 2)] for x, y in points],
            "polygon_norm": [round(float(v), 6) for v in (points / [width, height]).reshape(-1)]}


# 학습 라벨과 같은 형식에 신뢰도를 마지막 열로 붙인다(서버의 runs/<요청>/labels/<이미지>.txt).
def yolo_lines(boxes: list[dict[str, Any]]) -> str:
    """YOLO 라벨 + 신뢰도. 회전 박스는 YOLO OBB 형식(class x1 y1 … x4 y4 conf)"""
    lines = [" ".join([str(b["class_id"]), *(f"{v:.6f}" for v in b.get("polygon_norm") or b["bbox_yolo"]),
                       f"{b['confidence']:.4f}"]) for b in boxes]
    return "\n".join(lines) + ("\n" if lines else "")


# [신호 조건] 청크 하나의 판정 조건. 서버는 DB 의 최신 신호 상태·통계로 만들고, 쓴 통계 ID·임계값을 신호·판정에 남긴다.
#   - conf_thresholds / rules.tta_iou / rules.agreement_iou : 클래스별 ①②⑤ 임계값
#   - distance(③) / box(④)                                 : 같은 모델 가중치의 최신 거리 통계(없으면 None)
#   - box_active                                             : ④ 가 켜진 클래스와 (방식, k)
#   - stages / rules.closed / rules.audit_rates              : 클래스별 자동라벨링 단계(보정 안 된 클래스는 닫힘)
class SignalSetup:
    """청크 하나의 신호 계산 조건: TTA 변형, 클래스별 conf 임계값, 거리 통계(있으면), ④ B단계 상태, ⑤ 운영 모델(있으면)."""

    def __init__(self, *, task: str, enabled: bool, variants: list[sig.TtaVariant], match_iou: float,
                 conf_thresholds: dict[int, float], rules: RuleParams, distance: ImageDistance | None = None,
                 box: BoxDistance | None = None, box_active: dict[int, tuple[str, int | None]] | None = None,
                 default_k: int = 5, stats_id: str | None = None, production: dict[str, Any] | None = None,
                 stages: dict[int, dict[str, Any]] | None = None):
        self.task, self.enabled, self.variants, self.match_iou = task, enabled, list(variants), match_iou
        self.conf_thresholds, self.rules = dict(conf_thresholds), rules
        self.distance, self.box = distance, box
        self.box_active = dict(box_active or {})
        self.knn_k = {c: k for c, (_, k) in self.box_active.items() if k}
        self.default_k, self.stats_id = default_k, stats_id
        self.production = production  # ⑤ 운영 모델 참조 {"version", "class_map", …} 또는 None
        self.stages = dict(stages or {})

    # 작업 결과에 남기는 신호 설정 요약. 어떤 통계·단계로 판정했는지 나중에 확인할 수 있다.
    def summary(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "tta": [v.name for v in self.variants],
                "stats_version": self.stats_id,
                "box_distance": self.box is not None,
                "box_distance_active": sorted(self.box_active) if self.box is not None else [],
                "production": self.production["version"] if self.production else None,
                "stages": {str(c): st["stage"] for c, st in self.stages.items()}}

    # ④ 박스별 기록: kNN·MDS·RMDS 점수, 박스 크기 구간, (켜진 클래스면) 쓴 방식과 임계값.
    def box_records(self, features: np.ndarray, boxes: list[dict[str, Any]], height: int, width: int) -> list | None:
        """이미지 한 장 박스들의 ④ 기록. 통계와 특징 차원이 다르면(레이어 설정 변경) None."""
        if self.box is None:
            return None
        if len(boxes) and features.shape[1] != self.box.feature_dim:
            logger.warning("박스 특징 차원(%d)이 통계(%d)와 달라 ④ 를 건너뜁니다: 통계 %s", features.shape[1],
                           self.box.feature_dim, self.stats_id)
            return None
        classes = [b["class_id"] for b in boxes]
        scores = self.box.scores(features, classes, {c: self.knn_k.get(c, self.default_k) for c in set(classes)})
        out = []
        for box, score in zip(boxes, scores):
            active = None
            if box["class_id"] in self.box_active and score is not None:
                method, k = self.box_active[box["class_id"]]
                active = (method, self.box.threshold(box["class_id"], method, score["knn_k"] if method == "knn" else k))
            out.append(sig.box_distance_record(score, box_size(size_box(box), height, width), active))
        return out


def _xyxy(box: dict[str, Any]) -> list[float]:
    b = box["bbox"]
    return [b["x1"], b["y1"], b["x2"], b["y2"]]


def release(predictor) -> None:
    """예측기가 잡아 둔 특징맵(GPU 메모리)을 놓는다(release 가 있는 예측기만)."""
    fn = getattr(predictor, "release", None)
    if fn is not None:
        fn()


# [이미지 묶음 분석] 추론 횟수: 원본 1회 + TTA 변형 수 + (⑤ 있으면) 운영 모델 1회.
# ③④ 는 원본 추론 때 hook 으로 잡아 둔 neck 특징맵을 다시 쓰므로 추가 추론이 없다.
def analyze(predictor: Predictor, arrays: list[np.ndarray], params: dict[str, Any], cmap: dict[int, int],
            names: list[str], context: SignalSetup, keep_features: bool = False,
            production: Predictor | None = None) -> list[dict[str, Any]]:
    """이미지 묶음 하나: 원본 추론 → (신호를 켰으면) 같은 추론의 특징맵으로 ③④ → TTA ② → (운영 모델) ⑤ → 신호.
    params: 추론 조건 {"conf", "iou", "max_det"}. cmap: 모델 클래스 ID → 클래스 구성 ID. production: context.production 의 예측기.
    [{"boxes", "signals", "features"(keep_features 일 때 (박스 수, 차원) 또는 None)}]"""
    detections = predictor.predict(arrays, conf=params["conf"], iou=params["iou"], max_det=params["max_det"],
                                   classes=sorted(cmap))
    found = [to_boxes(dets, a.shape[1], a.shape[0], cmap, names) for a, dets in zip(arrays, detections)]
    if not context.enabled:
        return [{"boxes": boxes, "signals": None, "features": None} for boxes in found]
    distances = context.distance.distance(predictor.embed(arrays)) if context.distance is not None else None
    features = None
    if context.box is not None:
        if context.task == "obb":  # 회전 박스: 통계를 만든 방식(rotated·enclosing)으로 뽑는다
            features = predictor.box_features(arrays, [np.array([b["xywhr"] for b in boxes]).reshape(-1, 5) for boxes in found],
                                              context.box.levels, pooling=context.box.pooling)
        else:
            features = predictor.box_features(arrays, [np.array([_xyxy(b) for b in boxes]).reshape(-1, 4) for boxes in found],
                                              context.box.levels)
    # 특징맵(hook 으로 잡아 둔 텐서)을 놓아 GPU 메모리를 돌려준다. TTA 추론 전에 해야 메모리가 쌓이지 않는다.
    release(predictor)
    # ② TTA: 변형(좌우 뒤집기, 크기 변경)마다 다시 추론하고 좌표를 원본 기준으로 되돌린다.
    tta = {v.name: _tta_predictions(predictor, arrays, v, params, cmap, names) for v in context.variants}
    agreement = None
    if production is not None and context.production is not None:  # ⑤: 운영 모델 원본 추론(같은 conf·NMS), 클래스는 이름으로 맞춘다
        prod_map = {int(k): v for k, v in context.production["class_map"].items()}
        prod = production.predict(arrays, conf=params["conf"], iou=params["iou"], max_det=params["max_det"],
                                  classes=sorted(prod_map))
        agreement = [to_boxes(dets, a.shape[1], a.shape[0], prod_map, names) for a, dets in zip(arrays, prod)]
        release(production)
    out = []
    for index, (array, boxes) in enumerate(zip(arrays, found)):
        height, width = array.shape[:2]
        feats = as_matrix(features[index], len(boxes)) if features is not None else None
        distance = (distance_record(distances[index], context.distance.threshold, context.distance.method.value)
                    if distances is not None else None)
        signals = sig.compute(boxes, {name: f[index] for name, f in tta.items()}, context.variants,
                              context.conf_thresholds, context.match_iou, distance, context.stats_id,
                              context.box_records(feats, boxes, height, width) if feats is not None else None,
                              (context.production["version"], agreement[index]) if agreement is not None else None)
        out.append({"boxes": boxes, "signals": signals, "features": feats if keep_features else None})
    return out


def _tta_predictions(predictor: Predictor, arrays: list[np.ndarray], variant: sig.TtaVariant, params: dict[str, Any],
                     cmap: dict[int, int], names: list[str]) -> list[list[dict[str, Any]]]:
    detections = predictor.predict([sig.transform(a, variant) for a in arrays], conf=params["conf"], iou=params["iou"],
                                   max_det=params["max_det"], classes=sorted(cmap), imgsz=variant.imgsz)
    out = []
    for array, dets in zip(arrays, detections):
        height, width = array.shape[:2]
        out.append(to_boxes(sig.restore(dets, variant, width), width, height, cmap, names))
    return out
