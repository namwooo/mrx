"""검출 평가셋 평가·B단계 판정·임계값 보정의 계산 부분 (서버 LABEL_EVAL 이 부른다. DB 반영은 서버 몫).

정답 대조
    신뢰도 높은 박스부터 같은 클래스의 아직 짝이 없는 정답 중 IoU 가 가장 큰 것과 짝짓고 IoU ≥ 기준이면 맞은 박스다(COCO 방식).
    회전 박스(obb)는 정확한 회전 IoU(geometry)로 잰다. 틀린 박스는 종류를 남긴다:
        wrong_class   다른 클래스 정답과 IoU ≥ 기준        localization  같은 클래스 정답과 0.1 ≤ IoU < 기준
        duplicate     같은 클래스 정답이 이미 다른 박스와 짝  background    겹치는 정답이 없음

AUROC (평가 행 rows: 예측 박스 하나 = 행 하나, scores 는 모두 '클수록 틀렸을 가능성 높음')
    신호마다 틀린 박스 검출 AUROC 를 예측 클래스별·박스 크기 구간별로 계산한다. A단계 점수는 ①②③⑤ 의 백분위 순위 평균이고,
    ④ 를 더한 점수는 거기에 ④ 순위를 하나 더 넣어 평균한다(평가셋이 작아 가중치를 학습하지 않는다).

B단계 판정 (클래스별, 계획서 3.5절)
    조건 1  사람 확인 박스 ≥ StageBConfig.min_boxes
    조건 2  ④ 방식 중 단독 AUROC 가 가장 높은 방식을 더했을 때 AUROC 가 A단계보다 min_gain 이상 높다
    조건 3  같은 모델 버전에서 감사로 롤백된 적이 없다

임계값 보정 (calibrate)
    클래스별로 '자동 승인 박스의 오류율 ≤ target' 을 만족하면서 자동 승인되는 맞은 박스가 가장 많은 ① 신뢰도·② 매칭 IoU·
    ⑤ 일치 IoU 임계값을 찾는다. 이미지 단위 전후 결과는 simulate 로 따로 보고한다.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ...config import CalibrationConfig, StageBConfig
from ...features import box_size, size_bucket
from ...geometry import rotated_iou, size_box, truth_boxes
from ...metrics import auroc, average_precision, combined, error_bound, zero_error_boxes
from . import rules
from .signals import box_iou, score_agreement, score_tta

# A단계 결합 점수에 넣는 신호. ④ 는 따로 더해 보고(A+④) 개선폭으로 B단계를 판정한다.
A_SIGNALS = ("confidence", "tta", "image_distance", "model_agreement")  # 값이 없는 신호(통계·운영 모델 없음)는 빠진다
BUCKETS = ("small", "medium", "large")

__all__ = ["A_SIGNALS", "BUCKETS", "auroc", "combined", "average_precision", "error_bound", "zero_error_boxes",
           "box_scores", "candidate_thresholds", "evaluate_image", "size_px",
           "detection_summary", "match_golden", "summarize", "decide", "simulate", "calibrate"]


def detection_summary(rows: list[dict[str, Any]], truth: list[int], names: list[str], iou: float) -> dict[str, Any]:
    """평가셋 탐지 지표: 클래스별 AP 와 정답이 있는 클래스의 평균(map)"""
    classes = {}
    for class_id, name in enumerate(names):
        mine = [r for r in rows if r["class_id"] == class_id]
        ap = average_precision([r["confidence"] for r in mine], [not r["error"] for r in mine], truth[class_id])
        classes[name] = {"ap": _round(ap), "truth": truth[class_id], "predictions": len(mine),
                         "correct": sum(1 for r in mine if not r["error"])}
    values = [c["ap"] for c in classes.values() if c["ap"] is not None]
    return {"iou": iou, "map": _round(float(np.mean(values))) if values else None, "classes": classes}


# ── 정답 대조 ──────────────────────────────────────────────────────────────────
# [정답 대조] boxes 는 신뢰도 내림차순이어야 한다(높은 박스가 먼저 정답을 차지). 틀린 종류 판별 순서:
#   correct → duplicate(이미 짝지어진 같은 클래스 정답) → wrong_class → localization(IoU 0.1 이상) → background
def match_golden(boxes: list[dict[str, Any]], truth: np.ndarray, truth_classes: list[int],
                 iou_threshold: float) -> list[str]:
    """박스(신뢰도 내림차순)마다 correct 또는 틀린 종류. truth: 정답 xyxy (m, 4) 또는 회전 박스 xywhr (m, 5)"""
    if truth.ndim == 2 and truth.shape[1] == 5:
        iou = rotated_iou(np.array([b["xywhr"] for b in boxes], np.float64).reshape(-1, 5), truth)
    else:
        pred = np.array([[b["bbox"]["x1"], b["bbox"]["y1"], b["bbox"]["x2"], b["bbox"]["y2"]] for b in boxes]).reshape(-1, 4)
        iou = box_iou(pred, truth)
    same = np.array([[b["class_id"] == c for c in truth_classes] for b in boxes], bool).reshape(len(boxes), len(truth_classes))
    used = np.zeros(len(truth_classes), bool)
    kinds = []
    for i in range(len(boxes)):
        # 같은 클래스이면서 아직 짝이 없는 정답 중 IoU 최대인 것을 찾는다(나머지는 −1 로 가림).
        free = np.where(same[i] & ~used, iou[i], -1.0)
        j = int(free.argmax()) if len(truth_classes) else -1
        if j >= 0 and free[j] >= iou_threshold:
            used[j] = True
            kinds.append("correct")
        elif len(truth_classes) and (np.where(same[i] & used, iou[i], 0)).max() >= iou_threshold:
            kinds.append("duplicate")
        elif len(truth_classes) and (np.where(~same[i], iou[i], 0)).max() >= iou_threshold:
            kinds.append("wrong_class")
        elif len(truth_classes) and (np.where(same[i], iou[i], 0)).max() >= 0.1:
            kinds.append("localization")
        else:
            kinds.append("background")
    return kinds


# ── 평가 행 ────────────────────────────────────────────────────────────────────
def candidate_thresholds(conf_thresholds: dict[int, float | None], initial: float) -> dict[int, float]:
    """평가 대상(자동 승인 후보) 기준: 클래스별 min(① 임계값, 초기값). 보정으로 임계값이 올라가도(자동 승인 안 함 1.0 포함)
    초기값 이상 박스는 계속 평가한다(④ 판단이 막히지 않게)."""
    return {c: min(t if t is not None else 1.0, initial) for c, t in conf_thresholds.items()}


def size_px(box: dict[str, Any], height: int, width: int) -> float:
    """크기 구간용 √(w·h) (letterbox 입력 기준). 회전 박스는 감싸는 사각형이 아니라 회전 박스 자신의 w·h"""
    return box_size(size_box(box), height, width)


def box_scores(box, features: np.ndarray, classes: list[int]) -> list[dict[str, float | None]]:
    """평가용 ④ 점수: 박스마다 {knn@k…, mds, rmds} (그 클래스 통계가 없으면 모두 None, ④ 통계가 없으면 빈 dict)"""
    if box is None or not len(classes):
        return [{} for _ in classes]
    table, order = box.all_scores(features)
    slots = {c: i for i, c in enumerate(order)}
    out = []
    for row, c in enumerate(classes):
        slot = slots.get(c)
        out.append({key: (float(v[row, slot]) if slot is not None else None) for key, v in table.items()})
    return out


def evaluate_image(item: dict[str, Any], labels: list[dict[str, Any]], height: int, width: int, image: str, *,
                   iou: float, candidate: dict[int, float], num_classes: int, box=None) -> dict[str, Any]:
    """평가셋 이미지 한 장: analyze(keep_features=True) 결과 하나 + 정답 라벨 → 평가 행·이미지 정보·클래스별 정답/맞은 수.

    item: analyze 결과 {"boxes", "signals", "features"}, labels: 확정 라벨 [{class_id, bbox_yolo | polygon_norm}],
    image: 이미지 키(SHA-256), iou: 정답 대조 IoU, candidate: candidate_thresholds 결과, box: ④ BoxDistance(없으면 None).
    반환 {"rows": [박스 하나 = 행 하나], "image": calibrate·simulate 의 images 항목, "kinds": 박스별 대조 결과,
          "truth": [클래스별 정답 수], "matched": [클래스별 맞은 박스 수]}
    행의 scores 는 모두 '클수록 틀렸을 가능성 높음' 방향이다(① 은 1 − conf)."""
    boxes, signals = item["boxes"], item["signals"]
    truth_classes = [int(b["class_id"]) for b in labels]
    kinds = match_golden(boxes, truth_boxes(labels, height, width), truth_classes, iou)
    truth = [truth_classes.count(c) for c in range(num_classes)]
    matched = [sum(1 for b, k in zip(boxes, kinds) if k == "correct" and b["class_id"] == c) for c in range(num_classes)]
    feats = item["features"] if item["features"] is not None else np.zeros((len(boxes), 0))
    extras = box_scores(box, feats, [b["class_id"] for b in boxes])
    distance = signals["image_distance"]["value"] if signals["image_distance"] else None
    info = {"truth": len(truth_classes), "extra_top": signals["tta"]["extra_top"],
            "agreement_extra_top": (signals.get("agreement") or {}).get("extra_top") or [],
            "distance_exceeded": bool(signals["image_distance"] and signals["image_distance"]["exceeded"])}
    rows = []
    for b, record, kind, extra in zip(boxes, signals["boxes"], kinds, extras):
        size = round(size_px(b, height, width), 1)
        rows.append({"image": image, "class_id": b["class_id"], "confidence": b["confidence"],
                     "confidence_ok": record["confidence_ok"],
                     "candidate": b["confidence"] >= candidate.get(b["class_id"], 1.0),
                     "error": kind != "correct", "kind": kind,
                     "size": size, "size_bucket": size_bucket(size),
                     "tta_min_iou": record["tta_min_iou"], "tta_class_match": record["tta_class_match"],
                     "tta_missing": record["tta_missing"],
                     "agreement_iou": record.get("agreement_iou"),
                     "agreement_class_match": record.get("agreement_class_match"),
                     "agreement_missing": record.get("agreement_missing"),
                     "margin": (record.get("box_distance") or {}).get("margin"),
                     "scores": {"confidence": 1.0 - b["confidence"], "tta": score_tta(record),
                                "image_distance": distance, "model_agreement": score_agreement(record), **extra}})
    return {"rows": rows, "image": info, "kinds": kinds, "truth": truth, "matched": matched}


# ── 평가 ───────────────────────────────────────────────────────────────────────
# 같은 박스 모집단 안에서 신호별 단독 AUROC, A단계 결합 AUROC, A+④(방식별) 결합 AUROC 를 계산한다.
def _signal_table(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    """rows(같은 모집단)의 신호별 AUROC 와 A단계·A+④ 결합 AUROC."""
    errors = np.array([r["error"] for r in rows], bool)
    out: dict[str, Any] = {"boxes": len(rows), "errors": int(errors.sum()), "auroc": {}, "combined": {}}
    if not rows:
        return out
    available = {}
    for key in list(A_SIGNALS) + keys:
        values = [r["scores"].get(key) for r in rows]
        if all(v is not None for v in values):
            available[key] = np.array(values, np.float64)
            out["auroc"][key] = _round(auroc(available[key], errors))
        else:
            out["auroc"][key] = None
    base = [available[k] for k in A_SIGNALS if k in available]
    if base:
        out["combined"]["A"] = _round(auroc(combined(base), errors))
        for key in keys:
            if key in available:
                out["combined"][f"A+{key}"] = _round(auroc(combined(base + [available[key]]), errors))
    return out


def _round(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def summarize(rows: list[dict[str, Any]], names: list[str], keys: list[str], min_boxes: int) -> dict[str, Any]:
    """평가 대상(자동 승인 후보) 박스의 클래스별·크기 구간별 AUROC"""
    candidates = [r for r in rows if r["candidate"]]
    per_class = {}
    for class_id, name in enumerate(names):
        mine = [r for r in candidates if r["class_id"] == class_id]
        table = _signal_table(mine, keys)
        table["class_id"] = class_id
        # 틀린 박스·맞은 박스가 둘 다 충분해야 AUROC 를 믿을 수 있다. 부족하면 B단계 판정을 하지 않는다.
        table["sufficient"] = table["errors"] >= min_boxes and table["boxes"] - table["errors"] >= min_boxes
        table["kinds"] = {k: sum(1 for r in mine if r["kind"] == k) for k in
                          ("correct", "wrong_class", "localization", "duplicate", "background")}
        per_class[name] = table
    return {"population": "candidates (conf ≥ min(클래스 임계값, 초기값))", "boxes": len(rows), "candidates": len(candidates),
            "all": _signal_table(candidates, keys), "classes": per_class,
            "sizes": {b: _signal_table([r for r in candidates if r["size_bucket"] == b], keys) for b in BUCKETS}}


# [B단계 판정] ④ 방식 중 단독 AUROC 최고(best)를 고르고, A 대비 A+best 의 개선폭(gain)과 나머지 조건을 확인한다.
# reasons 가 비어 있으면 eligible(켤 수 있음).
def decide(class_id: int, name: str, table: dict[str, Any], keys: list[str], confirmed_boxes: int, has_stats: bool,
           rolled_back: bool, cfg: StageBConfig) -> dict[str, Any]:
    """클래스 하나의 B단계 판정 {"eligible", "reasons", "method", "knn_k", "best_key", "auroc_a", "auroc_with", "gain"}"""
    singles = {k: table["auroc"].get(k) for k in keys if table["auroc"].get(k) is not None}
    best = max(singles, key=lambda k: (singles[k], -keys.index(k))) if singles else None
    base = table["combined"].get("A")
    with_best = table["combined"].get(f"A+{best}") if best else None
    gain = round(with_best - base, 4) if with_best is not None and base is not None else None
    reasons = []
    if confirmed_boxes < cfg.min_boxes:
        reasons.append(f"confirmed_boxes {confirmed_boxes} < {cfg.min_boxes}")
    if not has_stats:
        reasons.append("no_class_stats")
    if not table.get("sufficient") or gain is None:
        reasons.append(f"eval_insufficient (틀린 {table['errors']}·맞은 {table['boxes'] - table['errors']}, "
                       f"각 {cfg.eval_min_boxes} 이상 필요)")
    elif gain < cfg.min_gain:
        reasons.append(f"no_improvement (gain {gain} < {cfg.min_gain})")
    if rolled_back:
        reasons.append("rolled_back_for_this_model")
    method = best.split("@")[0] if best else None
    knn_k = int(best.split("@")[1]) if best and best.startswith("knn@") else None
    return {"class_id": class_id, "class_name": name, "eligible": not reasons, "reasons": reasons, "best_key": best,
            "method": method, "knn_k": knn_k, "auroc_single": singles.get(best) if best else None, "auroc_a": base,
            "auroc_with": with_best, "gain": gain, "confirmed_boxes": confirmed_boxes}


# ── 임계값 보정 ────────────────────────────────────────────────────────────────
def _prepare(rows: list[dict[str, Any]], box_on: dict[int, tuple[str, str, float]], variants: int,
             hold_conf: float) -> list[dict[str, Any]]:
    """평가 박스마다 판정 규칙에 필요한 값: 보류 여부, ④ 초과·margin(켜진 클래스만)"""
    out = []
    for r in rows:
        exceeded = margin = None
        if r["class_id"] in box_on:
            key, method, threshold = box_on[r["class_id"]]
            value = r["scores"].get(key)
            exceeded = value is not None and value > threshold
            margin = (r["margin"] or {}).get(method)
        out.append({**r, "hold": rules.is_hold(r["confidence"], r["tta_missing"], variants, hold_conf),
                    "exceeded": exceeded, "box_margin": margin})
    return out


def _fails(p: dict[str, Any], conf: float, iou: float | None, margin_min: float, agree: float | None = None) -> list[str]:
    return rules.box_failures(p["confidence"], conf, p["tta_min_iou"], p["tta_class_match"], iou or 0.0, p["exceeded"],
                              p["box_margin"], margin_min, p.get("agreement_iou"), p.get("agreement_class_match"),
                              agree or 0.0)


def _fixed_gates(prepared: list[dict[str, Any]], images: dict[str, dict[str, Any]]) -> dict[str, bool]:
    """임계값과 무관한 이미지 게이트(③ 초과·보류 뺀 박스 0건·사라진 박스·⑤ 운영 모델에 없는 박스)"""
    kept: dict[str, list[dict[str, Any]]] = {sha: [] for sha in images}
    for p in prepared:
        if not p["hold"]:
            kept[p["image"]].append(p)
    return {sha: info["distance_exceeded"] or not kept[sha]
            or any(p["tta_missing"] or p.get("agreement_missing") for p in kept[sha])
            for sha, info in images.items()}


# [이미지 단위 모의 판정] 주어진 임계값으로 평가셋 이미지를 판정해 자동 승인 수·오류 수를 센다(보정 전후 비교, 루프 비교 지표).
def simulate(prepared: list[dict[str, Any]], images: dict[str, dict[str, Any]], conf: dict[int, float],
             iou: dict[int, float], margin_min: float, agree: dict[int, float] | None = None) -> dict[str, Any]:
    """평가셋 이미지에 판정 규칙(게이트 포함)을 적용한 결과. 자동 승인 이미지가 틀렸다 = 보류를 뺀 박스 중 틀린 박스가 있거나
    맞은 박스 수가 정답 수보다 적다(놓친 정답·보류된 정답 포함)."""
    fixed = _fixed_gates(prepared, images)
    by_image: dict[str, list[dict[str, Any]]] = {sha: [] for sha in images}
    for p in prepared:
        if not p["hold"]:
            by_image[p["image"]].append(p)
    approved = errors = 0
    for sha, info in images.items():
        kept = by_image[sha]
        extra = any(c >= conf.get(class_id, 1.01) for found in info["extra_top"].values() for class_id, c in found)
        extra = extra or any(c >= conf.get(class_id, 1.01) for class_id, c in info.get("agreement_extra_top") or [])
        if fixed[sha] or extra or any(_fails(p, conf.get(p["class_id"], 1.01), iou.get(p["class_id"]), margin_min,
                                             (agree or {}).get(p["class_id"])) for p in kept):
            continue
        approved += 1
        errors += any(p["error"] for p in kept) or sum(1 for p in kept if not p["error"]) < info["truth"]
    return {"images": len(images), "auto_approved": approved,
            "auto_approve_rate": round(approved / len(images), 4) if images else None,
            "errors": errors, "error_rate": round(errors / approved, 4) if approved else None}


def calibrate(rows: list[dict[str, Any]], images: dict[str, dict[str, Any]], names: list[str],
              conf_now: dict[int, float], iou_now: dict[int, float], box_on: dict[int, tuple[str, str, float]],
              variants: int, cfg: CalibrationConfig, closed: set[int] | frozenset[int] = frozenset(),
              agree_now: dict[int, float] | None = None) -> dict[str, Any]:
    """클래스별로 '자동 승인 박스 오류율 ≤ 목표'(cfg.bound: 95% 상한 또는 점추정)를 만족하면서 자동 승인되는
    맞은 박스가 가장 많은 (① 신뢰도, ② IoU, ⑤ IoU) 임계값을 찾는다.

    closed(지금 자동 승인이 열리지 않은 클래스)와, cfg.require_calibration 일 때 이번에 calibrated 가 아닌 클래스는 이미지 단위
    전후 판정(simulate)에서만 자동 승인하지 않는 것으로 본다. 클래스 기록(before·after)의 임계값은 실제 값 그대로다.

    박스 단위: 보류가 아니고 임계값과 무관한 게이트(③·0건·사라진 박스)에 걸리지 않은 박스 중 ①②④ 를 모두 채우는 박스가 자동 승인.
    ① 후보는 그 박스들의 신뢰도 값, ② 후보는 cfg.tta_ious, ⑤ 후보는 (운영 모델로 평가했을 때) cfg.agreement_ious.
    같은 수면 더 높은(보수적인) 임계값을 고른다.
        calibrated    찾음
        no_feasible   자동 승인 박스가 min_boxes 이상이면서 목표를 채우는 임계값이 없다 → ① 1.0(자동 승인 안 함)
        insufficient  클래스 평가 박스(보류 제외)가 min_boxes 미만 → 그대로 둔다"""
    target, min_boxes = cfg.target, cfg.min_boxes
    margin_min = cfg.box_margin
    prepared = _prepare(rows, box_on, variants, cfg.hold_conf)
    gated = _fixed_gates(prepared, images)
    ious = list(cfg.tta_ious) if variants else [None]
    agree_now = dict(agree_now or {})
    has_agreement = any(r.get("agreement_iou") is not None for r in rows)
    agrees = list(cfg.agreement_ious) if has_agreement else [None]
    classes, conf_after, iou_after, agree_after = {}, dict(conf_now), dict(iou_now), dict(agree_now)
    for class_id, name in enumerate(names):
        mine = [p for p in prepared if p["class_id"] == class_id and not p["hold"]]
        correct_total = sum(1 for p in mine if not p["error"])
        before = {"conf_threshold": conf_now.get(class_id), "tta_iou": iou_now.get(class_id),
                  "agreement_iou": agree_now.get(class_id)}
        out = {"status": "insufficient", "before": before, "after": before, "boxes": len(mine),
               "errors": len(mine) - correct_total, "approved": None, "approved_errors": None, "error_rate": None,
               "error_bound": None,
               "coverage": None, "box_distance": box_on[class_id][0] if class_id in box_on else None}
        if len(mine) >= min_boxes:
            best = None
            usable = [p for p in mine if not gated[p["image"]]]
            # [격자 탐색] ② IoU 후보 × ⑤ IoU 후보마다, 조건을 통과한 박스를 신뢰도 내림차순으로 하나씩 더하며
            # '지금 신뢰도를 ① 임계값으로 하면' 자동 승인 박스 수·오류 수를 센다. 목표를 만족하는 조합 중 맞은 박스가 가장 많은 것을 고른다.
            for iou in ious:
                for agree in agrees:
                    passing = sorted((p for p in usable if not _fails(p, 0.0, iou, margin_min, agree)),
                                     key=lambda p: -p["confidence"])
                    count = errors = 0
                    for i, p in enumerate(passing):
                        count += 1
                        errors += p["error"]
                        if i + 1 < len(passing) and passing[i + 1]["confidence"] == p["confidence"]:
                            continue  # 같은 신뢰도는 함께 들어간다
                        # 점추정이 목표 이하일 때만 상한을 계산한다(상한 ≥ 점추정)
                        if count >= min_boxes and errors <= target * count and error_bound(errors, count, cfg) <= target:
                            # 튜플 비교 순서 = 우선순위: 맞은 박스 수 → 높은 ② IoU → 높은 ⑤ IoU → 높은 ① 신뢰도(같은 수면 보수적인 값).
                            candidate = (count - errors, iou if iou is not None else -1.0,
                                         agree if agree is not None else -1.0, p["confidence"], count, errors)
                            best = max(best, candidate) if best else candidate
            if best:
                correct, iou, agree, threshold, count, errors = best
                after = {"conf_threshold": threshold, "tta_iou": iou if variants else before["tta_iou"],
                         "agreement_iou": agree if has_agreement else before["agreement_iou"]}
                out.update(status="calibrated", approved=count, approved_errors=errors,
                           error_rate=round(errors / count, 4), error_bound=round(error_bound(errors, count, cfg), 4),
                           coverage=round(correct / correct_total, 4))
            else:
                after = {"conf_threshold": 1.0, "tta_iou": before["tta_iou"], "agreement_iou": before["agreement_iou"]}
                out.update(status="no_feasible", approved=0, approved_errors=0, coverage=0.0)
            out["after"] = after
            conf_after[class_id], iou_after[class_id] = after["conf_threshold"], after["tta_iou"]
            if after["agreement_iou"] is not None:
                agree_after[class_id] = after["agreement_iou"]
        classes[name] = out
    never = 1.01  # simulate 에서 자동 승인하지 않는 클래스
    conf_before = {c: (never if c in closed else t) for c, t in conf_now.items()}
    if cfg.require_calibration:
        conf_after = {c: (t if classes[names[c]]["status"] == "calibrated" else never) if c < len(names) else t
                      for c, t in conf_after.items()}
    return {"target": target, "min_boxes": min_boxes, "hold_conf": cfg.hold_conf,
            "bound": cfg.bound, "confidence": cfg.confidence,
            "zero_error_boxes": zero_error_boxes(cfg),
            "tta_ious": ious if variants else [], "agreement_ious": agrees if has_agreement else [], "classes": classes,
            "held_boxes": sum(1 for p in prepared if p["hold"]),
            "images": {"before": simulate(prepared, images, conf_before, iou_now, margin_min, agree_now),
                       "after": simulate(prepared, images, conf_after, iou_after, margin_min, agree_after)}}
