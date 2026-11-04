"""분류 평가셋 평가·임계값 보정 (참조 구현). 평가 행 하나 = 평가셋 이미지 하나.

틀림 = 예측 클래스 ≠ 정답 클래스. 신호 점수(signals.scores)마다 틀린 이미지 검출 AUROC 를 예측 클래스별로 낸다.
보정은 검출과 같은 원칙이다: 클래스별로 '자동 승인 오류율(상한) ≤ target' 을 만족하면서 자동 승인되는 맞은 이미지가 가장 많은
① 신뢰도·② 점수 변화 허용치를 찾는다. 임계값과 무관한 사유(TTA 클래스 변화·③ 초과·⑤ 불일치·플러그인)는 고정 게이트로 본다.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ...config import CalibrationConfig
from ...metrics import auroc, combined, error_bound, zero_error_boxes
from . import signals as sig

A_SIGNALS = ("confidence", "tta", "image_distance", "model_agreement")


def evaluation_row(item: dict[str, Any], truth: int, image: str) -> dict[str, Any]:
    """analyze 결과 하나 + 정답 클래스 → 평가 행"""
    label, signals = item["label"], item["signals"]
    tta, agree, distance = signals["tta"], signals.get("agreement"), signals.get("image_distance")
    plugin_gate = any("error" in r or r.get("exceeded") for r in (signals.get("plugins") or {}).values())
    return {"image": image, "class_id": label["class_id"], "truth": int(truth), "confidence": label["confidence"],
            "error": label["class_id"] != int(truth), "tta_max_shift": tta["max_shift"],
            "gated": bool(tta["class_match"] is False or (distance and distance["exceeded"])
                          or (agree and not agree["class_match"]) or plugin_gate),
            "scores": sig.scores(label, signals)}


def _table(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = np.array([r["error"] for r in rows], bool)
    out: dict[str, Any] = {"images": len(rows), "errors": int(errors.sum()), "auroc": {}, "combined": None}
    keys = sorted({k for r in rows for k in r["scores"]})
    available = {}
    for key in keys:
        values = [r["scores"].get(key) for r in rows]
        if rows and all(v is not None for v in values):
            available[key] = np.array(values, np.float64)
            value = auroc(available[key], errors)
            out["auroc"][key] = round(value, 4) if value is not None else None
        else:
            out["auroc"][key] = None
    base = [available[k] for k in A_SIGNALS if k in available]
    if base:
        value = auroc(combined(base), errors)
        out["combined"] = round(value, 4) if value is not None else None
    return out


def summarize(rows: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    """전체·예측 클래스별 신호 AUROC 와 정확도"""
    correct = sum(1 for r in rows if not r["error"])
    return {"images": len(rows), "accuracy": round(correct / len(rows), 4) if rows else None, "all": _table(rows),
            "classes": {name: _table([r for r in rows if r["class_id"] == c]) for c, name in enumerate(names)}}


def simulate(rows: list[dict[str, Any]], conf: dict[int, float], shift: dict[int, float | None],
             closed: set[int] | frozenset[int] = frozenset()) -> dict[str, Any]:
    """주어진 임계값으로 평가셋 이미지를 판정한 자동 승인 수·오류 수"""
    approved = [r for r in rows if not r["gated"] and r["class_id"] not in closed
                and r["confidence"] >= conf.get(r["class_id"], 1.01)
                and (shift.get(r["class_id"]) is None or r["tta_max_shift"] is None
                     or r["tta_max_shift"] <= shift[r["class_id"]])]
    errors = sum(1 for r in approved if r["error"])
    return {"images": len(rows), "auto_approved": len(approved),
            "auto_approve_rate": round(len(approved) / len(rows), 4) if rows else None,
            "errors": errors, "error_rate": round(errors / len(approved), 4) if approved else None}


def calibrate(rows: list[dict[str, Any]], names: list[str], conf_now: dict[int, float], shift_now: dict[int, float],
              cfg: CalibrationConfig, tta_shifts: tuple[float, ...], closed: set[int] | frozenset[int] = frozenset()
              ) -> dict[str, Any]:
    """클래스별 (① 신뢰도, ② 점수 변화 허용치) 보정. 상태: calibrated · no_feasible(① 1.0) · insufficient"""
    has_tta = any(r["tta_max_shift"] is not None for r in rows)
    shifts = list(tta_shifts) if has_tta else [None]
    classes, conf_after, shift_after = {}, dict(conf_now), dict(shift_now)
    for class_id, name in enumerate(names):
        mine = [r for r in rows if r["class_id"] == class_id]
        correct_total = sum(1 for r in mine if not r["error"])
        before = {"conf_threshold": conf_now.get(class_id), "tta_shift": shift_now.get(class_id)}
        out = {"status": "insufficient", "before": before, "after": before, "images": len(mine),
               "errors": len(mine) - correct_total, "approved": None, "approved_errors": None, "error_bound": None,
               "coverage": None}
        if len(mine) >= cfg.min_boxes:
            best = None
            usable = [r for r in mine if not r["gated"]]
            for limit in shifts:
                passing = sorted((r for r in usable if limit is None or r["tta_max_shift"] is None
                                  or r["tta_max_shift"] <= limit), key=lambda r: -r["confidence"])
                count = errors = 0
                for i, r in enumerate(passing):
                    count += 1
                    errors += r["error"]
                    if i + 1 < len(passing) and passing[i + 1]["confidence"] == r["confidence"]:
                        continue  # 같은 신뢰도는 함께 들어간다
                    if count >= cfg.min_boxes and errors <= cfg.target * count and error_bound(errors, count, cfg) <= cfg.target:
                        # 맞은 이미지 수 → 작은(엄격한) 허용치 → 높은 신뢰도 순으로 고른다
                        candidate = (count - errors, -(limit if limit is not None else 1.0), r["confidence"], count, errors, limit)
                        best = max(best, candidate, key=lambda c: c[:3]) if best else candidate
            if best:
                correct, _, threshold, count, errors, limit = best
                after = {"conf_threshold": threshold, "tta_shift": limit if has_tta else before["tta_shift"]}
                out.update(status="calibrated", approved=count, approved_errors=errors,
                           error_bound=round(error_bound(errors, count, cfg), 4),
                           coverage=round(correct / correct_total, 4) if correct_total else 0.0)
            else:
                after = {"conf_threshold": 1.0, "tta_shift": before["tta_shift"]}
                out.update(status="no_feasible", approved=0, approved_errors=0, coverage=0.0)
            out["after"] = after
            conf_after[class_id], shift_after[class_id] = after["conf_threshold"], after["tta_shift"]
        classes[name] = out
    closed_after = set(closed) | ({c for c, n in enumerate(names) if classes[n]["status"] != "calibrated"}
                                 if cfg.require_calibration else set())
    return {"target": cfg.target, "min_images": cfg.min_boxes, "bound": cfg.bound, "confidence": cfg.confidence,
            "zero_error_images": zero_error_boxes(cfg), "tta_shifts": shifts if has_tta else [], "classes": classes,
            "images": {"before": simulate(rows, conf_now, shift_now, closed),
                       "after": simulate(rows, conf_after, shift_after, closed_after)}}
