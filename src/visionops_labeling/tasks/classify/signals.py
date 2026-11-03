"""분류 판정 신호 (참조 구현, 라벨링 확장 규격 1.0). 이미지 한 장 = 라벨 하나.

    ① confidence      예측 클래스 점수 ≥ 클래스별 임계값
    ② tta             어댑터가 선언한 변환(tasks.classify.tta)마다 다시 예측해 예측 클래스가 유지되는지(class_match)와
                      예측 클래스 점수가 얼마나 변하는지(shift = |변형 점수 − 원본 점수|)
    ③ image_distance  어댑터 embed 의 이미지 거리(distance.image). 거리 통계가 있을 때만
    ⑤ model_agreement 운영 모델의 예측 클래스가 같은지, 예측 클래스 점수 차(shift). 운영 모델이 없으면 null
    플러그인           모델 고유 신호(plugins). {"value", "threshold", "exceeded", "detail"} 또는 {"error"}

예측 클래스 정하기 (decision)
    binary  클래스 구성이 OK·NG 두 개면 NG 점수 ≥ threshold 일 때 NG (threshold: 요청값 → 어댑터 defaults → 0.5)
    argmax  그 밖에는 가장 큰 점수

predictions.signals (분류)
    {"version": 1, "task": "classify", "stats_version", "decision": {"rule", "threshold", "positive"},
     "conf_threshold", "confidence_ok",
     "tta": {"variants": [변형 이름], "results": {변형: {"class_id", "class_match", "shift"}}, "class_match", "max_shift"},
     "image_distance": {"value", "threshold", "exceeded", "method"} 또는 null,
     "agreement": {"model_version", "class_id", "class_match", "shift"} 또는 null,
     "plugins": {이름: 기록}}
"""
from __future__ import annotations

from typing import Any

import numpy as np

SIGNALS_VERSION = 1


def decision_rule(names: list[str], threshold: float | None) -> dict[str, Any]:
    """클래스 구성 이름으로 예측 클래스 정하는 방식. OK·NG 두 클래스면 이진(NG 점수 기준), 아니면 argmax."""
    if sorted(names) == ["NG", "OK"]:
        return {"rule": "binary", "threshold": 0.5 if threshold is None else float(threshold), "positive": names.index("NG")}
    return {"rule": "argmax", "threshold": None, "positive": None}


def predicted(row: np.ndarray, decision: dict[str, Any]) -> int:
    """점수 한 줄(클래스 구성 순서) → 예측 클래스 ID"""
    if decision["rule"] == "binary":
        positive = decision["positive"]
        return positive if float(row[positive]) >= decision["threshold"] else 1 - positive
    return int(np.argmax(row))


def label(row: np.ndarray, decision: dict[str, Any], names: list[str]) -> dict[str, Any]:
    """예측 라벨 {class_id, class_name, confidence(예측 클래스 점수), scores(클래스 구성 순서)}"""
    class_id = predicted(row, decision)
    return {"class_id": class_id, "class_name": names[class_id], "confidence": round(float(row[class_id]), 6),
            "scores": [round(float(v), 6) for v in row]}


def compute(row: np.ndarray, tta: dict[str, np.ndarray], decision: dict[str, Any], conf_threshold: float,
            distance: dict[str, Any] | None, stats_version: str | None,
            agreement: tuple[str, np.ndarray] | None = None, plugins: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """이미지 한 장의 신호. row·tta 값·agreement 점수는 클래스 구성 순서의 점수 한 줄."""
    pred = predicted(row, decision)
    conf = float(row[pred])
    results = {}
    for name, other in tta.items():
        cls = predicted(other, decision)
        results[name] = {"class_id": cls, "class_match": cls == pred, "shift": round(abs(float(other[pred]) - conf), 4)}
    agree = None
    if agreement is not None:
        version, other = agreement
        cls = predicted(other, decision)
        agree = {"model_version": version, "class_id": cls, "class_match": cls == pred,
                 "shift": round(abs(float(other[pred]) - conf), 4)}
    return {"version": SIGNALS_VERSION, "task": "classify", "stats_version": stats_version, "decision": decision,
            "conf_threshold": conf_threshold, "confidence_ok": conf >= conf_threshold,
            "tta": {"variants": list(tta), "results": results,
                    "class_match": all(r["class_match"] for r in results.values()) if results else None,
                    "max_shift": max((r["shift"] for r in results.values()), default=None)},
            "image_distance": distance, "agreement": agree, "plugins": dict(plugins or {})}


def scores(label: dict[str, Any], signals: dict[str, Any]) -> dict[str, float | None]:
    """신호를 '클수록 틀렸을 가능성이 높은' 점수로 (평가 AUROC 용). 값이 없는 신호는 None."""
    tta = signals["tta"]
    agree = signals.get("agreement")
    distance = signals.get("image_distance")
    out = {"confidence": 1.0 - label["confidence"],
           "tta": (tta["max_shift"] + (0.0 if tta["class_match"] else 1.0)) if tta["class_match"] is not None else None,
           "image_distance": distance["value"] if distance else None,
           "model_agreement": (agree["shift"] + (0.0 if agree["class_match"] else 1.0)) if agree else None}
    for name, record in (signals.get("plugins") or {}).items():
        out[f"plugin:{name}"] = record.get("value")
    return out
