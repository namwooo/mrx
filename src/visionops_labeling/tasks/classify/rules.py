"""분류 판정 규칙 (참조 구현). 이미지 한 장 = 라벨 하나라 사유는 모두 이미지 단위다.

자동 승인 (AUTO_APPROVED): 아래 사유가 하나도 없음. 그중 감사 비율만큼 감사 표본(audit)으로 검수에 보낸다.
    confidence        예측 클래스 점수 < 클래스 임계값
    tta_class         한 변형에서라도 예측 클래스가 바뀜
    tta_shift         변형 중 가장 큰 점수 변화 > 클래스 허용치
    image_distance    ③ 이미지 거리 > 임계값
    agreement_class   ⑤ 운영 모델의 예측 클래스가 다름
    plugin:<이름>      플러그인 값 > 임계값
    plugin_error:<이름> 플러그인 실패(그 이미지는 검수)
    not_calibrated    자동 승인이 열리지 않은 클래스(0단계, stages)

검수 우선순위: ③ 거리가 있고 열린 클래스면 거리(임계값 = 1), 아니면 불확실성(임계값에 가까운 점수·TTA·운영 모델 불일치).

predictions.judgment (분류). 서버 집계(검출과 같은 키)를 그대로 쓰도록 사유를 gate 에 넣고 boxes 는 빈 목록이다.
    {"version": 1, "task": "classify", "decision", "audit", "audit_rate", "priority", "priority_basis",
     "gate": [사유], "held": 0, "boxes": [], "params": {...}}
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...constants import AUTO_APPROVED, REVIEW
from ...metrics import audit_pick, relative

RULES_VERSION = 1
REASONS = ("confidence", "tta_class", "tta_shift", "image_distance", "agreement_class", "not_calibrated")


@dataclass
class ClassifyRuleParams:
    conf: dict[int, float] = field(default_factory=dict)        # ① 클래스별 신뢰도 임계값
    tta_shift: dict[int, float] = field(default_factory=dict)   # ② 클래스별 점수 변화 허용치
    default_conf: float = 0.9
    default_tta_shift: float = 0.1
    audit_rate: float = 0.05
    closed: set[int] = field(default_factory=set)                # 자동 승인이 열리지 않은 클래스 (0단계)
    audit_rates: dict[int, float] = field(default_factory=dict)  # 클래스 단계별 감사 비율

    def summary(self) -> dict[str, Any]:
        return {"conf": {str(c): v for c, v in sorted(self.conf.items())},
                "tta_shift": {str(c): v for c, v in sorted(self.tta_shift.items())},
                "default_conf": self.default_conf, "default_tta_shift": self.default_tta_shift,
                "audit_rate": self.audit_rate, "closed": sorted(self.closed),
                "audit_rates": {str(c): v for c, v in sorted(self.audit_rates.items())}}


def reasons(label: dict[str, Any], signals: dict[str, Any], params: ClassifyRuleParams) -> list[str]:
    """자동 승인을 막는 사유 목록(빈 목록이면 자동 승인 대상)."""
    cid = label["class_id"]
    out = []
    if label["confidence"] < params.conf.get(cid, params.default_conf):
        out.append("confidence")
    tta = signals["tta"]
    if tta["class_match"] is False:
        out.append("tta_class")
    if tta["max_shift"] is not None and tta["max_shift"] > params.tta_shift.get(cid, params.default_tta_shift):
        out.append("tta_shift")
    distance = signals.get("image_distance")
    if distance and distance["exceeded"]:
        out.append("image_distance")
    agree = signals.get("agreement")
    if agree and not agree["class_match"]:
        out.append("agreement_class")
    for name, record in sorted((signals.get("plugins") or {}).items()):
        if "error" in record:
            out.append(f"plugin_error:{name}")
        elif record.get("exceeded"):
            out.append(f"plugin:{name}")
    if cid in params.closed:
        out.append("not_calibrated")
    return out


def uncertainty(label: dict[str, Any], signals: dict[str, Any]) -> float:
    """[0, 1]. 점수가 판정 경계에 가까울수록(이진: NG 점수가 threshold 에 가까울수록, 다중: 1·2위 점수 차가 작을수록) 높고,
    TTA 클래스 변화·운영 모델 불일치는 1, TTA 점수 변화는 그 크기만큼."""
    decision, values = signals["decision"], label["scores"]
    if decision["rule"] == "binary":
        t = min(max(decision["threshold"], 0.01), 0.99)
        value = 1.0 - abs(values[decision["positive"]] - t) / max(t, 1.0 - t)
    else:
        top = sorted(values, reverse=True) + [0.0]
        value = 1.0 - (top[0] - top[1])
    tta = signals["tta"]
    unstable = 1.0 if tta["class_match"] is False else (tta["max_shift"] or 0.0)
    agree = signals.get("agreement")
    if agree and not agree["class_match"]:
        unstable = 1.0
    return round(max(value, unstable), 4)


def judge(label: dict[str, Any], signals: dict[str, Any], params: ClassifyRuleParams,
          audit_key: str | None = None) -> dict[str, Any]:
    """이미지 한 장의 판정 (predictions.judgment 형식)"""
    gate = reasons(label, signals, params)
    approved = not gate
    rate = params.audit_rates.get(label["class_id"], params.audit_rate)
    audit = approved and audit_key is not None and audit_pick(audit_key, rate)
    value, basis = None, None
    if not approved or audit:
        distance = signals.get("image_distance")
        far = relative(distance["value"], distance["threshold"]) if distance else None
        if far is not None and label["class_id"] not in params.closed:
            value, basis = round(far, 4), "distance"
        else:
            value, basis = uncertainty(label, signals), "uncertainty"
    return {"version": RULES_VERSION, "task": "classify", "decision": AUTO_APPROVED if approved else REVIEW, "audit": audit,
            "audit_rate": rate, "priority": value, "priority_basis": basis, "gate": gate, "held": 0, "boxes": [],
            "params": params.summary()}
