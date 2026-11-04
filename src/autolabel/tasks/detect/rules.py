"""검출 판정 규칙: 이미지 게이트 → 박스 보류·자동 승인 → 이미지 판정, 검수 우선순위, 감사 표본 (구축 계획 4.4절, M6).

자동라벨링이 이미지마다 신호(signals)를 계산한 뒤 판정한다. 신호를 끄면 판정하지 않는다(PREDICTED).

박스 (predictions.boxes 순서)
    hold    신뢰도 < hold_conf 이고 모든 TTA 변형에서 사라진 박스. 라벨에서 빼고 판정에 쓰지 않는다
            ('다른 신호에도 나타나지 않는' 아주 약한 박스. 거리는 '모른다'는 뜻이므로 거리만으로는 보류하지 않는다)
    auto    ① 신뢰도 ≥ 클래스 임계값 AND ② 모든 변형에서 같은 클래스로 매칭·최소 매칭 IoU ≥ 클래스 IoU 임계값
            AND (④ 켜진 클래스) 고른 방식의 거리 ≤ 임계값·margin ≥ box_margin
            AND (⑤ 운영 모델 있음) 운영 모델 박스와 같은 클래스로 매칭·매칭 IoU ≥ 클래스 ⑤ 임계값
    review  나머지 (reasons: confidence · tta_iou · tta_class · box_distance · box_margin · agreement_class · agreement_iou ·
            not_calibrated). not_calibrated: 자동 승인이 열리지 않은 클래스(0단계, stages). 보정 전에는 전부 검수한다(M8)
    TTA 를 끄면(변형 없음) ② 조건은, 운영 모델이 없으면 ⑤ 조건은 건너뛴다.

이미지 게이트 (하나라도 해당하면 박스와 무관하게 이미지 전체를 검수)
    image_distance  ③ 이미지 거리 > 임계값
    no_boxes        보류를 뺀 박스가 0건 (미검출 위험)
    tta_missing     보류가 아닌 박스가 한 변형에서라도 사라짐
    tta_extra       변형에서만 나온 박스 중 신뢰도 ≥ 그 클래스 임계값인 것이 있음
    model_disagreement  (⑤) 보류가 아닌 박스가 운영 모델에 없거나, 운영 모델에만 있는 박스 중 신뢰도 ≥ 그 클래스 임계값인 것이 있음

이미지 판정
    AUTO_APPROVED   게이트 통과 AND 보류를 뺀 모든 박스가 auto. 그중 감사 비율만큼 감사 표본(audit)으로 검수에 보낸다.
                    감사 비율은 박스 클래스의 단계별 비율 중 가장 큰 값(1단계 15%, 2단계 5%; 단계 미사용이면 audit_rate)
    REVIEW          나머지. 검수 우선순위(priority_basis):
                    distance     max(③ 거리, ④ 켜진 박스 거리) 를 임계값 기준으로 잰 값(1 = 임계값)
                    uncertainty  거리가 없거나 박스 클래스가 모두 0단계면(박스가 없으면 모든 클래스가 0단계면): 보류 아닌
                                 박스의 불확실성 최댓값 (box_uncertainty). 박스가 없으면 0.5

    감사 표본은 (요청 ID, 이미지 SHA-256) 해시로 고르므로 같은 요청을 다시 실행해도 같은 이미지가 뽑힌다.

predictions.judgment
    {"version": 3, "decision", "audit", "audit_rate", "priority", "priority_basis", "gate": [게이트 사유], "held": 보류 박스 수,
     "boxes": [{"verdict": auto·review·hold, "reasons": [...]}],
     "params": {"hold_conf", "box_margin", "tta_iou": {클래스 ID: 값}, "agreement_iou": {클래스 ID: 값}, "audit_rate",
                "closed": [0단계 클래스 ID], "audit_rates": {클래스 ID: 값}}}
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...config import RuleConfig
from ...constants import AUTO_APPROVED, REVIEW
from ...metrics import audit_pick, relative

# 판정 형식 버전. predictions.judgment.version 에 기록되어, 규칙이 바뀐 뒤에도 예전 결과를 구분할 수 있다.
RULES_VERSION = 3
GATES = ("image_distance", "no_boxes", "tta_missing", "tta_extra", "model_disagreement")
BOX_REASONS = ("confidence", "tta_iou", "tta_class", "box_distance", "box_margin", "agreement_class", "agreement_iou",
               "not_calibrated")
# 박스가 하나도 없는 이미지의 검수 우선순위(미검출일 수도, 정말 빈 이미지일 수도 있어 중간값).
EMPTY_UNCERTAINTY = 0.5

__all__ = ["AUTO_APPROVED", "REVIEW", "RULES_VERSION", "GATES", "BOX_REASONS", "EMPTY_UNCERTAINTY", "RuleParams",
           "rule_params", "is_hold", "box_failures", "relative", "priority", "box_uncertainty", "uncertainty",
           "audit_pick", "judge"]


# [판정 기준 묶음] 청크마다 신호 상태(서버 DB)에서 만든다. 판정 결과(judgment.params)에도 요약을 남긴다.
@dataclass
class RuleParams:
    conf: dict[int, float] = field(default_factory=dict)     # ① 클래스별 신뢰도 임계값 (tta_extra 게이트)
    tta_iou: dict[int, float] = field(default_factory=dict)  # ② 클래스별 최소 매칭 IoU 임계값
    agreement_iou: dict[int, float] = field(default_factory=dict)  # ⑤ 클래스별 운영 모델 일치 IoU 임계값
    hold_conf: float = 0.1
    box_margin: float = 0.0
    audit_rate: float = 0.05
    closed: set[int] = field(default_factory=set)             # 자동 승인이 열리지 않은 클래스 (0단계)
    audit_rates: dict[int, float] = field(default_factory=dict)  # 클래스 단계별 감사 비율 (없으면 audit_rate)

    def summary(self) -> dict[str, Any]:
        return {"hold_conf": self.hold_conf, "box_margin": self.box_margin, "audit_rate": self.audit_rate,
                "tta_iou": {str(c): v for c, v in sorted(self.tta_iou.items())},
                "agreement_iou": {str(c): v for c, v in sorted(self.agreement_iou.items())}, "closed": sorted(self.closed),
                "audit_rates": {str(c): v for c, v in sorted(self.audit_rates.items())}}


def rule_params(conf: dict[int, float], tta_iou: dict[int, float], agreement_iou: dict[int, float],
                stages: dict[int, dict[str, Any]], cfg: RuleConfig) -> RuleParams:
    """클래스별 임계값과 단계(stages.class_stages 결과)로 판정 기준을 만든다. 열리지 않은 클래스는 closed, 열린 클래스는 단계별 감사 비율."""
    return RuleParams(conf=dict(conf), tta_iou=dict(tta_iou), agreement_iou=dict(agreement_iou),
                      hold_conf=cfg.hold_conf, box_margin=cfg.box_margin, audit_rate=cfg.audit_rate,
                      closed={c for c, st in stages.items() if not st["open"]},
                      audit_rates={c: st["audit_rate"] for c, st in stages.items() if st["open"]})


# ── 박스 ───────────────────────────────────────────────────────────────────────
# 보류 박스는 라벨에서 빠진다(검수 묶음·학습 데이터에 들어가지 않음). 원본 결과에는 남아 있다.
def is_hold(confidence: float, tta_missing: list[str], variants: int, hold_conf: float) -> bool:
    """아주 약하고(신뢰도 < hold_conf) 모든 변형에서 사라진 박스. 변형이 없으면 신뢰도만 본다."""
    return confidence < hold_conf and len(tta_missing) == variants


# [박스 자동 승인 조건] 실패한 조건 이름을 모두 모아 돌려준다(검수 사유로 화면에 보인다).
def box_failures(confidence: float, conf_threshold: float, tta_min_iou: float | None, tta_class_match: bool | None,
                 tta_iou: float, box_exceeded: bool | None, box_margin: float | None, margin_min: float,
                 agreement: float | None = None, agreement_class_match: bool | None = None,
                 agreement_iou: float = 0.0) -> list[str]:
    """자동 승인 조건 중 못 채운 것. 빈 목록이면 auto. tta_min_iou None = TTA 없음, box_exceeded None = ④ 꺼짐,
    agreement None = ⑤ 없음(운영 모델 없음). agreement 는 운영 모델과의 매칭 IoU(없으면 0), agreement_iou 는 클래스 임계값."""
    reasons = []
    if confidence < conf_threshold:
        reasons.append("confidence")
    if tta_min_iou is not None:
        if not tta_class_match:
            reasons.append("tta_class")
        if tta_min_iou < tta_iou:
            reasons.append("tta_iou")
    if box_exceeded:
        reasons.append("box_distance")
    if box_exceeded is not None and box_margin is not None and box_margin < margin_min:
        reasons.append("box_margin")
    if agreement is not None:
        if not agreement_class_match:
            reasons.append("agreement_class")
        if agreement < agreement_iou:
            reasons.append("agreement_iou")
    return reasons


def _box_flags(record: dict[str, Any]) -> tuple[bool | None, float | None]:
    """④ 가 켜진 박스면 (임계값 초과, 고른 방식의 margin), 아니면 (None, None)"""
    box = record.get("box_distance") or {}
    if not box.get("active"):
        return None, None
    return bool(box.get("exceeded")), (box.get("margin") or {}).get(box.get("method"))


# ── 우선순위 ───────────────────────────────────────────────────────────────────
def priority(signals: dict[str, Any], keep: list[bool]) -> float:
    """검수 우선순위: ③ 이미지 거리와 ④ 가 켜진(보류 아닌) 박스 거리 중 임계값 대비 가장 먼 값. 거리가 없으면 0."""
    return round(max(_distances(signals, keep), default=0.0), 4)


def _distances(signals: dict[str, Any], keep: list[bool]) -> list[float]:
    values = []
    distance = signals.get("image_distance")
    if distance:
        values.append(relative(distance["value"], distance["threshold"]))
    for record, kept in zip(signals["boxes"], keep):
        box = record.get("box_distance") or {}
        if kept and box.get("active") and box.get("method"):
            values.append(relative(box[box["method"]], box.get("threshold")))
    return [v for v in values if v is not None]


def box_uncertainty(confidence: float, threshold: float | None, tta_min_iou: float | None, tta_missing: list,
                    tta_class_match: bool | None, agreement: float | None = None,
                    agreement_class_match: bool | None = None) -> float:
    """박스 하나의 불확실성 [0, 1]. 신뢰도는 클래스 임계값에 가까울수록 높다(임계값에서 1, 0 이나 1 에서 0 쪽).
    TTA 불안정(사라짐·클래스 바뀜 1, 아니면 1 − 최소 매칭 IoU)은 박스가 자동 승인 후보에 가까운 만큼(신뢰도 / 임계값)만 센다.
    낮은 conf(0.05)로 뽑은 배경 잡음 박스가 우선순위를 차지하지 않게 한다(COCO128 데모에서 확인)."""
    # 임계값을 0.01~0.99 로 묶어 0 나누기와 극단값을 피한다.
    t = 0.5 if threshold is None or threshold >= 1.0 else min(max(threshold, 0.01), 0.99)  # 1.0(자동 승인 안 함)은 초기값 기준
    # 신뢰도 불확실성: 임계값과 같으면 1, 임계값에서 가장 먼 쪽 끝(0 또는 1)이면 0 으로 선형 감소.
    value = 1.0 - abs(confidence - t) / max(t, 1.0 - t)
    unstable = 1.0 if tta_missing or tta_class_match is False else (1.0 - tta_min_iou if tta_min_iou is not None else 0.0)
    if agreement is not None:  # ⑤ 운영 모델과 다르면(없음·다른 클래스 1, 아니면 1 − IoU) 같은 비중으로
        unstable = max(unstable, 1.0 if not agreement_class_match else 1.0 - agreement)
    return max(value, unstable * min(1.0, confidence / t))


def uncertainty(boxes: list[dict[str, Any]], signals: dict[str, Any], keep: list[bool]) -> float:
    """0단계 검수 우선순위: 보류 아닌 박스의 불확실성(box_uncertainty) 최댓값. 박스가 없으면 EMPTY_UNCERTAINTY."""
    values = [box_uncertainty(box["confidence"], record["conf_threshold"], record["tta_min_iou"], record["tta_missing"],
                              record["tta_class_match"], record.get("agreement_iou"), record.get("agreement_class_match"))
              for box, record, kept in zip(boxes, signals["boxes"], keep) if kept]
    return round(max(values, default=EMPTY_UNCERTAINTY), 4)


# ── 이미지 ─────────────────────────────────────────────────────────────────────
# [이미지 판정 순서]
#   1) 박스마다: 보류인지 → 자동 승인 조건(①②④⑤) 확인 → 닫힌 클래스면 not_calibrated 추가
#   2) 이미지 게이트(③ 거리, 박스 없음, TTA 사라짐/추가, 운영 모델 불일치) 확인
#   3) 게이트 통과 + 남은 박스 모두 auto → AUTO_APPROVED, 그중 일부를 감사 표본으로
#   4) 검수로 가는 이미지(또는 감사 표본)는 우선순위 계산: 거리 기준, 0단계면 불확실성 기준
def judge(boxes: list[dict[str, Any]], signals: dict[str, Any], params: RuleParams,
          audit_key: str | None = None) -> dict[str, Any]:
    """이미지 한 장의 판정 (predictions.judgment 형식)"""
    variants = len(signals["tta"]["variants"])
    verdicts, keep = [], []
    for box, record in zip(boxes, signals["boxes"]):
        if is_hold(box["confidence"], record["tta_missing"], variants, params.hold_conf):
            verdicts.append({"verdict": "hold", "reasons": []})
            keep.append(False)
            continue
        exceeded, margin = _box_flags(record)
        reasons = box_failures(box["confidence"], record["conf_threshold"] if record["conf_threshold"] is not None else 0.0,
                               record["tta_min_iou"], record["tta_class_match"],
                               params.tta_iou.get(box["class_id"], 0.0), exceeded, margin, params.box_margin,
                               record.get("agreement_iou"), record.get("agreement_class_match"),
                               params.agreement_iou.get(box["class_id"], 0.0))
        if box["class_id"] in params.closed:
            reasons.append("not_calibrated")
        verdicts.append({"verdict": "review" if reasons else "auto", "reasons": reasons})
        keep.append(True)

    # 이미지 게이트: 하나라도 걸리면 박스가 모두 auto 여도 이미지 전체를 검수로 보낸다.
    gate = []
    distance = signals.get("image_distance")
    if distance and distance["exceeded"]:
        gate.append("image_distance")
    if not any(keep):
        gate.append("no_boxes")
    if any(k and record["tta_missing"] for k, record in zip(keep, signals["boxes"])):
        gate.append("tta_missing")
    extra = signals["tta"].get("extra_top") or {}
    # TTA 변형에서만 나온 박스가 임계값 이상 신뢰도면 원본이 놓친 물체일 수 있다(미검출 위험).
    if any(conf >= params.conf.get(class_id, 1.01) for found in extra.values() for class_id, conf in found):
        gate.append("tta_extra")
    agreement = signals.get("agreement")
    if agreement and (any(k and record.get("agreement_missing") for k, record in zip(keep, signals["boxes"]))
                      or any(conf >= params.conf.get(class_id, 1.01) for class_id, conf in agreement["extra_top"])):
        gate.append("model_disagreement")

    approved = not gate and all(v["verdict"] == "auto" for v in verdicts if v["verdict"] != "hold")
    # 감사 비율: 이미지 안 auto 박스 클래스들의 단계별 감사 비율 중 가장 큰 값(가장 조심해야 하는 클래스 기준).
    rate = max((params.audit_rates.get(b["class_id"], params.audit_rate)
                for b, v in zip(boxes, verdicts) if v["verdict"] == "auto"), default=params.audit_rate)
    audit = approved and audit_key is not None and audit_pick(audit_key, rate)
    value, basis = None, None
    if not approved or audit:
        distances = _distances(signals, keep)
        kept_classes = {b["class_id"] for b, k in zip(boxes, keep) if k}
        # 0단계 이미지: 박스 클래스가 모두 닫혀 있다(박스가 없으면 모든 클래스가 닫혀 있을 때). 한 묶음 안에서 거리·불확실성
        # 척도가 섞이지 않도록 0단계뿐인 시기에는 박스 없는 이미지도 불확실성으로 잰다
        stage0 = kept_classes <= params.closed if kept_classes else bool(params.conf) and set(params.conf) <= params.closed
        if distances and not stage0:
            value, basis = round(max(distances), 4), "distance"
        else:
            value, basis = uncertainty(boxes, signals, keep), "uncertainty"
    return {"version": RULES_VERSION, "decision": AUTO_APPROVED if approved else REVIEW, "audit": audit,
            "audit_rate": rate, "priority": value, "priority_basis": basis, "gate": gate,
            "held": sum(1 for v in verdicts if v["verdict"] == "hold"), "boxes": verdicts, "params": params.summary()}
