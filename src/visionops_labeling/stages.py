"""클래스별 자동라벨링 단계 (수정안 10/20 4.1절, M8). 단계는 시스템 전체가 아니라 클래스마다 신호 상태로 정해진다.
작업 종류와 무관하다(검출·분류 모두 ① 신뢰도 행의 보정 결과로 연다).

    0 보조 라벨링   자동 승인 없음(전부 검수, 사전 라벨 보조). 검수 우선순위는 불확실성(낮은 신뢰도·TTA 불안정)
    1 부분 자동     그 라벨링 모델로 한 평가셋 보정이 calibrated(오류율 95% 상한 ≤ 목표)인 클래스. 자동 승인,
                    감사 비율 audit_rate_open(15%) → 열린 뒤 감사 박스가 audit_settle_boxes 이상이면 기본 비율
    2 안정          1단계 + ④ B단계가 같은 모델 버전으로 켜짐(검출만). 감사 비율 audit_rate(5%)

단계 정하기 (① 신뢰도 행의 metrics)
    calibration  보정 결과 {status, model_version, …}. status=calibrated 이고 모델 버전이 같아야 연다
    closed       감사 롤백으로 닫힘 {model_version, …}. 같은 모델 버전에서는 다시 열지 않는다(새 모델의 보정으로 연다)
    open_audit   열린 뒤의 감사 누적(자동 승인 박스 수·틀린 수). 오류율이 목표를 넘으면 한 단계 내려간다
require_calibration 을 끄면 M6 처럼 모든 클래스를 연 것으로 보고(1단계 이상) 감사 비율은 audit_rate 하나다.

states 는 신호 상태 행 목록이다. 속성 signal·class_id·class_name·active·metrics 만 읽는다(서버의 SignalState 행 그대로 넘겨도 된다).
"""
from __future__ import annotations

from typing import Any, Protocol

from .config import StageConfig
from .constants import IMAGE_LEVEL, SignalName

# 단계 번호: 0 보조 라벨링 / 1 부분 자동 / 2 안정
ASSIST, PARTIAL, STABLE = 0, 1, 2


class SignalStateLike(Protocol):
    signal: str
    class_id: int
    class_name: str | None
    active: bool
    metrics: dict[str, Any]


# [단계 계산] 클래스별 ① 신뢰도 행의 metrics(calibration·closed·open_audit)와 ④ 상태로 단계·감사 비율을 정한다.
# 열림 판정 순서: 보정 불필요 설정 → 감사 롤백으로 닫힘 → 보정 안 됨 → 다른 모델로 보정됨 → 열림
def class_stages(states: list[SignalStateLike], model_version: str | None, cfg: StageConfig) -> dict[int, dict[str, Any]]:
    """클래스 ID → {class_id, class_name, stage, open, reason, audit_rate, open_audit}.
    model_version 이 None 이면 모델 버전 일치를 보지 않는다(목록 표시용: 각 클래스가 보정된 모델 기준)."""
    by = {(s.signal, s.class_id): s for s in states}
    out = {}
    for (signal, class_id), conf in sorted(by.items(), key=lambda kv: kv[0][1]):
        if signal != SignalName.CONFIDENCE.value or class_id == IMAGE_LEVEL:
            continue
        calibration = conf.metrics.get("calibration") or {}
        closed = conf.metrics.get("closed") or {}
        model = model_version or calibration.get("model_version")
        if not cfg.require_calibration:
            is_open, reason = True, "calibration_not_required"
        elif closed and closed.get("model_version") == model:
            is_open, reason = False, "audit_rollback"
        elif calibration.get("status") != "calibrated":
            is_open, reason = False, f"calibration_{calibration.get('status') or 'none'}"
        elif calibration.get("model_version") != model:
            is_open, reason = False, "calibrated_for_other_model"
        else:
            is_open, reason = True, "calibrated"
        box = by.get((SignalName.BOX_DISTANCE.value, class_id))
        box_on = bool(box and box.active and (model is None or box.metrics.get("model_version") == model))
        stage = (STABLE if box_on else PARTIAL) if is_open else ASSIST
        open_audit = conf.metrics.get("open_audit") or {"audited": 0, "errors": 0}
        settled = open_audit.get("audited", 0) >= cfg.audit_settle_boxes
        # 1단계로 막 열린 클래스는 감사 비율을 높게(15%) 두고, 감사 박스가 충분히 쌓이면 기본 비율(5%)로 낮춘다.
        rate = (cfg.audit_rate_open
                if cfg.require_calibration and stage == PARTIAL and not settled
                else cfg.audit_rate)
        out[class_id] = {"class_id": class_id, "class_name": conf.class_name, "stage": stage, "open": is_open,
                         "reason": reason, "audit_rate": rate, "open_audit": open_audit}
    return out
