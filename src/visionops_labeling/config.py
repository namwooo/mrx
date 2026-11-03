"""라이브러리 설정 묶음. 라이브러리는 환경 변수를 읽지 않고, 호출하는 쪽이 이 객체를 만들어 넘긴다.

기본값은 서버 설정(app.config.Settings 의 VISIONOPS_LABELING_*) 기본값과 같다(tests/test_visionops_labeling.py 가 확인).
서버는 app.labeling.configs 에서 Settings → 이 객체로 바꾼다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TtaConfig:
    """검출 ② TTA: 좌우 뒤집기 여부, 크기 변경 비율(640 기준), 같은 물체로 보는 최소 매칭 IoU."""
    flip: bool = True
    scales: tuple[float, ...] = (0.8,)
    match_iou: float = 0.3


@dataclass(frozen=True)
class RuleConfig:
    """검출 판정 규칙의 클래스 공통 값. 클래스별 임계값은 신호 상태(보정 결과)에서 온다."""
    hold_conf: float = 0.1       # 보류: 이 신뢰도 미만이고 모든 TTA 변형에서 사라진 박스
    box_margin: float = 0.0      # ④ 켜진 클래스의 최소 margin
    audit_rate: float = 0.05     # 기본 감사 비율


@dataclass(frozen=True)
class StageConfig:
    """클래스별 자동라벨링 단계(0 보조 · 1 부분 자동 · 2 안정)와 단계별 감사 비율."""
    require_calibration: bool = True   # 보정되지 않은 클래스는 자동 승인하지 않는다
    audit_rate: float = 0.05           # 2단계(또는 감사가 충분히 쌓인 1단계) 감사 비율
    audit_rate_open: float = 0.15      # 막 열린 1단계 감사 비율
    audit_settle_boxes: int = 150      # 열린 뒤 감사 박스가 이 수 이상이면 audit_rate 로 낮춘다


@dataclass(frozen=True)
class CalibrationConfig:
    """임계값 보정: 클래스별 '자동 승인 오류율 ≤ target' 을 만족하면서 자동 승인이 가장 많은 임계값을 찾는다."""
    target: float = 0.02                                       # 자동 승인 오류율 목표
    min_boxes: int = 20                                        # 클래스 평가 표본·자동 승인 표본 최소 수
    bound: Literal["clopper_pearson", "point"] = "clopper_pearson"
    confidence: float = 0.95                                   # Clopper-Pearson 신뢰 수준
    tta_ious: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9)    # 검출 ② 후보
    agreement_ious: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9)  # 검출 ⑤ 후보
    hold_conf: float = 0.1
    box_margin: float = 0.0
    require_calibration: bool = True


@dataclass(frozen=True)
class StageBConfig:
    """검출 ④ B단계 판정 조건."""
    min_boxes: int = 300        # 클래스의 사람 확인 박스 수
    min_gain: float = 0.01      # ①②③ 결합 대비 ④ 추가 시 AUROC 최소 개선폭
    eval_min_boxes: int = 3     # AUROC 계산 조건: 틀린·맞은 박스 각각


@dataclass(frozen=True)
class ClassifyConfig:
    """분류 자동라벨링 기본값 (참조 구현, 서버 연동 전). 클래스별 값은 보정 결과로 바뀐다."""
    conf_threshold: float = 0.9                    # ① 예측 클래스 점수가 이 값 이상이어야 자동 승인 후보
    threshold: float = 0.5                         # 이진(OK/NG) 판정 기준: NG 점수 ≥ threshold 면 NG (어댑터 defaults 가 우선)
    tta_shift: float = 0.1                         # ② 변환 뒤 예측 클래스 점수 변화 허용치
    tta_shifts: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3)  # 보정이 비교하는 ② 후보
