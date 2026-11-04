"""작업 종류와 무관한 평가·판정 계산: AUROC, 순위 결합, AP, 자동 승인 오류율 상한, 감사 표본 추첨, 임계값 기준 거리."""
from __future__ import annotations

import hashlib

import numpy as np

from .config import CalibrationConfig


# [AUROC] Mann-Whitney U 통계로 계산: (틀린 표본 순위 합 − 최소 가능 합) / (틀린 수 × 맞은 수).
# 0.5 = 무작위, 1.0 = 점수만으로 틀린 표본을 완벽히 가려냄.
def auroc(scores: list[float] | np.ndarray, errors: list[bool] | np.ndarray) -> float | None:
    """틀린 표본(True)이 맞은 표본보다 점수가 높을 확률(동점은 반). 한쪽이 없으면 None."""
    from scipy.stats import rankdata

    s, y = np.asarray(scores, np.float64), np.asarray(errors, bool)
    pos, neg = int(y.sum()), int((~y).sum())
    if pos == 0 or neg == 0:
        return None
    ranks = rankdata(s)
    return float((ranks[y].sum() - pos * (pos + 1) / 2) / (pos * neg))


# 신호마다 단위가 달라 값을 그대로 더할 수 없으므로, 각 신호를 0~1 순위로 바꾼 뒤 평균한다.
def combined(columns: list[np.ndarray]) -> np.ndarray:
    """신호 점수 열들의 백분위 순위 평균 (같은 모집단 안에서)"""
    from scipy.stats import rankdata

    return np.mean([rankdata(c) / len(c) for c in columns], axis=0)


# envelope: 오른쪽(재현율 큰 쪽)에서부터 정밀도 최댓값을 채운 곡선(COCO 식 보간).
def average_precision(confidences: list[float], correct: list[bool], truth: int) -> float | None:
    """AP (모든 점 보간): 신뢰도 내림차순 정밀도-재현율 곡선 아래 넓이. 정답이 없으면 None."""
    if truth <= 0:
        return None
    order = np.argsort(-np.asarray(confidences, np.float64), kind="stable")
    hits = np.asarray(correct, bool)[order]
    if not len(hits):
        return 0.0
    tp = np.cumsum(hits)
    precision = tp / np.arange(1, len(hits) + 1)
    recall = tp / truth
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    steps = np.diff(np.concatenate([[0.0], recall]))
    return float((steps * envelope).sum())


# Clopper-Pearson 단측 상한 = Beta(신뢰수준; 오류+1, 표본−오류) 분위. 표본이 적을수록 상한이 커져 보수적으로 판정된다.
def error_bound(errors: int, count: int, cfg: CalibrationConfig) -> float:
    """자동 승인 오류율의 보정 기준 값: 점추정 또는 Clopper-Pearson 단측 상한"""
    if count <= 0:
        return 1.0
    if cfg.bound == "point":
        return errors / count
    if errors >= count:
        return 1.0
    from scipy.stats import beta

    return float(beta.ppf(cfg.confidence, errors + 1, count - errors))


# 오류 0건일 때 상한 = 1 − (1−신뢰수준)^(1/n) ≤ 목표를 만족하는 최소 n. 예: 목표 2%, 신뢰 95% → 149.
def zero_error_boxes(cfg: CalibrationConfig) -> int | None:
    """오류 0건으로 목표를 증명하는 데 필요한 자동 승인 표본 수 (계획서 3.3절 표). 점추정이면 None"""
    if cfg.bound == "point" or cfg.target <= 0:
        return None
    return int(np.ceil(np.log(1 - cfg.confidence) / np.log(1 - cfg.target) - 1e-9))


# 해시 앞 12자리(48비트)를 0~1 값으로 바꿔 비교한다. 난수 대신 해시를 쓰므로 재실행해도 같은 이미지가 뽑힌다.
def audit_pick(key: str, rate: float) -> bool:
    """key(요청 ID:이미지 SHA-256)의 해시가 [0, 1) 에서 rate 미만이면 감사 표본."""
    if rate <= 0:
        return False
    digest = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:12], 16)
    return digest / float(1 << 48) < rate


# 예) 임계값 10, 거리 15 → 1.5 (임계값보다 50% 멀다). 거리 종류가 달라도 같은 척도로 비교하려고 쓴다.
def relative(value: float, threshold: float | None) -> float | None:
    """임계값 기준으로 잰 거리: 1 + (값 − 임계값) / |임계값|. 임계값이 양수면 값/임계값과 같고, RMDS 처럼 음수일 수 있는 점수에도 쓴다."""
    if threshold is None or threshold == 0:
        return None
    return 1.0 + (value - threshold) / abs(threshold)
