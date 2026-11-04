"""③ 이미지 거리 통계: 임베딩 → PCA → 대각 공분산 마할라노비스 거리 (방식에 따라 + PCA 잔차 한 축).

작업 종류와 무관하다(검출은 neck 특징 GAP, 분류는 어댑터 embed 를 넣는다). 통계는 사람이 확인한 확정 데이터(시드·검수)로만 만든다.

    pca_mahalanobis_residual (기본)
        거리² = Σ_k (z_k − 평균_k)² / 분산_k + ‖잔차‖² / 잔차 척도
        PCA 공간만 보면 확정 데이터에 없던 방향으로 벗어난 이미지(처음 보는 환경)가 투영되면서 가까워 보이므로,
        PCA 로 설명되지 않는 잔차 크기를 한 축으로 더한다. 잔차 척도와 임계값은 5겹 교차 검증의 '보지 않은 이미지'
        거리로 정한다. 학습에 쓴 이미지 자신의 거리는 실제보다 작게 나오기 때문이다(성분 수가 이미지 수에 가까우면
        모두 같은 값이 된다). 성분 수는 확정 이미지 수의 절반 이하(성분별 분산을 안정적으로 추정하기 위해).
    pca_mahalanobis_cv
        잔차 없이 PCA 공간 안의 거리만. 성분 수 상한·교차 검증 임계값은 위와 같다
    pca_mahalanobis (계획서 4.3절 그대로)
        PCA 공간 안의 거리만, 성분 수는 이미지 수 − 1 이하, 임계값은 통계에 쓴 이미지 자신의 거리 95분위
자세한 근거: 「이미지 거리(신호 ③) PCA 잔차 보강 설명」(2026-10-15).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..constants import DistanceMethod

FOLDS = 5


# [③ 이미지 거리 통계] fit → save(npz) → load → distance(새 이미지 임베딩).
class ImageDistance:
    """이미지 임베딩 PCA + 대각 공분산 마할라노비스 거리 (방식에 따라 + PCA 잔차 한 축)."""

    def __init__(self, method: DistanceMethod, mean: np.ndarray, components: np.ndarray, center: np.ndarray,
                 variance: np.ndarray, residual_scale: float, threshold: float, percentile: float, reference: np.ndarray):
        self.method = DistanceMethod(method)
        self.mean, self.components, self.center, self.variance = mean, components, center, variance
        self.residual_scale = float(residual_scale)  # 잔차를 쓰지 않는 방식은 0
        self.threshold, self.percentile, self.reference = float(threshold), float(percentile), reference
        self.explained: float | None = None  # fit 때만: PCA 설명 분산 비율 합

    @staticmethod
    def components_for(method: DistanceMethod, images: int, features: int, dim: int) -> int:
        """PCA 성분 수. 계획서 방식은 이미지 수 − 1, 교차 검증 방식은 이미지 수의 절반까지."""
        cap = images - 1 if method == DistanceMethod.PCA else images // 2
        return max(1, min(dim, cap, features))

    @staticmethod
    def _pca(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """(평균, 성분, 투영 평균, 투영 분산, 설명 분산 비율 합)"""
        from sklearn.decomposition import PCA

        pca = PCA(n_components=k, svd_solver="full").fit(x)
        z = pca.transform(x)
        variance = z.var(0, ddof=1)
        variance = np.maximum(variance, max(float(variance.max()) * 1e-6, 1e-12))  # 분산이 0 인 성분으로 나누지 않게
        return pca.mean_, pca.components_, z.mean(0), variance, float(pca.explained_variance_ratio_.sum())

    # 교차 검증 방식: 각 겹에서 나머지로 PCA 를 적합하고, 빠진 이미지(보지 않은 이미지)의 거리를 모아 임계값 분위를 구한다.
    # 잔차 방식은 먼저 보지 않은 이미지의 잔차 평균을 잔차 척도로 정한 뒤 거리를 계산한다.
    @classmethod
    def fit(cls, embeddings: np.ndarray, dim: int, percentile: float,
            method: DistanceMethod | str = DistanceMethod.PCA_RESIDUAL) -> ImageDistance:
        from sklearn.model_selection import KFold

        method = DistanceMethod(method)
        x = np.asarray(embeddings, np.float64)
        mean, components, center, variance, explained = cls._pca(x, cls.components_for(method, *x.shape, dim))
        if method == DistanceMethod.PCA:  # 통계에 쓴 이미지 자신의 거리
            item = cls(method, mean, components, center, variance, 0.0, 0.0, percentile, np.zeros(0))
            reference, scale = item.distance(x), 0.0
        else:
            folds = []  # (보지 않은 이미지 번호, 그 이미지 없이 만든 모델)
            for train, held in KFold(n_splits=min(FOLDS, len(x)), shuffle=True, random_state=0).split(x):
                k = cls.components_for(method, len(train), x.shape[1], dim)
                folds.append((held, cls(method, *cls._pca(x[train], k)[:4], 1.0, 0.0, percentile, np.zeros(0))))
            scale = 0.0
            if method == DistanceMethod.PCA_RESIDUAL:
                residuals = np.zeros(len(x))
                for held, model in folds:
                    residuals[held] = model.residual(x[held])
                scale = max(float(residuals.mean()), 1e-12)
            reference = np.zeros(len(x))
            for held, model in folds:
                model.residual_scale = scale
                reference[held] = model.distance(x[held])
        item = cls(method, mean, components, center, variance, scale, float(np.percentile(reference, percentile)),
                   percentile, reference)
        item.explained = explained
        return item

    # (PCA 좌표 z, PCA 로 설명되지 않는 잔차 벡터)를 함께 돌려준다.
    def _project(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        centered = np.asarray(embeddings, np.float64) - self.mean
        z = centered @ self.components.T
        return z, centered - z @ self.components

    def residual(self, embeddings: np.ndarray) -> np.ndarray:
        """PCA 로 설명되지 않는 잔차의 제곱 크기"""
        return (self._project(embeddings)[1] ** 2).sum(1)

    # 대각 마할라노비스: 성분별 (값−평균)²/분산 합. 잔차 방식이면 잔차² / 척도를 한 축으로 더한다.
    def distance(self, embeddings: np.ndarray) -> np.ndarray:
        z, rest = self._project(embeddings)
        squared = (((z - self.center) ** 2) / self.variance).sum(1)
        if self.method == DistanceMethod.PCA_RESIDUAL:
            squared = squared + (rest ** 2).sum(1) / self.residual_scale
        return np.sqrt(squared)

    @property
    def dim(self) -> int:
        return int(self.components.shape[0])

    def save(self, path: Path) -> None:
        partial = path.with_name(f".{path.stem}.{os.getpid()}.partial.npz")
        np.savez(partial, method=np.array(self.method.value), mean=self.mean, components=self.components,
                 center=self.center, variance=self.variance, residual_scale=self.residual_scale, threshold=self.threshold,
                 percentile=self.percentile, reference=self.reference)
        os.replace(partial, path)

    @classmethod
    def load(cls, path: Path) -> ImageDistance:
        with np.load(path, allow_pickle=False) as f:
            # method 가 없는 파일은 방식을 고를 수 있기 전(10/15)에 만든 통계로, 잔차 방식이다
            method = str(f["method"]) if "method" in f.files else DistanceMethod.PCA_RESIDUAL
            return cls(method, f["mean"], f["components"], f["center"], f["variance"], float(f["residual_scale"]),
                       float(f["threshold"]), float(f["percentile"]), f["reference"])


# ③ 이미지 거리 기록(값·임계값·초과 여부·방식). 검출·분류 신호가 같은 형식으로 남긴다.
def distance_record(value: float, threshold: float, method: str) -> dict:
    return {"value": round(float(value), 4), "threshold": round(float(threshold), 4), "exceeded": bool(value > threshold),
            "method": method}
