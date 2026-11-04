"""④ 박스 특징 거리 통계 (서버의 LABEL_STATS 가 ③ 과 함께 계산). 계획서 3.4절: 특징 추출은 공통, 거리 계산은 kNN·MDS·RMDS
세 가지를 모두 구현하고 평가셋 AUROC 로 클래스별로 고른다(tasks.detect.evaluation).

    kNN       x̂ = x / ‖x‖ → PCA → u = (PCA 결과) / ‖·‖. 예측 클래스의 확정 박스 u 중 k 번째로 가까운 이웃까지의 유클리드 거리
              (L2 정규화 필수, 참고 문헌 [1]. PCA 로 줄인 뒤 길이가 달라지므로 한 번 더 정규화한다)
    MDS       z = PCA(x) (정규화하지 않은 특징: 특징 크기도 정보다). (z − μ_c)ᵀ Σ⁻¹ (z − μ_c). 클래스별 평균 μ_c,
              공유 공분산 Σ 는 클래스 평균을 뺀 z 의 Ledoit-Wolf 수축 추정
    PCA 성분 수는 labeling_box_pca_dim 과 박스 수 ÷ 2 중 작은 값(계획: 32~64)
    RMDS      MDS − (z − μ_0)ᵀ Σ_0⁻¹ (z − μ_0). 배경 분포(전체 평균 μ_0·Ledoit-Wolf 공분산 Σ_0)와의 차 (Ren et al., 2021)
    margin    다른 클래스 중 가장 낮은 점수 − 예측 클래스 점수. 음수면 다른 클래스가 더 가깝다(클래스 혼동)

점수는 클수록 '확정 데이터와 다르다'. 클래스·방식별 임계값은 확정 박스의 교차 검증(이미지 단위 5겹, 같은 이미지의 박스는 같은 겹)
'보지 않은 박스' 점수의 분위(labeling_stats_percentile). 통계에 쓴 박스 자신의 점수는 실제보다 작게 나오기 때문이다(kNN k=1 이면 0).
확정 박스가 labeling_box_stats_min_boxes 보다 적은 클래스는 클래스 통계를 만들지 않는다(그 클래스 박스의 ④ 는 null).
배경 분포·PCA 에는 모든 확정 박스를 쓴다.

    (서버) labeling_root/stats/<모델 버전>/<통계 ID>/box_features.npz  (pickle 없음)
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

BOX_STATS_FILE = "box_features.npz"
# ④ 거리 방식 세 가지. 클래스마다 어느 방식을 쓸지는 LABEL_EVAL 이 평가셋 AUROC 로 고른다.
METHODS = ("knn", "mds", "rmds")
# 임계값을 정하는 교차 검증 겹 수(같은 이미지의 박스는 같은 겹에 넣는다: GroupKFold).
FOLDS = 5


# kNN 은 k 값마다 점수가 달라 'knn@5' 처럼 k 를 붙인 키로 구분한다.
def knn_key(k: int) -> str:
    return f"knn@{k}"


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


# [적합 결과] PCA(원 특징·정규화 특징) + 클래스 평균·공유 정밀도 행렬 + 배경 분포 + kNN 인덱스(정규화 확정 박스 벡터).
class _Core:
    """한 번의 적합 결과 (전체 또는 교차 검증의 한 겹)."""

    def __init__(self, mean, components, knn_mean, knn_components, classes, class_means, precision, bg_mean,
                 bg_precision, index, index_labels):
        self.mean, self.components = mean, components              # MDS·RMDS: 원래 특징의 PCA
        self.knn_mean, self.knn_components = knn_mean, knn_components  # kNN: L2 정규화한 특징의 PCA
        self.classes = [int(c) for c in classes]
        self.class_means, self.precision = class_means, precision
        self.bg_mean, self.bg_precision = bg_mean, bg_precision
        self.index, self.index_labels = index, index_labels
        self._slot = {c: i for i, c in enumerate(self.classes)}

    # Ledoit-Wolf: 표본이 적을 때도 역행렬이 안정적인 공분산 수축 추정. precision_ = 공분산의 역행렬.
    @classmethod
    def fit(cls, x: np.ndarray, labels: np.ndarray, classes: list[int], dim: int) -> _Core:
        from sklearn.covariance import LedoitWolf
        from sklearn.decomposition import PCA

        k = max(1, min(dim, len(x) // 2, x.shape[1]))
        pca = PCA(n_components=k, svd_solver="full").fit(x)
        z = pca.transform(x)
        knn_pca = PCA(n_components=k, svd_solver="full").fit(_normalize(x))
        keep = np.isin(labels, classes)
        means = np.stack([z[labels == c].mean(0) for c in classes])
        slot = {c: i for i, c in enumerate(classes)}
        within = z[keep] - means[[slot[int(c)] for c in labels[keep]]]
        precision = LedoitWolf().fit(within).precision_ if len(within) > 1 else np.eye(k)
        background = LedoitWolf().fit(z) if len(z) > 1 else None
        bg_precision = background.precision_ if background is not None else np.eye(k)
        u = _normalize(knn_pca.transform(_normalize(x[keep])))
        return cls(pca.mean_, pca.components_, knn_pca.mean_, knn_pca.components_, classes, means, precision,
                   z.mean(0), bg_precision, u.astype(np.float32), labels[keep].astype(np.int64))

    def project(self, features: np.ndarray) -> np.ndarray:
        return (np.asarray(features, np.float64) - self.mean) @ self.components.T

    def project_knn(self, features: np.ndarray) -> np.ndarray:
        return _normalize((_normalize(features) - self.knn_mean) @ self.knn_components.T)

    # einsum 으로 박스 n 개 × 클래스 c 개의 (z−μ)ᵀ P (z−μ) 를 한 번에 계산한다.
    def mahalanobis(self, z: np.ndarray) -> np.ndarray:
        """(n, 클래스 수) 클래스 평균까지의 제곱 마할라노비스 거리"""
        d = z[:, None, :] - self.class_means[None, :, :]
        return np.einsum("nck,kl,ncl->nc", d, self.precision, d)

    def background(self, z: np.ndarray) -> np.ndarray:
        d = z - self.bg_mean
        return np.einsum("nk,kl,nl->n", d, self.bg_precision, d)

    # ‖u−r‖² = ‖u‖² + ‖r‖² − 2u·r 로 거리 행렬을 만들고, np.partition 으로 k 번째 값만 뽑는다(전체 정렬보다 빠름).
    def knn(self, u: np.ndarray, k: int) -> np.ndarray:
        """(n, 클래스 수) 클래스 확정 박스 중 k 번째(박스 수보다 크면 가장 먼) 이웃까지의 거리. u: project_knn 결과"""
        out = np.zeros((len(u), len(self.classes)))
        for slot, c in enumerate(self.classes):
            ref = self.index[self.index_labels == c].astype(np.float64)
            d2 = np.maximum((u ** 2).sum(1)[:, None] + (ref ** 2).sum(1)[None, :] - 2 * u @ ref.T, 0.0)
            kth = min(k, len(ref)) - 1
            out[:, slot] = np.sqrt(np.partition(d2, kth, axis=1)[:, kth])
        return out

    def all_scores(self, features: np.ndarray, ks: list[int]) -> dict[str, np.ndarray]:
        """{방식 키: (n, 클래스 수)} — 방식 키는 knn@k·mds·rmds"""
        z = self.project(features)
        md = self.mahalanobis(z)
        u = self.project_knn(features)
        out = {knn_key(k): self.knn(u, k) for k in ks}
        out["mds"] = md
        out["rmds"] = md - self.background(z)[:, None]
        return out

    def slot(self, class_id: int) -> int | None:
        return self._slot.get(int(class_id))


# [④ 통계 객체] LABEL_STATS 가 fit → save, 자동라벨링·평가가 load → scores/threshold 로 쓴다.
class BoxDistance:
    """클래스별 kNN·MDS·RMDS 점수와 임계값."""

    def __init__(self, core: _Core, ks: list[int], counts: dict[int, int], thresholds: dict[str, np.ndarray],
                 percentile: float, levels: str, feature_dim: int, reference: dict[str, np.ndarray] | None = None,
                 pooling: str = "axis"):
        self.core, self.ks, self.counts = core, [int(k) for k in ks], {int(c): int(n) for c, n in counts.items()}
        self.thresholds = thresholds  # 방식 키 → (클래스 수,) core.classes 순서
        self.percentile, self.levels, self.feature_dim = float(percentile), str(levels), int(feature_dim)
        self.reference = reference or {}  # fit 때만: 교차 검증 점수 (박스 수,)
        self.pooling = str(pooling)       # 특징 방식: axis(사각형 박스 RoIAlign), 회전 박스 rotated·enclosing (app.labeling.features)

    @property
    def classes(self) -> list[int]:
        return self.core.classes

    @property
    def dim(self) -> int:
        return int(self.core.components.shape[0])

    # [적합 절차] 1) 박스가 min_boxes 이상인 클래스만 통계 대상  2) 전체 데이터로 _Core 적합
    # 3) GroupKFold 로 '보지 않은 박스' 점수(reference)를 모아  4) 클래스·방식별 percentile 분위를 임계값으로.
    @classmethod
    def fit(cls, features: np.ndarray, labels: np.ndarray, groups: np.ndarray, *, dim: int, ks: list[int],
            percentile: float, min_boxes: int, levels: str, pooling: str = "axis") -> BoxDistance | None:
        """클래스 통계를 만들 수 있는 클래스가 하나도 없으면 None."""
        from sklearn.model_selection import GroupKFold

        x = np.asarray(features, np.float64)
        labels, groups = np.asarray(labels, np.int64), np.asarray(groups)
        counts = {int(c): int((labels == c).sum()) for c in np.unique(labels)}
        classes = sorted(c for c, n in counts.items() if n >= min_boxes)
        if not classes or len(x) < 4:
            return None
        ks = sorted(set(int(k) for k in ks))
        core = _Core.fit(x, labels, classes, dim)
        keys = [knn_key(k) for k in ks] + ["mds", "rmds"]
        reference = {key: np.full(len(x), np.nan) for key in keys}
        n_groups = len(np.unique(groups))
        if n_groups >= 2:
            for train, held in GroupKFold(n_splits=min(FOLDS, n_groups)).split(x, labels, groups):
                fold_classes = [c for c in classes if (labels[train] == c).sum() >= 1]
                held = held[np.isin(labels[held], fold_classes)]
                if not len(held) or len(train) < 4:
                    continue
                fold = _Core.fit(x[train], labels[train], fold_classes, dim)
                scores = fold.all_scores(x[held], ks)
                slots = [fold.slot(c) for c in labels[held]]
                for key in keys:
                    reference[key][held] = scores[key][np.arange(len(held)), slots]
        own = core.all_scores(x, ks)  # 교차 검증 점수가 없는 클래스(한 이미지에만 있는 등)는 자기 점수로 대신한다
        thresholds = {}
        for key in keys:
            values = np.zeros(len(classes))
            for slot, c in enumerate(classes):
                mine = labels == c
                held = reference[key][mine]
                held = held[~np.isnan(held)]
                values[slot] = np.percentile(held if len(held) else own[key][mine, slot], percentile)
            thresholds[key] = values
        return cls(core, ks, counts, thresholds, percentile, levels, x.shape[1], reference, pooling)

    # 박스마다 예측 클래스 기준 점수와 margin(가장 가까운 다른 클래스 점수 − 내 클래스 점수)을 계산한다.
    def scores(self, features: np.ndarray, classes: list[int], k: int | dict[int, int] | None = None) -> list[dict | None]:
        """박스마다 {"knn", "knn_k", "mds", "rmds", "margin": {방식: 값}} 또는 None(그 클래스 통계 없음).
        k: kNN 의 k (클래스별 dict 가능). 통계에 없는 k 면 가장 가까운 후보를 쓴다."""
        if not len(features):
            return []
        wanted = {}
        for c in classes:
            want = k.get(int(c)) if isinstance(k, dict) else k
            wanted[int(c)] = min(self.ks, key=lambda v: abs(v - (want or self.ks[len(self.ks) // 2])))
        table = self.core.all_scores(np.asarray(features), sorted(set(wanted.values())))
        out = []
        for row, c in enumerate(classes):
            slot = self.core.slot(c)
            if slot is None:
                out.append(None)
                continue
            key = knn_key(wanted[int(c)])
            record = {"knn": float(table[key][row, slot]), "knn_k": wanted[int(c)],
                      "mds": float(table["mds"][row, slot]), "rmds": float(table["rmds"][row, slot]), "margin": {}}
            for method, name in (("knn", key), ("mds", "mds"), ("rmds", "rmds")):
                others = np.delete(table[name][row], slot)
                record["margin"][method] = float(others.min() - table[name][row, slot]) if len(others) else None
            out.append(record)
        return out

    def all_scores(self, features: np.ndarray) -> tuple[dict[str, np.ndarray], list[int]]:
        """평가용: ({방식 키: (n, 클래스 수)}, 클래스 순서)"""
        return self.core.all_scores(np.asarray(features), self.ks), self.classes

    def threshold(self, class_id: int, method: str, k: int | None = None) -> float | None:
        slot = self.core.slot(class_id)
        if slot is None:
            return None
        key = knn_key(k or self.ks[0]) if method == "knn" else method
        if key not in self.thresholds:
            key = knn_key(min(self.ks, key=lambda v: abs(v - (k or 0))))
        return float(self.thresholds[key][slot])

    def summary(self, names: list[str]) -> dict:
        per_class = {}
        for slot, c in enumerate(self.classes):
            per_class[names[c] if c < len(names) else str(c)] = {
                "class_id": c, "boxes": self.counts.get(c, 0),
                "thresholds": {key: round(float(v[slot]), 4) for key, v in self.thresholds.items()}}
        skipped = {names[c] if c < len(names) else str(c): n for c, n in self.counts.items() if c not in self.classes}
        return {"feature_dim": self.feature_dim, "pca_dim": self.dim, "levels": self.levels, "pooling": self.pooling,
                "knn_ks": self.ks,
                "percentile": self.percentile, "boxes": int(sum(self.counts.values())), "classes": per_class,
                "skipped_classes": skipped}

    # ── 저장·읽기 ────────────────────────────────────────────────────────────
    # npz(넘파이 배열 묶음)로 저장. allow_pickle 없이 읽을 수 있는 형식만 쓴다(임의 코드 실행 위험 없음).
    def save(self, path: Path) -> None:
        partial = path.with_name(f".{path.stem}.{os.getpid()}.partial.npz")
        keys = list(self.thresholds)
        counts = np.array(sorted(self.counts.items()), np.int64).reshape(-1, 2)
        c = self.core
        np.savez(partial, mean=c.mean, components=c.components, knn_mean=c.knn_mean, knn_components=c.knn_components,
                 classes=np.array(c.classes, np.int64),
                 class_means=c.class_means, precision=c.precision, bg_mean=c.bg_mean, bg_precision=c.bg_precision,
                 index=c.index, index_labels=c.index_labels, ks=np.array(self.ks, np.int64), counts=counts,
                 threshold_keys=np.array(keys), thresholds=np.stack([self.thresholds[k] for k in keys]),
                 percentile=self.percentile, levels=np.array(self.levels), feature_dim=self.feature_dim,
                 pooling=np.array(self.pooling))
        os.replace(partial, path)

    @classmethod
    def load(cls, path: Path) -> BoxDistance:
        with np.load(path, allow_pickle=False) as f:
            core = _Core(f["mean"], f["components"], f["knn_mean"], f["knn_components"], f["classes"], f["class_means"],
                         f["precision"], f["bg_mean"], f["bg_precision"], f["index"], f["index_labels"])
            thresholds = {str(k): f["thresholds"][i] for i, k in enumerate(f["threshold_keys"])}
            pooling = str(f["pooling"]) if "pooling" in f.files else "axis"  # O5 이전 통계는 사각형 박스만
            return cls(core, f["ks"].tolist(), {int(a): int(b) for a, b in f["counts"]}, thresholds,
                       float(f["percentile"]), str(f["levels"]), int(f["feature_dim"]), pooling=pooling)
