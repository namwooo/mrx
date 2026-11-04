"""분류 이미지 묶음 분석 (참조 구현): 원본 예측 → (신호를 켰으면) ③ 임베딩 거리 → ② TTA → ⑤ 운영 모델 → 플러그인 → 신호.

모델 호출 횟수: predict 1 + TTA 변형 수, embed 1(거리 통계가 있거나 특징을 남길 때), 운영 모델 predict 1(있을 때).
max_batch 를 넘는 묶음은 나눠 부른다. 판정은 rules.judge.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ... import plugins as plug
from ...config import ClassifyConfig
from ...distance.image import ImageDistance, distance_record
from . import signals as sig
from .model import ClassifierModel, ModelInfo, batched, class_map
from .rules import ClassifyRuleParams
from .tta import apply, expand


class ClassifySetup:
    """청크 하나의 분류 신호 계산 조건."""

    def __init__(self, *, names: list[str], info: ModelInfo, enabled: bool = True, threshold: float | None = None,
                 conf_thresholds: dict[int, float] | None = None, rules: ClassifyRuleParams | None = None,
                 transforms: tuple[str, ...] | None = None, distance: ImageDistance | None = None,
                 stats_id: str | None = None, production: dict[str, Any] | None = None,
                 plugins: list[plug.SignalPlugin] | None = None, plugin_thresholds: dict[str, float | None] | None = None,
                 cfg: ClassifyConfig = ClassifyConfig()):
        """transforms: 쓸 TTA 변환 이름(어댑터 선언 중 일부만 쓸 때). None 이면 어댑터 선언 전부.
        threshold: 이진 분류 NG 기준(요청값). None 이면 어댑터 defaults → cfg.threshold."""
        declared = tuple(info.tta)
        chosen = declared if transforms is None else tuple(t for t in transforms if t in declared)
        self.names, self.info, self.enabled = list(names), info, enabled
        self.decision = sig.decision_rule(self.names, threshold if threshold is not None
                                          else info.threshold if info.threshold is not None else cfg.threshold)
        self.variants = expand(chosen)
        self.conf_thresholds = {c: (conf_thresholds or {}).get(c, cfg.conf_threshold) for c in range(len(self.names))}
        self.rules = rules or ClassifyRuleParams(conf=dict(self.conf_thresholds), default_conf=cfg.conf_threshold,
                                                 default_tta_shift=cfg.tta_shift)
        self.distance, self.stats_id, self.production = distance, stats_id, production
        self.plugins = list(plugins or [])
        self.plugin_thresholds = dict(plugin_thresholds or {})

    def summary(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "task": "classify", "adapter": self.info.name, "decision": self.decision,
                "tta": self.variants, "stats_version": self.stats_id,
                "production": self.production["version"] if self.production else None,
                "plugins": [p.name for p in self.plugins]}


def _scores(model: ClassifierModel, arrays: list[np.ndarray], cols: list[int], max_batch: int) -> tuple[np.ndarray, np.ndarray]:
    """(모델 열 순서 점수, 클래스 구성 순서 점수). 점수는 0~1 유한한 값이어야 한다."""
    raw = batched(model.predict, arrays, max_batch)
    if raw.shape[1] != len(cols) or not np.isfinite(raw).all() or raw.min() < -1e-6 or raw.max() > 1 + 1e-6:
        raise ValueError(f"분류 점수가 규격(0~1, 열 {len(cols)}개)에 맞지 않습니다: 모양 {raw.shape}")
    return raw, raw[:, cols]


def analyze(model: ClassifierModel, arrays: list[np.ndarray], setup: ClassifySetup, *,
            production: ClassifierModel | None = None, keep_features: bool = False) -> list[dict[str, Any]]:
    """이미지 묶음 하나. [{"label", "signals"(신호를 끄면 None), "features"(keep_features 일 때 임베딩 한 줄)}]"""
    if not arrays:
        return []
    cols = class_map(list(model.class_names), setup.names)
    max_batch = setup.info.max_batch
    raw, scores = _scores(model, arrays, cols, max_batch)
    labels = [sig.label(row, setup.decision, setup.names) for row in scores]
    if not setup.enabled:
        return [{"label": lab, "signals": None, "features": None} for lab in labels]
    embeddings = None
    if setup.distance is not None or keep_features:
        embeddings = batched(model.embed, arrays, max_batch)
        if not np.isfinite(embeddings).all():
            raise ValueError("임베딩에 NaN·Inf 가 있습니다")
    distances = setup.distance.distance(embeddings) if setup.distance is not None else None
    tta = {v: _scores(model, [apply(a, v) for a in arrays], cols, max_batch)[1] for v in setup.variants}
    agreement = None
    if production is not None and setup.production is not None:
        prod_cols = class_map(list(production.class_names), setup.names)
        agreement = _scores(production, arrays, prod_cols, max_batch)[1]
    extra = plug.run(setup.plugins, model, arrays, raw, setup.plugin_thresholds) if setup.plugins else [{} for _ in arrays]
    out = []
    for i, lab in enumerate(labels):
        distance = (distance_record(distances[i], setup.distance.threshold, setup.distance.method.value)
                    if distances is not None else None)
        signals = sig.compute(scores[i], {v: s[i] for v, s in tta.items()}, setup.decision,
                              setup.conf_thresholds[lab["class_id"]], distance, setup.stats_id,
                              (setup.production["version"], agreement[i]) if agreement is not None else None, extra[i])
        out.append({"label": lab, "signals": signals,
                    "features": embeddings[i] if keep_features and embeddings is not None else None})
    return out
