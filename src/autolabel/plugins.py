"""신호 플러그인: 모델 고유 판정 신호를 외부(예: 제이비소프트)가 만들어 붙이는 확장 지점 (라벨링 확장 규격 1.0).

플랫폼이 계산하는 공통 신호(①②③⑤)에 더해, 모델을 아는 쪽이 '이 예측은 틀렸을 가능성이 높다'는 신호를 이미지마다 낸다.
판정·임계값 보정·감사·루프는 플랫폼이 한다. 플러그인은 값만 내고, 그 값이 임계값을 넘으면 플랫폼이 그 이미지를 검수로 보낸다.

    value      float, 클수록 틀렸을 가능성이 높다(NaN·Inf 금지)
    threshold  플랫폼이 정한다: 평가셋 보정 값 → 플러그인 default_threshold. 둘 다 없으면 기록만 하고 판정에 쓰지 않는다
    detail     JSON 으로 바꿀 수 있는 dict(4KB 이하). 검수 화면 표시용

등록: wheel 의 entry point 그룹 visionops.labeling_signals (discover). 같은 프로세스 안에서는 register 로도 된다(시험).
안전: 플러그인이 예외를 내거나 규격에 맞지 않는 값을 내면 그 묶음의 이미지마다 {"error": 사유} 를 남기고, 판정은 그 이미지를
검수로 보낸다(plugin_error:<이름>). 작업은 멈추지 않는다.
"""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .constants import LABELING_SPEC_VERSION

ENTRY_POINT_GROUP = "visionops.labeling_signals"
NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,39}$")
DETAIL_MAX_BYTES = 4096

logger = logging.getLogger("autolabel")


@dataclass(frozen=True)
class PluginSignal:
    """이미지 한 장의 플러그인 신호."""
    value: float
    detail: dict[str, Any] = field(default_factory=dict)


class SignalPlugin(Protocol):
    name: str                         # 소문자로 시작, 소문자·숫자·_ 2~40자. entry point 이름과 같다
    spec_version: str                 # 따르는 라벨링 확장 규격 버전 ("1.0")
    tasks: tuple[str, ...]            # 적용할 작업 종류 ("classify",)
    adapters: tuple[str, ...]         # 적용할 어댑터 이름. 빈 튜플이면 같은 작업 종류의 모든 모델
    default_threshold: float | None   # value 가 이보다 크면 검수. None 이면 기록만(평가로 효과를 본 뒤 플랫폼이 켠다)

    def compute(self, model: Any, images: list[np.ndarray], scores: np.ndarray) -> list[PluginSignal]:
        """images: 원본 이미지(어댑터 decode 그대로), scores: 같은 이미지의 model.predict 결과 (N, C).
        이미지 순서대로 PluginSignal 을 하나씩. images·scores 를 바꾸지 않는다."""
        ...


def plugin_problems(plugin: Any) -> list[str]:
    """플러그인 속성이 규격에 맞지 않는 이유 목록(빈 목록이면 통과). 등록·확인 도구가 쓴다."""
    problems = []
    name = getattr(plugin, "name", None)
    if not isinstance(name, str) or not NAME_PATTERN.match(name):
        problems.append(f"name 은 소문자로 시작하는 소문자·숫자·_ 2~40자여야 합니다: {name!r}")
    version = str(getattr(plugin, "spec_version", ""))
    if version.split(".")[0] != LABELING_SPEC_VERSION.split(".")[0]:
        problems.append(f"spec_version 주 버전이 {LABELING_SPEC_VERSION} 과 다릅니다: {version!r}")
    tasks = getattr(plugin, "tasks", None)
    if not isinstance(tasks, tuple) or not tasks or not all(isinstance(t, str) for t in tasks):
        problems.append("tasks 는 작업 종류 이름의 튜플이어야 합니다 (예: (\"classify\",))")
    adapters = getattr(plugin, "adapters", None)
    if not isinstance(adapters, tuple) or not all(isinstance(a, str) for a in adapters):
        problems.append("adapters 는 어댑터 이름의 튜플이어야 합니다 (모든 모델이면 빈 튜플)")
    threshold = getattr(plugin, "default_threshold", "missing")
    if threshold == "missing" or not (threshold is None or (isinstance(threshold, (int, float)) and math.isfinite(threshold))):
        problems.append("default_threshold 는 유한한 숫자 또는 None 이어야 합니다")
    if not callable(getattr(plugin, "compute", None)):
        problems.append("compute(model, images, scores) 메서드가 없습니다")
    return problems


def applies(plugin: SignalPlugin, task: str, adapter: str) -> bool:
    return task in plugin.tasks and (not plugin.adapters or adapter in plugin.adapters)


_registry: dict[str, SignalPlugin] = {}


def register(plugin: SignalPlugin, replace: bool = False) -> None:
    """같은 프로세스 안에서 플러그인을 등록한다. 규격에 맞지 않으면 ValueError."""
    problems = plugin_problems(plugin)
    if problems:
        raise ValueError(f"신호 플러그인이 규격에 맞지 않습니다: {problems}")
    if plugin.name in _registry and not replace:
        raise ValueError(f"이미 등록된 신호 플러그인입니다: {plugin.name}")
    _registry[plugin.name] = plugin


def unregister(name: str) -> None:
    _registry.pop(name, None)


def discover() -> list[str]:
    """entry point(visionops.labeling_signals)로 설치된 플러그인을 찾아 등록한다. 등록한 이름 목록.
    규격에 맞지 않거나 읽지 못한 플러그인은 경고 로그를 남기고 건너뛴다."""
    from importlib.metadata import entry_points

    found = []
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        try:
            obj = ep.load()
            plugin = obj() if isinstance(obj, type) else obj
            if plugin.name != ep.name:
                raise ValueError(f"entry point 이름({ep.name})과 name({plugin.name})이 다릅니다")
            register(plugin, replace=True)
            found.append(plugin.name)
        except Exception as error:  # noqa: BLE001 — 플러그인 하나 때문에 라벨링이 멈추지 않게
            logger.warning("신호 플러그인 %s 을 등록하지 못했습니다: %s", ep.name, error)
    return found


def registered(task: str | None = None, adapter: str | None = None) -> list[SignalPlugin]:
    """등록된 플러그인(이름순). task·adapter 를 주면 적용 대상만."""
    out = [p for _, p in sorted(_registry.items())]
    if task is not None:
        out = [p for p in out if applies(p, task, adapter or "")]
    return out


def _detail_error(detail: Any) -> str | None:
    if not isinstance(detail, dict):
        return "detail 은 dict 여야 합니다"
    try:
        size = len(json.dumps(detail, ensure_ascii=False, allow_nan=False).encode("utf-8"))
    except (TypeError, ValueError) as error:
        return f"detail 을 JSON 으로 바꿀 수 없습니다: {error}"
    return f"detail 이 {DETAIL_MAX_BYTES}바이트를 넘습니다({size})" if size > DETAIL_MAX_BYTES else None


def run(plugins: list[SignalPlugin], model: Any, images: list[np.ndarray], scores: np.ndarray,
        thresholds: dict[str, float | None] | None = None) -> list[dict[str, dict[str, Any]]]:
    """이미지마다 {플러그인 이름: {"value", "threshold", "exceeded", "detail"} 또는 {"error"}}.
    thresholds: 플랫폼이 정한 임계값(보정 값). 없으면 플러그인 default_threshold."""
    out: list[dict[str, dict[str, Any]]] = [{} for _ in images]
    for plugin in plugins:
        threshold = (thresholds or {}).get(plugin.name, plugin.default_threshold)
        try:
            results = list(plugin.compute(model, images, scores))
            if len(results) != len(images):
                raise ValueError(f"결과 수({len(results)})가 이미지 수({len(images)})와 다릅니다")
            records = []
            for item in results:
                value = getattr(item, "value", None)
                if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)) or not math.isfinite(value):
                    raise ValueError(f"value 는 유한한 숫자여야 합니다: {value!r}")
                detail = getattr(item, "detail", None) or {}
                error = _detail_error(detail)
                if error:
                    raise ValueError(error)
                records.append({"value": round(float(value), 6),
                                "threshold": round(float(threshold), 6) if threshold is not None else None,
                                "exceeded": bool(threshold is not None and float(value) > threshold), "detail": detail})
        except Exception as error:  # noqa: BLE001 — 플러그인 오류는 그 이미지들을 검수로 보내고 계속한다
            logger.warning("신호 플러그인 %s 실패: %s", plugin.name, error)
            records = [{"error": f"{type(error).__name__}: {error}"[:300]} for _ in images]
        for row, record in zip(out, records):
            row[plugin.name] = record
    return out
