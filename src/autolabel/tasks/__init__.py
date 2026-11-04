"""작업 종류별 라벨링 모듈 등록표.

작업 종류(클래스 구성의 task)마다 모듈 하나가 '분석(analyze) → 판정(judge)' 과 신호·판정 형식 버전을 갖는다.
새 작업 종류(분할·이상탐지)는 같은 모양의 모듈을 만들어 register_task 로 등록한다. 판정 결과는 모든 작업 종류가 같은 키
(decision·audit·priority·gate·held·boxes)를 써서 서버 집계·검수 순서·루프가 그대로 동작하게 한다.

    detect·obb   implemented  서버 운영 중 (tasks.detect)
    classify     reference    라이브러리 참조 구현, 서버 작업 연동 전 (tasks.classify)
    segment      planned      라벨 형식만 정함 (labels): 마스크 → 다각형, 픽셀·영역 신뢰도, TTA 마스크 IoU
    anomaly      planned      라벨 형식만 정함 (labels): 이상 점수 임계값 보정, OK/NG 검수
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from . import classify, detect

PLANNED = {"segment": "분할 라벨링 모듈은 아직 없습니다(마스크 → 다각형 라벨, 픽셀·영역 신뢰도, TTA 마스크 IoU 예정).",
           "anomaly": "이상탐지 라벨링 모듈은 아직 없습니다(이상 점수 임계값 보정, OK/NG 검수 예정)."}


class UnsupportedTask(ValueError):
    """라벨링 모듈이 없는 작업 종류"""


@dataclass(frozen=True)
class TaskModule:
    name: str                          # 모듈 이름
    tasks: tuple[str, ...]             # 처리하는 작업 종류(클래스 구성 task)
    status: Literal["implemented", "reference"]
    level: Literal["box", "image"]     # 라벨 단위: 박스(검출 계열) · 이미지(분류·이상탐지)
    signals: tuple[str, ...]           # 계산하는 공통 신호
    signals_version: int
    rules_version: int
    setup: type                        # 신호 조건 객체
    analyze: Callable[..., list[dict[str, Any]]]
    judge: Callable[..., dict[str, Any]]
    plugins: bool                      # 신호 플러그인(plugins)을 받는지


DETECT = TaskModule(name="detect", tasks=("detect", "obb"), status="implemented", level="box",
                    signals=("confidence", "tta", "image_distance", "box_distance", "model_agreement"),
                    signals_version=detect.SIGNALS_VERSION, rules_version=detect.RULES_VERSION,
                    setup=detect.SignalSetup, analyze=detect.analyze, judge=detect.judge, plugins=False)
CLASSIFY = TaskModule(name="classify", tasks=("classify",), status="reference", level="image",
                      signals=("confidence", "tta", "image_distance", "model_agreement"),
                      signals_version=classify.SIGNALS_VERSION, rules_version=classify.RULES_VERSION,
                      setup=classify.ClassifySetup, analyze=classify.analyze, judge=classify.judge, plugins=True)

_modules: dict[str, TaskModule] = {}


def register_task(module: TaskModule, replace: bool = False) -> None:
    for task in module.tasks:
        if task in _modules and not replace:
            raise ValueError(f"작업 종류 {task} 에 이미 라벨링 모듈({_modules[task].name})이 있습니다")
    for task in module.tasks:
        _modules[task] = module


def task_module(task: str) -> TaskModule:
    """작업 종류의 라벨링 모듈. 없으면 UnsupportedTask(이유 포함)."""
    if task in _modules:
        return _modules[task]
    raise UnsupportedTask(PLANNED.get(task, f"알 수 없는 작업 종류입니다: {task}"))


def available_tasks() -> dict[str, str]:
    """작업 종류 → 상태(implemented · reference · planned)"""
    return {**{t: m.status for t, m in _modules.items()}, **{t: "planned" for t in PLANNED if t not in _modules}}


register_task(DETECT)
register_task(CLASSIFY)
