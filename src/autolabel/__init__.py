"""autolabel: 검사 모델 자동라벨링의 판정 신호·판정 규칙·거리 통계·평가·보정 라이브러리.

DB·HTTP·작업 큐·환경 변수를 다루지 않는다(그것은 서버 app.labeling 의 몫). 입력은 이미지 배열·모델·설정 객체, 출력은
서버 DB 에 그대로 저장하는 dict(신호·판정 형식)다. 사용법·구조는 같은 폴더의 README.md.
"""
from .config import (CalibrationConfig, ClassifyConfig, RuleConfig, StageBConfig, StageConfig, TtaConfig)
from .constants import (AUTO_APPROVED, IMAGE_LEVEL, LABELING_SPEC_VERSION, LIBRARY_VERSION, REVIEW, DistanceMethod,
                        SignalName)
from .distance import BoxDistance, ImageDistance
from .images import read_image
from .labels import LABEL_KEY, label_error
from .stages import class_stages
from .tasks import TaskModule, UnsupportedTask, available_tasks, register_task, task_module

__version__ = LIBRARY_VERSION

__all__ = ["AUTO_APPROVED", "BoxDistance", "CalibrationConfig", "ClassifyConfig", "DistanceMethod", "IMAGE_LEVEL",
           "ImageDistance", "LABELING_SPEC_VERSION", "LABEL_KEY", "REVIEW", "RuleConfig", "SignalName", "StageBConfig",
           "StageConfig", "TaskModule", "TtaConfig", "UnsupportedTask", "available_tasks", "class_stages", "label_error",
           "read_image", "register_task", "task_module", "__version__"]
