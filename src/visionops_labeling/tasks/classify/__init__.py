"""분류(classify) 라벨링 참조 구현: 분류 어댑터(규격 1.0 LoadedModel)의 점수·임베딩·TTA 선언과 신호 플러그인으로
판정 신호를 계산하고 판정·평가·보정한다. 서버 자동라벨링 작업(AUTO_LABEL·LABEL_EVAL) 연동은 다음 단계다."""
from .analysis import ClassifySetup, analyze
from .model import BINARY_NAMES, ClassifierModel, ModelInfo, class_map
from .rules import RULES_VERSION, ClassifyRuleParams, judge
from .signals import SIGNALS_VERSION, decision_rule
from .tta import TRANSFORMS, expand

__all__ = ["BINARY_NAMES", "RULES_VERSION", "SIGNALS_VERSION", "TRANSFORMS", "ClassifierModel", "ClassifyRuleParams",
           "ClassifySetup", "ModelInfo", "analyze", "class_map", "decision_rule", "expand", "judge"]
