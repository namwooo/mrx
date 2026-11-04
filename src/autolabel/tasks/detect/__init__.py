"""검출(detect·obb) 라벨링: 예측기(ultralytics YOLO), 신호 ①~⑤, 판정 규칙, 평가·보정. 서버에서 운영 중인 구현이다."""
from .analysis import SignalSetup, analyze, obb_box, to_boxes, yolo_lines
from .rules import RULES_VERSION, RuleParams, judge, rule_params
from .signals import SIGNALS_VERSION, TtaVariant, compute, tta_variants

__all__ = ["RULES_VERSION", "SIGNALS_VERSION", "RuleParams", "SignalSetup", "TtaVariant", "analyze", "compute", "judge",
           "obb_box", "rule_params", "to_boxes", "tta_variants", "yolo_lines"]
