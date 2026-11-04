"""라이브러리 공통 상수: 모델 입력 크기, 작업 종류, 판정 신호 이름, 거리 방식.

서버의 라벨링 워커도 이 값을 그대로 쓴다.
"""
from __future__ import annotations

import enum

# 라이브러리 버전. 신호·판정 형식 버전(SIGNALS_VERSION·RULES_VERSION)과 별개로, 공개 API 가 바뀌면 올린다.
LIBRARY_VERSION = "0.1.0"
# 외부 개발 라벨링 확장(신호 플러그인) 규격 버전. 주 버전이 다르면 플랫폼이 플러그인 등록을 거부한다.
LABELING_SPEC_VERSION = "1.0"

INPUT_SIZE = 640  # 검출 모델 입력: 긴 변을 640 에 맞추고 나머지를 채운 640×640 letterbox (학습·추론 공통)

# 작업 종류. 라벨링 모듈이 있는 것은 tasks.available_tasks() 로 확인한다(검출·회전 박스·분류).
TASK_TYPES = ("detect", "obb", "segment", "classify", "anomaly")

IMAGE_LEVEL = -1  # 신호 상태의 class_id: 클래스와 무관한 이미지 단위 신호


# 판정 신호 5종. 자동 승인은 켜진 신호를 모두 통과한 박스·이미지에만 준다.
class SignalName(str, enum.Enum):
    CONFIDENCE = "confidence"          # ① 클래스별 신뢰도 임계값
    TTA = "tta"                        # ② TTA 일관성
    IMAGE_DISTANCE = "image_distance"  # ③ 이미지 임베딩 거리 (이미지 단위)
    BOX_DISTANCE = "box_distance"      # ④ 박스 특징 거리 kNN·MDS·RMDS (검출, 클래스별 B단계)
    MODEL_AGREEMENT = "model_agreement"  # ⑤ 운영 모델과의 일치 (운영 모델이 있을 때)


STAGE_B_SIGNALS = (SignalName.BOX_DISTANCE.value,)  # 조건을 채운 클래스만 켜는 신호. 감사 오류율이 목표를 넘으면 끈다(롤백)


class DistanceMethod(str, enum.Enum):
    """③ 이미지 거리 계산 방식 (distance.image)."""
    PCA = "pca_mahalanobis"                    # 계획서 방식: PCA 투영 마할라노비스, 통계에 쓴 이미지 자신의 거리로 임계값
    PCA_CV = "pca_mahalanobis_cv"              # 위 거리, 임계값만 교차 검증(보지 않은 이미지 거리)
    PCA_RESIDUAL = "pca_mahalanobis_residual"  # 기본: + PCA 잔차 한 축, 잔차 척도·임계값 교차 검증


# 판정 결과(이미지 단위)
AUTO_APPROVED, REVIEW = "AUTO_APPROVED", "REVIEW"
