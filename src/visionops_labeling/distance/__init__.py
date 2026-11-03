"""거리 통계: ③ 이미지 거리(모든 작업 종류), ④ 박스 특징 거리(검출)."""
from .box import BOX_STATS_FILE, METHODS, BoxDistance, knn_key
from .image import ImageDistance, distance_record

__all__ = ["BOX_STATS_FILE", "METHODS", "BoxDistance", "ImageDistance", "distance_record", "knn_key"]
