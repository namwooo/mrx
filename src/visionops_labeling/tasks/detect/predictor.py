"""검출 라벨링 모델 로드·일괄 추론 (사각형 박스 detect · 회전 박스 obb).

가중치 로드·ultralytics 호출은 추론 서버와 같은 엔진(visionops_models.ultralytics.engine.YoloEngine, 묶음 predict_batch)을
쓴다: 640×640 letterbox(rect=False), 같은 NMS. 여러 이미지를 한 번에 넣어도 이미지마다 같은 640×640 입력이 되므로 한 장씩
추론한 결과와 같다. 자동라벨링은 판정 신호를 위해 낮은 conf 로 추론한다. 박스 형식(아래 detections·obb_detections)은 라벨링용이다.

TTA(판정 신호 ②)는 같은 predict 를 변형한 이미지·입력 크기(imgsz)로 다시 부른다. 이미지 임베딩(③)과 박스 특징(④)은
neck 출력(P3·P4·P5)에 건 forward hook 으로 얻는다(features). 원본 추론(640) 때 잡은 특징맵을 같은 묶음의
임베딩·박스 특징에 다시 쓰므로 추가 추론이 없다. 추론 없이 부르면(거리 통계) 특징맵을 얻는 추론을 한 번 한다.

obb 결과는 xywhr(원본 픽셀, geometry 각도 규칙)·polygon(네 꼭짓점)·xyxy(감싸는 사각형)를 함께 준다.
torch·ultralytics·visionops_models 는 YoloPredictor·read_class_names 를 부를 때만 import 한다. 테스트는 같은 인터페이스의
가짜 예측기를 쓴다.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ...constants import INPUT_SIZE
from ...features import CapturedBatch, NeckCapture
from ...geometry import TASKS, obb_detection


# [예측기 인터페이스] 자동라벨링·통계·평가는 이 인터페이스에만 의존한다. 실제 구현 YoloPredictor, 테스트는 가짜 예측기.
class Predictor(Protocol):
    names: list[str]  # 모델 클래스 ID 순서의 이름
    task: str         # detect 또는 obb (가짜 예측기는 없으면 detect 로 본다)

    def predict(self, images: list[np.ndarray], *, conf: float, iou: float, max_det: int,
                classes: list[int] | None, imgsz: int = INPUT_SIZE) -> list[list[dict[str, Any]]]:
        """이미지(BGR)마다 [{"class_id": 모델 클래스 ID, "confidence", "xyxy": [x1, y1, x2, y2] 원본 픽셀}].
        obb 는 "xywhr": [cx, cy, w, h, θ] 원본 픽셀·라디안(각도 규칙), "polygon": [[x, y] × 4] 를 더 준다(xyxy 는 감싸는 사각형)."""
        ...

    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        """이미지(BGR)마다 임베딩 한 줄. (이미지 수, 차원) float32"""
        ...

    def box_features(self, images: list[np.ndarray], boxes: list[np.ndarray], levels: str = "all",
                     **kwargs: Any) -> list[np.ndarray]:
        """이미지(BGR)마다 원본 픽셀 박스 (n_i, 4) xyxy 의 특징 (n_i, 차원) float32. levels: 레벨 사용 방식(features 참고).
        회전 박스는 (n_i, 5) xywhr 와 pooling=rotated·enclosing 을 받는다."""
        ...


# [ultralytics 예측기] 가중치 로드·ultralytics 호출은 추론 서버와 같은 엔진(visionops_models YoloEngine)을 쓴다.
# 엔진을 워밍업 없이 올린 뒤(첫 predict 전에 hook 을 걸어야 한다) neck 레이어에 hook 을 걸고,
# predict 결과(탐지)와 같은 추론에서 잡은 특징맵으로 embed(③)·box_features(④)를 계산한다.
class YoloPredictor:
    def __init__(self, weights: Path, device: str, offline: bool = True, embed_layers: list[int] | None = None,
                 feature_layers: list[int] | None = None):
        # 학습·추론 라이브러리가 있어야 실제 예측기를 쓸 수 있다(라벨링 계산만 쓰는 쪽은 필요 없다)
        from visionops_models.ultralytics.engine import EngineConfig, YoloEngine

        self.engine = YoloEngine(None, "labeling")  # task 는 가중치를 따른다(detect·obb)
        try:
            self.engine.load(EngineConfig(device=device, offline=offline, warmup=False), Path(weights))
        except RuntimeError as error:
            raise RuntimeError(f"자동라벨링 모델을 올리지 못했습니다: {error}") from error
        self.model = self.engine.yolo
        self.task = self.engine.task
        self.device = device
        self.embed_layers = list(embed_layers or [16, 19, 22])
        self.feature_layers = list(feature_layers or [16, 19, 22])
        net = self.model.model.model
        for kind, layers in (("임베딩", self.embed_layers), ("박스 특징", self.feature_layers)):
            if max(layers) >= len(net):
                raise RuntimeError(f"{kind} 레이어 {layers} 가 모델 레이어 수({len(net)})를 넘습니다: {weights}")
        # 임베딩용·박스 특징용 레이어를 합쳐 한 번만 hook 을 건다.
        hooked = sorted(set(self.embed_layers) | set(self.feature_layers))
        # 첫 predict 전에 건다. ultralytics 는 predictor 를 만들 때 모델을 복사해 쓰는데(AutoBackend), 복사본에도 hook 이 따라간다
        self.capture = NeckCapture([net[i] for i in hooked], hooked)
        self._batch: CapturedBatch | None = None  # 마지막 원본 추론 묶음의 특징맵
        self.names = [self.engine.names[i] for i in range(len(self.engine.names))]

    # 묶음 추론. 엔진 predict_batch 가 묶음 전체를 한 번에 GPU 에 넣는다(batch=이미지 수, 640×640 letterbox).
    def predict(self, images, *, conf, iou, max_det, classes, imgsz=INPUT_SIZE):
        if not images:
            return []
        keep = imgsz == INPUT_SIZE  # 원본 입력 크기 추론만 특징맵을 남긴다(TTA 크기 변경은 버린다)
        self._batch = None
        if keep:
            self.capture.start()
        try:
            results = self.engine.predict_batch(images, conf=conf, iou=iou, max_det=max_det, classes=classes,
                                                imgsz=imgsz)
        finally:
            maps = self.capture.stop()
        if keep:
            if set(maps) != set(self.capture.indices):
                raise RuntimeError(f"neck hook 이 특징맵을 잡지 못했습니다 (잡은 레이어 {sorted(maps)}, 필요 {self.capture.indices}). "
                                   "ultralytics 버전을 확인하세요.")
            self._batch = CapturedBatch(images, maps)
        return [obb_detections(r.obb) if self.task == "obb" else detections(r.boxes) for r in results]

    def _captured(self, images: list[np.ndarray]) -> CapturedBatch:
        """이 묶음의 특징맵. 직전 원본 추론과 같은 배열 목록이면 그대로 쓰고, 아니면 특징맵을 얻는 추론을 한 번 한다."""
        if self._batch is None or not self._batch.matches(images):
            self.predict(images, conf=1.0, iou=0.7, max_det=1, classes=None)  # 탐지 결과는 쓰지 않는다
        return self._batch

    def embed(self, images):
        if not images:
            return np.zeros((0, 0), np.float32)
        return self._captured(images).embed(self.embed_layers)

    def box_features(self, images, boxes, levels="all", pooling="rotated"):
        if not images:
            return []
        return self._captured(images).box_features(self.feature_layers, boxes, levels, pooling=pooling)

    def release(self) -> None:
        """묶음 처리가 끝나면 특징맵(GPU 메모리)을 놓는다."""
        self._batch = None


# ultralytics 결과 객체 → 파이썬 dict 목록(모델 클래스 번호 기준, 클래스 구성 번호로는 analysis.to_boxes 가 바꾼다).
def detections(boxes) -> list[dict[str, Any]]:
    """ultralytics Boxes → [{"class_id", "confidence", "xyxy"}]"""
    if boxes is None or len(boxes) == 0:
        return []
    return [{"class_id": int(c), "confidence": float(s), "xyxy": [float(v) for v in b]}
            for b, s, c in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist())]


def obb_detections(obb) -> list[dict[str, Any]]:
    """ultralytics OBB → [{"class_id", "confidence", "xywhr", "polygon", "xyxy"(감싸는 사각형)}]. 각도는 geometry 규칙으로 맞춘다."""
    if obb is None or len(obb) == 0:
        return []
    out = []
    for r, s, c in zip(obb.xywhr.cpu().tolist(), obb.conf.cpu().tolist(), obb.cls.cpu().tolist()):
        out.append(obb_detection(int(c), float(s), *r))
    return out


# 가중치 (경로, 크기, 수정 시각) → (클래스 이름, task). 요청 접수 때마다 가중치를 다시 열지 않도록 캐시한다.
_names_cache: dict[tuple[str, int, int], tuple[list[str], str]] = {}
_names_lock = threading.Lock()


def _read_weights_info(weights: Path) -> tuple[list[str], str]:
    stat = weights.stat()
    key = (str(weights.resolve()), stat.st_size, stat.st_mtime_ns)
    with _names_lock:
        if key not in _names_cache:
            os.environ.setdefault("YOLO_OFFLINE", "1")
            from ultralytics import YOLO

            model = YOLO(str(weights))
            if model.task not in TASKS:
                raise RuntimeError(f"자동라벨링은 사각형 박스(detect)·회전 박스(obb) 모델만 지원합니다 (가중치 task={model.task}): {weights}")
            _names_cache[key] = ([str(model.names[i]) for i in range(len(model.names))], model.task)
        names, task = _names_cache[key]
        return list(names), task


def read_class_names(weights: Path) -> list[str]:
    """가중치의 클래스 이름(CPU 로 읽는다). 서버가 요청 접수 때 클래스 구성과 맞는지 확인하는 데 쓴다."""
    return _read_weights_info(weights)[0]


def read_model_task(weights: Path) -> str:
    """가중치의 모델 종류(detect·obb). read_class_names 와 같은 캐시를 쓴다."""
    return _read_weights_info(weights)[1]
