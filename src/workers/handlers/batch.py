from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Iterator

from core.jobs import Job
from inference.predictor import RawDetection

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

OUTPUT_DIR = Path(os.environ.get("BATCH_OUTPUT_DIR", "./data/batch_outputs"))

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "./models/yolo11"))

CHUNK_SIZE = int(os.environ.get("BATCH_CHUNK_SIZE", "16"))

def handle_batch_job(job: Job) -> Dict[str, Any]:
    job_id = job.id
    payload = job.payload
    model = payload["model"]
    input_path = payload["input_path"]
    params = payload.get("params", {})

    images = _list_images(payload["input_path"])
    if not images:
        raise ValueError(f"no images found in {payload['input_path']}")

    logger.info("batch job %s: model=%s input=%s params=%s", job.id, model, input_path, params)

    model = _load_model(payload["model"])

    out_dir = OUTPUT_DIR / str(job_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "predictions.jsonl"
    tmp_path = out_path.with_suffix(".jsonl.part")

    with tmp_path.open("w", encoding="utf-8") as f:
        for chunk in _chunks(images, CHUNK_SIZE):
            paths, arrays = [], []
            for path in chunk:
                array = _read_image(path)
                if array is None:
                    continue
                paths.append(path)
                arrays.append(array)

            if not arrays:
                continue

            detections_per_image = model.predict(arrays)

            for path, detections in zip(paths, detections_per_image):
                record = _to_record(path, detections, model.model.names)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    tmp_path.replace(out_path)

    return {
        "output_path": str(out_path),
    }


@lru_cache(maxsize=2)
def _load_model(model: str):
    from inference.yolo import YoloPredictor

    logger.info("model loading: %s", model)
    return YoloPredictor(model)


def _list_images(input_path: str) -> List[Path]:
    path = Path(input_path)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    raise FileNotFoundError(f"input_path not found: {input_path}")


def _chunks(items: List[Path], size: int) -> Iterator[List[Path]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _read_image(path: Path):
    import cv2

    return cv2.imread(str(path))


def _to_record(image_path: Path, detections: List[RawDetection], names: List[str]) -> Dict[str, Any]:
    return {
        "image": str(image_path),
        "detections": [
            {
                "class_id": d["class_id"],
                "class_name": names[d["class_id"]],
                "confidence": round(d["confidence"], 4),
                "bbox_xyxy": [round(v, 2) for v in d["xyxy"]],
            }
            for d in detections
        ],
    }