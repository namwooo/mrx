from __future__ import annotations

import json
import logging
import os
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping

from core.jobs import Job
from workers.handlers.images import chunks, list_images, read_image

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(os.environ.get("LABELING_OUTPUT_DIR", "./data/labeling_outputs"))

CHUNK_SIZE = int(os.environ.get("LABELING_CHUNK_SIZE", "16"))

HISTOGRAM_BINS = 10


def handle_labeling_job(job: Job) -> Dict[str, Any]:
    payload = job.payload
    params = payload.get("params") or {}

    images = list_images(payload["input_path"])
    if not images:
        raise ValueError(f"no images found in {payload['input_path']}")

    logger.info("labeling job %s: model=%s input=%s images=%d params=%s",
                job.id, payload["model"], payload["input_path"], len(images), params)

    model = _load_model(payload["model"])
    names = model.model.names

    out_dir = OUTPUT_DIR / str(job.id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "labels.jsonl"
    tmp_path = out_path.with_suffix(".jsonl.part")

    summary = _Summary()
    with tmp_path.open("w", encoding="utf-8") as f:
        for chunk in chunks(images, CHUNK_SIZE):
            paths, arrays = [], []
            for path in chunk:
                array = read_image(path)
                if array is None:
                    logger.warning("labeling job %s: cannot read %s", job.id, path)
                    summary.unreadable += 1
                    continue
                paths.append(path)
                arrays.append(array)

            if not arrays:
                continue

            try:
                detections_per_image = model.predict(
                    arrays,
                    conf=params.get("conf", 0.25),
                    iou=params.get("iou", 0.7),
                    max_det=params.get("max_det", 300),
                )
            finally:
                model.release()  # free feature maps (GPU memory) after every chunk

            for path, array, detections in zip(paths, arrays, detections_per_image):
                height, width = array.shape[:2]
                record = _to_record(path, width, height, detections, names)
                summary.add(record)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if summary.images == 0:
        tmp_path.unlink(missing_ok=True)
        raise ValueError(f"none of the {len(images)} images in {payload['input_path']} could be read")

    tmp_path.replace(out_path)

    return {"output_path": str(out_path), **summary.to_dict()}


@lru_cache(maxsize=2)
def _load_model(model: str):
    from inference.yolo import YOLO

    logger.info("model loading: %s", model)
    return YOLO(model_path=model)


def _to_record(
        image_path: Path,
        width: int,
        height: int,
        detections: List[Dict[str, Any]],
        names: Mapping[int, str],
) -> Dict[str, Any]:
    return {
        "image": str(image_path),
        "width": width,
        "height": height,
        "labels": [
            {
                "class_id": d["class_id"],
                "class_name": names[d["class_id"]],
                "confidence": round(d["confidence"], 4),
                "bbox_xyxy": [round(v, 2) for v in d["xyxy"]],
                "bbox_xywhn": _xywhn(d["xyxy"], width, height),
            }
            for d in detections
        ],
    }


def _xywhn(xyxy: List[float], width: int, height: int) -> List[float]:
    """Box as normalized center x, center y, width, height (YOLO label format)."""
    x1, y1, x2, y2 = xyxy
    return [
        round((x1 + x2) / 2 / width, 6),
        round((y1 + y2) / 2 / height, 6),
        round((x2 - x1) / width, 6),
        round((y2 - y1) / height, 6),
    ]


class _Summary:
    def __init__(self) -> None:
        self.images = 0
        self.images_without_labels = 0
        self.unreadable = 0
        self.labels = 0
        self.class_counts: Counter = Counter()
        self.histogram = [0] * HISTOGRAM_BINS

    def add(self, record: Dict[str, Any]) -> None:
        self.images += 1
        labels = record["labels"]
        if not labels:
            self.images_without_labels += 1
        for label in labels:
            self.labels += 1
            self.class_counts[label["class_name"]] += 1
            index = min(int(label["confidence"] * HISTOGRAM_BINS), HISTOGRAM_BINS - 1)
            self.histogram[index] += 1

    def to_dict(self) -> Dict[str, Any]:
        step = 1 / HISTOGRAM_BINS
        return {
            "num_images": self.images,
            "num_images_without_labels": self.images_without_labels,
            "num_unreadable_images": self.unreadable,
            "num_labels": self.labels,
            "class_counts": dict(self.class_counts.most_common()),
            "confidence_histogram": {
                f"{i * step:.1f}-{(i + 1) * step:.1f}": count for i, count in enumerate(self.histogram)
            },
        }
