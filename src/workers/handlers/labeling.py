from __future__ import annotations

import logging
import os
from collections import Counter
from functools import lru_cache
from typing import Any, Dict, List, Mapping

from core.boxes import xywhn
from core.db import SessionLocal
from core.jobs import Job
from core.orm import LabelingImageRow, LabelingRow, ModelRow, Suggestion
from sqlalchemy import update
from workers.handlers.images import chunks, list_images, read_image

logger = logging.getLogger(__name__)

CHUNK_SIZE = int(os.environ.get("LABELING_CHUNK_SIZE", "16"))

APPROVE_CONF = float(os.environ.get("LABELING_APPROVE_CONF", "0.9"))

HISTOGRAM_BINS = 10


def handle_labeling_job(job: Job) -> Dict[str, Any]:
    labeling_id = job.payload["labeling_id"]

    with SessionLocal() as session:  # closed before the labeling: it can take hours
        record = session.get(LabelingRow, labeling_id)
        if record is None:
            raise ValueError(f"labeling '{labeling_id}' not found")
        model_row = session.get(ModelRow, record.model_id)
        if model_row is None or not model_row.weights_path:
            raise ValueError(f"model '{record.model_id}' has no weights")

    params = record.params
    images = list_images(record.source_path)
    if not images:
        raise ValueError(f"no images found in {record.source_path}")

    logger.info("labeling %s: model=%s source=%s images=%d params=%s",
                labeling_id, model_row.weights_path, record.source_path, len(images), params)

    model = _load_model(model_row.weights_path)
    names = model.model.names
    _save_start(labeling_id, len(images), names)

    summary = _Summary()
    done = 0
    for chunk in chunks(images, CHUNK_SIZE):
        paths, arrays = [], []
        for path in chunk:
            array = read_image(path)
            if array is None:
                logger.warning("labeling %s: cannot read %s", labeling_id, path)
                summary.unreadable += 1
                continue
            paths.append(path)
            arrays.append(array)

        rows = []
        if arrays:
            detections_per_image = model.predict(
                arrays,
                conf=params["conf"],
                iou=params["iou"],
                max_det=params["max_det"],
            )
            for path, array, detections in zip(paths, arrays, detections_per_image):
                height, width = array.shape[:2]
                labels = _to_labels(width, height, detections, names)
                summary.add(labels)
                rows.append(LabelingImageRow(
                    labeling_id=labeling_id,
                    image_path=str(path),
                    width=width,
                    height=height,
                    labels=labels,
                    num_labels=len(labels),
                    suggested=suggest(labels).value,
                ))

        done += len(chunk)
        _save_chunk(labeling_id, rows, done)

    if summary.images == 0:
        raise ValueError(f"none of the {len(images)} images in {record.source_path} could be read")

    return summary.to_dict()


def suggest(labels: List[Dict[str, Any]]) -> Suggestion:
    """The system's verdict. An image without labels may be a missed defect, so a person looks at it."""
    if labels and all(label["confidence"] >= APPROVE_CONF for label in labels):
        return Suggestion.APPROVED
    return Suggestion.REVIEW


@lru_cache(maxsize=2)
def _load_model(model: str):
    from inference.yolo import YoloPredictor

    logger.info("model loading: %s", model)
    return YoloPredictor(model)


def _save_start(labeling_id: int, total: int, names: Mapping[int, str]) -> None:
    with SessionLocal() as session:
        session.execute(
            update(LabelingRow)
            .where(LabelingRow.id == labeling_id)
            .values(images_total=total, class_names={str(i): name for i, name in names.items()})
        )
        session.commit()


def _save_chunk(labeling_id: int, rows: List[LabelingImageRow], done: int) -> None:
    with SessionLocal() as session:
        session.add_all(rows)
        session.execute(update(LabelingRow).where(LabelingRow.id == labeling_id).values(images_done=done))
        session.commit()


def _to_labels(
        width: int,
        height: int,
        detections: List[Dict[str, Any]],
        names: Mapping[int, str],
) -> List[Dict[str, Any]]:
    return [
        {
            "class_id": d["class_id"],
            "class_name": names[d["class_id"]],
            "confidence": round(d["confidence"], 4),
            "bbox_xyxy": [round(v, 2) for v in d["xyxy"]],
            "bbox_xywhn": xywhn(d["xyxy"], width, height),
        }
        for d in detections
    ]


class _Summary:
    def __init__(self) -> None:
        self.images = 0
        self.images_without_labels = 0
        self.unreadable = 0
        self.labels = 0
        self.class_counts: Counter = Counter()
        self.histogram = [0] * HISTOGRAM_BINS

    def add(self, labels: List[Dict[str, Any]]) -> None:
        self.images += 1
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
