from __future__ import annotations

import logging
import os
from collections import Counter
from functools import lru_cache
from typing import Any, Dict, List, Mapping

from autolabel.config import RuleConfig, TtaConfig
from autolabel.constants import AUTO_APPROVED
from autolabel.tasks import detect
from core.db import SessionLocal
from core.jobs import Job
from core.orm import LabelingImageRow, LabelingRow, ModelRow, Suggestion
from sqlalchemy import update
from workers.handlers.images import chunks, list_images, read_image

logger = logging.getLogger(__name__)

CHUNK_SIZE = int(os.environ.get("LABELING_CHUNK_SIZE", "16"))

APPROVE_CONF = float(os.environ.get("LABELING_APPROVE_CONF", "0.9"))

TTA_IOU = float(os.environ.get("LABELING_TTA_IOU", "0.5"))

AUDIT_RATE = float(os.environ.get("LABELING_AUDIT_RATE", "0.05"))

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
    class_names = model.model.names
    _save_start(labeling_id, len(images), class_names)

    names = [class_names[i] for i in range(len(class_names))]
    class_map = {i: i for i in range(len(names))}
    setup = _signal_setup(class_map)

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
            analyzed = detect.analyze(model, arrays, params, class_map, names, setup)
            for path, array, item in zip(paths, arrays, analyzed):
                height, width = array.shape[:2]
                judgment = detect.judge(item["boxes"], item["signals"], setup.rules, f"{labeling_id}:{path}")
                labels = _to_labels(item["boxes"], judgment)
                summary.add(labels)
                rows.append(LabelingImageRow(
                    labeling_id=labeling_id,
                    image_path=str(path),
                    width=width,
                    height=height,
                    labels=labels,
                    num_labels=len(labels),
                    suggested=_suggestion(judgment).value,
                    predictions={"boxes": item["boxes"], "signals": item["signals"], "judgment": judgment},
                ))

        done += len(chunk)
        _save_chunk(labeling_id, rows, done)

    if summary.images == 0:
        raise ValueError(f"none of the {len(images)} images in {record.source_path} could be read")

    return summary.to_dict()


def _signal_setup(class_map: Mapping[int, int]) -> detect.SignalSetup:
    """Confidence and TTA only, with one threshold for every class.
    The other signals and per-class calibration need statistics this server does not keep yet."""
    tta = TtaConfig()
    conf = {class_id: APPROVE_CONF for class_id in class_map.values()}
    tta_iou = {class_id: TTA_IOU for class_id in class_map.values()}
    return detect.SignalSetup(
        task="detect",
        enabled=True,
        variants=detect.tta_variants(tta),
        match_iou=tta.match_iou,
        conf_thresholds=conf,
        rules=detect.rule_params(conf, tta_iou, {}, {}, RuleConfig(audit_rate=AUDIT_RATE)),
    )


def _suggestion(judgment: Dict[str, Any]) -> Suggestion:
    """Images picked for the audit sample go to a person even though the rules approved them."""
    if judgment["decision"] == AUTO_APPROVED and not judgment["audit"]:
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


def _to_labels(boxes: List[Dict[str, Any]], judgment: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Held boxes stay in predictions but never become labels."""
    return [
        {
            "class_id": box["class_id"],
            "class_name": box["class_name"],
            "confidence": round(box["confidence"], 4),
            "bbox_xyxy": [box["bbox"][key] for key in ("x1", "y1", "x2", "y2")],
            "bbox_xywhn": box["bbox_yolo"],
        }
        for box, verdict in zip(boxes, judgment["boxes"])
        if verdict["verdict"] != "hold"
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
