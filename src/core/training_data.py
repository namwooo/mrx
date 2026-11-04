"""Which images a training learns from. The API checks with it, the worker builds the dataset with it."""
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

from core.orm import LabelingImageRow, ReviewResult, Suggestion
from sqlalchemy import select
from sqlalchemy.orm import Session


@dataclass(frozen=True)
class TrainingImage:
    image_id: int
    image_path: str
    labels: List[Dict[str, Any]]
    split: str


def pick_split(image_path: str, split: Dict[str, float]) -> str:
    """The hash decides, so an image stays in its split across trainings. Test comes first: changing the
    train and val ratios never moves a test image."""
    position = int(hashlib.sha256(image_path.encode()).hexdigest(), 16) / 2 ** 256
    if position < split["test"]:
        return "test"
    return "val" if position < split["test"] + split["val"] else "train"


def _usable_labels(image: LabelingImageRow) -> List[Dict[str, Any]] | None:
    if image.result == ReviewResult.MODIFY.value:
        return image.reviewed_labels
    if image.result == ReviewResult.OK.value:
        return image.labels
    if image.result is None and image.suggested == Suggestion.APPROVED.value:
        return image.labels
    return None


def select_images(session: Session, labeling_ids: Sequence[int], split: Dict[str, float]) -> List[TrainingImage]:
    rows = session.scalars(
        select(LabelingImageRow)
        .where(LabelingImageRow.labeling_id.in_(labeling_ids))
        .order_by(LabelingImageRow.labeling_id.desc(), LabelingImageRow.id)
    )

    images, seen = [], set()
    for row in rows:
        if row.image_path in seen:  # the newest labeling comes first and decides, even when it discards the image
            continue
        seen.add(row.image_path)
        labels = _usable_labels(row)
        if labels is not None:
            images.append(TrainingImage(row.id, row.image_path, labels, pick_split(row.image_path, split)))
    return images


def write_dataset(images: Sequence[TrainingImage], names: Dict[int, str], root: Path) -> Path:
    """Builds the folder layout YOLO expects: links to the images and label files from the stored labels."""
    import yaml

    for image in images:
        image_dir, label_dir = root / "images" / image.split, root / "labels" / image.split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        # Named by id: two source folders can hold files with the same name.
        os.symlink(Path(image.image_path).resolve(), image_dir / f"{image.image_id}{Path(image.image_path).suffix}")
        lines = [
            " ".join(str(value) for value in [label["class_id"], *label["bbox_xywhn"]])
            for label in image.labels
        ]
        (label_dir / f"{image.image_id}.txt").write_text("\n".join(lines), encoding="utf-8")

    config = {"path": str(root.resolve()), "train": "images/train", "val": "images/val", "names": names}
    if any(image.split == "test" for image in images):
        config["test"] = "images/test"

    data_yaml = root / "dataset.yaml"
    data_yaml.write_text(yaml.safe_dump(config), encoding="utf-8")
    return data_yaml
