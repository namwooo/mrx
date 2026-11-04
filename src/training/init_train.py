"""Runs one init training. The worker starts this module with the training Python.

It runs in the training environment, so it knows nothing of the database or core. It reads a request file and reports
to the worker through lines on stdout that start with MRX_.
"""
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from training.validation import validate_split

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}  # the formats training.validation checks
CLASSES_FILE = "classes.txt"
VAL_RATIO = 0.2
SPLIT_SEED = 0
METRIC_NAMES = {
    "metrics/mAP50(B)": "mAP50",
    "metrics/mAP50-95(B)": "mAP50-95",
    "metrics/precision(B)": "precision",
    "metrics/recall(B)": "recall",
}


def report(kind: str, payload: Dict[str, Any]) -> None:
    print(f"MRX_{kind} {json.dumps(payload)}", flush=True)


def resolve_classes(classes: Optional[List[Dict[str, Any]]], label_dir: Path) -> Dict[int, str]:
    if classes:
        known = {c["class_id"]: c["class_name"] for c in classes}
    elif (label_dir / CLASSES_FILE).is_file():
        lines = (label_dir / CLASSES_FILE).read_text(encoding="utf-8").splitlines()
        known = {i: line.strip() for i, line in enumerate(lines) if line.strip()}
    else:
        known = {}
        for label in label_dir.glob("*.txt"):
            for line in label.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    known[int(line.split()[0])] = f"class_{int(line.split()[0])}"

    if not known:
        raise ValueError(f"no classes found in {label_dir}")
    # YOLO needs ids without gaps: an id nobody named still gets a name.
    return {i: known.get(i, f"class_{i}") for i in range(max(known) + 1)}


def prepare_dataset(image_dir: Path, label_dir: Path, names: Dict[int, str], root: Path) -> Path:
    """Builds the folder layout YOLO expects from links to the originals, with a fixed train and val split."""
    images = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if len(images) < 2:
        raise ValueError(f"need at least 2 images in {image_dir}, found {len(images)}")

    random.Random(SPLIT_SEED).shuffle(images)
    val_count = max(1, int(len(images) * VAL_RATIO))

    for split, files in (("val", images[:val_count]), ("train", images[val_count:])):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for image in files:
            (root / "images" / split / image.name).symlink_to(image.resolve())
            label = label_dir / f"{image.stem}.txt"
            if label.is_file():
                (root / "labels" / split / label.name).symlink_to(label.resolve())

    data_yaml = root / "dataset.yaml"
    data_yaml.write_text(
        yaml.safe_dump({"path": str(root.resolve()), "train": "images/train", "val": "images/val", "names": names}),
        encoding="utf-8",
    )
    return data_yaml


def report_epoch(trainer: Any) -> None:
    # Fires once more for the final evaluation, with the epoch already past the last one.
    if trainer.epoch >= trainer.epochs:
        report("PROGRESS", {"epochs_done": trainer.epochs})
        return

    values = {**trainer.metrics, **trainer.label_loss_items(trainer.tloss, prefix="train")}
    metrics = {f"val/{METRIC_NAMES[k]}" if k in METRIC_NAMES else k: float(v) for k, v in values.items()}
    report("PROGRESS", {"epochs_done": trainer.epoch + 1, "metrics": metrics})


def train(request: Dict[str, Any], data_yaml: Path, output_dir: Path) -> Dict[str, Any]:
    import torch
    from ultralytics import YOLO

    params = request["params"]
    model = YOLO(request["weights_path"])
    model.add_callback("on_fit_epoch_end", report_epoch)

    results = model.train(
        data=str(data_yaml),
        epochs=params["epochs"],
        batch=params["batch_size"],
        imgsz=params["image_size"],
        lr0=params["learning_rate"],
        optimizer="AdamW",  # the default 'auto' picks its own learning rate and ignores lr0
        device=0 if torch.cuda.is_available() else "cpu",
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,  # the folder already holds the dataset
        plots=False,
        workers=4,
        amp=True,
    )

    metrics = {name: float(results.results_dict[key]) for key, name in METRIC_NAMES.items()}
    return {"metrics": metrics, "weights_path": str(Path(results.save_dir).resolve() / "weights" / "best.pt")}


def run(request: Dict[str, Any]) -> Dict[str, Any]:
    output_dir = Path(request["output_dir"])
    image_dir, label_dir = Path(request["image_path"]), Path(request["label_path"])

    names = resolve_classes(request["params"].get("classes"), label_dir)
    root = output_dir / "dataset"
    data_yaml = prepare_dataset(image_dir, label_dir, names, root)

    for split in ("images/train", "images/val"):
        if not validate_split(str(root), split, len(names)):
            raise ValueError(f"dataset check failed for {split}: see the training log")

    return train(request, data_yaml, output_dir)


def main(request_path: str) -> None:
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    try:
        report("RESULT", run(request))
    except Exception as exc:
        report("ERROR", {"message": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    main(sys.argv[1])
