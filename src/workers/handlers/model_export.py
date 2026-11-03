from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict

from core.db import SessionLocal
from core.jobs import Job
from core.orm import ModelRow

logger = logging.getLogger(__name__)

CONVERTED_FORMATS = {"onnx", "torchscript"}
HASH_CHUNK_SIZE = 1024 * 1024


def handle_model_export_job(job: Job) -> Dict[str, Any]:
    model_id = job.payload["model_id"]
    export_format = job.payload["format"]

    with SessionLocal() as session:
        model = session.get(ModelRow, model_id)  # closed before the export: it can take minutes
    if model is None:
        raise ValueError(f"model '{model_id}' not found")
    if not model.weights_path:
        raise ValueError(f"model '{model_id}' has no weights")

    weights = Path(model.weights_path)
    if export_format == "pt":
        exported = weights
    elif export_format in CONVERTED_FORMATS:
        logger.info("model export job %s: converting %s to %s", job.id, weights, export_format)
        exported = _convert(weights, export_format)
    else:
        raise ValueError(f"export format '{export_format}' is not supported yet")

    return {"file_path": str(exported), "file_hash": f"sha256:{_sha256(exported)}"}


def _convert(weights: Path, export_format: str) -> Path:
    from ultralytics import YOLO

    produced = Path(YOLO(str(weights)).export(format=export_format))
    target = weights.with_name(f"model{produced.suffix}")
    produced.replace(target)
    return target


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()
