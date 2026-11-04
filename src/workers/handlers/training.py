from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from core import jobs as job_repo
from core.db import SessionLocal
from core.jobs import Job
from core.orm import JobStatus, LabelingRow, ModelRow, TrainingMetricRow, TrainingRow
from core.training_data import select_images, write_dataset
from sqlalchemy import update
from workers.handlers.process import run_process

logger = logging.getLogger(__name__)

TRAIN_PYTHON = os.environ.get("TRAIN_PYTHON", sys.executable)
TRAIN_MODULE = "training.retrain"


def handle_training_job(job: Job) -> Optional[Dict[str, Any]]:
    training_id = job.payload["training_id"]

    with SessionLocal() as session:  # closed before the training: it can take hours
        record = session.get(TrainingRow, training_id)
        if record is None:
            raise ValueError(f"training '{training_id}' not found")

        images = select_images(session, record.labeling_ids, record.split)
        labeling = session.get(LabelingRow, record.labeling_ids[0])
        names = {int(class_id): name for class_id, name in labeling.class_names.items()}

        if record.base_model_id is None:
            weights = f"{record.params['model_arch']}.yaml"  # learns from scratch
        else:
            weights = session.get(ModelRow, record.base_model_id).weights_path

    output_dir = Path(record.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = write_dataset(images, names, output_dir / "dataset")

    request_path = output_dir / "request.json"
    request_path.write_text(
        json.dumps({
            "weights_path": weights,
            "data_yaml": str(data_yaml),
            "params": record.params,
            "output_dir": str(output_dir),
            "has_test": any(image.split == "test" for image in images),
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("training %s: %d images, starting with %s", training_id, len(images), TRAIN_PYTHON)
    return run_process(
        _train_command(request_path),
        output_dir / "train.log",
        lambda message: _save_progress(training_id, message),
        lambda: _is_canceled(job.id),
    )


def _train_command(request_path: Path) -> List[str]:
    return [TRAIN_PYTHON, "-m", TRAIN_MODULE, str(request_path)]


def _is_canceled(job_id: int) -> bool:
    return job_repo.get_job(job_id).status == JobStatus.CANCELED.value


def _save_progress(training_id: int, message: Dict[str, Any]) -> None:
    now = datetime.now(timezone.utc)
    epoch = message["epochs_done"]
    with SessionLocal() as session:
        session.execute(update(TrainingRow).where(TrainingRow.id == training_id).values(epochs_done=epoch))
        session.add_all([
            TrainingMetricRow(training_id=training_id, epoch=epoch, name=name, value=value, recorded_at=now)
            for name, value in message.get("metrics", {}).items()
        ])
        session.commit()
