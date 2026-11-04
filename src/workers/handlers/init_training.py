from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from core.db import SessionLocal
from core.jobs import Job
from core.orm import InitTrainingRow, ModelRow
from sqlalchemy import update
from workers.handlers.process import run_process

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(os.environ.get("INIT_TRAINING_DIR", "./data/init_trainings"))

# The training environment can differ from the worker's: locally, TRAIN_PYTHON=.venv-train/bin/python.
TRAIN_PYTHON = os.environ.get("TRAIN_PYTHON", sys.executable)

TRAIN_MODULE = "training.init_train"


def handle_init_training_job(job: Job) -> Dict[str, Any]:
    init_training_id = job.payload["init_training_id"]

    with SessionLocal() as session:
        record = session.get(InitTrainingRow, init_training_id)
        if record is None:
            raise ValueError(f"init training '{init_training_id}' not found")
        model = session.get(ModelRow, record.model_id)
        if model is None or not model.weights_path:
            raise ValueError(f"model '{record.model_id}' has no weights")

    output_dir = OUTPUT_DIR.resolve() / str(init_training_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    request_path = output_dir / "request.json"
    request_path.write_text(
        json.dumps({
            "weights_path": model.weights_path,
            "image_path": record.image_path,
            "label_path": record.label_path,
            "params": record.params,
            "output_dir": str(output_dir),
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("init training %s: starting with %s", init_training_id, TRAIN_PYTHON)
    return run_process(
        _train_command(request_path),
        output_dir / "train.log",
        lambda message: _save_progress(init_training_id, message["epochs_done"]),
    )


def _train_command(request_path: Path) -> List[str]:
    return [TRAIN_PYTHON, "-m", TRAIN_MODULE, str(request_path)]


def _save_progress(init_training_id: int, epochs_done: int) -> None:
    with SessionLocal() as session:
        session.execute(
            update(InitTrainingRow)
            .where(InitTrainingRow.id == init_training_id)
            .values(epochs_done=epochs_done)
        )
        session.commit()
