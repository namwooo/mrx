from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def handle_batch_job(job: Dict[str, Any]) -> Dict[str, Any]:
    payload = job["payload"]
    model = payload["model"]
    input_path = payload["input_path"]
    params = payload.get("params", {})

    logger.info("batch job %s: model=%s input=%s params=%s", job["id"], model, input_path, params)

    # TODO: inference 모듈의 실제 추론 함수를 여기서 호출 (with 동운님)
    raise NotImplementedError("No YOLO Inference module yet")
