"""Runs a training in its own process: the training Python can differ from the worker's."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

SRC_DIR = Path(__file__).resolve().parents[2]
MESSAGE_PREFIX = "MRX_"
STOP_CHECK_SECONDS = float(os.environ.get("TRAINING_STOP_CHECK_SECONDS", "2"))


def run_process(
        command: List[str],
        log_path: Path,
        on_progress: Callable[[Dict[str, Any]], None],
        should_stop: Optional[Callable[[], bool]] = None,
) -> Optional[Dict[str, Any]]:
    """Returns what the process reported as its result, or None when should_stop ended it."""
    result, error = None, None
    finished = threading.Event()
    stopped = threading.Event()

    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONPATH": str(SRC_DIR), "PYTHONUNBUFFERED": "1"},
        )

        def watch() -> None:
            while not finished.wait(STOP_CHECK_SECONDS):
                try:
                    if should_stop():
                        stopped.set()
                        process.terminate()
                        return
                except Exception:  # the database can be briefly busy: check again next round
                    logger.exception("could not check whether the training should stop")

        if should_stop:
            threading.Thread(target=watch, daemon=True).start()

        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                if not line.startswith(MESSAGE_PREFIX):
                    continue

                kind, payload = line.removeprefix(MESSAGE_PREFIX).split(" ", 1)
                payload = json.loads(payload)
                if kind == "PROGRESS":
                    on_progress(payload)
                elif kind == "RESULT":
                    result = payload
                elif kind == "ERROR":
                    error = payload["message"]
            process.wait()
        finally:
            finished.set()
            if process.poll() is None:
                process.kill()
                process.wait()

    if stopped.is_set():
        return None
    if process.returncode != 0 or result is None:
        reason = error or f"training exited with code {process.returncode}"
        raise RuntimeError(f"{reason} (log: {log_path})")
    return result
