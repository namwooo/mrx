from __future__ import annotations

import argparse
import logging
import os
import signal
import sys

from core.jobs import JobType
from workers.handlers import HANDLERS
from workers.worker import Worker

logger = logging.getLogger(__name__)


def stop_on_signals(worker: Worker) -> None:
    def on_signal(signum, _frame) -> None:
        if worker.stopping:
            raise KeyboardInterrupt  # the second signal gives up on the running job
        logger.info("%s: finishing the current job, send it again to stop now", signal.Signals(signum).name)
        worker.stop()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m workers")
    parser.add_argument(
        "job_type",
        choices=[t.value for t in HANDLERS],
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=float(os.environ.get("WORKER_POLL_INTERVAL", "2")),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    job_type = JobType(args.job_type)
    worker = Worker(job_type, HANDLERS[job_type], poll_interval=args.poll_interval)
    stop_on_signals(worker)
    try:
        worker.run_forever()
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
