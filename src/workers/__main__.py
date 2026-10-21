from __future__ import annotations

import argparse
import logging
import os

from core.jobs import JobType
from workers.handlers import HANDLERS
from workers.worker import Worker


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
    Worker(job_type, HANDLERS[job_type], poll_interval=args.poll_interval).run_once()


if __name__ == "__main__":
    main()
