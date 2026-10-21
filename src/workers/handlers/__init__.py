from typing import Dict

from core.jobs import JobType
from workers.handlers.batch import handle_batch_job
from workers.worker import JobHandler

HANDLERS: Dict[JobType, JobHandler] = {
    JobType.BATCH: handle_batch_job,
}