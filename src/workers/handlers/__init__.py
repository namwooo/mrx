from typing import Dict

from core.jobs import JobType
from workers.handlers.batch import handle_batch_job
from workers.handlers.labeling import handle_labeling_job
from workers.handlers.model_export import handle_model_export_job
from workers.worker import JobHandler

HANDLERS: Dict[JobType, JobHandler] = {
    JobType.BATCH: handle_batch_job,
    JobType.LABELING: handle_labeling_job,
    JobType.MODEL_EXPORT: handle_model_export_job,
}