from typing import Dict

from core.jobs import JobType
from workers.handlers.init_training import handle_init_training_job
from workers.handlers.labeling import handle_labeling_job
from workers.handlers.model_export import handle_model_export_job
from workers.handlers.training import handle_training_job
from workers.worker import JobHandler

HANDLERS: Dict[JobType, JobHandler] = {
    JobType.INIT_TRAINING: handle_init_training_job,
    JobType.LABELING: handle_labeling_job,
    JobType.MODEL_EXPORT: handle_model_export_job,
    JobType.TRAINING: handle_training_job,
}