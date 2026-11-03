from typing import Optional

import visionops_labeling
from core import jobs as job_repo
from fastapi import APIRouter, Query, Path
from app.schemas.labeling import LabelingCreate
from app.services import labeling as labeling_service

router = APIRouter(
    prefix="/labelings",
    tags=["Auto Labeling"]
)


@router.post("/", summary="Create auto labeling job", status_code=201)
def create_labeling_job(body: LabelingCreate):
    return labeling_service.create_labeling_job(body)


@router.get("/", summary="List all auto labeling jobs", status_code=200)
def list_labeling_jobs(
        status: Optional[job_repo.JobStatus] = Query(None),
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0)
):
    return labeling_service.list_labeling_jobs(status=status, limit=limit, offset=offset)


@router.get("/{labeling_id}", summary="Get labeling job status and result")
def get_labeling_job(labeling_id: int = Path(ge=1)):
    return labeling_service.get_labeling_job(labeling_id)


@router.delete("/{labeling_id}", summary="Cancel queued labeling job")
def cancel_labeling_job(labeling_id: int = Path(ge=1)):
    return labeling_service.cancel_labeling_job(labeling_id)

@router.post("/test")
def test():
    from visionops_labeling.tasks import detect
    from visionops_labeling.config import TtaConfig, RuleConfig
    from visionops_labeling.stages import class_stages

    setup = detect.SignalSetup(
        task="detect",
        enabled=True,
        variants=detect.tta_variants(TtaConfig()),
        match_iou=0.3,
        conf_thresholds=conf,
        rules=detect.rule_params(
            conf,
            tta_iou,
            agree_iou,
            class_stages(states, model_version, stage_cfg),
            RuleConfig()),
        distance=image_distance,
        box=box_distance,
        box_active=box_active,
        stats_id=stats_id,
        production=production_ref)

    for item in detect.analyze(
            predictor,
            arrays,
            {"conf": 0.05, "iou": 0.7, "max_det": 300},
            cmap,
            names,
            setup,
            production=production_predictor):
        judgment = detect.judge(item["boxes"], item["signals"], setup.rules, f"{batch_id}:{sha256}")