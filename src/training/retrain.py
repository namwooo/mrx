"""Runs one retraining. The worker starts this module with the training Python, after it built the dataset."""
import json
import sys
from pathlib import Path
from typing import Any, Dict

from training.init_train import METRIC_NAMES, report, train


def evaluate(weights: str, request: Dict[str, Any], data_yaml: Path, output_dir: Path) -> Dict[str, float]:
    """Scores the best weights on the test images, which the training never saw."""
    from ultralytics import YOLO

    results = YOLO(weights).val(
        data=str(data_yaml),
        split="test",
        imgsz=request["params"]["image_size"],
        batch=request["params"]["batch_size"],
        project=str(output_dir),
        name="test",
        exist_ok=True,
        plots=False,
    )
    return {name: float(results.results_dict[key]) for key, name in METRIC_NAMES.items()}


def run(request: Dict[str, Any]) -> Dict[str, Any]:
    data_yaml, output_dir = Path(request["data_yaml"]), Path(request["output_dir"])

    result = train(request, data_yaml, output_dir)
    if request["has_test"]:
        result["metrics"] = evaluate(result["weights_path"], request, data_yaml, output_dir)
    return result


def main(request_path: str) -> None:
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    try:
        report("RESULT", run(request))
    except Exception as exc:
        report("ERROR", {"message": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    main(sys.argv[1])
