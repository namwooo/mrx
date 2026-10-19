import os
import sys
import gc
import uuid

import torch
from ultralytics import YOLO

BASE_MODEL_PATH = "/Users/youngchankim/projects/mrx/models/yolo11/yolo11n.pt"
DATASET_YAML_PATH = "/Users/youngchankim/projects/mrx/data/dataset.yaml"
OUTPUT_DIR = "/Users/youngchankim/projects/mrx/artifacts/runs"
RUN_ID = str(uuid.uuid4())


def check_environment():
    print("🖥️ [Training] Checking hardware acceleration...")
    if torch.cuda.is_available():
        print(f"✅ CUDA Available: {torch.cuda.get_device_name(0)}")
        return 0  # Use GPU 0
    else:
        print("⚠️ CUDA NOT Available. Falling back to CPU mode (Not recommended for YOLO).")
        return "cpu"


def main():
    print("🏋️ [Training] Initializing YOLO Training Process...")

    # Check if local files exist to prevent internet connection attempts
    if not os.path.exists(BASE_MODEL_PATH):
        print(f"❌ [Error] Base model weight not found at {BASE_MODEL_PATH}")
        print("Please download 'yolov11n.pt' from an internet-connected PC and transfer it first.")
        sys.exit(1)

    if not os.path.exists(DATASET_YAML_PATH):
        print(f"❌ [Error] Dataset configuration file not found at {DATASET_YAML_PATH}")
        sys.exit(1)

    # Select Computing Device (GPU vs CPU)
    device = check_environment()

    try:
        # Load Local Model
        # YOLO internal logic detects that this is a local path and won't trigger GitHub downloads.
        model = YOLO(BASE_MODEL_PATH)

        print(f"🚀 [Training] Starting YOLO training loop using data: {DATASET_YAML_PATH}")
        model.train(
            data=DATASET_YAML_PATH,
            epochs=5,
            imgsz=640,
            batch=16,
            device=device,
            project=OUTPUT_DIR,
            name=RUN_ID,
            plots=False,
            workers=4,

            # 💡 Critical Air-Gapped Option: Disable remote tracking tools like WandB
            # This prevents code execution from freezing or crashing due to network connection timeouts.
            amp=True,  # Automatic Mixed Precision training (saves VRAM and boosts speed)
        )
        print("✅ [Training] Core training process completed successfully.")

    except Exception as e:
        print(f"❌ [Training] A critical error occurred during training: {str(e)}")
        sys.exit(1)

    finally:
        # Force Memory Cleanup before exit
        # Explicitly delete objects to ensure smooth VRAM reclamation back to the OS.
        if 'model' in locals():
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("🧹 [Training] Local cache flushed. Exiting training process script.")


if __name__ == "__main__":
    main()
