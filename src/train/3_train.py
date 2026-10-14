import os
import sys
import gc
import torch
from ultralytics import YOLO

BASE_MODEL_PATH = "/Users/youngchankim/projects/mrx/models/yolo11/yolo11n.pt"
DATASET_YAML_PATH = "/Users/youngchankim/projects/mrx/data/dataset.yaml"
OUTPUT_DIR = "./runs/detect"
RUN_NAME = "closed_network_run"


def check_environment():
    """Verify GPU availability before starting the heavy training process."""
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

    # 2. Select Computing Device (GPU vs CPU)
    device = check_environment()

    try:
        # 3. Load Local Model
        # YOLO internal logic detects that this is a local path and won't trigger GitHub downloads.
        model = YOLO(BASE_MODEL_PATH)

        # 4. Execute Accelerated Training
        print(f"🚀 [Training] Starting YOLO training loop using data: {DATASET_YAML_PATH}")
        model.train(
            data=DATASET_YAML_PATH,  # Path to local dataset.yaml
            epochs=5,  # Total training epochs
            imgsz=640,  # Input image size
            batch=16,  # Batch size (adjust based on your VRAM size)
            device=device,  # Target GPU ID or 'cpu'
            project=OUTPUT_DIR,  # Directory to save results
            name=RUN_NAME,  # Experiment folder name
            plots=True,  # Generate local evaluation graphs (loss, mAP, etc.)
            workers=4,  # Multi-threading for data loading (adjust based on CPU cores)

            # 💡 Critical Air-Gapped Option: Disable remote tracking tools like WandB
            # This prevents code execution from freezing or crashing due to network connection timeouts.
            amp=True,  # Automatic Mixed Precision training (saves VRAM and boosts speed)
        )
        print("✅ [Training] Core training process completed successfully.")

    except Exception as e:
        print(f"❌ [Training] A critical error occurred during training: {str(e)}")
        sys.exit(1)

    finally:
        # 5. Force Memory Cleanup before exit
        # Explicitly delete objects to ensure smooth VRAM reclamation back to the OS.
        if 'model' in locals():
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("🧹 [Training] Local cache flushed. Exiting training process script.")


if __name__ == "__main__":
    main()
