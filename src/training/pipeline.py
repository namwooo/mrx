import subprocess
import sys

def run_step(step_name, script_path, venv_python_path):
    print(f"=========================================")
    print(f"🔄 [Pipeline] Starting Stage: {step_name}")
    print(f"=========================================")

    result = subprocess.run(
        [venv_python_path, script_path],
        capture_output=False,
        text=True
    )

    if result.returncode != 0:
        print(f"❌ [Pipeline] Critical error occurred in [{step_name}]. Terminating pipeline.")
        sys.exit(result.returncode)

    print(f"✅ [Pipeline] Stage [{step_name}] completed successfully.\n")


if __name__ == "__main__":
    # Each stage can be configured to point to different virtual environments if necessary.
    TRAIN_VENV = "./.venv-train/bin/python"

    # run_step("1. Dataset Download & Preparation", "./src/train/1_data_prep.py", TRAIN_VENV)
    run_step("2. Data Format Validation", "./src/train/2_validation.py", TRAIN_VENV)
    run_step("3. YOLO GPU Accelerated Training", "./src/train/3_train.py", TRAIN_VENV)
    # run_step("4. Production Server Model Deployment", "./src/train/4_deploy.py", TRAIN_VENV)