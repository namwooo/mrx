import os
import sys

import yaml
from PIL import Image


DATASET_YAML_PATH = "/Users/youngchankim/projects/mrx/data/dataset.yaml"


def load_dataset_config(yaml_path):
    """Loads the dataset.yaml file and returns the configuration dictionary."""
    if not os.path.exists(yaml_path):
        print(f"❌ [Validation] Dataset YAML file not found at: {yaml_path}")
        sys.exit(1)

    with open(yaml_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"❌ [Validation] Failed to parse YAML file: {str(e)}")
            sys.exit(1)


def validate_split(root_path, split_dir, num_classes):
    """Validates images and corresponding YOLO label text files for a specific split (train/val)."""
    images_dir = os.path.join(root_path, split_dir)
    # Automatically map 'images/...' to 'labels/...' directory structure
    labels_dir = os.path.join(root_path, split_dir.replace("images", "labels"))

    print(f"🔍 [Validation] Checking split: {split_dir}")
    print(f"   - Target Images: {images_dir}")
    print(f"   - Target Labels: {labels_dir}")

    if not os.path.exists(images_dir):
        print(f"❌ [Validation] Image directory does not exist: {images_dir}")
        return False

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"⚠️ [Validation] No valid images found in: {images_dir}")
        return False

    corrupted_images = 0
    invalid_labels = 0
    total_checked = 0

    for img_name in image_files:
        total_checked += 1
        base_name, _ = os.path.splitext(img_name)
        img_path = os.path.join(images_dir, img_name)
        label_path = os.path.join(labels_dir, f"{base_name}.txt")

        # 1. Verify Image Integrity (Check for corrupted files)
        try:
            with Image.open(img_path) as img:
                img.verify()  # Verifies the file integrity without loading full pixel data
        except Exception:
            print(f"   🚨 Corrupted image file detected: {img_path}")
            corrupted_images += 1
            continue

        # 2. Verify YOLO Label File Existence & Content Format
        if not os.path.exists(label_path):
            # Missing label can either be treated as background or an error.
            # In this pipeline, we flag it as an invalid label layout.
            print(f"   🚨 Missing matching label file: {label_path}")
            invalid_labels += 1
            continue

        try:
            with open(label_path, "r") as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, 1):
                parts = line.strip().split()
                if not parts:
                    continue  # Skip empty lines

                # YOLO format must have exactly 5 elements: [class_id, x_center, y_center, width, height]
                if len(parts) != 5:
                    print(
                        f"   🚨 Invalid YOLO format in {label_path} (Line {line_num}): Expected 5 elements, got {len(parts)}")
                    invalid_labels += 1
                    break

                # Verify Class ID range boundaries
                class_id = int(parts[0])
                if class_id < 0 or class_id >= num_classes:
                    print(
                        f"   🚨 Out-of-bounds Class ID in {label_path} (Line {line_num}): Class {class_id} (Max allowed: {num_classes - 1})")
                    invalid_labels += 1
                    break

                # Verify bounding box normalized float ranges (0.0 to 1.0)
                coords = [float(x) for x in parts[1:]]
                if any(coord < 0.0 or coord > 1.0 for coord in coords):
                    print(
                        f"   🚨 Out-of-range bbox coordinates in {label_path} (Line {line_num}): Must be between 0.0 and 1.0")
                    invalid_labels += 1
                    break
        except Exception as e:
            print(f"   🚨 Failed to read or parse label file {label_path}: {str(e)}")
            invalid_labels += 1

    print(
        f"📊 [Validation Summary for {split_dir}]: Checked {total_checked} files. Corrupted Images: {corrupted_images}, Invalid Labels: {invalid_labels}")
    return corrupted_images == 0 and invalid_labels == 0

def main():
    print("📋 [Validation] Loading dataset configurations for pre-flight checks...")
    config = load_dataset_config(DATASET_YAML_PATH)

    root_path = config.get('path')
    train_dir = config.get('train')
    val_dir = config.get('val')
    num_classes = len(config.get('names', {}))

    if not root_path or not train_dir or not val_dir or num_classes == 0:
        print("❌ [Validation] Missing required fields (path, train, val, names) in dataset.yaml")
        sys.exit(1)

    # Validate Train and Validation datasets sequentially
    train_ok = validate_split(root_path, train_dir, num_classes)
    val_ok = validate_split(root_path, val_dir, num_classes)

    if train_ok and val_ok:
        print("✅ [Validation] All data format checks passed successfully. Proceeding to training phase.")
        sys.exit(0)  # Exit code 0 signals success to pipeline_master.py
    else:
        print("❌ [Validation] Data corruption or incorrect formatting detected. Pipeline halted.")
        sys.exit(1)  # Non-zero exit code halts the pipeline execution loop


if __name__ == "__main__":
    main()