from pathlib import Path
from typing import Iterator, List

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(input_path: str) -> List[Path]:
    path = Path(input_path)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    raise FileNotFoundError(f"input_path not found: {input_path}")


def chunks(items: List[Path], size: int) -> Iterator[List[Path]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def read_image(path: Path):
    import cv2

    return cv2.imread(str(path))
