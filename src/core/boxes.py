from typing import List, Sequence


def xywhn(xyxy: Sequence[float], width: int, height: int) -> List[float]:
    """Box as normalized center x, center y, width, height (YOLO label format)."""
    x1, y1, x2, y2 = xyxy
    return [
        round((x1 + x2) / 2 / width, 6),
        round((y1 + y2) / 2 / height, 6),
        round((x2 - x1) / width, 6),
        round((y2 - y1) / height, 6),
    ]
