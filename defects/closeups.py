import numpy as np

from defects.marks import factors, scale_box
from defects.prediction import Instance

MARGIN = 0.35
MIN_PAD = 16


def bounds(box, width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    pad = max(MIN_PAD, MARGIN * max(x1 - x0, y1 - y0))
    left, top = max(0, int(np.floor(x0 - pad))), max(0, int(np.floor(y0 - pad)))
    right, bottom = min(width, int(np.ceil(x1 + pad))), min(height, int(np.ceil(y1 + pad)))
    if right <= left:
        right = min(width, left + 1)
    if bottom <= top:
        bottom = min(height, top + 1)
    return left, top, right, bottom


def crops(display, found: list[Instance], size: tuple[int, int]) -> list[tuple[np.ndarray, str]]:
    pixels = np.asarray(display)
    height, width = pixels.shape[:2]
    sx, sy = factors(size, (width, height))
    out = []
    for item in sorted(found, key=lambda entry: -entry.score):
        left, top, right, bottom = bounds(scale_box(item.box, sx, sy), width, height)
        out.append((pixels[top:bottom, left:right].copy(), f"{item.label} {item.score:.2f}"))
    return out
