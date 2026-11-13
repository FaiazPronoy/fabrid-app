import math

import numpy as np

from defects import marks
from defects.marks import factors, scale_box
from defects.prediction import Instance

CONTEXT = 3.0
MIN_SIDE = 200
LIMIT = 24


def span(low: float, high: float, limit: int) -> tuple[int, int]:
    length = max(1.0, high - low)
    start, end = low - CONTEXT * length, high + CONTEXT * length
    if end - start < MIN_SIDE:
        grow = (MIN_SIDE - (end - start)) / 2
        start, end = start - grow, end + grow
    start, end = math.floor(start), math.ceil(end)
    if end - start >= limit:
        return 0, limit
    if start < 0:
        start, end = 0, end - start
    if end > limit:
        start, end = start - (end - limit), limit
    return max(0, start), min(limit, end)


def bounds(box, width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    left, right = span(x0, x1, width)
    top, bottom = span(y0, y1, height)
    return left, top, right, bottom


def crops(
    display, found: list[Instance], size: tuple[int, int], style: str = marks.DEFAULT_STYLE
) -> list[tuple[np.ndarray, str]]:
    pixels = np.asarray(display)
    height, width = pixels.shape[:2]
    sx, sy = factors(size, (width, height))
    out = []
    for item in sorted(found, key=lambda entry: -entry.score)[:LIMIT]:
        left, top, right, bottom = bounds(scale_box(item.box, sx, sy), width, height)
        marked = marks.render(pixels, [item], size, style, labels=False)
        out.append((marked[top:bottom, left:right].copy(), f"{item.label} {item.score:.2f}"))
    return out
