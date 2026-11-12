import math

import numpy as np
from PIL import Image, ImageDraw

from defects.draw import font
from defects.labels import COLORS
from defects.palette import text_on
from defects.prediction import Instance

STYLES = {
    "mask": "Mask and box",
    "box": "Box only",
    "outline": "Outline",
    "circle": "Circle",
}
DEFAULT_STYLE = "mask"
ALPHA = 0.40
CIRCLE_MARGIN = 0.15


def factors(size: tuple[int, int], target: tuple[int, int]) -> tuple[float, float]:
    return target[0] / size[0], target[1] / size[1]


def scale_box(box, sx: float, sy: float) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box
    return x0 * sx, y0 * sy, x1 * sx, y1 * sy


def scaled_mask(item: Instance, sx: float, sy: float, width: int, height: int):
    rows, cols = item.mask.shape
    if rows == 0 or cols == 0:
        return np.zeros((0, 0), dtype=bool), (0, 0)
    ox, oy = item.origin
    x0, y0 = math.floor(ox * sx), math.floor(oy * sy)
    x1, y1 = math.ceil((ox + cols) * sx), math.ceil((oy + rows) * sy)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((0, 0), dtype=bool), (0, 0)
    picture = Image.fromarray(item.mask.astype(np.uint8) * 255)
    resized = picture.resize((x1 - x0, y1 - y0), Image.Resampling.NEAREST)
    return np.asarray(resized) > 127, (x0, y0)


def edge(mask: np.ndarray, thickness: int) -> np.ndarray:
    padded = np.pad(mask, 1)
    inner = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    line = mask & ~inner
    for _ in range(max(0, thickness - 1)):
        grown = np.pad(line, 1)
        line = (
            grown[1:-1, 1:-1]
            | grown[:-2, 1:-1]
            | grown[2:, 1:-1]
            | grown[1:-1, :-2]
            | grown[1:-1, 2:]
        )
    return line


def paint(canvas: np.ndarray, item: Instance, sx, sy, thickness: int, filled: bool) -> None:
    height, width = canvas.shape[:2]
    part, (x0, y0) = scaled_mask(item, sx, sy, width, height)
    if part.size == 0:
        return
    pad = thickness
    local = np.pad(part, pad)
    line = edge(local, thickness)
    color = np.array(COLORS[item.label], dtype=np.float32)
    lx0, ly0 = x0 - pad, y0 - pad
    cx0, cy0 = max(0, lx0), max(0, ly0)
    cx1, cy1 = min(width, lx0 + local.shape[1]), min(height, ly0 + local.shape[0])
    window = canvas[cy0:cy1, cx0:cx1]
    area = local[cy0 - ly0 : cy1 - ly0, cx0 - lx0 : cx1 - lx0]
    border = line[cy0 - ly0 : cy1 - ly0, cx0 - lx0 : cx1 - lx0]
    if filled:
        blend = window[area].astype(np.float32) * (1 - ALPHA) + color * ALPHA
        window[area] = np.round(blend).astype(np.uint8)
    window[border] = color.astype(np.uint8)


def label(draw: ImageDraw.ImageDraw, item: Instance, box, text_font, pad: int, size) -> None:
    color = COLORS[item.label]
    text = f"{item.label} {item.score:.2f}"
    left, top, right, bottom = draw.textbbox((0, 0), text, font=text_font)
    tag_w, tag_h = right - left + 2 * pad, bottom - top + 2 * pad
    x0, y0, _, y1 = box
    tag_y = y0 - tag_h if y0 - tag_h >= 0 else min(y1, size[1] - tag_h)
    tag_x = min(max(0, x0), max(0, size[0] - tag_w))
    draw.rectangle((tag_x, tag_y, tag_x + tag_w, tag_y + tag_h), fill=color)
    draw.text((tag_x + pad - left, tag_y + pad - top), text, fill=text_on(color), font=text_font)


def render(display, found: list[Instance], size: tuple[int, int], style: str = DEFAULT_STYLE):
    if style not in STYLES:
        raise ValueError(f"unknown mark style {style}")
    canvas = np.array(display, dtype=np.uint8, copy=True)
    height, width = canvas.shape[:2]
    sx, sy = factors(size, (width, height))
    longest = max(width, height)
    thickness = max(2, round(longest / 350))
    ordered = sorted(found, key=lambda item: item.score)
    if style in ("mask", "outline"):
        for item in ordered:
            paint(canvas, item, sx, sy, thickness, filled=style == "mask")
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    text_font = font(max(14, round(longest / 40)))
    pad = max(3, thickness)
    for item in ordered:
        box = scale_box(item.box, sx, sy)
        color = COLORS[item.label]
        if style in ("mask", "box"):
            draw.rectangle(box, outline=color, width=thickness)
        if style == "circle":
            x0, y0, x1, y1 = box
            grow_x, grow_y = (x1 - x0) * CIRCLE_MARGIN + pad, (y1 - y0) * CIRCLE_MARGIN + pad
            draw.ellipse(
                (x0 - grow_x, y0 - grow_y, x1 + grow_x, y1 + grow_y), outline=color, width=thickness
            )
        label(draw, item, box, text_font, pad, image.size)
    return np.asarray(image)
