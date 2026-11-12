import numpy as np
from PIL import Image, ImageDraw, ImageFont

from defects.labels import COLORS
from defects.prediction import Instance

ALPHA = 0.45


def font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def outline(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1)
    inner = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return mask & ~inner


def fill(canvas: np.ndarray, item: Instance) -> None:
    if item.mask.size == 0:
        return
    height, width = canvas.shape[:2]
    plane = item.full_mask(width, height)
    color = np.array(COLORS[item.label], dtype=np.float32)
    region = canvas[plane].astype(np.float32)
    canvas[plane] = np.round(region * (1 - ALPHA) + color * ALPHA).astype(np.uint8)
    canvas[outline(plane)] = color.astype(np.uint8)


def annotate(pixels: np.ndarray, found: list[Instance]) -> np.ndarray:
    canvas = np.array(pixels, dtype=np.uint8, copy=True)
    ordered = sorted(found, key=lambda item: item.score)
    for item in ordered:
        fill(canvas, item)
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    longest = max(image.size)
    width = max(2, round(longest / 400))
    text_font = font(max(12, round(longest / 45)))
    for item in ordered:
        color = COLORS[item.label]
        x0, y0, x1, y1 = item.box
        draw.rectangle((x0, y0, x1, y1), outline=color, width=width)
        text = f"{item.label} {item.score:.2f}"
        left, top, right, bottom = draw.textbbox((0, 0), text, font=text_font)
        text_w, text_h = right - left, bottom - top
        pad = max(2, width)
        tag_y = y0 - text_h - 2 * pad if y0 - text_h - 2 * pad >= 0 else y0
        tag_x = min(max(0, x0), max(0, image.width - text_w - 2 * pad))
        draw.rectangle(
            (tag_x, tag_y, tag_x + text_w + 2 * pad, tag_y + text_h + 2 * pad), fill=color
        )
        draw.text(
            (tag_x + pad - left, tag_y + pad - top), text, fill=(255, 255, 255), font=text_font
        )
    return np.asarray(image)
