import io
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

DISPLAY_SIDE = 2048
STRETCH_PERCENTILES = (2, 98)


class ImageError(Exception):
    pass


def load(source) -> Image.Image:
    if source is None or (isinstance(source, str | Path) and not str(source)):
        raise ImageError("No image received. Choose a photo or take one with the camera.")
    if isinstance(source, Image.Image):
        image = source
    else:
        if isinstance(source, bytes):
            source = io.BytesIO(source)
        elif not Path(source).is_file():
            raise ImageError("The uploaded file could not be found on the server. Try again.")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                image = Image.open(source)
                image.load()
        except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
            raise ImageError("The image is too large to process. Use a smaller photo.") from error
        except UnidentifiedImageError as error:
            raise ImageError(
                "This file is not a readable image. Use a JPEG, PNG or WebP photo. "
                "iPhone HEIC files need the camera set to Most Compatible."
            ) from error
        except (OSError, ValueError, SyntaxError) as error:
            raise ImageError(
                "The image file is damaged or incomplete. Try another photo."
            ) from error
    try:
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")
    except (OSError, ValueError) as error:
        raise ImageError("The image could not be converted to colour pixels.") from error


def fit(image: Image.Image, side: int) -> Image.Image:
    width, height = image.size
    longest = max(width, height)
    if longest <= side:
        return image.copy()
    scale = side / longest
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def prepare(source, side: int) -> tuple[Image.Image, np.ndarray]:
    image = load(source)
    display = fit(image, DISPLAY_SIDE)
    work = np.ascontiguousarray(np.asarray(fit(image, side), dtype=np.uint8))
    return display, work


def stretch(pixels: np.ndarray) -> np.ndarray:
    low, high = np.percentile(pixels, STRETCH_PERCENTILES)
    if high - low < 1:
        return np.array(pixels, dtype=np.uint8, copy=True)
    scaled = (pixels.astype(np.float32) - low) * (255.0 / (high - low))
    return np.ascontiguousarray(np.clip(np.round(scaled), 0, 255).astype(np.uint8))
