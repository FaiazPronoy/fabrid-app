import numpy as np
import pytest
from PIL import Image

from conftest import photo_bytes
from defects import imageio


def exif_with_orientation(value: int) -> bytes:
    exif = Image.Exif()
    exif[0x0112] = value
    return exif.tobytes()


def test_exif_rotation_is_applied(tmp_path):
    path = tmp_path / "rotated.jpg"
    path.write_bytes(photo_bytes(64, 48, exif=exif_with_orientation(6)))
    image = imageio.load(path)
    assert image.size == (48, 64)
    pixels = np.asarray(image).astype(int)
    assert pixels[0, 0, 0] < 40
    assert pixels[-1, 0, 0] > 215


def test_plain_image_keeps_size(photo):
    assert imageio.load(photo).size == (64, 48)


def test_large_photo_is_downscaled_for_work(tmp_path):
    path = tmp_path / "big.jpg"
    path.write_bytes(photo_bytes(4000, 3000))
    display, work = imageio.prepare(path, 640)
    assert work.shape == (480, 640, 3)
    assert max(display.size) == imageio.DISPLAY_SIDE


def test_small_photo_is_not_upscaled(photo):
    display, work = imageio.prepare(photo, 640)
    assert work.shape == (48, 64, 3)
    assert display.size == (64, 48)


def test_corrupt_upload_gives_readable_error(tmp_path):
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"this is not an image")
    with pytest.raises(imageio.ImageError, match="not a readable image"):
        imageio.load(path)


def test_truncated_jpeg_gives_readable_error(tmp_path):
    path = tmp_path / "cut.jpg"
    path.write_bytes(photo_bytes(400, 300)[:600])
    with pytest.raises(imageio.ImageError):
        imageio.load(path)


def test_missing_input_gives_readable_error():
    with pytest.raises(imageio.ImageError, match="No image"):
        imageio.load(None)


def test_greyscale_and_alpha_become_rgb(tmp_path):
    path = tmp_path / "alpha.png"
    Image.new("RGBA", (10, 8), (10, 20, 30, 0)).save(path)
    assert imageio.load(path).mode == "RGB"


def test_downscale_uses_antialiasing(tmp_path):
    stripes = np.zeros((1280, 1280, 3), dtype=np.uint8)
    stripes[:, ::2] = 255
    path = tmp_path / "stripes.png"
    Image.fromarray(stripes).save(path)
    _, work = imageio.prepare(path, 640)
    assert work.shape == (640, 640, 3)
    assert 100 < work.mean() < 155
    assert work.std() < 20


def test_portrait_photo_long_side_is_limited(tmp_path):
    path = tmp_path / "tall.jpg"
    path.write_bytes(photo_bytes(900, 1600))
    _, work = imageio.prepare(path, 640)
    assert work.shape == (640, 360, 3)


def test_contrast_stretch_uses_percentiles():
    rng = np.random.default_rng(0)
    pixels = rng.integers(100, 150, (50, 50, 3)).astype(np.uint8)
    out = imageio.stretch(pixels)
    low, high = np.percentile(out, (2, 98))
    assert low <= 3 and high >= 252
    assert out.dtype == np.uint8 and out.shape == pixels.shape
    assert not np.array_equal(out, pixels)


def test_contrast_stretch_of_flat_image_is_unchanged():
    flat = np.full((10, 10, 3), 77, dtype=np.uint8)
    assert np.array_equal(imageio.stretch(flat), flat)
