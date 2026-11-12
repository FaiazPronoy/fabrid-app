import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from defects.history import History
from defects.models import Slot
from defects.registry import MODELS
from defects.service import Workspace
from fakes import FakePredictor, square

SESSION = "0123456789abcdef0123456789abcdef"


def photo_bytes(width=64, height=48, fmt="JPEG", exif=None) -> bytes:
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    pixels[..., 0] = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    buffer = io.BytesIO()
    options = {"exif": exif} if exif is not None else {}
    Image.fromarray(pixels).save(buffer, format=fmt, **options)
    return buffer.getvalue()


@pytest.fixture
def photo(tmp_path) -> Path:
    path = tmp_path / "photo.png"
    path.write_bytes(photo_bytes(fmt="PNG"))
    return path


@pytest.fixture
def known():
    return [
        square("hole", 0.9, 4, 4, 20, 16),
        square("stain", 0.6, 30, 10, 50, 30),
        square("hole", 0.3, 40, 34, 60, 44),
        square("needle mark", 0.1, 2, 30, 10, 40),
    ]


@pytest.fixture
def workspace(tmp_path, known):
    fake = FakePredictor(known)
    slots = {
        MODELS[0].key: Slot(MODELS[0], predictor=fake, load_seconds=0.0),
        MODELS[1].key: Slot(MODELS[1], reason="weights file weights/fabrimssa_net.pth not found"),
    }
    space = Workspace(slots, History(tmp_path / "history"), {})
    space.fake = fake
    return space
