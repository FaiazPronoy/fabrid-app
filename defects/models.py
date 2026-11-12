import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from defects.registry import ModelSpec


def ultralytics_builder(spec, path):
    from defects import yolo

    return yolo.load(spec, path)


def fabrimssa_builder(spec, path):
    from defects import rcnn

    return rcnn.load_fabrimssa(spec, path)


def maskrcnn_builder(spec, path):
    from defects import rcnn

    return rcnn.load_maskrcnn(spec, path)


BUILDERS = {
    "ultralytics": ultralytics_builder,
    "fabrimssa": fabrimssa_builder,
    "maskrcnn": maskrcnn_builder,
}


@dataclass
class Slot:
    spec: ModelSpec
    predictor: object | None = None
    reason: str | None = None
    load_seconds: float | None = None
    verification: str | None = None

    @property
    def ready(self) -> bool:
        return self.predictor is not None


def weights_path(spec: ModelSpec, root: Path, folder: Path | None = None) -> Path:
    if folder is not None:
        return Path(folder) / Path(spec.weights).name
    return Path(root) / spec.weights


def shown_path(spec: ModelSpec, folder: Path | None = None) -> str:
    return spec.weights if folder is None else str(weights_path(spec, Path(), folder))


def load_one(
    spec: ModelSpec, root: Path, builders=BUILDERS, warm: bool = True, folder: Path | None = None
) -> Slot:
    path = weights_path(spec, root, folder)
    if not path.is_file():
        return Slot(spec, reason=f"weights file {shown_path(spec, folder)} not found")
    if spec.kind not in builders:
        return Slot(spec, reason=f"no loader for model kind {spec.kind}")
    started = time.perf_counter()
    try:
        predictor = builders[spec.kind](spec, path)
        if warm:
            predictor.predict(np.full((256, 256, 3), 128, dtype=np.uint8))
    except ImportError as error:
        return Slot(spec, reason=f"required package missing: {error.name or error}")
    except Exception as error:
        traceback.print_exc()
        return Slot(spec, reason=f"{type(error).__name__}: {error}")
    return Slot(spec, predictor=predictor, load_seconds=time.perf_counter() - started)


def load_all(
    specs, root: Path, builders=BUILDERS, warm: bool = True, folder: Path | None = None
) -> dict[str, Slot]:
    slots = {}
    for spec in specs:
        slot = load_one(spec, root, builders, warm, folder)
        status = "ready" if slot.ready else f"unavailable: {slot.reason}"
        print(f"{spec.name}: {status}", flush=True)
        slots[spec.key] = slot
    return slots
