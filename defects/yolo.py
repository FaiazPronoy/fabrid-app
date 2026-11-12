from pathlib import Path

import numpy as np
from PIL import Image

from defects.labels import CLASSES
from defects.prediction import Instance, ModelInput, instances, to_numpy
from defects.registry import ModelSpec


def check_names(names, offset: int) -> None:
    listed = dict(enumerate(names)) if isinstance(names, list | tuple) else dict(names)
    expected = {index + offset: name for index, name in enumerate(CLASSES)}
    found = {int(k): str(v) for k, v in listed.items()}
    if found != expected:
        raise ValueError(f"weights name classes {found}, expected {expected}")


def fit_masks(masks: np.ndarray, height: int, width: int) -> np.ndarray:
    if masks.shape[1:] == (height, width):
        return masks > 0.5
    resized = [
        np.asarray(Image.fromarray((m > 0.5).astype(np.uint8) * 255).resize((width, height))) > 127
        for m in masks
    ]
    return np.stack(resized) if resized else np.zeros((0, height, width), dtype=bool)


class UltralyticsPredictor:
    def __init__(self, model, spec: ModelSpec):
        check_names(model.names, spec.class_offset)
        self.model = model
        self.spec = spec

    def run(self, pixels: np.ndarray):
        settings = self.spec.settings
        return self.model.predict(
            np.ascontiguousarray(pixels[..., ::-1]),
            imgsz=settings["imgsz"],
            conf=settings["conf"],
            iou=settings["iou"],
            max_det=settings["max_det"],
            retina_masks=True,
            device="cpu",
            verbose=False,
        )[0]

    def predict(self, pixels: np.ndarray) -> list[Instance]:
        result = self.run(pixels)
        height, width = pixels.shape[:2]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0 or result.masks is None:
            return []
        masks = fit_masks(to_numpy(result.masks.data), height, width)
        return instances(boxes.xyxy, boxes.conf, boxes.cls, masks, self.spec.class_offset)

    def model_input(self, pixels: np.ndarray) -> ModelInput:
        predictor = getattr(self.model, "predictor", None)
        if predictor is None:
            self.run(pixels)
            predictor = self.model.predictor
        shown = predictor.pre_transform([np.ascontiguousarray(pixels[..., ::-1])])[0]
        height, width = pixels.shape[:2]
        note = (
            f"Model input: letterboxed by Ultralytics from {width} x {height} to "
            f"{shown.shape[1]} x {shown.shape[0]} (target size {self.spec.settings['imgsz']},"
            " grey padding)."
        )
        return ModelInput(np.ascontiguousarray(shown[..., ::-1]), note)


def load(spec: ModelSpec, path: Path) -> UltralyticsPredictor:
    from ultralytics import YOLO

    model = YOLO(str(path))
    if model.task != "segment":
        raise ValueError(f"{path.name} is a {model.task} model, expected a segmentation model")
    return UltralyticsPredictor(model, spec)
