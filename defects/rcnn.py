from pathlib import Path

import numpy as np

from defects.checkpoint import load_strict
from defects.prediction import Instance, ModelInput, instances
from defects.registry import ModelSpec


class TorchvisionPredictor:
    def __init__(self, model, spec: ModelSpec):
        self.model = model.eval()
        self.spec = spec

    def tensor(self, pixels: np.ndarray):
        import torch

        owned = np.array(pixels, copy=True, order="C")
        return torch.from_numpy(owned).permute(2, 0, 1).float() / 255

    def predict(self, pixels: np.ndarray) -> list[Instance]:
        import torch

        with torch.inference_mode():
            output = self.model([self.tensor(pixels)])[0]
        masks = output["masks"][:, 0] > self.spec.settings["mask_threshold"]
        return instances(
            output["boxes"], output["scores"], output["labels"], masks, self.spec.class_offset
        )

    def model_input(self, pixels: np.ndarray) -> ModelInput:
        import torch

        transform = self.model.transform
        with torch.inference_mode():
            batch, _ = transform([self.tensor(pixels)])
        mean = torch.tensor(transform.image_mean).view(3, 1, 1)
        std = torch.tensor(transform.image_std).view(3, 1, 1)
        shown = (batch.tensors[0] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        height, width = batch.image_sizes[0]
        pad_h, pad_w = shown.shape[0] - height, shown.shape[1] - width
        note = (
            f"Model input: resized to {width} x {height} (long side {self.spec.settings['imgsz']})"
            f" and padded by {pad_w} x {pad_h} px to a multiple of 32. The model also"
            " normalises each colour channel; that step is undone here so the image can be shown."
        )
        return ModelInput(np.round(shown * 255).astype(np.uint8), note)


def finish(model, path: Path, spec: ModelSpec) -> TorchvisionPredictor:
    from torchvision.models.detection._utils import overwrite_eps

    overwrite_eps(model, 0.0)
    load_strict(model, path)
    return TorchvisionPredictor(model, spec)


def load_fabrimssa(spec: ModelSpec, path: Path) -> TorchvisionPredictor:
    from defects.fabrimssa import build_hybrid

    return finish(build_hybrid(spec.settings), path, spec)


def load_maskrcnn(spec: ModelSpec, path: Path) -> TorchvisionPredictor:
    from defects.fabrimssa import build_maskrcnn

    return finish(build_maskrcnn(spec.settings), path, spec)
