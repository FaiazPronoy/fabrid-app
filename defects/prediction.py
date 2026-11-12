from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from defects.labels import class_name


@dataclass(frozen=True)
class Instance:
    label: str
    score: float
    box: tuple[float, float, float, float]
    mask: np.ndarray = field(repr=False)
    origin: tuple[int, int] = (0, 0)

    @property
    def area(self) -> int:
        return int(self.mask.sum())

    def full_mask(self, width: int, height: int) -> np.ndarray:
        plane = np.zeros((height, width), dtype=bool)
        x, y = self.origin
        rows, cols = self.mask.shape
        plane[y : y + rows, x : x + cols] = self.mask[: height - y, : width - x]
        return plane


@dataclass
class ModelInput:
    pixels: np.ndarray | None
    note: str


class Predictor(Protocol):
    def predict(self, pixels: np.ndarray) -> list[Instance]: ...

    def model_input(self, pixels: np.ndarray) -> ModelInput: ...


def to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def crop(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0:
        return np.zeros((0, 0), dtype=bool), (0, 0)
    part = mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    return part.astype(bool).copy(), (int(cols[0]), int(rows[0]))


def instances(boxes, scores, classes, masks, offset: int) -> list[Instance]:
    boxes, scores = to_numpy(boxes).reshape(-1, 4), to_numpy(scores).reshape(-1)
    classes, masks = to_numpy(classes).reshape(-1), to_numpy(masks).astype(bool)
    if not len(boxes) == len(scores) == len(classes) == len(masks):
        raise ValueError("model output has boxes, scores, classes and masks of different lengths")
    found = []
    for box, score, index, mask in zip(boxes, scores, classes, masks, strict=True):
        part, origin = crop(mask)
        found.append(
            Instance(
                label=class_name(int(index), offset),
                score=float(score),
                box=tuple(float(v) for v in box),
                mask=part,
                origin=origin,
            )
        )
    return sorted(found, key=lambda item: -item.score)


def select(found: list[Instance], threshold: float, classes) -> list[Instance]:
    chosen = set(classes)
    kept = [item for item in found if item.score >= threshold and item.label in chosen]
    return sorted(kept, key=lambda item: -item.score)
