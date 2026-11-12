import numpy as np

from defects.prediction import Instance, ModelInput, crop


def square(label: str, score: float, x0: int, y0: int, x1: int, y1: int, size=(64, 48)):
    width, height = size
    plane = np.zeros((height, width), dtype=bool)
    plane[y0:y1, x0:x1] = True
    part, origin = crop(plane)
    return Instance(label, score, (float(x0), float(y0), float(x1), float(y1)), part, origin)


class FakePredictor:
    def __init__(self, found=None):
        self.found = list(found or [])
        self.calls = 0
        self.seen = []

    def predict(self, pixels: np.ndarray) -> list[Instance]:
        self.calls += 1
        self.seen.append(np.array(pixels, copy=True))
        return list(self.found)

    def model_input(self, pixels: np.ndarray) -> ModelInput:
        return ModelInput(np.array(pixels, copy=True), "Model input: unchanged test image.")
