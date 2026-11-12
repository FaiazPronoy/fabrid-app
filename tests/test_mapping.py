from types import SimpleNamespace

import numpy as np
import pytest

from defects.labels import CLASSES, class_name
from defects.prediction import instances
from defects.registry import MODELS, find
from defects.yolo import UltralyticsPredictor, check_names


def test_class_order_is_fixed():
    assert CLASSES == ("needle mark", "hole", "missing yarn", "stain")


def test_ultralytics_indices_start_at_zero():
    offset = find("yolov8m_seg_aug").class_offset
    assert [class_name(i, offset) for i in range(4)] == list(CLASSES)
    with pytest.raises(ValueError):
        class_name(4, offset)


def test_maskrcnn_indices_skip_background():
    offset = find("fabrimssa_net").class_offset
    assert [class_name(i, offset) for i in range(1, 5)] == list(CLASSES)
    with pytest.raises(ValueError, match="no defect class"):
        class_name(0, offset)
    with pytest.raises(ValueError):
        class_name(5, offset)


def test_rcnn_output_labels_map_to_names():
    masks = np.zeros((4, 10, 10), dtype=bool)
    for index in range(4):
        masks[index, index : index + 2, 0:3] = True
    boxes = np.array([[0, i, 3, i + 2] for i in range(4)], dtype=float)
    found = instances(boxes, [0.9, 0.8, 0.7, 0.6], [1, 2, 3, 4], masks, offset=1)
    assert [item.label for item in found] == list(CLASSES)


def test_rcnn_background_label_is_refused():
    masks = np.ones((1, 4, 4), dtype=bool)
    with pytest.raises(ValueError):
        instances(np.zeros((1, 4)), [0.5], [0], masks, offset=1)


class Boxes(SimpleNamespace):
    def __len__(self):
        return len(self.cls)


def fake_yolo(classes, names=None):
    count = len(classes)
    masks = np.zeros((count, 20, 30), dtype=np.float32)
    for index in range(count):
        masks[index, 2 * index : 2 * index + 2, 0:5] = 1.0
    boxes = Boxes(
        xyxy=np.array([[0, 2 * i, 5, 2 * i + 2] for i in range(count)], dtype=float),
        conf=np.linspace(0.9, 0.5, count),
        cls=np.array(classes, dtype=float),
    )
    result = SimpleNamespace(boxes=boxes, masks=SimpleNamespace(data=masks))
    model = SimpleNamespace(
        names=names if names is not None else dict(enumerate(CLASSES)),
        predict=lambda *args, **kwargs: [result],
    )
    return model


def test_ultralytics_predictor_maps_names():
    spec = find("yolov8m_seg_aug")
    predictor = UltralyticsPredictor(fake_yolo([3, 2, 1, 0]), spec)
    found = predictor.predict(np.zeros((20, 30, 3), dtype=np.uint8))
    assert [item.label for item in found] == ["stain", "missing yarn", "hole", "needle mark"]
    assert found[0].area == 10


def test_ultralytics_predictor_rejects_wrong_names():
    spec = find("yolov8m_seg_aug")
    with pytest.raises(ValueError, match="expected"):
        UltralyticsPredictor(fake_yolo([0], names={0: "hole", 1: "stain"}), spec)


def test_check_names_accepts_list():
    check_names(list(CLASSES), 0)


def test_registry_entries_are_complete():
    keys = [spec.key for spec in MODELS]
    assert len(set(keys)) == len(keys)
    assert MODELS[0].name == "YOLOv8m-seg (augmented)"
    assert MODELS[1].name == "FabriMSSA-Net"
    for spec in MODELS:
        assert spec.weights.startswith("weights/")
        assert spec.class_offset in (0, 1)
