import warnings

import numpy as np
import pytest

from conftest import SESSION
from defects import views
from defects.labels import CLASSES
from fakes import square

YOLO = "yolov8m_seg_aug"
HYBRID = "fabrimssa_net"


@pytest.fixture
def one_hole(workspace):
    workspace.fake.found = [square("hole", 0.2, 4, 4, 20, 16)]
    return workspace


def test_switching_model_does_not_refilter_other_models_analysis(one_hole, photo):
    first = one_hole.analyse(SESSION, str(photo), YOLO, 0.2, list(CLASSES))
    assert len(first.rows) == 1
    before = one_hole.history.entries(SESSION)[0]
    result = one_hole.refilter(SESSION, first.entry, 0.65, list(CLASSES), HYBRID)
    assert result.ok and result.skipped
    assert result.annotated is None and result.rows == []
    analysis = one_hole.lookup(SESSION, first.entry)
    assert analysis.threshold == 0.2 and len(analysis.shown()) == 1
    after = one_hole.history.entries(SESSION)[0]
    assert after == before
    assert after["total"] == 1 and after["threshold"] == 0.2
    assert one_hole.dashboard(SESSION)[0].startswith("1 analyses in this session, 1 detections")


def test_class_filter_under_other_model_is_ignored(one_hole, photo):
    first = one_hole.analyse(SESSION, str(photo), YOLO, 0.2, list(CLASSES))
    one_hole.refilter(SESSION, first.entry, 0.2, ["stain"], HYBRID)
    assert one_hole.history.entries(SESSION)[0]["classes"] == list(CLASSES)


def test_same_model_still_refilters(one_hole, photo):
    first = one_hole.analyse(SESSION, str(photo), YOLO, 0.2, list(CLASSES))
    result = one_hole.refilter(SESSION, first.entry, 0.65, list(CLASSES), YOLO)
    assert not result.skipped and result.rows == []
    assert one_hole.history.entries(SESSION)[0]["total"] == 0
    assert one_hole.fake.calls == 1


def test_skipped_filter_leaves_every_output_unchanged(one_hole, photo):
    first = one_hole.analyse(SESSION, str(photo), YOLO, 0.2, list(CLASSES))
    result = one_hole.refilter(SESSION, first.entry, 0.65, list(CLASSES), HYBRID)
    shown = views.pick(views.values(result, SESSION, first.entry, False, "keep"), views.LIGHT)
    assert shown[:2] == (SESSION, first.entry)
    assert set(shown[2:]) == {"keep"}


def test_download_files_have_short_names(workspace, photo):
    first = workspace.analyse(SESSION, str(photo), YOLO, 0.25, list(CLASSES))
    assert first.png.name == "annotated.png" and first.csv.name == "results.csv"
    assert first.png.parent.parent.name == first.entry
    path, _ = workspace.report(SESSION, first.entry)
    assert path.name == "report.pdf" and path.parent == first.png.parent
    second = workspace.analyse(SESSION, str(photo), YOLO, 0.25, list(CLASSES))
    assert second.png != first.png and first.png.is_file()


def test_rcnn_tensor_from_read_only_array_without_warning():
    torch = pytest.importorskip("torch")
    from defects.rcnn import TorchvisionPredictor

    pixels = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    pixels.setflags(write=False)
    predictor = TorchvisionPredictor.__new__(TorchvisionPredictor)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tensor = predictor.tensor(pixels)
    expected = torch.tensor(pixels.copy()).permute(2, 0, 1).float() / 255
    assert torch.equal(tensor, expected)
    assert not pixels.flags.writeable
