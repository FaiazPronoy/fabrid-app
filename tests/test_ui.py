import pytest

from conftest import SESSION
from defects.labels import CLASSES

gr = pytest.importorskip("gradio")


def test_interface_builds(workspace):
    from defects.ui import about, build

    demo = build(workspace)
    assert isinstance(demo, gr.Blocks)
    assert "not seen in training" in about(workspace) or "not evaluated yet" in about(workspace)


def test_frame_has_columns_when_empty():
    from defects.views import frame

    assert list(frame([]).columns)[1] == "class"


def test_full_flow_through_service(workspace, photo):
    result = workspace.analyse(SESSION, str(photo), "yolov8m_seg_aug", 0.25, list(CLASSES))
    assert result.ok
