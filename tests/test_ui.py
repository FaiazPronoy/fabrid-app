import pytest

from conftest import SESSION
from defects.labels import CLASSES

gr = pytest.importorskip("gradio")


def test_classic_interface_builds(workspace):
    from defects.classic import about, build

    demo = build(workspace)
    assert isinstance(demo, gr.Blocks)
    assert "not seen in training" in about(workspace) or "not evaluated yet" in about(workspace)


def test_redesigned_interface_builds_and_records_handlers(workspace):
    from defects.ui import build, launch_options

    demo = build(workspace)
    assert isinstance(demo, gr.Blocks)
    names = {wire.name for wire in demo.wires}
    assert {"analyse", "filter_threshold", "model", "compare", "history_open"} <= names
    options = launch_options()
    assert options["footer_links"] == [] and "fd-verdict" in options["css"]


def test_interface_check_passes():
    from defects import uicheck

    assert uicheck.run(launch=False) == []


def test_frame_has_columns_when_empty():
    from defects.views import frame

    assert list(frame([]).columns)[1] == "class"


def test_full_flow_through_service(workspace, photo):
    result = workspace.analyse(SESSION, str(photo), "yolov8m_seg_aug", 0.25, list(CLASSES))
    assert result.ok
