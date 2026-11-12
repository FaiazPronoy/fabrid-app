import csv

import pytest

from conftest import SESSION
from defects.labels import CLASSES

YOLO = "yolov8m_seg_aug"
HYBRID = "fabrimssa_net"


def test_analyse_counts_and_table(workspace, photo):
    result = workspace.analyse(SESSION, str(photo), YOLO, 0.25, list(CLASSES))
    assert result.ok
    assert [row[1] for row in result.rows] == ["hole", "stain", "hole"]
    assert result.rows[0][7] == 16 * 12
    assert result.rows[0][8] == pytest.approx(100 * 192 / (64 * 48), abs=1e-3)
    assert "3 defects found" in result.message
    assert "| hole | 2 |" in result.message
    assert "| total | 3 |" in result.message
    assert "inference time" in result.message
    assert result.annotated.shape == (48, 64, 3)


def test_threshold_refilters_without_inference(workspace, photo):
    first = workspace.analyse(SESSION, str(photo), YOLO, 0.25, list(CLASSES))
    assert workspace.fake.calls == 1
    higher = workspace.refilter(SESSION, first.entry, 0.5, list(CLASSES))
    assert [row[1] for row in higher.rows] == ["hole", "stain"]
    lower = workspace.refilter(SESSION, first.entry, 0.05, list(CLASSES))
    assert len(lower.rows) == 4
    assert workspace.fake.calls == 1


def test_class_filter_refilters_without_inference(workspace, photo):
    first = workspace.analyse(SESSION, str(photo), YOLO, 0.05, list(CLASSES))
    only = workspace.refilter(SESSION, first.entry, 0.05, ["stain", "needle mark"])
    assert [row[1] for row in only.rows] == ["stain", "needle mark"]
    assert "Showing only: needle mark, stain." in only.message
    assert workspace.fake.calls == 1


def test_empty_result_state(workspace, photo):
    workspace.fake.found = []
    result = workspace.analyse(SESSION, str(photo), YOLO, 0.25, list(CLASSES))
    assert result.ok
    assert result.rows == []
    assert "No defect found above confidence 0.25" in result.message


def test_threshold_above_all_scores_is_empty(workspace, photo):
    first = workspace.analyse(SESSION, str(photo), YOLO, 0.25, list(CLASSES))
    result = workspace.refilter(SESSION, first.entry, 0.95, list(CLASSES))
    assert result.rows == []
    assert "No defect found above confidence 0.95" in result.message


def test_no_class_selected(workspace, photo):
    first = workspace.analyse(SESSION, str(photo), YOLO, 0.25, list(CLASSES))
    result = workspace.refilter(SESSION, first.entry, 0.25, [])
    assert result.ok and result.rows == []
    assert "No classes selected" in result.message


def test_corrupt_upload_does_not_crash(workspace, tmp_path):
    path = tmp_path / "bad.png"
    path.write_bytes(b"\x89PNG broken")
    result = workspace.analyse(SESSION, str(path), YOLO, 0.25, list(CLASSES))
    assert not result.ok
    assert "not a readable image" in result.message
    assert workspace.fake.calls == 0


def test_unavailable_model_is_reported(workspace, photo):
    result = workspace.analyse(SESSION, str(photo), HYBRID, 0.5, list(CLASSES))
    assert not result.ok
    assert "FabriMSSA-Net is unavailable" in result.message
    assert "not found" in result.message


def test_predictor_failure_is_reported(workspace, photo):
    def broken(pixels):
        raise RuntimeError("out of memory")

    workspace.fake.predict = broken
    result = workspace.analyse(SESSION, str(photo), YOLO, 0.25, list(CLASSES))
    assert not result.ok
    assert "out of memory" in result.message


def test_refilter_before_analysis(workspace):
    result = workspace.refilter(SESSION, None, 0.5, list(CLASSES))
    assert not result.ok
    assert "Analyse a photo first" in result.message


def test_choices_mark_unavailable(workspace):
    labels = dict(workspace.choices())
    assert labels["YOLOv8m-seg (augmented)"] == YOLO
    assert labels["FabriMSSA-Net (unavailable)"] == HYBRID
    assert "not found" in workspace.info(HYBRID)
    assert "not evaluated yet" in workspace.info(YOLO)


def test_downloads_are_written(workspace, photo):
    result = workspace.analyse(SESSION, str(photo), YOLO, 0.25, list(CLASSES))
    assert result.png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    with result.csv.open(encoding="utf-8") as handle:
        lines = list(csv.reader(handle))
    assert lines[0][1] == "class"
    assert len(lines) == 4


def test_model_input_view_is_returned(workspace, photo):
    result = workspace.analyse(SESSION, str(photo), YOLO, 0.25, list(CLASSES))
    assert result.model_input.shape == (48, 64, 3)
    assert result.input_note.startswith("Model input")


def test_memory_keeps_a_bounded_number_of_sessions(workspace, photo):
    from defects import service

    for index in range(service.KEPT_SESSIONS + 3):
        session = f"{index:032x}"
        workspace.analyse(session, str(photo), YOLO, 0.25, list(CLASSES))
    assert len(workspace.current) == service.KEPT_SESSIONS
    oldest = f"{0:032x}"
    entry = workspace.history.entries(oldest)[0]["entry"]
    assert workspace.refilter(oldest, entry, 0.5, list(CLASSES)).ok
