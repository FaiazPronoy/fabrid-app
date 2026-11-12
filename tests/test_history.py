import numpy as np
import pytest

from conftest import SESSION
from defects.history import History, pack, unpack
from defects.labels import CLASSES
from fakes import square


def test_pack_round_trip():
    found = [square("hole", 0.9, 1, 2, 5, 9), square("stain", 0.4, 10, 10, 30, 12)]
    back = unpack(pack(found))
    assert [(i.label, i.score, i.box, i.origin) for i in back] == [
        (i.label, i.score, i.box, i.origin) for i in found
    ]
    for a, b in zip(found, back, strict=True):
        assert np.array_equal(a.mask, b.mask)


def test_pack_of_empty_result():
    assert unpack(pack([])) == []


def test_saved_analysis_reopens(workspace, photo):
    first = workspace.analyse(SESSION, str(photo), "yolov8m_seg_aug", 0.25, list(CLASSES))
    workspace.refilter(SESSION, first.entry, 0.5, ["hole"])
    entries = workspace.history.entries(SESSION)
    assert len(entries) == 1
    meta = entries[0]
    assert meta["model_name"] == "YOLOv8m-seg (augmented)"
    assert meta["threshold"] == 0.5
    assert meta["counts"]["hole"] == 1 and meta["total"] == 1
    assert workspace.history.thumb(SESSION, first.entry).is_file()

    workspace.current.clear()
    reopened = workspace.reopen(SESSION, first.entry)
    assert reopened.ok
    assert reopened.threshold == 0.5 and reopened.classes == ["hole"]
    assert [row[1] for row in reopened.rows] == ["hole"]
    assert "No new inference was run" in reopened.message
    assert not np.array_equal(reopened.annotated, first.annotated)
    again = workspace.refilter(SESSION, first.entry, 0.05, list(CLASSES))
    assert len(again.rows) == 4
    assert "No new inference was run" in again.message
    assert workspace.fake.calls == 1


def test_entries_newest_first(workspace, photo):
    a = workspace.analyse(SESSION, str(photo), "yolov8m_seg_aug", 0.25, list(CLASSES))
    b = workspace.analyse(SESSION, str(photo), "yolov8m_seg_aug", 0.25, list(CLASSES))
    order = [m["entry"] for m in workspace.history.entries(SESSION)]
    assert sorted([a.entry, b.entry], reverse=True) == order


def test_sessions_are_separate(workspace, photo):
    workspace.analyse(SESSION, str(photo), "yolov8m_seg_aug", 0.25, list(CLASSES))
    assert workspace.history.entries("f" * 32) == []


def test_bad_ids_are_refused(tmp_path):
    history = History(tmp_path)
    with pytest.raises(ValueError):
        history.folder("../etc")
    with pytest.raises(ValueError):
        history.folder(SESSION, "../../x")


def test_reopen_unknown_entry_is_reported(workspace):
    result = workspace.reopen(SESSION, "20260101-000000-000000-abcdef")
    assert not result.ok
    assert "could not be opened" in result.message
