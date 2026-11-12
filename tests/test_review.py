import csv
import os
import time
from pathlib import Path

import numpy as np

from conftest import SESSION, photo_bytes
from defects import card, views
from defects.history import History, session_id
from defects.labels import CLASSES
from defects.report import heading
from defects.service import Workspace, cleanup

YOLO = "yolov8m_seg_aug"
HYBRID = "fabrimssa_net"
ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT.parent / "fabrid-benchmark"


def big_photo(tmp_path):
    path = tmp_path / "big.jpg"
    path.write_bytes(photo_bytes(4000, 3000))
    return path


def test_model_gets_image_scaled_to_its_input_size(workspace, tmp_path):
    result = workspace.analyse(SESSION, str(big_photo(tmp_path)), YOLO, 0.25, list(CLASSES))
    assert workspace.fake.seen[-1].shape == (480, 640, 3)
    assert result.annotated.shape == (480, 640, 3)
    assert "analysis size 640 x 480 px" in result.message
    with result.csv.open(encoding="utf-8") as handle:
        assert next(csv.reader(handle))[0] == "number"


def test_small_photo_is_not_enlarged_for_the_model(workspace, photo):
    workspace.analyse(SESSION, str(photo), YOLO, 0.25, list(CLASSES))
    assert workspace.fake.seen[-1].shape == (48, 64, 3)


def test_contrast_stretch_is_off_by_default(workspace, photo):
    result = workspace.analyse(SESSION, str(photo), YOLO, 0.25, list(CLASSES))
    assert "Contrast stretch before inference: off." in result.message
    assert workspace.history.entries(SESSION)[0]["stretch"] is False


def test_contrast_stretch_changes_model_input_and_is_recorded(workspace, tmp_path):
    path = tmp_path / "dull.png"
    rng = np.random.default_rng(1)
    from PIL import Image

    Image.fromarray(rng.integers(100, 140, (48, 64, 3)).astype(np.uint8)).save(path)
    result = workspace.analyse(SESSION, str(path), YOLO, 0.25, list(CLASSES), stretch=True)
    fed = workspace.fake.seen[-1]
    assert fed.max() >= 250 and fed.min() <= 5
    assert np.array_equal(result.model_input, fed)
    assert result.input_note.startswith("Contrast stretched")
    assert "on (approximation, not validated)" in result.message
    meta = workspace.history.entries(SESSION)[0]
    assert meta["stretch"] is True
    reopened = workspace.reopen(SESSION, meta["entry"])
    assert "on (approximation" in reopened.message
    analysis = workspace.lookup(SESSION, meta["entry"])
    assert int(analysis.work.min()) >= 95 and int(analysis.work.max()) <= 145
    assert "Contrast stretch before inference: on" in "\n".join(heading(analysis, "", _now()))


def _now():
    from datetime import datetime

    return datetime(2026, 11, 16, 10, 0, 0)


def test_filter_update_does_not_resend_images(workspace, photo):
    first = workspace.analyse(SESSION, str(photo), YOLO, 0.25, list(CLASSES))
    assert first.display is not None and first.model_input is not None
    light = workspace.refilter(SESSION, first.entry, 0.5, list(CLASSES))
    assert light.light and light.display is None and light.model_input is None
    assert light.annotated is not None and light.png and light.csv
    shown = views.pick(views.values(light, SESSION, first.entry, False, "keep"), views.LIGHT)
    assert len(shown) == len(views.LIGHT)
    assert "original" not in views.LIGHT and "model_input" not in views.LIGHT
    assert "input_note" not in views.LIGHT


def test_full_view_holds_every_output(workspace, photo):
    first = workspace.analyse(SESSION, str(photo), YOLO, 0.25, list(CLASSES))
    chosen = views.values(first, SESSION, "", True, "keep")
    assert chosen["original"] is first.display
    assert chosen["entry"] == first.entry
    failed = views.values(
        workspace.analyse(SESSION, None, YOLO, 0.25, list(CLASSES)), SESSION, "x", True, "keep"
    )
    assert failed["original"] is None and failed["entry"] == "x"
    assert failed["status"].startswith("**Could not complete:**")


def test_session_is_created_when_missing(workspace, photo):
    for missing in ("", None, "../x", "ABC"):
        created = workspace.session(missing)
        assert len(created) == 32 and created == session_id(created)
    assert workspace.session(SESSION) == SESSION
    session = workspace.session("")
    result = workspace.analyse(session, str(photo), YOLO, 0.25, list(CLASSES))
    assert result.ok
    assert workspace.refilter(session, result.entry, 0.5, list(CLASSES)).ok
    assert workspace.report(session, result.entry)[0] is not None
    assert len(workspace.listing(session)[0]) == 1
    assert workspace.dashboard(session)[0].startswith("1 analyses")
    fresh = workspace.session("")
    assert workspace.listing(fresh) == ([], [])
    assert workspace.dashboard(fresh)[0].startswith("0 analyses")
    assert workspace.refilter(fresh, None, 0.5, list(CLASSES)).message == "Analyse a photo first."


def test_threshold_comes_from_model_card(tmp_path):
    cards = card.read(ROOT / "model_card.json")
    from defects.models import Slot
    from defects.registry import MODELS

    slots = {spec.key: Slot(spec, reason="x") for spec in MODELS}
    space = Workspace(slots, History(tmp_path), cards)
    assert space.threshold_for(HYBRID) == 0.65
    assert space.threshold_for(YOLO) == 0.2


def test_threshold_matches_benchmark_file():
    if not BENCHMARK.is_dir():
        return
    cards = card.read(ROOT / "model_card.json")
    for entry in cards.values():
        chosen = entry.get("default_threshold")
        if not chosen:
            continue
        source = BENCHMARK / chosen["source"].removeprefix("fabrid-benchmark/")
        with source.open(encoding="utf-8") as handle:
            values = [float(row[chosen["column"]]) for row in csv.DictReader(handle)]
        assert len(values) == len(CLASSES)
        assert round(round(np.mean(values) / 0.05) * 0.05, 2) == chosen["value"]


def test_old_sessions_are_pruned(tmp_path):
    history = History(tmp_path)
    old, new = "a" * 32, "b" * 32
    for name in (old, new):
        (tmp_path / name / "x").mkdir(parents=True)
        (tmp_path / name / "x" / "meta.json").write_text("{}", encoding="utf-8")
    stamp = time.time() - 25 * 3600
    for path in (tmp_path / old).rglob("*"):
        os.utime(path, (stamp, stamp))
    (tmp_path / "notes.txt").write_text("keep", encoding="utf-8")
    assert history.prune() == 1
    assert not (tmp_path / old).exists()
    assert (tmp_path / new).exists() and (tmp_path / "notes.txt").exists()
    assert "removed 0 sessions" in cleanup(history)


def test_cleanup_failure_is_reported(tmp_path):
    history = History(tmp_path)

    def broken(*args, **kwargs):
        raise PermissionError("locked")

    history.prune = broken
    message = cleanup(history)
    assert "cleanup failed" in message and "locked" in message


def test_prune_without_history_folder(tmp_path):
    assert History(tmp_path / "none").prune() == 0
