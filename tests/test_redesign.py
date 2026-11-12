import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import app
from conftest import SESSION, photo_bytes
from defects import branding, closeups, marks, palette, screens, summary, uicheck
from defects.history import History
from defects.labels import CLASSES, COLORS
from defects.models import Slot
from defects.registry import MODELS
from defects.service import Result, Workspace
from fakes import FakePredictor, square

ROOT = Path(__file__).resolve().parents[1]
YOLO, HYBRID = MODELS[0].key, MODELS[1].key


def ring(size=(64, 48)):
    width, height = size
    yy, xx = np.mgrid[0:height, 0:width]
    plane = ((xx - 32) / 12) ** 2 + ((yy - 24) / 9) ** 2 < 1
    from defects.prediction import Instance, crop

    part, origin = crop(plane)
    return Instance("hole", 0.9, (20.0, 15.0, 44.0, 33.0), part, origin)


def grey(width, height):
    return np.full((height, width, 3), 128, dtype=np.uint8)


def test_scaled_mask_covers_the_same_region():
    item = square("stain", 0.8, 10, 12, 20, 18)
    part, origin = marks.scaled_mask(item, 4.0, 4.0, 256, 192)
    assert origin == (40, 48)
    assert np.array_equal(part, np.kron(item.mask, np.ones((4, 4), dtype=bool)))
    assert marks.scale_box(item.box, 4.0, 4.0) == (40.0, 48.0, 80.0, 72.0)


def test_scaled_mask_at_non_integer_factor():
    item = square("stain", 0.8, 10, 10, 20, 20)
    part, (x0, y0) = marks.scaled_mask(item, 2.5, 2.5, 200, 200)
    assert (x0, y0) == (25, 25) and part.shape == (25, 25) and part.all()


def test_mask_style_fills_and_boxes_at_display_scale():
    out = marks.render(grey(256, 192), [ring()], (64, 48), "mask")
    color = np.array(COLORS["hole"], dtype=np.float32)
    center = out[96, 128].astype(np.float32)
    assert np.allclose(center, np.round(128 * 0.6 + color * 0.4), atol=1)
    assert np.array_equal(out[131, 120], COLORS["hole"])
    assert np.array_equal(out[5, 5], [128, 128, 128])


def test_box_style_leaves_inside_unchanged():
    out = marks.render(grey(256, 192), [ring()], (64, 48), "box")
    assert np.array_equal(out[96, 128], [128, 128, 128])
    assert np.array_equal(out[131, 120], COLORS["hole"])


def test_outline_style_draws_only_the_mask_edge():
    out = marks.render(grey(256, 192), [ring()], (64, 48), "outline")
    assert np.array_equal(out[96, 128], [128, 128, 128])
    column = out[:, 128]
    edge = [y for y in range(100, 192) if np.array_equal(column[y], COLORS["hole"])]
    assert edge and 125 <= min(edge) <= 135
    assert np.array_equal(out[131, 82], [128, 128, 128])


def test_circle_style_rings_the_box():
    out = marks.render(grey(256, 192), [ring()], (64, 48), "circle")
    assert np.array_equal(out[96, 128], [128, 128, 128])
    row = out[96]
    hits = [x for x in range(0, 128) if np.array_equal(row[x], COLORS["hole"])]
    assert hits and max(hits) < 80


def test_styles_differ_and_bad_style_is_refused():
    renders = {s: marks.render(grey(256, 192), [ring()], (64, 48), s) for s in marks.STYLES}
    keys = list(renders)
    for a, b in zip(keys, keys[1:], strict=False):
        assert not np.array_equal(renders[a], renders[b])
    with pytest.raises(ValueError):
        marks.render(grey(64, 48), [], (64, 48), "glow")


def test_label_text_sized_from_image():
    small = marks.render(grey(256, 192), [ring()], (64, 48), "box")
    large = marks.render(grey(1024, 768), [ring()], (64, 48), "box")
    tag = np.array(COLORS["hole"])
    small_rows = np.where((small[:, :, :] == tag).all(-1).any(1))[0]
    large_rows = np.where((large[:, :, :] == tag).all(-1).any(1))[0]
    assert large_rows.size > 3 * small_rows.size


def test_closeups_count_bounds_and_captions():
    display = np.arange(256 * 192 * 3, dtype=np.uint32).reshape(192, 256, 3).astype(np.uint8)
    found = [square("hole", 0.6, 4, 4, 10, 10), square("stain", 0.97, 50, 30, 64, 48)]
    crops = closeups.crops(display, found, (64, 48))
    assert [caption for _, caption in crops] == ["stain 0.97", "hole 0.60"]
    stain, hole = crops[0][0], crops[1][0]
    assert stain.shape[0] <= 192 and stain.shape[1] <= 256
    assert hole.shape[0] >= 24 and hole.shape[1] >= 24
    assert np.array_equal(hole, display[: hole.shape[0], : hole.shape[1]])
    assert closeups.crops(display, [], (64, 48)) == []


def test_verdict_and_tiles(workspace, photo):
    result = workspace.analyse(SESSION, str(photo), YOLO, 0.25, list(CLASSES), style="mask")
    assert result.verdict["kind"] == "found"
    assert result.verdict["title"] == "3 defects found above confidence 0.25"
    assert "YOLOv8m-seg (augmented)" in result.verdict["detail"]
    tiles = {tile["name"]: tile for tile in result.tiles}
    assert tiles["hole"]["count"] == 2 and not tiles["hole"]["muted"]
    assert tiles["missing yarn"]["muted"]
    empty = workspace.refilter(SESSION, result.entry, 0.95, list(CLASSES), YOLO, "mask")
    assert empty.verdict == {**empty.verdict, "kind": "clear"}
    assert empty.verdict["title"] == "No defect found above confidence 0.95"
    assert len(empty.tiles) == 4 and all(tile["muted"] for tile in empty.tiles)


def test_styled_result_is_at_display_size(workspace, tmp_path):
    path = tmp_path / "big.jpg"
    path.write_bytes(photo_bytes(1600, 1200))
    result = workspace.analyse(SESSION, str(path), YOLO, 0.05, list(CLASSES), style="mask")
    assert result.annotated.shape == (1200, 1600, 3)
    assert np.asarray(result.display).shape == result.annotated.shape
    assert result.rows[0][3:7] == [4, 4, 20, 16]
    assert len(result.closeups) == 4
    classic = workspace.refilter(SESSION, result.entry, 0.05, list(CLASSES), YOLO)
    assert classic.annotated.shape == (480, 640, 3) and classic.display is None


def test_restyle_does_not_touch_history(workspace, photo):
    first = workspace.analyse(SESSION, str(photo), YOLO, 0.25, list(CLASSES), style="mask")
    before = workspace.history.entries(SESSION)
    again = workspace.restyle(SESSION, first.entry, "circle")
    assert again.ok and again.style == "circle" and again.display is not None
    assert workspace.history.entries(SESSION) == before
    assert workspace.fake.calls == 1


def test_compare_runs_every_available_model_and_saves(tmp_path, photo):
    found = [square("hole", 0.9, 4, 4, 20, 16), square("stain", 0.7, 30, 10, 50, 30)]
    slots = {
        YOLO: Slot(MODELS[0], predictor=FakePredictor(found)),
        HYBRID: Slot(MODELS[1], predictor=FakePredictor(found[:1])),
    }
    cards = {
        YOLO: {"default_threshold": {"value": 0.2}},
        HYBRID: {"default_threshold": {"value": 0.8}},
    }
    space = Workspace(slots, History(tmp_path), cards)
    results, sentence = space.compare(SESSION, str(photo))
    assert [key for key, _ in results] == [YOLO, HYBRID]
    assert all(result.ok for _, result in results)
    assert results[1][1].threshold == 0.8
    assert len(space.history.entries(SESSION)) == 2
    assert sentence == (
        "The counts agree for needle mark (0), hole (1), missing yarn (0). "
        "The counts differ for stain (YOLOv8m-seg (augmented) 1, FabriMSSA-Net 0)."
    )


def test_compare_with_unavailable_model(workspace, photo):
    results, sentence = workspace.compare(SESSION, str(photo))
    assert not results[1][1].ok and "unavailable" in results[1][1].message
    assert sentence == "Only one model could run, so there is nothing to compare."


def test_compare_sentence_when_all_agree():
    tally = dict.fromkeys(CLASSES, 0)
    sentence = summary.compare_sentence([("A", tally), ("B", dict(tally))])
    assert sentence.startswith("The models report the same count for every class")
    assert "not an accuracy measurement" in summary.COMPARE_NOTE


def test_history_cards_and_clearing(workspace, photo):
    first = workspace.analyse(SESSION, str(photo), YOLO, 0.25, list(CLASSES))
    cards = workspace.cards_for(SESSION)
    assert len(cards) == 1 and cards[0]["entry"] == first.entry
    assert "YOLOv8m-seg (augmented) | 3 defects: hole 2, stain 1 at 0.25" in cards[0]["caption"]
    gallery, choices, empty = screens.history_values(cards)
    assert len(gallery) == 1 and choices[0][1] == first.entry and empty == ""
    assert screens.entry_at(cards, 0) == first.entry
    assert screens.entry_at(cards, 5) is None and screens.entry_at(cards, None) is None
    assert workspace.clear(SESSION) == 1
    assert workspace.cards_for(SESSION) == [] and SESSION not in workspace.current
    assert "History is empty" in screens.history_values([])[2]
    assert workspace.clear(SESSION) == 0


def test_slider_follows_the_analysis_model(workspace, photo):
    first = workspace.analyse(SESSION, str(photo), YOLO, 0.35, list(CLASSES))
    assert workspace.slider_for(SESSION, first.entry, YOLO) == 0.35
    assert workspace.slider_for(SESSION, first.entry, HYBRID) == workspace.threshold_for(HYBRID)
    workspace.current.clear()
    reopened = workspace.reopen(SESSION, first.entry, "mask")
    assert reopened.threshold == 0.35
    assert workspace.slider_for(SESSION, first.entry, YOLO) == 0.35


def test_screen_values_for_success_failure_and_skip(workspace, photo):
    good = workspace.analyse(SESSION, str(photo), YOLO, 0.25, list(CLASSES), style="mask")
    shown = screens.values(good, SESSION, "", True, "keep", lambda flag: ("visible", flag))
    assert set(shown) == set(screens.INSPECT)
    assert shown["result_box"] == ("visible", True) and shown["entry"] == good.entry
    assert shown["comparison"][1] is good.annotated and shown["report_note"] == ""
    assert "3 defects found" in shown["status"]
    bad = workspace.analyse(SESSION, None, YOLO, 0.25, list(CLASSES), style="mask")
    failed = screens.values(bad, SESSION, good.entry, True, "keep", lambda flag: flag)
    assert failed["entry"] == "" and failed["result_box"] is False
    assert failed["comparison"] is None and "could not be completed" in failed["status"]
    skipped = screens.values(Result(True, "", skipped=True), SESSION, "x", False, "keep", bool)
    assert set(screens.pick(skipped, screens.LIGHT)[2:]) == {"keep"}
    assert "model_input" not in screens.LIGHT


def test_model_text_and_about(workspace):
    text = screens.model_text(workspace, YOLO)
    assert "Limits." in text and "Default confidence threshold" in text
    titles = [title for title, _, _ in screens.about_sections(workspace, branding.Branding())]
    assert titles == [
        "What the app does",
        "The four defect types",
        "The models and their validation figures",
        "How verification works",
        "Limits",
        "Team",
    ]


def test_compare_values_layout(workspace, photo):
    results, sentence = workspace.compare(SESSION, str(photo))
    out = screens.compare_values(workspace, results, sentence)
    assert len(out) == 2 * len(results) + 1
    assert out[0] is not None and out[2] is None
    assert "not an accuracy measurement" in out[-1]


def test_chips_and_header(workspace):
    workspace.slots[YOLO].verification = "Verified against the benchmark reference (6 images)."
    chips = workspace.chips()
    assert [chip["label"] for chip in chips] == ["Verified", "Unavailable"]
    assert "not found" in chips[1]["reason"]
    workspace.slots[YOLO].verification = "Not verified, no reference pack."
    assert workspace.chips()[0]["label"] == "Not verified"
    html = summary.header_html(branding.Branding(team=("A <b>",)), workspace.chips())
    assert "A &lt;b&gt;" in html and "Unavailable" in html


def test_picker_labels_use_model_card(tmp_path):
    slots = {spec.key: Slot(spec, predictor=FakePredictor()) for spec in MODELS}
    space = Workspace(slots, History(tmp_path), {YOLO: {"value": 0.187115}})
    labels = {key: label for label, key in space.picker()}
    assert labels[YOLO].endswith("validation mask AP50 0.187")
    assert labels[HYBRID].endswith("not evaluated yet")


def test_branding_with_missing_fields(tmp_path):
    assert branding.read(tmp_path / "none.json") == branding.Branding()
    path = tmp_path / "b.json"
    path.write_text(json.dumps({"title": " ", "team": "A, , B", "supervisor": "Dr X"}), "utf-8")
    read = branding.read(path)
    assert read.title == branding.TITLE and read.subtitle == ""
    assert read.lines() == ["Team: A, B", "Supervisor: Dr X"]
    path.write_text("[1, 2]", encoding="utf-8")
    assert branding.read(path) == branding.Branding()
    shipped = branding.read(ROOT / "branding.json")
    assert shipped.lines() == [] and shipped.title == branding.TITLE


def test_theme_contrast_and_colour_blind_separation():
    for name, ratio in palette.text_pairs().items():
        assert ratio >= palette.MIN_TEXT_CONTRAST, name
    for simulation, distance in palette.separation().items():
        assert distance >= palette.MIN_DISTANCE, simulation
    old = {"a": (230, 25, 75), "b": (0, 130, 200), "c": (245, 160, 0), "d": (60, 180, 75)}
    assert palette.separation(old)["deuteranopia"] < palette.MIN_DISTANCE


def test_report_carries_branding(workspace, photo):
    workspace.branding = branding.Branding(institution="Textile Department")
    first = workspace.analyse(SESSION, str(photo), YOLO, 0.25, list(CLASSES))
    from defects import report

    analysis = workspace.lookup(SESSION, first.entry)
    path = report.write(analysis, "headline", photo.parent / "r.pdf", branding=workspace.branding)
    data = path.read_bytes()
    assert data.startswith(b"%PDF")
    assert data.count(b"/Type /Page") - data.count(b"/Type /Pages") == 2


def wire_calls(path: Path) -> dict[str, ast.Call]:
    calls = {}
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "wire":
            calls[node.args[0].value] = node
    return calls


def test_every_interface_handler_has_a_check_scenario():
    names = set(wire_calls(ROOT / "defects" / "ui.py"))
    state = {"keys": [YOLO, HYBRID], "good": "g", "bad": "b", "session": "s", "entry": "e"}
    planned = {name for name, _ in uicheck.plan(state)}
    assert names == planned


def test_analyse_and_compare_cannot_be_queued_twice():
    calls = wire_calls(ROOT / "defects" / "ui.py")
    for name in ("analyse_lock", "compare_lock"):
        options = {k.arg: k.value.value for k in calls[name].keywords}
        assert options == {"queue": False, "trigger_mode": "once"}
    assert {"analyse_unlock", "compare_unlock"} <= set(calls)


def test_classic_interface_uses_the_same_service_calls(workspace):
    tree = ast.parse((ROOT / "defects" / "classic.py").read_text(encoding="utf-8"))
    used = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "workspace"
    }
    assert {"analyse", "refilter", "reopen", "report", "listing", "dashboard", "session"} <= used
    for name in used:
        assert hasattr(workspace, name), name
    for name in ("analyse", "refilter", "reopen", "report", "listing", "dashboard"):
        assert callable(getattr(Workspace, name))


def test_stub_predictor_for_the_check():
    found = [uicheck.instance("hole", 0.9, (1, 2, 5, 6))]
    stub = uicheck.Stub(found)
    assert stub.predict(np.zeros((4, 4, 3))) == found
    assert stub.model_input(np.zeros((4, 4, 3))).note.startswith("Model input")


def test_classic_flag_and_addresses():
    assert app.options(["--classic"], env={}).classic
    assert not app.options([], env={}).classic
    launched = (None, "http://127.0.0.1:7860/", "https://x.gradio.live")
    assert app.addresses(SimpleNamespace(), launched) == (launched[2], launched[1])
    demo = SimpleNamespace(share_url="https://y.gradio.live", local_url="http://l")
    assert app.addresses(demo, None) == ("https://y.gradio.live", "http://l")


def test_launcher_falls_back_to_classic():
    spec = importlib.util.spec_from_file_location("launch", ROOT / "kaggle" / "launch.py")
    launch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch)
    assert launch.ui_check()[1:] == ["-u", "-m", "defects.uicheck"]
    command = ["python", "app.py", "--share"]
    assert launch.choose(command, True) == command
    assert launch.choose(command, False) == [*command, "--classic"]
