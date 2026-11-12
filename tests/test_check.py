import json

import pytest

from conftest import photo_bytes
from defects import check, examples, verify
from defects.models import Slot
from defects.registry import MODELS
from fakes import FakePredictor, square

FOUND = [square("hole", 0.9, 4, 4, 20, 16)]
CARD = {
    "metric": "mask AP50",
    "value": 0.2,
    "interval": [0.1, 0.3],
    "seed": 0,
    "evaluator": "benchmark evaluator",
    "source": "x",
    "field": "y",
    "default_threshold": {"value": 0.3},
}


def build(tmp_path, packs=True, weights=True):
    root = tmp_path / "app"
    for spec in MODELS:
        path = root / spec.weights
        path.parent.mkdir(parents=True, exist_ok=True)
        if weights:
            path.write_bytes(spec.key.encode())
        if not packs:
            continue
        pack = root / "reference" / spec.key
        (pack / "images").mkdir(parents=True)
        for name in ("b.jpg", "a.jpg"):
            (pack / "images" / name).write_bytes(photo_bytes())
        listed = [
            {"label": i.label, "score": i.score, "box": list(i.box), "mask_area": i.area}
            for i in FOUND
        ]
        data = {
            "model_key": spec.key,
            "weights_sha256": verify.sha256(path) if weights else "0" * 64,
            "min_score": 0.05,
            "images": [
                {"file": name, "width": 64, "height": 48, "detections": listed}
                for name in ("b.jpg", "a.jpg")
            ],
        }
        (pack / "detections.json").write_text(json.dumps(data), encoding="utf-8")
    return root


def slots_for(fakes):
    return {spec.key: Slot(spec, predictor=fake) for spec, fake in zip(MODELS, fakes, strict=True)}


def cards():
    return {spec.key: dict(CARD) for spec in MODELS}


def test_ready_report(tmp_path):
    root = build(tmp_path)
    fakes = [FakePredictor(FOUND), FakePredictor(FOUND)]
    report = check.run(root, slots_for(fakes), cards(), tmp_path / "history")
    assert report.ok, report.text()
    text = report.text()
    assert text.endswith("READY")
    assert "Verified against the benchmark reference (2 images)" in text
    assert f"sha256 {verify.sha256(root / MODELS[0].weights)}" in text
    assert "Default threshold 0.30" in text
    assert "GB free for the history folder" in text
    assert "(b.jpg)" in text
    assert all(fake.calls == 2 + check.RUNS for fake in fakes)


def test_missing_weights_file_is_a_problem(tmp_path):
    root = build(tmp_path)
    (root / MODELS[1].weights).unlink()
    report = check.run(root, slots_for([FakePredictor(FOUND), FakePredictor(FOUND)]), cards())
    assert not report.ok
    assert any("not found" in p for p in report.problems)
    assert "NOT READY" in report.text()


def test_failed_verification_is_a_problem_and_not_timed(tmp_path):
    root = build(tmp_path)
    wrong = FakePredictor([square("stain", 0.9, 4, 4, 20, 16)])
    report = check.run(root, slots_for([FakePredictor(FOUND), wrong]), cards(), tmp_path)
    assert any("Failed verification" in p for p in report.problems)
    assert "b.jpg: reference hole" in report.text()
    assert wrong.calls == 2


def test_missing_pack_is_a_problem(tmp_path):
    root = build(tmp_path, packs=False)
    report = check.run(root, slots_for([FakePredictor(), FakePredictor()]), cards(), tmp_path)
    assert any("no reference pack" in p for p in report.problems)
    assert any("no reference image to time" in p for p in report.problems)


def test_unloaded_model_is_a_problem(tmp_path):
    root = build(tmp_path)
    slots = slots_for([FakePredictor(FOUND), FakePredictor(FOUND)])
    slots[MODELS[0].key] = Slot(MODELS[0], reason="required package missing: torch")
    report = check.run(root, slots, cards(), tmp_path)
    assert any("not loaded, required package missing: torch" in p for p in report.problems)


def test_card_without_threshold_or_entry_is_a_problem(tmp_path):
    root = build(tmp_path)
    partial = {MODELS[0].key: {**CARD, "default_threshold": None}}
    report = check.run(root, slots_for([FakePredictor(FOUND), FakePredictor(FOUND)]), partial)
    assert any("no default threshold" in p for p in report.problems)
    assert any("no evaluated entry" in p for p in report.problems)


def test_low_disk_space_is_a_problem(tmp_path, monkeypatch):
    root = build(tmp_path)
    monkeypatch.setattr(check, "MIN_FREE_GB", 10**9)
    report = check.run(root, slots_for([FakePredictor(FOUND), FakePredictor(FOUND)]), cards())
    assert any("GB free for the history folder" in p for p in report.problems)


def test_timing_is_median_of_three_runs(tmp_path, monkeypatch):
    root = build(tmp_path)
    clock = iter([0.0, 1.0, 10.0, 15.0, 20.0, 22.0])
    monkeypatch.setattr(check.time, "perf_counter", lambda: next(clock))
    fake = FakePredictor(FOUND)
    assert check.timing(fake, root / "reference" / MODELS[0].key / "images" / "a.jpg", 640) == 2.0
    assert fake.calls == 3


def test_first_image_falls_back_to_folder(tmp_path):
    pack = tmp_path / "key" / "images"
    pack.mkdir(parents=True)
    (pack / "z.jpg").write_bytes(b"x")
    (pack / "c.jpg").write_bytes(b"x")
    assert check.first_image(tmp_path, "key").name == "c.jpg"
    assert check.first_image(tmp_path, "none") is None


def test_command_line_fails_without_weights(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["check", "--root", str(tmp_path)])
    assert check.main() == 1


def test_examples_are_deduplicated_and_sorted(tmp_path):
    root = build(tmp_path)
    (root / "reference" / MODELS[0].key / "images" / "notes.txt").write_text("x", encoding="utf-8")
    (root / "reference" / MODELS[1].key / "images" / "c.png").write_bytes(b"x")
    found = examples.listing(root / "reference")
    assert [path.name for path in found] == ["a.jpg", "b.jpg", "c.png"]


def test_no_examples_without_reference_images(tmp_path):
    assert examples.listing(tmp_path / "reference") == []


def test_example_loads_like_an_upload(workspace, tmp_path):
    root = build(tmp_path)
    path = examples.listing(root / "reference")[0]
    result = workspace.analyse("a" * 32, str(path), MODELS[0].key, 0.25, ["hole"])
    assert result.ok and workspace.fake.calls == 1


def test_interface_shows_examples_row(workspace, tmp_path):
    gr = pytest.importorskip("gradio")
    from defects.ui import build as interface

    root = build(tmp_path)
    demo = interface(workspace, examples.listing(root / "reference"))
    assert isinstance(demo, gr.Blocks)
