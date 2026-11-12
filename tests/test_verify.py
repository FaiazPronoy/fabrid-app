import json

from conftest import photo_bytes
from defects import verify
from defects.models import Slot
from defects.registry import MODELS
from defects.verify import Detection, compare
from fakes import FakePredictor, square

SPEC = MODELS[0]
BASE = Detection("hole", 0.80, (10.0, 10.0, 50.0, 50.0), 1000)


def test_identical_detections_pass():
    assert compare([BASE], [BASE]) == []


def test_small_differences_within_tolerance_pass():
    near = Detection("hole", 0.815, (10.5, 10.0, 50.5, 50.0), 1080)
    assert compare([BASE], [near]) == []


def test_wrong_label_fails_both_ways():
    other = Detection("stain", 0.80, BASE.box, 1000)
    problems = compare([BASE], [other])
    assert len(problems) == 2
    assert problems[0].startswith("reference hole")
    assert "no hole detection" in problems[0]
    assert problems[1].startswith("extra stain")


def test_shifted_box_fails():
    shifted = Detection("hole", 0.80, (14.0, 10.0, 54.0, 50.0), 1000)
    problems = compare([BASE], [shifted])
    assert len(problems) == 2
    assert "box IoU 0.818" in problems[0]


def test_missing_detection_fails():
    problems = compare([BASE], [])
    assert problems == [
        f"reference {BASE.describe()} not reproduced (no hole detection on the other side)"
    ]


def test_extra_detection_fails():
    problems = compare([], [BASE])
    assert len(problems) == 1 and problems[0].startswith("extra hole 0.800")


def test_score_drift_fails():
    drift = Detection("hole", 0.83, BASE.box, 1000)
    problems = compare([BASE], [drift])
    assert len(problems) == 2
    assert "score difference 0.030" in problems[0]


def test_mask_area_outside_ten_percent_fails():
    larger = Detection("hole", 0.80, BASE.box, 1150)
    assert len(compare([BASE], [larger])) == 2


def test_low_scores_are_ignored_but_can_partner():
    weak = Detection("stain", 0.20, (0.0, 0.0, 5.0, 5.0), 20)
    assert compare([weak], []) == []
    assert compare([], [weak]) == []
    edge = Detection("hole", 0.305, BASE.box, 1000)
    below = Detection("hole", 0.29, BASE.box, 1000)
    assert compare([edge], [below]) == []


def make_pack(tmp_path, found, sha=None, detections=None, size=(64, 48)):
    root = tmp_path / "app"
    weights = root / SPEC.weights
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"weights bytes")
    pack = root / "reference" / SPEC.key
    (pack / "images").mkdir(parents=True)
    (pack / "images" / "a.jpg").write_bytes(photo_bytes(*size))
    listed = detections
    if listed is None:
        listed = [
            {"label": i.label, "score": i.score, "box": list(i.box), "mask_area": i.area}
            for i in found
        ]
    data = {
        "model_key": SPEC.key,
        "weights_sha256": sha or verify.sha256(weights),
        "min_score": 0.05,
        "images": [{"file": "a.jpg", "width": size[0], "height": size[1], "detections": listed}],
    }
    (pack / "detections.json").write_text(json.dumps(data), encoding="utf-8")
    return root


def test_matching_pack_verifies(tmp_path):
    found = [square("hole", 0.9, 4, 4, 20, 16), square("stain", 0.1, 30, 10, 50, 30)]
    root = make_pack(tmp_path, found)
    fake = FakePredictor(found)
    verdict = verify.verify(SPEC, fake, root, root / "reference")
    assert verdict.status == verify.VERIFIED
    assert verdict.text == "Verified against the benchmark reference (1 images)."
    assert fake.seen[0].shape == (48, 64, 3)


def test_wrong_weights_hash_is_reported_first(tmp_path):
    found = [square("hole", 0.9, 4, 4, 20, 16)]
    root = make_pack(tmp_path, found, sha="0" * 64)
    fake = FakePredictor(found)
    verdict = verify.verify(SPEC, fake, root, root / "reference")
    assert verdict.status == verify.FAILED
    assert "the weights file is not the one the reference was made from" in verdict.text
    assert fake.calls == 0


def test_differing_prediction_reports_per_image(tmp_path):
    found = [square("hole", 0.9, 4, 4, 20, 16)]
    root = make_pack(tmp_path, found)
    fake = FakePredictor([square("stain", 0.9, 4, 4, 20, 16)])
    verdict = verify.verify(SPEC, fake, root, root / "reference")
    assert verdict.status == verify.FAILED
    assert "1 of 1 images differ" in verdict.text
    assert len(verdict.problems["a.jpg"]) == 2
    assert "a.jpg: reference hole" in verdict.report(SPEC.name)


def test_size_mismatch_is_reported(tmp_path):
    root = make_pack(tmp_path, [], detections=[])
    listing = root / "reference" / SPEC.key / "detections.json"
    data = json.loads(listing.read_text(encoding="utf-8"))
    data["images"][0]["width"] = 640
    listing.write_text(json.dumps(data), encoding="utf-8")
    verdict = verify.verify(SPEC, FakePredictor(), root, root / "reference")
    assert "differs from reference 640 x 48" in verdict.problems["a.jpg"][0]


def test_missing_pack_is_not_verified(tmp_path):
    verdict = verify.verify(SPEC, FakePredictor(), tmp_path, tmp_path / "reference")
    assert verdict.status == verify.MISSING
    assert verdict.text == "Not verified, no reference pack."


def test_apply_sets_status_and_disables_failed_model(tmp_path):
    found = [square("hole", 0.9, 4, 4, 20, 16)]
    root = make_pack(tmp_path, found)
    slots = {
        MODELS[0].key: Slot(MODELS[0], predictor=FakePredictor([])),
        MODELS[1].key: Slot(MODELS[1], predictor=FakePredictor([])),
    }
    verdicts = verify.apply(slots, root, root / "reference")
    failed = slots[MODELS[0].key]
    assert not failed.ready
    assert failed.reason.startswith("Failed verification")
    assert verdicts[MODELS[1].key].status == verify.MISSING
    assert slots[MODELS[1].key].ready
    assert slots[MODELS[1].key].verification == "Not verified, no reference pack."


def test_status_line_shows_verification(workspace):
    workspace.slots[MODELS[0].key].verification = "Not verified, no reference pack."
    assert "Not verified, no reference pack." in workspace.info(MODELS[0].key)
    assert "Unavailable" in workspace.info(MODELS[1].key)


def test_command_line_exits_non_zero_without_models(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["verify", "--root", str(tmp_path)])
    assert verify.main() == 1
