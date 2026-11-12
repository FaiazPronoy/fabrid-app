import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT.parent / "fabrid-benchmark"
README = (ROOT / "README.md").read_text(encoding="utf-8")


def pins(text: str) -> dict[str, str]:
    found = {}
    for line in text.splitlines():
        match = re.match(r"^([A-Za-z0-9_.-]+)==(\S+)$", line.strip())
        if match:
            found[match.group(1).lower()] = match.group(2)
    return found


def fallback_block() -> str:
    section = README.split("### Fallback: Gradio 5", 1)[1]
    return section.split("```", 2)[1]


def test_fallback_pins_are_complete_and_compatible():
    current = pins((ROOT / "requirements.txt").read_text(encoding="utf-8"))
    fallback = pins(fallback_block())
    assert set(fallback) == set(current)
    assert fallback["gradio"] == "5.50.0"
    assert int(fallback["pandas"].split(".")[0]) < 3
    assert int(fallback["pillow"].split(".")[0]) < 12
    changed = {name for name in current if current[name] != fallback[name]}
    assert changed == {"gradio", "pandas", "pillow"}
    assert "--extra-index-url https://download.pytorch.org/whl/cpu" in fallback_block()


def test_no_hugging_face_space_setup_left():
    assert not README.startswith("---")
    for stale in ("sdk_version", "hf.space", "Create the Space", "git lfs", "Git LFS"):
        assert stale not in README, stale
    assert not (ROOT / ".gitattributes").exists()
    assert not (ROOT / "packages.txt").exists()


def test_no_hosting_specific_wording_in_texts():
    from defects import texts

    shown = [texts.TITLE, texts.INTRO, texts.LIMITS, texts.HISTORY_NOTE]
    sources = [
        README,
        *[path.read_text(encoding="utf-8") for path in (ROOT / "defects").glob("*.py")],
    ]
    for text in shown + sources:
        assert not re.search(r"Hugging ?Face|\bSpaces?\b|hf\.space", text), text[:80]
    assert "lost when the server restarts" in texts.HISTORY_NOTE


def test_weights_are_ignored_by_git():
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "weights/*.pt" in ignored and "weights/*.pth" in ignored


def test_kaggle_section_covers_the_run():
    section = README.split("## Run on Kaggle", 1)[1].split("\n## ", 1)[0]
    for needed in (
        "New Dataset",
        "Add Input",
        "GITHUB_TOKEN",
        "Accelerator: None",
        "Internet: On",
        "Save & Run All",
        "PUBLIC URL:",
        "12 hours",
        "changes on every run",
        "NOT READY",
    ):
        assert needed in section, needed


def test_readme_kaggle_commands_match_launcher():
    import importlib.util

    spec = importlib.util.spec_from_file_location("launch", ROOT / "kaggle" / "launch.py")
    launch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch)
    check, app = launch.commands(Path("<folder>"), Path("/kaggle/working/data"))
    assert str(launch.DATA) == str(Path("/kaggle/working/data"))
    assert "python -m defects.check --weights <folder> --data /kaggle/working/data" in README
    assert "python app.py --share --weights <folder> --data /kaggle/working/data" in README
    assert check[2:4] == ["-m", "defects.check"] and app[2:4] == ["app.py", "--share"]


def test_phone_camera_note_for_plain_http():
    plan = README.split("## Demo day plan", 1)[1].split("## Features", 1)[0]
    assert "webcam button inside the page does not work" in plan
    assert "Take photo" in plan


def test_no_stale_analysis_size():
    assert "1280" not in README
    for path in (ROOT / "defects").glob("*.py"):
        assert "1280" not in path.read_text(encoding="utf-8"), path.name


def test_yolo_card_uses_common_evaluator_once_available():
    source = BENCHMARK / "results/evaluation/yolov8m_aug_session_seed0/metrics.json"
    entry = json.loads((ROOT / "model_card.json").read_text(encoding="utf-8"))["yolov8m_seg_aug"]
    if source.is_file():
        assert entry["source"].endswith("evaluation/yolov8m_aug_session_seed0/metrics.json")
        assert entry["field"] == "mask_ap50.mean"
    else:
        assert entry["field"] == "mask.overall.AP50"
        assert "has not scored this run yet" in entry["evaluator"]


def test_demo_day_checklist_comes_first_in_the_plan():
    plan = README.split("## Demo day plan", 1)[1]
    checklist = plan.split("### Full plan", 1)[0]
    assert "### Demo day checklist" in checklist
    for step in (
        "Start the Kaggle run",
        "PUBLIC URL",
        "python -m defects.check",
        "What to show",
        "accuracy",
        "Offline",
    ):
        assert step in checklist
    assert "not a final inspection" in checklist
