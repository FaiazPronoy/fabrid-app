from pathlib import Path

import app
from defects import check
from defects.models import load_all, shown_path, weights_path
from defects.registry import MODELS
from fakes import FakePredictor

YOLO, HYBRID = MODELS


def test_defaults_without_arguments_or_environment():
    args = app.options([], env={})
    assert args.weights is None
    assert args.data == app.ROOT / "data"
    assert not args.share and not args.lan and args.port is None


def test_weights_and_data_from_arguments(tmp_path):
    args = app.options(
        ["--share", "--weights", str(tmp_path / "w"), "--data", str(tmp_path / "d")], env={}
    )
    assert args.share
    assert args.weights == tmp_path / "w"
    assert args.data == tmp_path / "d"


def test_weights_and_data_from_environment(tmp_path):
    env = {"FABRID_APP_WEIGHTS": str(tmp_path / "w"), "FABRID_APP_DATA": str(tmp_path / "d")}
    args = app.options([], env=env)
    assert args.weights == tmp_path / "w"
    assert args.data == tmp_path / "d"


def test_arguments_win_over_environment(tmp_path):
    env = {"FABRID_APP_WEIGHTS": str(tmp_path / "env"), "FABRID_APP_DATA": str(tmp_path / "env")}
    args = app.options(["--weights", "arg_w", "--data", "arg_d"], env=env)
    assert args.weights == Path("arg_w") and args.data == Path("arg_d")


def test_empty_environment_values_are_ignored():
    args = app.options([], env={"FABRID_APP_WEIGHTS": "", "FABRID_APP_DATA": ""})
    assert args.weights is None and args.data == app.ROOT / "data"


def test_weights_path_uses_registry_file_name(tmp_path):
    assert weights_path(YOLO, tmp_path) == tmp_path / "weights" / "yolov8m_seg_aug.pt"
    folder = tmp_path / "input" / "fabrid-weights"
    assert weights_path(YOLO, tmp_path, folder) == folder / "yolov8m_seg_aug.pt"
    assert weights_path(HYBRID, tmp_path, folder) == folder / "fabrimssa_net.pth"
    assert shown_path(YOLO) == "weights/yolov8m_seg_aug.pt"


def test_models_load_from_weights_folder(tmp_path):
    folder = tmp_path / "elsewhere"
    folder.mkdir()
    (folder / "yolov8m_seg_aug.pt").write_bytes(b"x")
    fake = FakePredictor()
    slots = load_all(
        MODELS, tmp_path / "code", builders={"ultralytics": lambda s, p: fake}, folder=folder
    )
    assert slots[YOLO.key].ready
    assert str(folder / "fabrimssa_net.pth") in slots[HYBRID.key].reason


def test_check_reads_weights_folder(tmp_path):
    folder = tmp_path / "w"
    folder.mkdir()
    (folder / "yolov8m_seg_aug.pt").write_bytes(b"abc")
    report = check.run(tmp_path, slots={}, cards={}, history=tmp_path, folder=folder)
    text = report.text()
    assert f"{folder / 'yolov8m_seg_aug.pt'}, 0.0 MB, sha256" in text
    assert f"{folder / 'fabrimssa_net.pth'} not found" in text


def test_public_line():
    url = "https://abc.gradio.live"
    assert app.public_line(True, url, "http://127.0.0.1:7860") == f"PUBLIC URL: {url}"
    warning = app.public_line(True, None, "http://127.0.0.1:7860")
    assert warning.startswith("WARNING: no public link") and "PUBLIC URL" not in warning
    assert app.public_line(False, None, "http://127.0.0.1:7860").startswith("LOCAL URL:")
