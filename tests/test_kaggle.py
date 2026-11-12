import ast
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("kaggle_launch", ROOT / "kaggle" / "launch.py")
launch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launch)
REQUIREMENTS = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
NOTEBOOK = json.loads((ROOT / "kaggle" / "demo.ipynb").read_text(encoding="utf-8"))
CELL = "".join(NOTEBOOK["cells"][0]["source"])
SECRETS = {"token", "encoded", "hidden", "auth"}


def test_pins_read_requirements():
    found = launch.pins(REQUIREMENTS)
    assert found["torch"] == "2.11.0+cpu" and found["gradio"]
    assert "--extra-index-url" not in " ".join(found)


def test_matching_preinstalled_torch_is_kept():
    present = {"torch": "2.11.0+cu126", "torchvision": "0.26.0+cu126"}
    kept, notes = launch.plan(REQUIREMENTS, present)
    names = launch.pins(kept)
    assert "torch" not in names and "torchvision" not in names
    assert "ultralytics" in names and kept[0].startswith("--extra-index-url")
    assert notes[0] == "keeping preinstalled torch 2.11.0+cu126 (pin 2.11.0+cpu)"


def test_other_torch_versions_are_replaced_by_pins():
    kept, notes = launch.plan(REQUIREMENTS, {"torch": "2.6.0+cu124", "torchvision": None})
    names = launch.pins(kept)
    assert names["torch"] == "2.11.0+cpu" and names["torchvision"] == "0.26.0+cpu"
    assert notes == [
        "installing torch==2.11.0+cpu (found 2.6.0+cu124)",
        "installing torchvision==0.26.0+cpu (found none)",
    ]


def test_version_report():
    wanted = {"torch": "2.11.0+cpu", "gradio": "6.31.0", "numpy": "2.1.3"}
    lines = launch.versions(wanted, {"torch": "2.11.0+cu126", "gradio": "6.31.0", "numpy": None})
    assert lines[0].endswith("ok") and lines[1].endswith("ok")
    assert lines[2] == "  numpy: not installed (pin 2.1.3) MISSING"
    assert launch.versions({"pandas": "3.0.3"}, {"pandas": "2.2.0"})[0].endswith("differs from pin")


def test_weights_folder_is_found_anywhere_under_input(tmp_path):
    names = ["yolov8m_seg_aug.pt", "fabrimssa_net.pth"]
    partial = tmp_path / "a-other"
    partial.mkdir()
    (partial / names[0]).write_bytes(b"x")
    full = tmp_path / "datasets" / "user" / "fabrid-weights"
    full.mkdir(parents=True)
    for name in names:
        (full / name).write_bytes(b"x")
    assert launch.find_weights(tmp_path, names) == full
    assert launch.find_weights(partial, names) is None
    assert launch.find_weights(tmp_path / "missing", names) is None


def test_commands():
    check, app = launch.commands(Path("/kaggle/input/w"), Path("/kaggle/working/data"))
    assert check[1:4] == ["-u", "-m", "defects.check"]
    assert app[1:4] == ["-u", "app.py", "--share"]
    for command in (check, app):
        expected = ["--weights", str(Path("/kaggle/input/w"))]
        assert command[-4:] == [*expected, "--data", str(Path("/kaggle/working/data"))]
    _, bare = launch.commands(None, Path("/d"))
    assert "--weights" not in bare


def test_stream_passes_output_and_code(tmp_path, capsys):
    code = launch.stream([sys.executable, "-c", "print('hello'); raise SystemExit(3)"], tmp_path)
    assert code == 3
    assert "hello" in capsys.readouterr().out


def test_banner():
    assert launch.banner("WARNING: x").splitlines()[1] == "WARNING: x"


def test_notebook_cell_compiles_and_clones_the_app():
    tree = ast.parse(CELL)
    assert "https://github.com/FaiazPronoy/fabrid-app.git" in CELL
    assert "http.https://github.com/.extraheader=AUTHORIZATION: basic" in CELL
    assert 'get_secret("GITHUB_TOKEN")' in CELL
    assert "kaggle" in CELL and "launch.py" in CELL
    assert isinstance(tree, ast.Module)


def test_notebook_never_prints_the_token():
    for node in ast.walk(ast.parse(CELL)):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "print":
            used = {n.id for arg in node.args for n in ast.walk(arg) if isinstance(n, ast.Name)}
            assert not used & SECRETS, ast.unparse(node)
    assert 'os.environ["GITHUB_TOKEN"]' not in CELL


def test_notebook_redacts_clone_errors():
    assert 'message.replace(secret, "***")' in CELL
