import socket
import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from defects.labels import CLASSES
from defects.prediction import Instance, ModelInput, crop

SIZE = (320, 240)


class Stub:
    def __init__(self, found):
        self.found = found

    def predict(self, pixels):
        return list(self.found)

    def model_input(self, pixels):
        return ModelInput(np.array(pixels, copy=True), "Model input: check image.")


def instance(label: str, score: float, box: tuple[int, int, int, int]) -> Instance:
    plane = np.zeros((SIZE[1], SIZE[0]), dtype=bool)
    x0, y0, x1, y1 = box
    plane[y0:y1, x0:x1] = True
    part, origin = crop(plane)
    return Instance(label, score, tuple(float(v) for v in box), part, origin)


def workspace(root: Path):
    from defects.history import History
    from defects.models import Slot
    from defects.registry import MODELS
    from defects.service import Workspace

    found = [instance("hole", 0.9, (40, 40, 90, 80)), instance("stain", 0.4, (150, 100, 220, 160))]
    slots = {}
    for index, spec in enumerate(MODELS):
        slots[spec.key] = Slot(
            spec,
            predictor=Stub(found[: index + 1]),
            verification="Verified against the benchmark reference (1 images).",
        )
    cards = {
        spec.key: {
            "metric": "mask AP50",
            "value": 0.2,
            "interval": [0.1, 0.3],
            "seed": 0,
            "evaluator": "check",
            "source": "check",
            "field": "check",
            "default_threshold": {"value": 0.25},
        }
        for spec in MODELS
    }
    return Workspace(slots, History(root / "history"), cards)


def photos(root: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(0)
    pixels = rng.integers(60, 200, (SIZE[1] * 2, SIZE[0] * 2, 3)).astype(np.uint8)
    good = root / "photo.jpg"
    Image.fromarray(pixels).save(good, quality=90)
    bad = root / "broken.jpg"
    bad.write_bytes(b"not an image")
    return good, bad


def free_port() -> int:
    with socket.socket() as handle:
        handle.bind(("127.0.0.1", 0))
        return handle.getsockname()[1]


def plan(state: dict) -> list[tuple[str, callable]]:
    first, second = state["keys"][0], state["keys"][-1]
    every = list(CLASSES)
    return [
        ("load", lambda: ("",)),
        ("analyse_lock", lambda: ()),
        (
            "analyse",
            lambda: (str(state["bad"]), first, 0.25, every, "mask", False, state["session"], ""),
        ),
        (
            "analyse",
            lambda: (str(state["good"]), first, 0.25, every, "mask", False, state["session"], ""),
        ),
        ("analyse_unlock", lambda: ()),
        ("analyse_history", lambda: (state["session"],)),
        ("filter_threshold", lambda: (0.5, every, "mask", first, state["session"], state["entry"])),
        (
            "filter_classes",
            lambda: (0.25, ["hole"], "mask", first, state["session"], state["entry"]),
        ),
        ("style", lambda: ("circle", state["session"], state["entry"])),
        ("model", lambda: (second, state["session"], state["entry"])),
        ("report", lambda: (state["session"], state["entry"])),
        ("compare_lock", lambda: ()),
        ("compare", lambda: (str(state["good"]), False, "mask", state["session"])),
        ("compare_unlock", lambda: ()),
        ("compare_history", lambda: (state["session"],)),
        ("history_tab", lambda: (state["session"],)),
        ("history_refresh", lambda: (state["session"],)),
        ("history_pick", lambda: (state["session"], SimpleNamespace(index=0))),
        ("history_choose", lambda: (state["session"], state["entry"])),
        (
            "history_open",
            lambda: (state["session"], state["entry"], state["entry"], 0.5, every, second, "mask"),
        ),
        ("dashboard_tab", lambda: (state["session"],)),
        ("dashboard_refresh", lambda: (state["session"],)),
        ("clear_ask", lambda: ()),
        ("clear_cancel", lambda: ()),
        ("clear_confirm", lambda: (state["session"], state["entry"])),
    ]


def remember(name: str, args: tuple, result, state: dict) -> None:
    if name == "load":
        state["session"] = result
    if name == "analyse" and args[0] == str(state["good"]):
        state["entry"] = result[1]
        before, after = result[5]
        if np.asarray(before).shape != np.asarray(after).shape:
            raise AssertionError("before and after images differ in size")
        if not state["entry"]:
            raise AssertionError("a good photo produced no analysis entry")
    if name == "analyse" and args[0] == str(state["bad"]) and result[1] != "":
        raise AssertionError("a broken photo kept an analysis entry")


def check_handlers(demo, state: dict) -> list[str]:
    wires = {wire.name: wire for wire in demo.wires}
    planned = plan(state)
    missing = sorted(set(wires) - {name for name, _ in planned})
    failures = [f"{name}: no check scenario for this handler" for name in missing]
    for name, arguments in planned:
        wire = wires.get(name)
        if wire is None:
            failures.append(f"{name}: handler not found in the interface")
            continue
        args = arguments()
        try:
            result = wire.fn(*args)
            expected = len(wire.outputs)
            if expected > 1 and (not isinstance(result, tuple) or len(result) != expected):
                count = len(result) if isinstance(result, tuple) else 1
                raise AssertionError(f"returned {count} values for {expected} outputs")
            remember(name, args, result, state)
        except Exception as error:
            traceback.print_exc()
            failures.append(f"{name}: {type(error).__name__}: {error}")
    return failures


def check_launch(demo) -> list[str]:
    from defects.ui import launch_options

    try:
        demo.launch(
            server_name="127.0.0.1",
            server_port=free_port(),
            prevent_thread_lock=True,
            quiet=True,
            **launch_options(),
        )
        demo.close()
    except Exception as error:
        traceback.print_exc()
        return [f"launch: {type(error).__name__}: {error}"]
    return []


def run(launch: bool = True) -> list[str]:
    from defects.ui import build

    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        good, bad = photos(root)
        space = workspace(root)
        demo = build(space, [good])
        state = {"keys": list(space.slots), "good": good, "bad": bad, "session": "", "entry": ""}
        failures = check_handlers(demo, state)
        if launch and not failures:
            failures += check_launch(demo)
    return failures


def main() -> int:
    try:
        import gradio
    except ImportError:
        print("UI CHECK FAILED: gradio is not installed")
        return 1
    print(f"gradio {gradio.__version__}", flush=True)
    try:
        failures = run(launch="--no-launch" not in sys.argv)
    except Exception as error:
        traceback.print_exc()
        failures = [f"build: {type(error).__name__}: {error}"]
    if failures:
        print("UI CHECK FAILED")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("UI CHECK PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
