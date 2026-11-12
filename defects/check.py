import argparse
import json
import shutil
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from defects import card, imageio, settings, verify
from defects.models import shown_path, weights_path
from defects.registry import MODELS

ROOT = Path(__file__).resolve().parents[1]
MIN_FREE_GB = 1.0
RUNS = 3


@dataclass
class Report:
    lines: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def add(self, text: str) -> None:
        self.lines.append(text)

    def fail(self, text: str) -> None:
        self.lines.append(f"PROBLEM: {text}")
        self.problems.append(text)

    @property
    def ok(self) -> bool:
        return not self.problems

    def text(self) -> str:
        verdict = "READY" if self.ok else f"NOT READY: {len(self.problems)} problems"
        return "\n".join([*self.lines, "", verdict])


def first_image(reference: Path, key: str) -> Path | None:
    pack = reference / key
    listing = pack / "detections.json"
    if listing.is_file():
        images = json.loads(listing.read_text(encoding="utf-8")).get("images") or []
        if images:
            return pack / "images" / images[0]["file"]
    found = sorted((pack / "images").glob("*")) if (pack / "images").is_dir() else []
    return found[0] if found else None


def timing(predictor, path: Path, side: int, runs: int = RUNS) -> float:
    _, work = imageio.prepare(path, side)
    seconds = []
    for _ in range(runs):
        started = time.perf_counter()
        predictor.predict(work)
        seconds.append(time.perf_counter() - started)
    return statistics.median(seconds)


def existing(path: Path) -> Path:
    path = Path(path).resolve()
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def weights_section(report: Report, specs, root: Path, folder: Path | None = None) -> None:
    report.add("Weights files")
    for spec in specs:
        path = weights_path(spec, root, folder)
        shown = shown_path(spec, folder)
        if not path.is_file():
            report.fail(f"{spec.name}: {shown} not found")
            continue
        size = path.stat().st_size / 2**20
        report.add(f"  {spec.name}: {shown}, {size:.1f} MB, sha256 {verify.sha256(path)}")


def models_section(report: Report, slots: dict, verdicts: dict) -> None:
    report.add("Models")
    for key, slot in slots.items():
        name = slot.spec.name
        if key in verdicts:
            verdict = verdicts[key]
            line = verdict.report(name).replace("\n", "\n  ")
            if verdict.status == verify.VERIFIED:
                report.add(f"  {line}")
            else:
                report.fail(line)
        elif not slot.ready:
            report.fail(f"{name}: not loaded, {slot.reason}")


def card_section(report: Report, specs, cards: dict) -> None:
    report.add("Model card")
    for spec in specs:
        entry = cards.get(spec.key)
        if not entry or entry.get("value") is None:
            report.fail(f"{spec.name}: no evaluated entry in model_card.json")
            continue
        chosen = card.threshold(entry)
        if chosen is None:
            report.fail(f"{spec.name}: no default threshold in model_card.json")
            continue
        report.add(f"  {spec.name}: {card.headline(entry)} Default threshold {chosen:.2f}.")


def disk_section(report: Report, history: Path) -> None:
    report.add("Disk")
    free = shutil.disk_usage(existing(history)).free / 2**30
    text = f"{free:.1f} GB free for the history folder {history}"
    if free < MIN_FREE_GB:
        report.fail(f"only {text}, need at least {MIN_FREE_GB:.0f} GB")
    else:
        report.add(f"  {text}")


def timing_section(report: Report, slots: dict, reference: Path) -> None:
    report.add(f"Inference time on CPU, median of {RUNS} runs on the first reference image")
    for key, slot in slots.items():
        if not slot.ready:
            continue
        path = first_image(reference, key)
        if path is None or not path.is_file():
            report.fail(f"{slot.spec.name}: no reference image to time")
            continue
        try:
            seconds = timing(slot.predictor, path, slot.spec.settings["imgsz"])
        except Exception as error:
            report.fail(f"{slot.spec.name}: inference failed, {type(error).__name__}: {error}")
            continue
        report.add(f"  {slot.spec.name}: {seconds:.2f} s ({path.name})")


def run(
    root: Path = ROOT,
    slots: dict | None = None,
    cards: dict | None = None,
    history: Path | None = None,
    specs=MODELS,
    folder: Path | None = None,
) -> Report:
    from defects.models import load_all

    reference = root / "reference"
    report = Report()
    weights_section(report, specs, root, folder)
    if slots is None:
        slots = load_all(specs, root, folder=folder)
    verdicts = verify.apply(slots, root, reference, quiet=True, folder=folder)
    models_section(report, slots, verdicts)
    card_section(report, specs, card.read(root / "model_card.json") if cards is None else cards)
    disk_section(report, history or root / "data" / "history")
    timing_section(report, slots, reference)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Readiness report for the demo")
    parser.add_argument("--root", type=Path, default=ROOT)
    settings.add_folders(parser)
    args = parser.parse_args()
    folder, data = settings.folders(args, args.root)
    report = run(args.root, history=data / "history", folder=folder)
    print(report.text())
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
