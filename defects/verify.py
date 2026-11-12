import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from defects import imageio, settings
from defects.labels import CLASSES
from defects.models import weights_path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reference"
MIN_SCORE = 0.30
MIN_IOU = 0.90
MAX_SCORE_DIFF = 0.02
MAX_AREA_SHARE = 0.10
VERIFIED = "verified"
FAILED = "failed"
MISSING = "missing"


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    box: tuple[float, float, float, float]
    mask_area: int

    def describe(self) -> str:
        box = ", ".join(f"{v:.0f}" for v in self.box)
        return f"{self.label} {self.score:.3f} at [{box}] area {self.mask_area}"


@dataclass
class Verdict:
    status: str
    message: str
    images: int = 0
    problems: dict[str, list[str]] = field(default_factory=dict)

    @property
    def text(self) -> str:
        if self.status == VERIFIED:
            return f"Verified against the benchmark reference ({self.images} images)."
        if self.status == MISSING:
            return "Not verified, no reference pack."
        return f"Failed verification against the benchmark reference: {self.message}"

    def report(self, name: str) -> str:
        lines = [f"{name}: {self.text}"]
        for image, problems in self.problems.items():
            lines.extend(f"  {image}: {problem}" for problem in problems)
        return "\n".join(lines)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def iou(a, b) -> float:
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def partners(reference: Detection, found: Detection) -> bool:
    return (
        reference.label == found.label
        and iou(reference.box, found.box) >= MIN_IOU
        and abs(reference.score - found.score) <= MAX_SCORE_DIFF
        and abs(reference.mask_area - found.mask_area) <= MAX_AREA_SHARE * reference.mask_area
    )


def closest(item: Detection, others: list[Detection]) -> str:
    same = [other for other in others if other.label == item.label]
    if not same:
        return f"no {item.label} detection on the other side"
    best = max(same, key=lambda other: iou(item.box, other.box))
    return (
        f"closest {item.label}: box IoU {iou(item.box, best.box):.3f}, "
        f"score difference {abs(item.score - best.score):.3f}, "
        f"mask area {best.mask_area} against {item.mask_area}"
    )


def compare(reference: list[Detection], found: list[Detection]) -> list[str]:
    problems = []
    for item in reference:
        if item.score >= MIN_SCORE and not any(partners(item, other) for other in found):
            problems.append(f"reference {item.describe()} not reproduced ({closest(item, found)})")
    for item in found:
        if item.score >= MIN_SCORE and not any(partners(other, item) for other in reference):
            problems.append(
                f"extra {item.describe()} not in reference ({closest(item, reference)})"
            )
    return problems


def detections(found) -> list[Detection]:
    return [Detection(i.label, i.score, tuple(i.box), i.area) for i in found]


def parse(item: dict) -> Detection:
    if item["label"] not in CLASSES:
        raise ValueError(f"unknown label {item['label']}")
    return Detection(
        item["label"], float(item["score"]), tuple(item["box"]), int(item["mask_area"])
    )


def verify(
    spec, predictor, root: Path = ROOT, reference: Path = REFERENCE, folder: Path | None = None
) -> Verdict:
    pack = reference / spec.key
    listing = pack / "detections.json"
    if not listing.is_file():
        return Verdict(MISSING, "no reference pack")
    try:
        data = json.loads(listing.read_text(encoding="utf-8"))
        if data.get("model_key") != spec.key:
            return Verdict(FAILED, f"the reference pack is for {data.get('model_key')}")
        weights = weights_path(spec, root, folder)
        if sha256(weights) != data["weights_sha256"]:
            return Verdict(FAILED, "the weights file is not the one the reference was made from")
        problems = {}
        for image in data["images"]:
            _, work = imageio.prepare(pack / "images" / image["file"], spec.settings["imgsz"])
            height, width = work.shape[:2]
            found = []
            if (width, height) != (image["width"], image["height"]):
                found_problems = [
                    f"analysis size {width} x {height} differs from reference "
                    f"{image['width']} x {image['height']}"
                ]
            else:
                found = detections(predictor.predict(work))
                expected = [parse(item) for item in image["detections"]]
                found_problems = compare(expected, found)
            if found_problems:
                problems[image["file"]] = found_problems
    except Exception as error:
        return Verdict(FAILED, f"the reference pack could not be checked: {error}")
    count = len(data["images"])
    if problems:
        failed = len(problems)
        return Verdict(FAILED, f"{failed} of {count} images differ", count, problems)
    if count == 0:
        return Verdict(FAILED, "the reference pack lists no images")
    return Verdict(VERIFIED, "ok", count)


def apply(
    slots: dict,
    root: Path = ROOT,
    reference: Path = REFERENCE,
    quiet: bool = False,
    folder: Path | None = None,
) -> dict[str, Verdict]:
    verdicts = {}
    for key, slot in slots.items():
        if not slot.ready:
            continue
        verdict = verify(slot.spec, slot.predictor, root, reference, folder)
        verdicts[key] = verdict
        slot.verification = verdict.text
        if verdict.status == FAILED:
            slot.predictor = None
            slot.reason = verdict.text
        if not quiet:
            print(verdict.report(slot.spec.name), flush=True)
    return verdicts


def main() -> int:
    from defects.models import load_all
    from defects.registry import MODELS

    parser = argparse.ArgumentParser(description="Check the models against reference packs")
    parser.add_argument("--root", type=Path, default=ROOT)
    settings.add_folders(parser)
    args = parser.parse_args()
    folder, _ = settings.folders(args, args.root)
    slots = load_all(MODELS, args.root, warm=False, folder=folder)
    verdicts = apply(slots, args.root, args.root / "reference", folder=folder)
    unloaded = [slot for slot in slots.values() if slot.reason and slot.spec.key not in verdicts]
    failed = [v for v in verdicts.values() if v.status == FAILED]
    return 1 if failed or unloaded else 0


if __name__ == "__main__":
    sys.exit(main())
