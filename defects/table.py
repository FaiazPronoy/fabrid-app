import csv
from pathlib import Path

from defects.labels import CLASSES
from defects.prediction import Instance

COLUMNS = (
    "number",
    "class",
    "confidence",
    "x0",
    "y0",
    "x1",
    "y1",
    "mask area (px)",
    "mask area (% of image)",
)


def rows(found: list[Instance], size: tuple[int, int]) -> list[list]:
    total = size[0] * size[1]
    out = []
    for number, item in enumerate(found, start=1):
        x0, y0, x1, y1 = (round(v) for v in item.box)
        share = 100 * item.area / total if total else 0.0
        out.append(
            [number, item.label, round(item.score, 3), x0, y0, x1, y1, item.area, round(share, 3)]
        )
    return out


def counts(found: list[Instance]) -> dict[str, int]:
    tally = dict.fromkeys(CLASSES, 0)
    for item in found:
        tally[item.label] += 1
    return tally


def write_csv(found: list[Instance], size: tuple[int, int], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        writer.writerows(rows(found, size))
    return path
