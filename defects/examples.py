from pathlib import Path

SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
LABEL = "Validation images from the dataset, from capture sessions not used in training"


def listing(reference: Path) -> list[Path]:
    chosen: dict[str, Path] = {}
    for path in sorted(Path(reference).glob("*/images/*")):
        if path.is_file() and path.suffix.lower() in SUFFIXES:
            chosen.setdefault(path.name, path)
    return [chosen[name] for name in sorted(chosen)]
