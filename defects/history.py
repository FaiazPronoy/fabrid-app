import json
import re
import secrets
import shutil
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from defects.labels import CLASSES
from defects.prediction import Instance, ModelInput, select
from defects.table import counts

SESSION_PATTERN = re.compile(r"^[0-9a-f]{32}$")
ENTRY_PATTERN = re.compile(r"^\d{8}-\d{6}-\d{6}-[0-9a-f]{6}$")
THUMB_SIDE = 256
MAX_AGE_HOURS = 24


@dataclass
class Analysis:
    model_key: str
    model_name: str
    seconds: float
    display: Image.Image
    work: np.ndarray
    found: list[Instance]
    model_input: ModelInput
    threshold: float
    classes: list[str] = field(default_factory=lambda: list(CLASSES))
    entry: str = ""
    created: str = ""
    reopened: bool = False
    stretch: bool = False

    @property
    def size(self) -> tuple[int, int]:
        return self.work.shape[1], self.work.shape[0]

    def shown(self) -> list[Instance]:
        return select(self.found, self.threshold, self.classes)


def session_id(candidate=None) -> str:
    if isinstance(candidate, str) and SESSION_PATTERN.match(candidate):
        return candidate
    return uuid.uuid4().hex


def new_entry(now: datetime | None = None) -> str:
    return f"{(now or datetime.now()):%Y%m%d-%H%M%S-%f}-{secrets.token_hex(3)}"


def pack(found: list[Instance]) -> dict[str, np.ndarray]:
    return {
        "labels": np.array([CLASSES.index(i.label) for i in found], dtype=np.int64),
        "scores": np.array([i.score for i in found], dtype=np.float64),
        "boxes": np.array([i.box for i in found], dtype=np.float64).reshape(-1, 4),
        "origins": np.array([i.origin for i in found], dtype=np.int64).reshape(-1, 2),
        "shapes": np.array([i.mask.shape for i in found], dtype=np.int64).reshape(-1, 2),
        "bits": np.concatenate([i.mask.ravel() for i in found] or [np.zeros(0, dtype=bool)]),
    }


def unpack(data) -> list[Instance]:
    found, start = [], 0
    bits = data["bits"].astype(bool)
    for label, score, box, origin, shape in zip(
        data["labels"], data["scores"], data["boxes"], data["origins"], data["shapes"], strict=True
    ):
        size = int(shape[0] * shape[1])
        mask = bits[start : start + size].reshape(int(shape[0]), int(shape[1]))
        start += size
        found.append(
            Instance(
                label=CLASSES[int(label)],
                score=float(score),
                box=tuple(float(v) for v in box),
                mask=mask,
                origin=(int(origin[0]), int(origin[1])),
            )
        )
    return found


class History:
    def __init__(self, root: Path):
        self.root = Path(root)

    def folder(self, session: str, entry: str | None = None) -> Path:
        if not SESSION_PATTERN.match(session or ""):
            raise ValueError("invalid session id")
        if entry is None:
            return self.root / session
        if not ENTRY_PATTERN.match(entry or ""):
            raise ValueError("invalid history entry id")
        return self.root / session / entry

    def meta(self, analysis: Analysis) -> dict:
        shown = analysis.shown()
        return {
            "entry": analysis.entry,
            "created": analysis.created,
            "model_key": analysis.model_key,
            "model_name": analysis.model_name,
            "seconds": analysis.seconds,
            "size": list(analysis.size),
            "threshold": analysis.threshold,
            "classes": list(analysis.classes),
            "counts": counts(shown),
            "total": len(shown),
            "input_note": analysis.model_input.note,
            "stretch": analysis.stretch,
        }

    def save(self, session: str, analysis: Analysis) -> dict:
        if not analysis.entry:
            analysis.entry = new_entry()
        if not analysis.created:
            analysis.created = datetime.now().isoformat(timespec="seconds")
        folder = self.folder(session, analysis.entry)
        partial = folder.with_name(folder.name + ".partial")
        shutil.rmtree(partial, ignore_errors=True)
        partial.mkdir(parents=True)
        Image.fromarray(analysis.work).save(partial / "work.png")
        analysis.display.save(partial / "display.jpg", quality=92)
        thumb = analysis.display.copy()
        thumb.thumbnail((THUMB_SIDE, THUMB_SIDE))
        thumb.save(partial / "thumb.jpg", quality=85)
        if analysis.model_input.pixels is not None:
            Image.fromarray(analysis.model_input.pixels).save(partial / "input.png")
        np.savez_compressed(partial / "instances.npz", **pack(analysis.found))
        meta = self.meta(analysis)
        (partial / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        partial.replace(folder)
        return meta

    def update(self, session: str, analysis: Analysis) -> dict:
        folder = self.folder(session, analysis.entry)
        meta = self.meta(analysis)
        (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    def entries(self, session: str) -> list[dict]:
        base = self.folder(session)
        if not base.is_dir():
            return []
        found = []
        for meta_path in base.glob("*/meta.json"):
            try:
                found.append(json.loads(meta_path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        return sorted(found, key=lambda m: m["entry"], reverse=True)

    def thumb(self, session: str, entry: str) -> Path:
        return self.folder(session, entry) / "thumb.jpg"

    def found(self, session: str, entry: str) -> list[Instance]:
        with np.load(self.folder(session, entry) / "instances.npz", allow_pickle=False) as data:
            return unpack(data)

    def open(self, session: str, entry: str) -> Analysis:
        folder = self.folder(session, entry)
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        with Image.open(folder / "work.png") as image:
            work = np.asarray(image.convert("RGB"))
        with Image.open(folder / "display.jpg") as image:
            display = image.convert("RGB")
        pixels = None
        if (folder / "input.png").is_file():
            with Image.open(folder / "input.png") as image:
                pixels = np.asarray(image.convert("RGB"))
        return Analysis(
            model_key=meta["model_key"],
            model_name=meta["model_name"],
            seconds=meta["seconds"],
            display=display,
            work=work,
            found=self.found(session, entry),
            model_input=ModelInput(pixels, meta["input_note"]),
            threshold=meta["threshold"],
            classes=meta["classes"],
            entry=meta["entry"],
            created=meta["created"],
            stretch=bool(meta.get("stretch", False)),
        )

    def clear(self, session: str) -> int:
        folder = self.folder(session)
        if not folder.is_dir():
            return 0
        count = len(list(folder.glob("*/meta.json")))
        shutil.rmtree(folder)
        return count

    def prune(self, max_age_hours: float = MAX_AGE_HOURS, now: float | None = None) -> int:
        if not self.root.is_dir():
            return 0
        limit = (now if now is not None else time.time()) - max_age_hours * 3600
        removed = 0
        for folder in self.root.iterdir():
            if not folder.is_dir() or not SESSION_PATTERN.match(folder.name):
                continue
            stamps = [path.stat().st_mtime for path in folder.rglob("*")]
            newest = max(stamps, default=folder.stat().st_mtime)
            if newest < limit:
                shutil.rmtree(folder)
                removed += 1
        return removed
