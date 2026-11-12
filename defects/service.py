import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from defects import card, charts, closeups, imageio, marks, report, summary
from defects.branding import Branding
from defects.draw import annotate
from defects.history import Analysis, History, session_id
from defects.labels import CLASSES
from defects.prediction import ModelInput
from defects.table import counts, rows, write_csv

KEPT_SESSIONS = 32
STRETCH_LABEL = "Contrast stretch (approximation of the training export)"
STRETCH_NOTE = (
    "Rescales brightness between the 2nd and 98th percentile of the photo before inference. "
    "The training images were contrast stretched by Roboflow with unknown settings; this "
    "approximation has not been validated. Applies at the next Analyse."
)


@dataclass
class Result:
    ok: bool
    message: str
    entry: str | None = None
    display: Image.Image | None = None
    annotated: np.ndarray | None = None
    rows: list[list] = field(default_factory=list)
    model_input: np.ndarray | None = None
    input_note: str = ""
    png: Path | None = None
    csv: Path | None = None
    threshold: float | None = None
    classes: list[str] | None = None
    model_key: str | None = None
    light: bool = False
    skipped: bool = False
    verdict: dict | None = None
    tiles: list[dict] = field(default_factory=list)
    closeups: list[tuple[np.ndarray, str]] = field(default_factory=list)
    style: str | None = None


def clean_classes(classes) -> list[str]:
    chosen = set(classes or [])
    return [name for name in CLASSES if name in chosen]


def clamp(threshold) -> float:
    return min(1.0, max(0.0, float(threshold)))


def stretch_text(used: bool) -> str:
    if used:
        return "Contrast stretch before inference: on (approximation, not validated)."
    return "Contrast stretch before inference: off."


def text_summary(analysis: Analysis) -> str:
    shown = analysis.shown()
    tally = counts(shown)
    threshold = f"{analysis.threshold:.2f}"
    lines = []
    if not analysis.classes:
        lines.append("**No classes selected.** Tick at least one class to see detections.")
    elif not shown:
        lines.append(f"**No defect found above confidence {threshold}** in the selected classes.")
    else:
        listed = ", ".join(f"{name} {n}" for name, n in tally.items() if n)
        noun = "defect" if len(shown) == 1 else "defects"
        lines.append(f"**{len(shown)} {noun} found** above confidence {threshold}: {listed}.")
    if analysis.classes and len(analysis.classes) < len(CLASSES):
        lines.append(f"Showing only: {', '.join(analysis.classes)}.")
    width, height = analysis.size
    if analysis.reopened:
        lines.append(
            f"Reopened from history, analysed at {analysis.created} with {analysis.model_name}. "
            f"No new inference was run. Inference time then: {analysis.seconds:.2f} s."
        )
    else:
        lines.append(
            f"{analysis.model_name}: inference time {analysis.seconds:.2f} s on CPU, "
            f"analysis size {width} x {height} px."
        )
    lines.append(stretch_text(analysis.stretch))
    lines.append("")
    lines.append("| class | count |")
    lines.append("|---|---|")
    lines.extend(f"| {name} | {n} |" for name, n in tally.items())
    lines.append(f"| total | {len(shown)} |")
    return "\n".join(lines)


class Workspace:
    def __init__(self, slots: dict, history: History, cards: dict, branding=None):
        self.slots = slots
        self.history = history
        self.cards = cards
        self.branding = branding or Branding()
        self.current: dict[str, Analysis] = {}

    @staticmethod
    def session(candidate=None) -> str:
        return session_id(candidate)

    def remember(self, session: str, analysis: Analysis) -> None:
        self.current.pop(session, None)
        self.current[session] = analysis
        while len(self.current) > KEPT_SESSIONS:
            self.current.pop(next(iter(self.current)))

    def choices(self) -> list[tuple[str, str]]:
        out = []
        for key, slot in self.slots.items():
            label = slot.spec.name if slot.ready else f"{slot.spec.name} (unavailable)"
            out.append((label, key))
        return out

    def default_key(self) -> str:
        return next(iter(self.slots))

    def status(self, key: str) -> str:
        slot = self.slots.get(key)
        if slot is None:
            return "Unknown model."
        if not slot.ready:
            return f"Unavailable: {slot.reason}."
        return f"Loaded. {slot.verification or 'Not verified.'}"

    def info(self, key: str) -> str:
        slot = self.slots.get(key)
        if slot is None:
            return "Unknown model."
        entry = self.cards.get(key)
        return f"**{slot.spec.name}**. {self.status(key)}\n\n{card.headline(entry)}"

    def threshold_for(self, key: str) -> float:
        chosen = card.threshold(self.cards.get(key))
        if chosen is not None:
            return chosen
        slot = self.slots.get(key)
        return slot.spec.default_threshold if slot else 0.5

    def analyse(
        self, session: str, source, key: str, threshold, classes, stretch=False, style=None
    ) -> Result:
        slot = self.slots.get(key)
        if slot is None:
            return Result(False, "Choose a model first.")
        if not slot.ready:
            return Result(False, f"{slot.spec.name} is unavailable: {slot.reason}.")
        try:
            display, work = imageio.prepare(source, slot.spec.settings["imgsz"])
        except imageio.ImageError as error:
            return Result(False, str(error))
        fed = imageio.stretch(work) if stretch else work
        try:
            started = time.perf_counter()
            found = slot.predictor.predict(fed)
            seconds = time.perf_counter() - started
        except Exception as error:
            traceback.print_exc()
            return Result(False, f"The model failed on this image: {type(error).__name__}: {error}")
        try:
            model_input = slot.predictor.model_input(fed)
        except Exception as error:
            model_input = ModelInput(None, f"Model input view not available: {error}")
        if stretch:
            model_input = ModelInput(
                model_input.pixels,
                "Contrast stretched (2nd to 98th percentile, approximation) before the model "
                f"step. {model_input.note}",
            )
        analysis = Analysis(
            model_key=key,
            model_name=slot.spec.name,
            seconds=seconds,
            display=display,
            work=work,
            found=found,
            model_input=model_input,
            threshold=clamp(threshold),
            classes=clean_classes(classes),
            stretch=bool(stretch),
        )
        try:
            self.history.save(session, analysis)
        except Exception as error:
            traceback.print_exc()
            return Result(False, f"The result could not be saved: {error}")
        self.remember(session, analysis)
        return self.render(session, analysis, style=style)

    def lookup(self, session: str, entry: str | None) -> Analysis | None:
        if not entry:
            return None
        analysis = self.current.get(session)
        if analysis is not None and analysis.entry == entry:
            return analysis
        analysis = self.history.open(session, entry)
        self.remember(session, analysis)
        return analysis

    def refilter(
        self,
        session: str,
        entry: str | None,
        threshold,
        classes,
        model_key: str | None = None,
        style=None,
    ) -> Result:
        try:
            analysis = self.lookup(session, entry)
        except Exception as error:
            return Result(False, f"The stored analysis could not be read: {error}")
        if analysis is None:
            return Result(False, "Analyse a photo first.")
        if model_key is not None and model_key != analysis.model_key:
            return Result(True, "", entry=analysis.entry, light=True, skipped=True)
        analysis.threshold = clamp(threshold)
        analysis.classes = clean_classes(classes)
        try:
            self.history.update(session, analysis)
        except Exception:
            traceback.print_exc()
        return self.render(session, analysis, light=True, style=style)

    def restyle(self, session: str, entry: str | None, style: str) -> Result:
        try:
            analysis = self.lookup(session, entry)
        except Exception as error:
            return Result(False, f"The stored analysis could not be read: {error}")
        if analysis is None:
            return Result(False, "Analyse a photo first.")
        return self.render(session, analysis, light=True, style=style)

    def reopen(self, session: str, entry: str | None, style=None) -> Result:
        try:
            analysis = self.history.open(session, entry or "")
        except Exception as error:
            return Result(False, f"That history entry could not be opened: {error}")
        analysis.reopened = True
        self.remember(session, analysis)
        return self.render(session, analysis, style=style)

    def draw(self, analysis: Analysis, style=None) -> np.ndarray:
        shown = analysis.shown()
        if style is None:
            return annotate(analysis.work, shown)
        return marks.render(analysis.display, shown, analysis.size, style)

    def render(self, session: str, analysis: Analysis, light: bool = False, style=None) -> Result:
        shown = analysis.shown()
        annotated = self.draw(analysis, style)
        exports = self.history.folder(session, analysis.entry) / "exports"
        exports.mkdir(parents=True, exist_ok=True)
        png = exports / "annotated.png"
        Image.fromarray(annotated).save(png)
        csv_path = write_csv(shown, analysis.size, exports / "results.csv")
        return Result(
            ok=True,
            message=text_summary(analysis),
            entry=analysis.entry,
            display=None if light and style is None else analysis.display,
            annotated=annotated,
            rows=rows(shown, analysis.size),
            model_input=None if light else analysis.model_input.pixels,
            input_note="" if light else analysis.model_input.note,
            png=png,
            csv=csv_path,
            threshold=analysis.threshold,
            classes=list(analysis.classes),
            model_key=analysis.model_key,
            light=light,
            verdict=summary.verdict(analysis),
            tiles=summary.tiles(shown, analysis.classes),
            closeups=[]
            if style is None
            else closeups.crops(analysis.display, shown, analysis.size),
            style=style,
        )

    def report(self, session: str, entry: str | None) -> tuple[Path | None, str]:
        try:
            analysis = self.lookup(session, entry)
        except Exception as error:
            return None, f"The stored analysis could not be read: {error}"
        if analysis is None:
            return None, "Analyse a photo first."
        folder = self.history.folder(session, analysis.entry) / "exports"
        try:
            path = report.write(
                analysis,
                card.headline(self.cards.get(analysis.model_key)),
                folder / "report.pdf",
                branding=self.branding,
            )
        except Exception as error:
            traceback.print_exc()
            return None, f"The PDF report could not be created: {error}"
        return path, "PDF report ready."

    def listing(self, session: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        choices, gallery = [], []
        for meta in self.history.entries(session):
            when = meta["created"].replace("T", " ")
            stretched = ", contrast stretch" if meta.get("stretch") else ""
            label = (
                f"{when} | {meta['model_name']} | threshold {meta['threshold']:.2f} | "
                f"{meta['total']} shown{stretched}"
            )
            choices.append((label, meta["entry"]))
            thumb = self.history.thumb(session, meta["entry"])
            if thumb.is_file():
                gallery.append((str(thumb), f"{when}, {meta['total']} shown"))
        return choices, gallery

    def dashboard(self, session: str):
        records = charts.collect(self.history, session)
        total = sum(len(record.shown) for record in records)
        text = (
            f"{len(records)} analyses in this session, {total} detections shown at the "
            "threshold and classes each analysis was last viewed with."
        )
        return (
            text,
            charts.class_chart(records),
            charts.image_chart(records),
            charts.confidence_chart(records),
        )

    def compare(self, session: str, source, stretch=False, style=marks.DEFAULT_STYLE):
        results = []
        for key, slot in self.slots.items():
            if not slot.ready:
                results.append(
                    (key, Result(False, f"{slot.spec.name} is unavailable: {slot.reason}."))
                )
                continue
            threshold = self.threshold_for(key)
            results.append(
                (key, self.analyse(session, source, key, threshold, CLASSES, stretch, style))
            )
        tallies = [
            (self.slots[key].spec.name, {tile["name"]: tile["count"] for tile in result.tiles})
            for key, result in results
            if result.ok
        ]
        return results, summary.compare_sentence(tallies)

    def slider_for(self, session: str, entry: str | None, key: str) -> float:
        analysis = self.current.get(session)
        if analysis is not None and entry and analysis.entry == entry and analysis.model_key == key:
            return analysis.threshold
        return self.threshold_for(key)

    def chips(self) -> list[dict]:
        return [summary.chip(slot) for slot in self.slots.values()]

    def picker(self) -> list[tuple[str, str]]:
        out = []
        for key, slot in self.slots.items():
            state = summary.chip(slot)["label"]
            entry = self.cards.get(key) or {}
            figure = (
                f"validation mask AP50 {entry['value']:.3f}"
                if entry.get("value") is not None
                else "not evaluated yet"
            )
            out.append((f"{slot.spec.name}  |  {state}  |  {figure}", key))
        return out

    def cards_for(self, session: str) -> list[dict]:
        out = []
        for meta in self.history.entries(session):
            thumb = self.history.thumb(session, meta["entry"])
            out.append(
                {
                    "entry": meta["entry"],
                    "thumb": str(thumb) if thumb.is_file() else None,
                    "caption": summary.card_caption(meta),
                }
            )
        return out

    def clear(self, session: str) -> int:
        self.current.pop(session, None)
        return self.history.clear(session)

    def stats(self, session: str) -> list[tuple[str, str]]:
        return summary.stat_tiles(charts.collect(self.history, session))


def cleanup(history: History) -> str:
    try:
        removed = history.prune()
    except Exception as error:
        return f"History cleanup failed, continuing: {type(error).__name__}: {error}"
    return f"History cleanup: removed {removed} sessions older than 24 hours."
