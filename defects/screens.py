from defects import card, summary
from defects.labels import CLASSES, COLORS
from defects.palette import hexcode, text_on
from defects.service import STRETCH_NOTE, Result
from defects.texts import HISTORY_NOTE, LIMITS

INSPECT = (
    "session",
    "entry",
    "status",
    "result_box",
    "tiles",
    "comparison",
    "details",
    "closeups",
    "model_input",
    "input_note",
    "png",
    "csv",
    "pdf",
    "report_note",
)
LIGHT = tuple(name for name in INSPECT if name not in ("model_input", "input_note"))
DEFECTS = {
    "needle mark": "A small puncture or row of punctures left by a sewing or knitting needle.",
    "hole": "An opening where yarns are broken or missing through the fabric.",
    "missing yarn": "A line where a warp or weft yarn is absent, often running across the fabric.",
    "stain": "A local discolouration such as oil, dirt or dye.",
}
HISTORY_EMPTY = "No analyses in this session yet. Results appear here after each Analyse."


def values(result: Result, session: str, entry: str, clear: bool, keep, visible) -> dict:
    if result.skipped:
        kept = dict.fromkeys(INSPECT, keep)
        kept.update(session=session, entry=entry)
        return kept
    if result.ok:
        pair = (result.display, result.annotated) if result.display is not None else None
        return {
            "session": session,
            "entry": result.entry,
            "status": summary.banner_html(result.verdict),
            "result_box": visible(True),
            "tiles": summary.tiles_html(result.tiles),
            "comparison": pair,
            "details": summary.table_html(result.rows),
            "closeups": list(result.closeups),
            "model_input": result.model_input,
            "input_note": result.input_note,
            "png": str(result.png) if result.png else None,
            "csv": str(result.csv) if result.csv else None,
            "pdf": None,
            "report_note": "",
        }
    status = summary.error_html(result.message)
    if clear:
        return {
            "session": session,
            "entry": "",
            "status": status,
            "result_box": visible(False),
            "tiles": "",
            "comparison": None,
            "details": "",
            "closeups": [],
            "model_input": None,
            "input_note": "",
            "png": None,
            "csv": None,
            "pdf": None,
            "report_note": "",
        }
    kept = dict.fromkeys(INSPECT, keep)
    kept.update(session=session, entry=entry, status=status)
    return kept


def pick(chosen: dict, names) -> tuple:
    return tuple(chosen[name] for name in names)


def model_text(workspace, key: str) -> str:
    slot = workspace.slots.get(key)
    if slot is None:
        return "Unknown model."
    entry = workspace.cards.get(key)
    return "\n\n".join(
        [
            f"**{slot.spec.name}**. {workspace.status(key)}",
            card.headline(entry),
            card.details(entry),
            f"Default confidence threshold: {workspace.threshold_for(key):.2f}.",
            f"**Limits.** {LIMITS}",
        ]
    )


def compare_values(workspace, results: list[tuple[str, Result]], sentence: str) -> list:
    out = []
    for key, result in results:
        name = workspace.slots[key].spec.name
        image = result.annotated if result.ok else None
        out.extend([image, summary.compare_html(name, result, result.tiles if result.ok else [])])
    note = (
        f"<div class='fd-compare-summary'><div>{summary.escape(sentence)}</div>"
        f"<div class='fd-help'>{summary.escape(summary.COMPARE_NOTE)}</div></div>"
    )
    out.append(note)
    return out


def history_values(cards: list[dict]) -> tuple[list, list, str]:
    gallery = [(item["thumb"], item["caption"]) for item in cards if item["thumb"]]
    choices = [(item["caption"], item["entry"]) for item in cards]
    empty = summary.empty_html("History is empty", HISTORY_EMPTY) if not cards else ""
    return gallery, choices, empty


def entry_at(cards: list[dict], index) -> str | None:
    shown = [item for item in cards if item["thumb"]]
    if isinstance(index, list | tuple):
        index = index[0] if index else None
    if not isinstance(index, int) or not 0 <= index < len(shown):
        return None
    return shown[index]["entry"]


def defect_lines() -> str:
    rows = []
    for name in CLASSES:
        color = COLORS[name]
        rows.append(
            f"<div class='fd-defect'><span class='fd-chip' style='background:{hexcode(color)};"
            f"color:{hexcode(text_on(color))}'>{summary.escape(name)}</span>"
            f"<span>{summary.escape(DEFECTS[name])}</span></div>"
        )
    return f"<div class='fd-defects'>{''.join(rows)}</div>"


def about_sections(workspace, branding) -> list[tuple[str, str, bool]]:
    models = "\n\n".join(model_text(workspace, key) for key in workspace.slots)
    team = "\n\n".join(branding.lines()) or "No team details were configured."
    return [
        (
            "What the app does",
            "The app runs fabric defect segmentation models on one photo. Each model marks every "
            "defect it finds with a mask, a box, the class and a confidence, and the app lists "
            "them. Every result is computed live when you press Analyse.",
            True,
        ),
        ("The four defect types", defect_lines(), True),
        ("The models and their validation figures", models, False),
        (
            "How verification works",
            "At start, each model is run on a small reference pack of validation images made by "
            "the benchmark's own code. The weights file must match by sha256, and every detection "
            "with confidence 0.30 or more must be reproduced with the same class, a box overlap "
            "of at least 0.90, a confidence within 0.02 and a mask area within 10 percent. A model "
            "that fails is switched off.",
            False,
        ),
        ("Limits", f"{LIMITS}\n\n{STRETCH_NOTE}\n\n{HISTORY_NOTE}", False),
        ("Team", team, False),
    ]
