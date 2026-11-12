from html import escape

from defects.labels import CLASSES, COLORS
from defects.palette import VERDICTS, hexcode, text_on
from defects.table import counts

COMPARE_NOTE = (
    "This compares the outputs of the models on one photo. It is not an accuracy measurement."
)
EMPTY_TITLE = "No result yet"
EMPTY_TEXT = "Choose a photo or an example on the left, pick a model and press Analyse."


def noun(count: int) -> str:
    return "defect" if count == 1 else "defects"


def verdict(analysis) -> dict:
    shown = analysis.shown()
    threshold = f"{analysis.threshold:.2f}"
    if not analysis.classes:
        kind, title = "clear", "No classes selected"
    elif not shown:
        kind, title = "clear", f"No defect found above confidence {threshold}"
    else:
        kind, title = "found", f"{len(shown)} {noun(len(shown))} found above confidence {threshold}"
    width, height = analysis.size
    details = [
        analysis.model_name,
        f"inference {analysis.seconds:.2f} s on CPU",
        f"analysis size {width} x {height} px",
    ]
    notes = []
    if analysis.classes and len(analysis.classes) < len(CLASSES):
        notes.append(f"Showing only: {', '.join(analysis.classes)}.")
    if analysis.reopened:
        notes.append(f"Reopened from history, analysed at {analysis.created.replace('T', ' ')}.")
        notes.append("No new inference was run.")
    if analysis.stretch:
        notes.append("Contrast stretch was on (approximation, not validated).")
    return {"kind": kind, "title": title, "detail": " | ".join(details), "notes": notes}


def tiles(found, classes=CLASSES) -> list[dict]:
    tally = counts(found)
    chosen = set(classes)
    return [
        {
            "name": name,
            "count": tally[name],
            "color": hexcode(COLORS[name]),
            "text": hexcode(text_on(COLORS[name])),
            "muted": tally[name] == 0 or name not in chosen,
        }
        for name in CLASSES
    ]


def compare_sentence(results: list[tuple[str, dict[str, int]]]) -> str:
    if len(results) < 2:
        return "Only one model could run, so there is nothing to compare."
    agree, differ = [], []
    for name in CLASSES:
        values = [tally[name] for _, tally in results]
        if len(set(values)) == 1:
            agree.append(f"{name} ({values[0]})")
        else:
            parts = ", ".join(f"{model} {tally[name]}" for model, tally in results)
            differ.append(f"{name} ({parts})")
    if not differ:
        return f"The models report the same count for every class: {', '.join(agree)}."
    sentence = f"The counts differ for {'; '.join(differ)}."
    if agree:
        sentence = f"The counts agree for {', '.join(agree)}. " + sentence
    return sentence


def card_caption(meta: dict) -> str:
    when = meta["created"].replace("T", " ")[:16]
    total = meta["total"]
    found = [f"{name} {n}" for name, n in meta["counts"].items() if n]
    result = f"{total} {noun(total)}: {', '.join(found)}" if found else "no defect shown"
    return f"{when} | {meta['model_name']} | {result} at {meta['threshold']:.2f}"


def stat_tiles(records) -> list[tuple[str, str]]:
    tally = dict.fromkeys(CLASSES, 0)
    for record in records:
        for item in record.shown:
            tally[item.label] += 1
    total = sum(tally.values())
    top = max(tally.values(), default=0)
    frequent = ", ".join(name for name, n in tally.items() if n == top) if top else "none"
    return [
        ("Analyses", str(len(records))),
        ("Defects shown", str(total)),
        ("Most frequent class", frequent),
    ]


def chip(slot) -> dict:
    if not slot.ready:
        return {
            "name": slot.spec.name,
            "label": "Unavailable",
            "kind": "bad",
            "reason": slot.reason or "not loaded",
        }
    text = slot.verification or "Not verified."
    if text.startswith("Verified"):
        return {"name": slot.spec.name, "label": "Verified", "kind": "ok", "reason": text}
    return {"name": slot.spec.name, "label": "Not verified", "kind": "warn", "reason": text}


def banner_html(result: dict) -> str:
    colors = VERDICTS[result["kind"]]
    notes = "".join(f"<div class='fd-note'>{escape(note)}</div>" for note in result["notes"])
    return (
        f"<div class='fd-verdict fd-{result['kind']}' style='background:{hexcode(colors['fill'])};"
        f"border-color:{hexcode(colors['edge'])};color:{hexcode(colors['text'])}'>"
        f"<div class='fd-verdict-title'>{escape(result['title'])}</div>"
        f"<div class='fd-verdict-detail'>{escape(result['detail'])}</div>{notes}</div>"
    )


def error_html(message: str) -> str:
    colors = VERDICTS["error"]
    return (
        f"<div class='fd-verdict fd-error' style='background:{hexcode(colors['fill'])};"
        f"border-color:{hexcode(colors['edge'])};color:{hexcode(colors['text'])}'>"
        "<div class='fd-verdict-title'>The analysis could not be completed</div>"
        f"<div class='fd-verdict-detail'>{escape(message)}</div></div>"
    )


def empty_html(title: str = EMPTY_TITLE, text: str = EMPTY_TEXT) -> str:
    return (
        f"<div class='fd-empty'><div class='fd-empty-title'>{escape(title)}</div>"
        f"<div class='fd-empty-text'>{escape(text)}</div></div>"
    )


def tiles_html(items: list[dict]) -> str:
    cells = []
    for item in items:
        style = "" if item["muted"] else f" style='background:{item['color']};color:{item['text']}'"
        muted = " fd-muted" if item["muted"] else ""
        cells.append(
            f"<div class='fd-tile{muted}'{style}><div class='fd-tile-count'>{item['count']}</div>"
            f"<div class='fd-tile-name'>{escape(item['name'])}</div></div>"
        )
    return f"<div class='fd-tiles'>{''.join(cells)}</div>"


def table_html(listed: list[list]) -> str:
    if not listed:
        return "<div class='fd-help'>No defect is shown at the current settings.</div>"
    head = (
        "<tr><th>#</th><th>class</th><th>confidence</th><th>box x0, y0, x1, y1</th>"
        "<th>mask area (px)</th><th>share of image</th></tr>"
    )
    rows = []
    for number, name, score, x0, y0, x1, y1, area, share in listed:
        color = COLORS[name]
        swatch = (
            f"<span class='fd-chip' style='background:{hexcode(color)};"
            f"color:{hexcode(text_on(color))}'>{escape(name)}</span>"
        )
        rows.append(
            f"<tr><td>{number}</td><td>{swatch}</td><td>{score:.3f}</td>"
            f"<td>{x0}, {y0}, {x1}, {y1}</td><td>{area}</td><td>{share:.2f}%</td></tr>"
        )
    return (
        "<div class='fd-table-wrap'><table class='fd-table'>"
        f"<thead>{head}</thead><tbody>{''.join(rows)}</tbody></table></div>"
        "<div class='fd-help'>Coordinates and areas are in analysis pixels.</div>"
    )


def chips_html(items: list[dict]) -> str:
    cells = []
    for item in items:
        cells.append(
            f"<details class='fd-status fd-{item['kind']}' title='{escape(item['reason'])}'>"
            f"<summary>{escape(item['name'])}: {escape(item['label'])}</summary>"
            f"<div class='fd-status-reason'>{escape(item['reason'])}</div></details>"
        )
    return f"<div class='fd-statuses'>{''.join(cells)}</div>"


def header_html(branding, items: list[dict]) -> str:
    lines = "".join(f"<div class='fd-brand-line'>{escape(line)}</div>" for line in branding.lines())
    subtitle = (
        f"<div class='fd-subtitle'>{escape(branding.subtitle)}</div>" if branding.subtitle else ""
    )
    return (
        "<div class='fd-header'><div class='fd-brand'>"
        f"<div class='fd-title'>{escape(branding.title)}</div>{subtitle}{lines}</div>"
        f"{chips_html(items)}</div>"
    )


def stat_html(items: list[tuple[str, str]]) -> str:
    cells = "".join(
        f"<div class='fd-stat'><div class='fd-stat-value'>{escape(value)}</div>"
        f"<div class='fd-stat-name'>{escape(name)}</div></div>"
        for name, value in items
    )
    return f"<div class='fd-stats'>{cells}</div>"


def step_html(number: int, title: str, helper: str) -> str:
    return (
        f"<div class='fd-step'><span class='fd-step-number'>{number}</span>"
        f"<span class='fd-step-title'>{escape(title)}</span></div>"
        f"<div class='fd-help'>{escape(helper)}</div>"
    )


def compare_html(name: str, result, chosen_tiles: list[dict] | None) -> str:
    if result is None or not result.ok:
        message = result.message if result is not None else "not run"
        return f"<div class='fd-compare-head'>{escape(name)}</div>{error_html(message)}"
    return (
        f"<div class='fd-compare-head'>{escape(name)}</div>{banner_html(result.verdict)}"
        f"{tiles_html(chosen_tiles or [])}"
    )
