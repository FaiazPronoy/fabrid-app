import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch

from defects import marks, summary
from defects.branding import Branding
from defects.history import Analysis
from defects.palette import ACCENT, MUTED, TEXT, VERDICTS, hexcode
from defects.table import rows
from defects.texts import LIMITS

A4 = (8.27, 11.69)
ROWS_PER_PAGE = 40
HEADERS = ("no.", "class", "confidence", "x0", "y0", "x1", "y1", "area (px)", "area (%)")
WIDTHS = (0.06, 0.18, 0.12, 0.09, 0.09, 0.09, 0.09, 0.14, 0.14)
LEFT = 0.07


def stretch_line(used: bool) -> str:
    if used:
        return "Contrast stretch before inference: on (approximation, not validated)"
    return "Contrast stretch before inference: off"


def heading(analysis: Analysis, headline: str, now: datetime) -> list[str]:
    width, height = analysis.size
    return [
        f"Model: {analysis.model_name}",
        f"Confidence threshold: {analysis.threshold:.2f}",
        f"Classes shown: {', '.join(analysis.classes) or 'none'}",
        f"Analysed: {analysis.created.replace('T', ' ')}",
        f"Report created: {now:%Y-%m-%d %H:%M:%S}",
        f"Inference time: {analysis.seconds:.2f} s on CPU, analysis size {width} x {height} px",
        stretch_line(analysis.stretch),
        headline,
    ]


def color(rgb) -> tuple[float, float, float]:
    return tuple(v / 255 for v in rgb)


TITLE_CHARS = 52
SUBTITLE_CHARS = 100
LINE_CHARS = 112


def header_lines(branding: Branding) -> list[tuple[str, float, str, tuple]]:
    lines = [(part, 17, "bold", TEXT) for part in textwrap.wrap(branding.title, TITLE_CHARS)]
    for part in textwrap.wrap(branding.subtitle, SUBTITLE_CHARS):
        lines.append((part, 9.5, "normal", MUTED))
    for line in branding.lines():
        lines.extend((part, 8.5, "normal", MUTED) for part in textwrap.wrap(line, LINE_CHARS))
    return lines


def header(figure: Figure, branding: Branding) -> float:
    top = 0.965
    for text, size, weight, ink in header_lines(branding):
        step = size / 72 / A4[1] * 1.35
        top -= step
        figure.text(LEFT, top, text, fontsize=size, weight=weight, color=color(ink))
    top -= 0.008
    figure.add_artist(
        FancyBboxPatch(
            (LEFT, top - 0.004),
            0.86,
            0.0015,
            boxstyle="square,pad=0",
            transform=figure.transFigure,
            color=color(ACCENT),
        )
    )
    return top - 0.02


def banner(figure: Figure, verdict: dict, top: float) -> float:
    colors = VERDICTS[verdict["kind"]]
    figure.add_artist(
        FancyBboxPatch(
            (LEFT, top - 0.05),
            0.86,
            0.048,
            boxstyle="round,pad=0.004",
            transform=figure.transFigure,
            facecolor=color(colors["fill"]),
            edgecolor=color(colors["edge"]),
            linewidth=1.2,
        )
    )
    figure.text(
        LEFT + 0.015,
        top - 0.02,
        verdict["title"],
        fontsize=13,
        weight="bold",
        color=color(colors["text"]),
    )
    figure.text(
        LEFT + 0.015, top - 0.04, verdict["detail"], fontsize=8.5, color=color(colors["text"])
    )
    return top - 0.07


def tile_row(figure: Figure, tiles: list[dict], top: float) -> float:
    width = 0.86 / len(tiles)
    for index, tile in enumerate(tiles):
        x = LEFT + index * width
        fill = (0.93, 0.945, 0.96) if tile["muted"] else tile["color"]
        ink = (0.37, 0.41, 0.46) if tile["muted"] else tile["text"]
        figure.add_artist(
            FancyBboxPatch(
                (x + 0.005, top - 0.055),
                width - 0.01,
                0.05,
                boxstyle="round,pad=0.003",
                transform=figure.transFigure,
                facecolor=fill,
                edgecolor="none",
            )
        )
        figure.text(
            x + width / 2,
            top - 0.027,
            str(tile["count"]),
            fontsize=15,
            weight="bold",
            ha="center",
            color=ink,
        )
        figure.text(x + width / 2, top - 0.047, tile["name"], fontsize=8, ha="center", color=ink)
    return top - 0.07


def first_page(analysis: Analysis, headline: str, now: datetime, branding: Branding) -> Figure:
    figure = Figure(figsize=A4)
    top = header(figure, branding)
    top = banner(figure, summary.verdict(analysis), top)
    shown = analysis.shown()
    annotated = marks.render(analysis.display, shown, analysis.size, marks.DEFAULT_STYLE)
    image_top = top - 0.01
    for position, (title, pixels) in enumerate(
        (("Original", np.asarray(analysis.display)), ("Detected defects", annotated))
    ):
        ax = figure.add_axes((LEFT + position * 0.44, image_top - 0.25, 0.42, 0.24))
        ax.imshow(pixels)
        ax.set_title(title, fontsize=9.5, color=color(TEXT))
        ax.axis("off")
    top = tile_row(figure, summary.tiles(shown, analysis.classes), image_top - 0.265)
    lines = heading(analysis, headline, now)
    for index, line in enumerate(lines[:-1]):
        figure.text(LEFT, top - index * 0.018, line, fontsize=8.5, color=color(TEXT), va="top")
    card_top = top - (len(lines) - 1) * 0.018 - 0.004
    wrapped = textwrap.fill(lines[-1], 110)
    figure.text(LEFT, card_top, wrapped, fontsize=8.5, color=color(TEXT), va="top")
    limits_top = card_top - 0.05
    figure.text(LEFT, limits_top, "Limits", fontsize=10, weight="bold", color=color(TEXT), va="top")
    figure.text(
        LEFT, limits_top - 0.02, LIMITS, fontsize=8, wrap=True, va="top", color=color(MUTED)
    )
    return figure


def table_page(chunk: list[list], page: int, pages: int, branding: Branding) -> Figure:
    figure = Figure(figsize=A4)
    figure.text(LEFT, 0.955, branding.title, fontsize=10, color=color(MUTED))
    figure.text(
        LEFT,
        0.93,
        f"Detections (page {page} of {pages})",
        fontsize=13,
        weight="bold",
        color=color(TEXT),
    )
    ax = figure.add_axes((0.05, 0.05, 0.9, 0.86))
    ax.axis("off")
    if not chunk:
        ax.text(0.0, 0.98, "No defect found above the threshold.", fontsize=10, va="top")
        return figure
    table = ax.table(
        cellText=[[str(v) for v in row] for row in chunk],
        colLabels=list(HEADERS),
        colWidths=list(WIDTHS),
        loc="upper center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.3)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor(hexcode((226, 231, 238)))
        if row == 0:
            cell.set_facecolor(hexcode((238, 241, 245)))
            cell.set_text_props(weight="bold")
    return figure


def write(
    analysis: Analysis,
    headline: str,
    path: Path,
    now: datetime | None = None,
    branding: Branding | None = None,
) -> Path:
    now = now or datetime.now()
    branding = branding or Branding()
    listed = rows(analysis.shown(), analysis.size)
    chunks = [listed[i : i + ROWS_PER_PAGE] for i in range(0, len(listed), ROWS_PER_PAGE)] or [[]]
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        pdf.savefig(first_page(analysis, headline, now, branding))
        for page, chunk in enumerate(chunks, start=1):
            pdf.savefig(table_page(chunk, page, len(chunks), branding))
    return path
