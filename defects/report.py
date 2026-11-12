from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from defects.draw import annotate
from defects.history import Analysis
from defects.table import counts, rows
from defects.texts import LIMITS


def stretch_line(used: bool) -> str:
    if used:
        return "Contrast stretch before inference: on (approximation, not validated)"
    return "Contrast stretch before inference: off"


A4 = (8.27, 11.69)
ROWS_PER_PAGE = 32
HEADERS = ("no.", "class", "confidence", "x0", "y0", "x1", "y1", "area (px)", "area (%)")
WIDTHS = (0.06, 0.18, 0.12, 0.09, 0.09, 0.09, 0.09, 0.14, 0.14)


def heading(analysis: Analysis, headline: str, now: datetime) -> list[str]:
    shown = analysis.shown()
    tally = ", ".join(f"{name} {n}" for name, n in counts(shown).items())
    width, height = analysis.size
    return [
        f"Model: {analysis.model_name}",
        f"Confidence threshold: {analysis.threshold:.2f}",
        f"Classes shown: {', '.join(analysis.classes) or 'none'}",
        f"Analysed: {analysis.created.replace('T', ' ')}",
        f"Report created: {now:%Y-%m-%d %H:%M:%S}",
        f"Inference time: {analysis.seconds:.2f} s on CPU, analysis size {width} x {height} px",
        f"Defects shown: {len(shown)} ({tally})",
        stretch_line(analysis.stretch),
        headline,
    ]


def first_page(analysis: Analysis, headline: str, now: datetime) -> Figure:
    figure = Figure(figsize=A4)
    figure.text(0.07, 0.95, "Fabric defect analysis", fontsize=18, weight="bold")
    lines = heading(analysis, headline, now)
    for index, line in enumerate(lines):
        figure.text(0.07, 0.91 - index * 0.022, line, fontsize=9, wrap=True)
    annotated = annotate(analysis.work, analysis.shown())
    for position, (title, pixels) in enumerate(
        (("Original", np.asarray(analysis.display)), ("Annotated", annotated))
    ):
        ax = figure.add_axes((0.05 + position * 0.47, 0.30, 0.43, 0.38))
        ax.imshow(pixels)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    figure.text(0.07, 0.22, "Limits", fontsize=10, weight="bold")
    figure.text(0.07, 0.205, LIMITS, fontsize=8, wrap=True, va="top")
    return figure


def table_page(chunk: list[list], page: int, pages: int) -> Figure:
    figure = Figure(figsize=A4)
    figure.text(0.07, 0.95, f"Detections (page {page} of {pages})", fontsize=12, weight="bold")
    ax = figure.add_axes((0.05, 0.05, 0.9, 0.87))
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
    table.set_fontsize(7)
    table.scale(1, 1.3)
    return figure


def write(analysis: Analysis, headline: str, path: Path, now: datetime | None = None) -> Path:
    now = now or datetime.now()
    listed = rows(analysis.shown(), analysis.size)
    chunks = [listed[i : i + ROWS_PER_PAGE] for i in range(0, len(listed), ROWS_PER_PAGE)] or [[]]
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        pdf.savefig(first_page(analysis, headline, now))
        for page, chunk in enumerate(chunks, start=1):
            pdf.savefig(table_page(chunk, page, len(chunks)))
    return path
