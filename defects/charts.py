from dataclasses import dataclass

import numpy as np
from matplotlib.figure import Figure

from defects.labels import CLASSES, COLORS
from defects.palette import ACCENT, BORDER, MUTED, TEXT
from defects.prediction import Instance, select


@dataclass
class Record:
    meta: dict
    shown: list[Instance]

    @property
    def label(self) -> str:
        return self.meta["created"].replace("T", " ")[11:19] or self.meta["entry"]


def collect(history, session: str) -> list[Record]:
    records = []
    for meta in reversed(history.entries(session)):
        found = history.found(session, meta["entry"])
        records.append(Record(meta, select(found, meta["threshold"], meta["classes"])))
    return records


def per_class(records: list[Record]) -> dict[str, int]:
    tally = dict.fromkeys(CLASSES, 0)
    for record in records:
        for item in record.shown:
            tally[item.label] += 1
    return tally


def per_image(records: list[Record]) -> list[tuple[str, int]]:
    return [(record.label, len(record.shown)) for record in records]


def scores(records: list[Record]) -> list[float]:
    return [item.score for record in records for item in record.shown]


def colour(name: str) -> tuple[float, float, float]:
    return tuple(v / 255 for v in COLORS[name])


def rgb(color) -> tuple[float, float, float]:
    return tuple(v / 255 for v in color)


def figure_for() -> Figure:
    figure = Figure(figsize=(6, 3.4), layout="constrained")
    figure.patch.set_facecolor("white")
    return figure


def tidy(ax, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=rgb(TEXT))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(rgb(BORDER))
    ax.tick_params(colors=rgb(MUTED), labelsize=9)
    ax.yaxis.label.set_color(rgb(MUTED))
    ax.xaxis.label.set_color(rgb(MUTED))
    ax.grid(axis="y", color=rgb(BORDER), linewidth=0.8)
    ax.set_axisbelow(True)


def empty(figure: Figure, text: str) -> Figure:
    ax = figure.subplots()
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=12, color=rgb(MUTED))
    return figure


def class_chart(records: list[Record]) -> Figure:
    figure = figure_for()
    if not records:
        return empty(figure, "No analyses in this session yet")
    tally = per_class(records)
    ax = figure.subplots()
    bars = ax.bar(list(tally), list(tally.values()), color=[colour(n) for n in tally])
    ax.bar_label(bars, color=rgb(TEXT))
    tidy(ax, "Defects per class")
    ax.set_ylabel("count")
    ax.yaxis.get_major_locator().set_params(integer=True)
    return figure


def image_chart(records: list[Record]) -> Figure:
    figure = figure_for()
    if not records:
        return empty(figure, "No analyses in this session yet")
    pairs = per_image(records)
    ax = figure.subplots()
    positions = np.arange(len(pairs))
    bars = ax.bar(positions, [n for _, n in pairs], color=rgb(ACCENT))
    ax.bar_label(bars, color=rgb(TEXT))
    ax.set_xticks(positions, [label for label, _ in pairs], rotation=45, ha="right")
    tidy(ax, "Defects per image (time of analysis)")
    ax.set_ylabel("count")
    ax.yaxis.get_major_locator().set_params(integer=True)
    return figure


def confidence_chart(records: list[Record]) -> Figure:
    figure = figure_for()
    values = scores(records)
    if not values:
        return empty(figure, "No detections above the thresholds used")
    ax = figure.subplots()
    ax.hist(values, bins=np.linspace(0, 1, 21), color=rgb(ACCENT), edgecolor="white")
    ax.set_xlim(0, 1)
    tidy(ax, "Confidence of shown detections")
    ax.set_xlabel("confidence")
    ax.set_ylabel("count")
    return figure
