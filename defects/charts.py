from dataclasses import dataclass

import numpy as np
from matplotlib.figure import Figure

from defects.labels import CLASSES, COLORS
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


def empty(figure: Figure, text: str) -> Figure:
    ax = figure.subplots()
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=12)
    return figure


def class_chart(records: list[Record]) -> Figure:
    figure = Figure(figsize=(6, 3.6), layout="constrained")
    if not records:
        return empty(figure, "No analyses in this session yet")
    tally = per_class(records)
    ax = figure.subplots()
    bars = ax.bar(list(tally), list(tally.values()), color=[colour(n) for n in tally])
    ax.bar_label(bars)
    ax.set_title("Defects per class")
    ax.set_ylabel("count")
    ax.yaxis.get_major_locator().set_params(integer=True)
    return figure


def image_chart(records: list[Record]) -> Figure:
    figure = Figure(figsize=(6, 3.6), layout="constrained")
    if not records:
        return empty(figure, "No analyses in this session yet")
    pairs = per_image(records)
    ax = figure.subplots()
    positions = np.arange(len(pairs))
    bars = ax.bar(positions, [n for _, n in pairs], color="#4b5563")
    ax.bar_label(bars)
    ax.set_xticks(positions, [label for label, _ in pairs], rotation=45, ha="right")
    ax.set_title("Defects per image (time of analysis)")
    ax.set_ylabel("count")
    ax.yaxis.get_major_locator().set_params(integer=True)
    return figure


def confidence_chart(records: list[Record]) -> Figure:
    figure = Figure(figsize=(6, 3.6), layout="constrained")
    values = scores(records)
    if not values:
        return empty(figure, "No detections above the thresholds used")
    ax = figure.subplots()
    ax.hist(values, bins=np.linspace(0, 1, 21), color="#4b5563", edgecolor="white")
    ax.set_xlim(0, 1)
    ax.set_title("Confidence of shown detections")
    ax.set_xlabel("confidence")
    ax.set_ylabel("count")
    return figure
