import ast
import re
from pathlib import Path

import numpy as np

from conftest import SESSION
from defects import branding, closeups, marks, report, screens, summary
from defects.labels import CLASSES, COLORS
from fakes import square

ROOT = Path(__file__).resolve().parents[1]
CSS = (ROOT / "defects" / "style.css").read_text(encoding="utf-8")
UI = (ROOT / "defects" / "ui.py").read_text(encoding="utf-8")
YOLO = "yolov8m_seg_aug"


def rules() -> list[tuple[str, str]]:
    found = []
    for match in re.finditer(r"([^{}]+)\{([^{}]*)\}", CSS):
        selector = " ".join(match.group(1).split())
        selector = selector.split("@media (max-width: 767px)")[-1].strip()
        found.append((selector, match.group(2)))
    return found


def rule(selector: str) -> str:
    bodies = [body for name, body in rules() if name == selector]
    assert bodies, selector
    return bodies[0]


def test_crop_window_has_context_and_minimum_size():
    left, right = closeups.span(500, 510, 2048)
    assert left <= 500 - 30 and right >= 510 + 30
    assert right - left >= closeups.MIN_SIDE
    assert closeups.span(400, 500, 2048) == (100, 800)
    assert closeups.span(0, 10, 2048) == (0, closeups.MIN_SIDE)
    assert closeups.span(2040, 2048, 2048) == (2048 - closeups.MIN_SIDE, 2048)
    assert closeups.span(0, 150, 300) == (0, 300)


def test_crop_draws_the_defect_mark_without_a_tag():
    display = np.full((960, 1280, 3), 128, dtype=np.uint8)
    found = [square("hole", 0.61, 100, 100, 104, 104), square("stain", 0.97, 600, 440, 640, 480)]
    crops = closeups.crops(display, found, (640, 480))
    assert [caption for _, caption in crops] == ["stain 0.97", "hole 0.61"]
    hole = crops[1][0]
    assert hole.shape[:2] == (closeups.MIN_SIDE, closeups.MIN_SIDE)
    assert (hole == np.array(COLORS["hole"])).all(-1).any()
    assert not (hole == np.array(COLORS["stain"])).all(-1).any()
    plain = marks.render(display, found[:1], (640, 480), "box", labels=False)
    tagged = marks.render(display, found[:1], (640, 480), "box")
    assert not np.array_equal(plain, tagged)
    many = [square("hole", 0.5, 4 * i, 4, 4 * i + 2, 6) for i in range(30)]
    assert len(closeups.crops(display, many, (640, 480))) == closeups.LIMIT


def test_history_captions_are_short_and_newest_first(workspace, photo):
    workspace.fake.found = []
    workspace.analyse(SESSION, str(photo), YOLO, 0.25, list(CLASSES))
    workspace.fake.found = [square("hole", 0.9, 4, 4, 20, 16)]
    second = workspace.analyse(SESSION, str(photo), YOLO, 0.25, list(CLASSES))
    cards = workspace.cards_for(SESSION)
    assert cards[0]["entry"] == second.entry
    assert cards[0]["caption"].endswith("| 1 defect")
    assert cards[1]["caption"].endswith("| no defect")
    assert all(len(card["caption"]) <= 40 for card in cards)
    assert len(cards[0]["caption"].split(" | ")[0]) == 5


def test_scroll_script_targets_the_result():
    script = screens.scroll_js("fd-result", True)
    assert script.startswith("() =>") and "getElementById('fd-result')" in script
    assert ".fd-error" in script and "scrollIntoView" in script
    assert ".fd-error" not in screens.scroll_js("fd-compare-result", False)


def test_scroll_runs_after_analyse_and_compare():
    targets = []
    for node in ast.walk(ast.parse(UI)):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "then":
            for keyword in node.keywords:
                if keyword.arg == "js":
                    targets.append(keyword.value.args[0].value)
    assert sorted(targets) == ["fd-compare-result", "fd-result"]
    assert 'elem_id="fd-result"' in UI and 'elem_id="fd-compare-result"' in UI


def test_only_tables_scroll_sideways():
    for selector, body in rules():
        if re.search(r"overflow(-x)?:\s*(auto|scroll)", body):
            assert selector == ".fd-table-wrap", selector
    panel = [body for selector, body in rules() if selector.startswith(".fd-panel,")]
    assert panel and "overflow-x: clip" in panel[0]
    assert "overflow" not in rule(".gradio-container")
    assert "overflow-x: hidden" in rule("html, body")


def test_gallery_grids_threshold_and_header_in_stylesheet():
    assert "repeat(4, minmax(0, 1fr))" in rule(".fd-crops .grid-container")
    assert "140px" in rule(".fd-crops .grid-container")
    assert "repeat(5, minmax(0, 1fr))" in rule(".fd-history .grid-container")
    assert "max-height: none" in rule(".fd-crops .grid-wrap, .fd-history .grid-wrap")
    phone = CSS.split("@media (max-width: 767px)")[1]
    assert ".fd-history .grid-container" in phone and "repeat(2, minmax(0, 1fr))" in phone
    assert "5.5em" in rule('.fd-threshold input[type="number"]')
    assert "outline" in rule(".fd-history .thumbnail-item.selected, .fd-history button.selected")
    assert "overflow-wrap: anywhere" in rule(".fd-title")
    assert "min-width: 0" in rule(".fd-brand")


def test_galleries_open_as_grids():
    assert len(re.findall(r"(?<!allow_)preview=False", UI)) == 2
    assert 'elem_classes=["fd-crops"]' in UI and 'elem_classes=["fd-history"]' in UI
    assert "columns=5" in UI and 'elem_classes=["fd-threshold"]' in UI


def test_long_branding_wraps_in_header_and_pdf():
    long = branding.Branding(
        title="Automated Detection and Instance Segmentation of Knitted Fabric Defects Using "
        "Deep Learning Models",
        team=(
            "Mohammad Abdullah Al Mamun",
            "Faiaz Abrar Zaman Pronoy",
            "Sinthia Khandoker",
            "Purnata Zaman",
        ),
        supervisor="Professor Doctor Long Name Example",
    )
    lines = report.header_lines(long)
    titles = [text for text, size, _, _ in lines if size == 17]
    assert len(titles) >= 2 and all(len(text) <= report.TITLE_CHARS for text in titles)
    joined = " ".join(text for text, _, _, _ in lines)
    for name in long.team:
        assert name in joined
    assert all(len(text) <= report.LINE_CHARS for text, size, _, _ in lines if size == 8.5)
    html = summary.header_html(long, [])
    assert "Purnata Zaman" in html and "fd-brand-line" in html
