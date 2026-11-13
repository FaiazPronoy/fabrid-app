import ast
import re
from pathlib import Path

from defects import appearance, palette, summary

ROOT = Path(__file__).resolve().parents[1]
CSS = (ROOT / "defects" / "style.css").read_text(encoding="utf-8")
UI = (ROOT / "defects" / "ui.py").read_text(encoding="utf-8")


def rules() -> dict[str, str]:
    found = {}
    for match in re.finditer(r"([^{}]+)\{([^{}]*)\}", CSS):
        selector = " ".join(match.group(1).split())
        selector = selector.split("@media (max-width: 767px)")[-1].strip()
        found.setdefault(selector, match.group(2))
    return found


def variables(body: str) -> dict[str, str]:
    return dict(re.findall(r"(--fd-[a-z-]+):\s*(#[0-9a-f]{6});", body))


def test_both_modes_meet_contrast():
    for mode in ("light", "dark"):
        for name, ratio in palette.mode_pairs(mode).items():
            assert ratio >= palette.MIN_TEXT_CONTRAST, (mode, name)


def test_stylesheet_variables_match_palette():
    css = rules()
    assert variables(css[":root"]) == palette.css_variables("light")
    assert variables(css[":root.dark, body.dark"]) == palette.css_variables("dark")
    assert "color-scheme: light" in css[":root"]
    assert "color-scheme: dark" in css[":root.dark, body.dark"]


def test_no_fixed_verdict_colours_left():
    for kind in ("clear", "found", "error"):
        body = next(
            body for selector, body in rules().items() if f".fd-verdict.fd-{kind}" in selector
        )
        for part in ("fill", "edge", "text"):
            assert f"var(--fd-{kind}-{part})" in body
    banner = summary.banner_html({"kind": "found", "title": "t", "detail": "d", "notes": []})
    assert "style=" not in banner and "fd-verdict fd-found" in banner
    assert "style=" not in summary.error_html("x")
    assert "var(--fd-accent-fill)" in rules()[".fd-step-number"]


def test_theme_follows_both_modes():
    values = appearance.theme_values()
    for name, token in appearance.THEME_TOKENS.items():
        assert values[name] == palette.hexcode(palette.MODES["light"][token])
        assert values[f"{name}_dark"] == palette.hexcode(palette.MODES["dark"][token])
    assert values["body_background_fill"] != values["body_background_fill_dark"]


def test_mode_script_starts_light_and_remembers():
    script = appearance.MODE_JS
    assert script.strip().startswith("() =>")
    assert f"'{appearance.MODE_KEY}'" in script
    assert "=== 'dark' ? 'dark' : 'light'" in script
    assert "localStorage.setItem" in script and "MutationObserver" in script
    assert "window.fdToggleMode" in script and "fdToggleMode" in appearance.TOGGLE_JS


def test_toggle_button_runs_the_script_only():
    calls = [
        node
        for node in ast.walk(ast.parse(UI))
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", "") == "click"
        and getattr(node.func.value, "id", "") == "mode_button"
    ]
    assert len(calls) == 1
    call = calls[0]
    assert isinstance(call.args[0], ast.Constant) and call.args[0].value is None
    assert [k.value.id for k in call.keywords if k.arg == "js"] == ["TOGGLE_JS"]
    assert 'elem_id="fd-mode"' in UI


def test_every_accordion_carries_the_chevron_class():
    accordions = [
        node
        for node in ast.walk(ast.parse(UI))
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "Accordion"
    ]
    assert len(accordions) == 7
    for node in accordions:
        classes = [k.value for k in node.keywords if k.arg == "elem_classes"]
        assert classes and [e.value for e in classes[0].elts] == ["fd-acc"]


def test_chevron_header_rules():
    css = rules()
    header = css[".fd-acc .label-wrap"]
    assert "cursor: pointer" in header and "min-height: 44px" in header
    assert "background" in css[".fd-acc .label-wrap:hover"]
    assert "display: none" in css[".fd-acc .label-wrap .icon"]
    assert "rotate(-45deg)" in css[".fd-acc .label-wrap::after"]
    assert "rotate(45deg)" in css[".fd-acc .label-wrap.open::after"]
    assert "margin-bottom: 0" in css[".fd-acc .label-wrap:not(.open)"]


def test_clip_spares_buttons_and_the_threshold():
    selector = next(s for s in rules() if s.startswith(".fd-panel,"))
    for part in (":not(button)", ":not(button *)", ":not(.fd-threshold)", ":not(.fd-threshold *)"):
        assert part in selector, part
    assert "5.5em" in rules()['.fd-threshold input[type="number"]']


def test_threshold_box_and_reset_are_not_cut():
    css = rules()
    container = css[".fd-threshold .tab-like-container"]
    assert "overflow: visible" in container and "height: auto" in container
    assert "40px" in css['.fd-threshold input[type="number"]']
    reset = css[".fd-threshold .reset-button"]
    assert "min-width: 40px" in reset and "min-height: 40px" in reset
