import traceback
from dataclasses import dataclass
from pathlib import Path

import gradio as gr

from defects import examples, marks, screens, summary
from defects.branding import Branding
from defects.labels import CLASSES
from defects.palette import ACCENT, ACCENT_HOVER, BACKGROUND, BORDER, CARD, MUTED, TEXT, hexcode
from defects.service import STRETCH_LABEL, STRETCH_NOTE, Result, Workspace
from defects.texts import LIMITS

CSS_PATH = Path(__file__).with_name("style.css")
MIN_THRESHOLD = 0.05
FONTS = ["ui-sans-serif", "system-ui", "Segoe UI", "Roboto", "Helvetica", "Arial", "sans-serif"]
LIGHT_JS = """
() => {
  const strip = () => {
    document.documentElement.classList.remove('dark');
    document.body.classList.remove('dark');
  };
  strip();
  const watch = new MutationObserver(strip);
  watch.observe(document.documentElement, {attributes: true, attributeFilter: ['class']});
  watch.observe(document.body, {attributes: true, attributeFilter: ['class']});
}
"""


def show(flag: bool):
    return gr.update(visible=flag)


@dataclass
class Wire:
    name: str
    fn: object
    inputs: list
    outputs: list


def theme():
    light = {
        "body_background_fill": hexcode(BACKGROUND),
        "body_text_color": hexcode(TEXT),
        "body_text_color_subdued": hexcode(MUTED),
        "background_fill_primary": hexcode(CARD),
        "background_fill_secondary": hexcode(BACKGROUND),
        "block_background_fill": hexcode(CARD),
        "block_border_color": hexcode(BORDER),
        "border_color_primary": hexcode(BORDER),
        "button_primary_background_fill": hexcode(ACCENT),
        "button_primary_background_fill_hover": hexcode(ACCENT_HOVER),
        "button_primary_text_color": "#ffffff",
    }
    both = {**light, **{f"{name}_dark": value for name, value in light.items()}}
    base = gr.themes.Base(primary_hue="indigo", neutral_hue="slate", radius_size="md", font=FONTS)
    return base.set(**both)


def launch_options() -> dict:
    return {
        "theme": theme(),
        "css": CSS_PATH.read_text(encoding="utf-8"),
        "js": LIGHT_JS,
        "footer_links": [],
    }


def build(
    workspace: Workspace, samples: list[Path] | None = None, branding: Branding | None = None
) -> gr.Blocks:
    branding = branding or workspace.branding
    default_key = workspace.default_key()
    wires: list[Wire] = []

    def wire(name, event, fn, inputs, outputs, **options):
        wires.append(Wire(name, fn, list(inputs), list(outputs)))
        return event(fn, inputs=inputs, outputs=outputs, **options)

    def respond(action, session, entry, clear, names):
        try:
            result = action()
        except Exception as error:
            traceback.print_exc()
            result = Result(False, f"unexpected error, {type(error).__name__}: {error}")
        chosen = screens.values(result, session, entry, clear, gr.update(), show)
        return screens.pick(chosen, names), result

    def on_load(session):
        return workspace.session(session)

    def lock():
        return gr.update(interactive=False, value="Working...")

    def unlock_analyse():
        return gr.update(interactive=True, value="Analyse")

    def unlock_compare():
        return gr.update(interactive=True, value="Run both models")

    def on_analyse(image, key, threshold, classes, style, stretch, session, entry):
        session = workspace.session(session)
        shown, _ = respond(
            lambda: workspace.analyse(session, image, key, threshold, classes, stretch, style),
            session,
            entry,
            True,
            screens.INSPECT,
        )
        return shown

    def on_filter(threshold, classes, style, key, session, entry):
        session = workspace.session(session)
        if not entry:
            return screens.pick(
                screens.values(
                    Result(True, "", skipped=True), session, entry, False, gr.update(), show
                ),
                screens.LIGHT,
            )
        shown, _ = respond(
            lambda: workspace.refilter(session, entry, threshold, classes, key, style),
            session,
            entry,
            False,
            screens.LIGHT,
        )
        return shown

    def on_style(style, session, entry):
        session = workspace.session(session)
        if not entry:
            return screens.pick(
                screens.values(
                    Result(True, "", skipped=True), session, entry, False, gr.update(), show
                ),
                screens.LIGHT,
            )
        shown, _ = respond(
            lambda: workspace.restyle(session, entry, style), session, entry, False, screens.LIGHT
        )
        return shown

    def on_model(key, session, entry):
        session = workspace.session(session)
        return screens.model_text(workspace, key), workspace.slider_for(session, entry, key)

    def on_report(session, entry):
        session = workspace.session(session)
        try:
            path, message = workspace.report(session, entry)
        except Exception as error:
            traceback.print_exc()
            return session, None, f"The PDF report could not be created: {error}"
        return session, (str(path) if path else None), message

    def on_compare(image, stretch, style, session):
        session = workspace.session(session)
        try:
            results, sentence = workspace.compare(session, image, stretch, style)
        except Exception as error:
            traceback.print_exc()
            failed = Result(False, f"unexpected error, {type(error).__name__}: {error}")
            results, sentence = [(key, failed) for key in workspace.slots], ""
        return (session, *screens.compare_values(workspace, results, sentence))

    def on_history(session):
        session = workspace.session(session)
        try:
            cards = workspace.cards_for(session)
        except Exception:
            traceback.print_exc()
            cards = []
        gallery, choices, empty = screens.history_values(cards)
        return session, gallery, gr.update(choices=choices, value=None), empty

    def on_choose(session, chosen):
        session = workspace.session(session)
        return gr.update(selected_index=screens.index_of(workspace.cards_for(session), chosen))

    def on_pick(session, evt: gr.SelectData):
        session = workspace.session(session)
        chosen = screens.entry_at(workspace.cards_for(session), getattr(evt, "index", None))
        return gr.update(value=chosen) if chosen else gr.update()

    def on_open(session, chosen, entry, threshold, classes, key, style):
        session = workspace.session(session)
        if not chosen:
            failed = Result(False, "Choose an analysis from the list first.")
            kept = screens.values(failed, session, entry, False, gr.update(), show)
            return (*screens.pick(kept, screens.INSPECT), threshold, classes, key, gr.update())
        shown, result = respond(
            lambda: workspace.reopen(session, chosen, style),
            session,
            entry,
            False,
            screens.INSPECT,
        )
        if not result.ok:
            return (*shown, threshold, classes, key, gr.update())
        return (
            *shown,
            result.threshold,
            result.classes,
            result.model_key,
            gr.update(selected="inspect"),
        )

    def on_clear_ask():
        return show(True)

    def on_clear_cancel():
        return show(False)

    def on_clear(session, entry):
        session = workspace.session(session)
        try:
            workspace.clear(session)
        except Exception:
            traceback.print_exc()
        gallery, choices, empty = screens.history_values([])
        return (
            session,
            "",
            gallery,
            gr.update(choices=choices, value=None),
            empty,
            show(False),
            summary.empty_html(),
            show(False),
        )

    def on_dashboard(session):
        session = workspace.session(session)
        try:
            text, by_class, by_image, confidence = workspace.dashboard(session)
            tiles = summary.stat_html(workspace.stats(session))
        except Exception as error:
            traceback.print_exc()
            return session, summary.error_html(str(error)), None, None, None
        if text.startswith("0 analyses"):
            tiles = summary.empty_html(
                "Nothing to chart yet", "Analyse a photo and the charts fill in here."
            )
        return session, tiles, by_class, by_image, confidence

    with gr.Blocks(title=branding.title) as demo:
        session = gr.State("")
        entry = gr.State("")
        gr.HTML(summary.header_html(branding, workspace.chips()))
        with gr.Tabs() as tabs:
            with gr.Tab("Inspect", id="inspect"), gr.Row(equal_height=False):
                with gr.Column(scale=5, min_width=300, elem_classes=["fd-panel"]):
                    gr.HTML(
                        summary.step_html(1, "Photo", "Upload, take a photo, or pick an example.")
                    )
                    photo = gr.Image(
                        type="filepath",
                        sources=["upload", "webcam"],
                        show_label=False,
                        height=280,
                    )
                    if samples:
                        gr.Examples(
                            examples=[[str(path)] for path in samples],
                            inputs=[photo],
                            label=examples.LABEL,
                            examples_per_page=12,
                            cache_examples=False,
                        )
                    gr.HTML(
                        summary.step_html(
                            2, "Model", "Each line shows the status and the validation figure."
                        )
                    )
                    model = gr.Radio(
                        choices=workspace.picker(),
                        value=default_key,
                        show_label=False,
                        elem_classes=["fd-picker"],
                    )
                    with gr.Accordion("More about this model", open=False):
                        model_info = gr.Markdown(screens.model_text(workspace, default_key))
                    gr.HTML(
                        summary.step_html(3, "Analyse", "Runs the chosen model on the photo now.")
                    )
                    analyse = gr.Button("Analyse", variant="primary", elem_classes=["fd-analyse"])
                    with gr.Accordion("Display and advanced settings", open=False):
                        threshold = gr.Slider(
                            minimum=MIN_THRESHOLD,
                            maximum=0.95,
                            step=0.05,
                            value=workspace.threshold_for(default_key),
                            label="Confidence threshold",
                            elem_classes=["fd-threshold"],
                        )
                        classes = gr.CheckboxGroup(
                            choices=list(CLASSES), value=list(CLASSES), label="Classes to show"
                        )
                        style = gr.Radio(
                            choices=[(label, key) for key, label in marks.STYLES.items()],
                            value=marks.DEFAULT_STYLE,
                            label="Mark style",
                        )
                        stretch = gr.Checkbox(value=False, label=STRETCH_LABEL)
                        gr.Markdown(STRETCH_NOTE)
                with gr.Column(
                    scale=7, min_width=300, elem_classes=["fd-panel"], elem_id="fd-result"
                ):
                    status = gr.HTML(summary.empty_html())
                    with gr.Column(visible=False) as result_box:
                        tiles = gr.HTML()
                        comparison = gr.ImageSlider(
                            label="Before and after: drag the handle to compare",
                            type="numpy",
                            buttons=["fullscreen", "download"],
                            interactive=False,
                            max_height=620,
                        )
                        with gr.Accordion("Defect details", open=False):
                            details = gr.HTML()
                        with gr.Accordion("Defect close-ups", open=False):
                            closeups = gr.Gallery(
                                show_label=False,
                                columns=4,
                                height="auto",
                                object_fit="contain",
                                allow_preview=True,
                                preview=False,
                                interactive=False,
                                elem_classes=["fd-crops"],
                            )
                        with gr.Accordion("Model input", open=False):
                            model_input = gr.Image(
                                label="Model input", interactive=False, buttons=["fullscreen"]
                            )
                            input_note = gr.Markdown("")
                        with (
                            gr.Accordion("Downloads", open=False),
                            gr.Column(elem_classes=["fd-downloads"]),
                        ):
                            png_file = gr.File(label="Annotated image", interactive=False)
                            csv_file = gr.File(label="Defect table", interactive=False)
                            report = gr.Button("Create PDF report")
                            pdf_file = gr.File(label="PDF report", interactive=False)
                            report_note = gr.Markdown("")
            with gr.Tab("Compare", id="compare"), gr.Row(equal_height=False):
                with gr.Column(scale=4, min_width=300, elem_classes=["fd-panel"]):
                    gr.HTML(summary.step_html(1, "Photo", "One photo for every available model."))
                    compare_photo = gr.Image(
                        type="filepath",
                        sources=["upload", "webcam"],
                        show_label=False,
                        height=260,
                    )
                    if samples:
                        gr.Examples(
                            examples=[[str(path)] for path in samples],
                            inputs=[compare_photo],
                            label=examples.LABEL,
                            examples_per_page=12,
                            cache_examples=False,
                        )
                    gr.HTML(
                        summary.step_html(2, "Run", "Each model uses its own default threshold.")
                    )
                    compare_button = gr.Button(
                        "Run both models", variant="primary", elem_classes=["fd-analyse"]
                    )
                    gr.Markdown(summary.COMPARE_NOTE)
                with gr.Column(
                    scale=8, min_width=300, elem_classes=["fd-panel"], elem_id="fd-compare-result"
                ):
                    compare_outputs = []
                    with gr.Row(equal_height=False):
                        for slot in workspace.slots.values():
                            with gr.Column(min_width=260):
                                compare_outputs.append(
                                    gr.Image(
                                        label=slot.spec.name,
                                        interactive=False,
                                        buttons=["fullscreen", "download"],
                                    )
                                )
                                compare_outputs.append(
                                    gr.HTML(summary.empty_html(slot.spec.name, "Not run yet."))
                                )
                    compare_summary = gr.HTML()
            with (
                gr.Tab("History", id="history") as history_tab,
                gr.Column(elem_classes=["fd-panel"]),
            ):
                gr.HTML(
                    summary.step_html(1, "This session", "Pick a card, then open it in Inspect.")
                )
                history_empty = gr.HTML(
                    summary.empty_html("History is empty", screens.HISTORY_EMPTY)
                )
                gallery = gr.Gallery(
                    show_label=False,
                    columns=5,
                    height="auto",
                    object_fit="cover",
                    allow_preview=False,
                    preview=False,
                    interactive=False,
                    elem_classes=["fd-history"],
                )
                chosen = gr.Dropdown(choices=[], label="Analysis to open")
                with gr.Row():
                    reopen = gr.Button("Open in Inspect", variant="primary")
                    refresh = gr.Button("Refresh")
                    clear_ask = gr.Button("Clear this session", variant="stop")
                with gr.Row(visible=False) as confirm_row:
                    gr.Markdown(
                        "Delete every analysis of this browser session? This cannot be undone."
                    )
                    clear_yes = gr.Button("Yes, clear", variant="stop")
                    clear_no = gr.Button("Cancel")
            with (
                gr.Tab("Dashboard", id="dashboard") as dashboard_tab,
                gr.Column(elem_classes=["fd-panel"]),
            ):
                stats = gr.HTML(
                    summary.empty_html(
                        "Nothing to chart yet", "Analyse a photo and the charts fill in here."
                    )
                )
                dashboard_refresh = gr.Button("Refresh charts")
                by_class = gr.Plot(show_label=False)
                by_image = gr.Plot(show_label=False)
                confidence = gr.Plot(show_label=False)
            with gr.Tab("About", id="about"), gr.Column(elem_classes=["fd-panel"]):
                for title, text, opened in screens.about_sections(workspace, branding):
                    with gr.Accordion(title, open=opened):
                        if text.startswith("<div"):
                            gr.HTML(text)
                        else:
                            gr.Markdown(text)
        gr.HTML(f"<div class='fd-limits'><b>Limits.</b> {summary.escape(LIMITS)}</div>")

        parts = {
            "session": session,
            "entry": entry,
            "status": status,
            "result_box": result_box,
            "tiles": tiles,
            "comparison": comparison,
            "details": details,
            "closeups": closeups,
            "model_input": model_input,
            "input_note": input_note,
            "png": png_file,
            "csv": csv_file,
            "pdf": pdf_file,
            "report_note": report_note,
        }
        inspect = [parts[name] for name in screens.INSPECT]
        light = [parts[name] for name in screens.LIGHT]
        history_outputs = [session, gallery, chosen, history_empty]
        dashboard_outputs = [session, stats, by_class, by_image, confidence]

        wire("load", demo.load, on_load, [session], [session])
        started = wire(
            "analyse_lock", analyse.click, lock, [], [analyse], queue=False, trigger_mode="once"
        )
        ran = wire(
            "analyse",
            started.then,
            on_analyse,
            [photo, model, threshold, classes, style, stretch, session, entry],
            inspect,
            show_progress="full",
        )
        done = wire("analyse_unlock", ran.then, unlock_analyse, [], [analyse])
        wire("analyse_history", done.then, on_history, [session], history_outputs)
        done.then(None, js=screens.scroll_js("fd-result", True))
        filter_inputs = [threshold, classes, style, model, session, entry]
        wire(
            "filter_threshold",
            threshold.change,
            on_filter,
            filter_inputs,
            light,
            trigger_mode="always_last",
            show_progress="minimal",
        )
        wire(
            "filter_classes",
            classes.change,
            on_filter,
            filter_inputs,
            light,
            trigger_mode="always_last",
            show_progress="minimal",
        )
        wire(
            "style",
            style.change,
            on_style,
            [style, session, entry],
            light,
            trigger_mode="always_last",
            show_progress="minimal",
        )
        wire("model", model.change, on_model, [model, session, entry], [model_info, threshold])
        wire("report", report.click, on_report, [session, entry], [session, pdf_file, report_note])
        compare_started = wire(
            "compare_lock",
            compare_button.click,
            lock,
            [],
            [compare_button],
            queue=False,
            trigger_mode="once",
        )
        compared = wire(
            "compare",
            compare_started.then,
            on_compare,
            [compare_photo, stretch, style, session],
            [session, *compare_outputs, compare_summary],
            show_progress="full",
        )
        compare_done = wire("compare_unlock", compared.then, unlock_compare, [], [compare_button])
        wire("compare_history", compare_done.then, on_history, [session], history_outputs)
        compare_done.then(None, js=screens.scroll_js("fd-compare-result", False))
        wire("history_tab", history_tab.select, on_history, [session], history_outputs)
        wire("history_refresh", refresh.click, on_history, [session], history_outputs)
        wire("history_pick", gallery.select, on_pick, [session], [chosen])
        wire("history_choose", chosen.change, on_choose, [session, chosen], [gallery])
        wire(
            "history_open",
            reopen.click,
            on_open,
            [session, chosen, entry, threshold, classes, model, style],
            [*inspect, threshold, classes, model, tabs],
        )
        wire("clear_ask", clear_ask.click, on_clear_ask, [], [confirm_row])
        wire("clear_cancel", clear_no.click, on_clear_cancel, [], [confirm_row])
        wire(
            "clear_confirm",
            clear_yes.click,
            on_clear,
            [session, entry],
            [session, entry, gallery, chosen, history_empty, confirm_row, status, result_box],
        )
        wire("dashboard_tab", dashboard_tab.select, on_dashboard, [session], dashboard_outputs)
        wire(
            "dashboard_refresh", dashboard_refresh.click, on_dashboard, [session], dashboard_outputs
        )
    demo.wires = wires
    return demo
