import traceback
from pathlib import Path

import gradio as gr

from defects import card, examples, views
from defects.labels import CLASSES, COLORS
from defects.service import STRETCH_LABEL, STRETCH_NOTE, Result, Workspace
from defects.texts import HISTORY_NOTE, INTRO, LIMITS, TITLE

MIN_THRESHOLD = 0.05


def legend() -> str:
    parts = []
    for name in CLASSES:
        red, green, blue = COLORS[name]
        swatch = f"<span style='color: rgb({red},{green},{blue})'>&#9632;</span>"
        parts.append(f"{swatch} {name}")
    return "Colours: " + " &nbsp; ".join(parts)


def about(workspace: Workspace) -> str:
    lines = [f"### Limits\n\n{LIMITS}", "", "### Models", ""]
    for key, slot in workspace.slots.items():
        entry = workspace.cards.get(key)
        lines.append(f"**{slot.spec.name}**: {workspace.status(key)}")
        lines.append("")
        lines.append(card.headline(entry))
        lines.append("")
        lines.append(card.details(entry))
        lines.append("")
        lines.append(f"Default confidence threshold: {workspace.threshold_for(key):.2f}.")
        lines.append("")
    lines.append("### How a photo is processed")
    lines.append("")
    lines.append(
        "The photo is turned upright using its EXIF orientation and scaled with antialiasing "
        "(Lanczos) so its long side is at most the model's input size (640 px for both models). "
        "Smaller photos are not enlarged. Boxes, mask areas, the table and the annotated image "
        "all use this analysis size. The image exactly as the model receives it is shown under "
        "Model input. Everything runs on the CPU and the inference time covers the model call "
        f"only. Detections down to confidence {MIN_THRESHOLD} are kept, so the threshold slider "
        "and the class filter work without running the model again."
    )
    lines.append("")
    lines.append(f"{STRETCH_LABEL}: off by default. {STRETCH_NOTE}")
    lines.append("")
    lines.append(HISTORY_NOTE)
    return "\n".join(lines)


def build(workspace: Workspace, samples: list[Path] | None = None) -> gr.Blocks:
    default_key = workspace.default_key()

    def respond(action, session: str, entry: str, clear: bool, names) -> tuple:
        try:
            result = action()
        except Exception as error:
            traceback.print_exc()
            result = Result(False, f"unexpected error, {type(error).__name__}: {error}")
        chosen = views.values(result, session, entry, clear, gr.update())
        return views.pick(chosen, names), result

    def on_analyse(image, key, threshold, classes, stretch, session, entry):
        session = workspace.session(session)
        shown, _ = respond(
            lambda: workspace.analyse(session, image, key, threshold, classes, stretch),
            session,
            entry,
            True,
            views.FULL,
        )
        return shown

    def on_filter(threshold, classes, key, session, entry):
        session = workspace.session(session)
        if not entry:
            return views.unchanged(session, entry, views.LIGHT, gr.update())
        shown, _ = respond(
            lambda: workspace.refilter(session, entry, threshold, classes, key),
            session,
            entry,
            False,
            views.LIGHT,
        )
        return shown

    def on_model(key):
        return workspace.info(key), workspace.threshold_for(key)

    def on_report(session, entry):
        session = workspace.session(session)
        try:
            path, message = workspace.report(session, entry)
        except Exception as error:
            traceback.print_exc()
            return session, None, f"The PDF report could not be created: {error}"
        return session, (str(path) if path else None), message

    def on_listing(session):
        session = workspace.session(session)
        try:
            choices, gallery = workspace.listing(session)
        except Exception:
            traceback.print_exc()
            choices, gallery = [], []
        return session, gr.update(choices=choices, value=None), gallery

    def on_open(session, chosen, entry, threshold, classes):
        session = workspace.session(session)
        if not chosen:
            kept = views.values(
                Result(False, "Choose an analysis from the list first."),
                session,
                entry,
                False,
                gr.update(),
            )
            return (*views.pick(kept, views.FULL), threshold, classes, gr.update())
        shown, result = respond(
            lambda: workspace.reopen(session, chosen), session, entry, False, views.FULL
        )
        if not result.ok:
            return (*shown, threshold, classes, gr.update())
        return (*shown, result.threshold, result.classes, gr.update(selected="analyse"))

    def on_dashboard(session):
        session = workspace.session(session)
        try:
            return (session, *workspace.dashboard(session))
        except Exception as error:
            traceback.print_exc()
            return session, f"The dashboard could not be drawn: {error}", None, None, None

    def on_load(session):
        return workspace.session(session)

    with gr.Blocks(title=TITLE) as demo:
        session = gr.State("")
        entry = gr.State("")
        gr.Markdown(f"# {TITLE}\n\n{INTRO}")
        with gr.Tabs() as tabs:
            with gr.Tab("Analyse", id="analyse"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=280):
                        photo = gr.Image(
                            type="filepath",
                            sources=["upload", "webcam"],
                            label="Photo (upload, or use the camera)",
                        )
                        if samples:
                            gr.Examples(
                                examples=[[str(path)] for path in samples],
                                inputs=[photo],
                                label=examples.LABEL,
                                cache_examples=False,
                            )
                        model = gr.Dropdown(
                            choices=workspace.choices(), value=default_key, label="Model"
                        )
                        model_info = gr.Markdown(workspace.info(default_key))
                        stretch = gr.Checkbox(value=False, label=STRETCH_LABEL)
                        gr.Markdown(STRETCH_NOTE)
                        analyse = gr.Button("Analyse", variant="primary")
                    with gr.Column(scale=1, min_width=280):
                        threshold = gr.Slider(
                            minimum=MIN_THRESHOLD,
                            maximum=0.95,
                            step=0.05,
                            value=workspace.threshold_for(default_key),
                            label="Confidence threshold",
                        )
                        classes = gr.CheckboxGroup(
                            choices=list(CLASSES), value=list(CLASSES), label="Classes to show"
                        )
                        gr.Markdown(legend())
                        status = gr.Markdown("Choose a photo and press Analyse.")
                with gr.Row():
                    original = gr.Image(label="Original", interactive=False)
                    annotated = gr.Image(label="Detected defects", interactive=False)
                table = gr.Dataframe(
                    value=views.frame([]), label="Detections (pixels at analysis size)"
                )
                with gr.Accordion("Model input", open=False):
                    model_input = gr.Image(label="Model input", interactive=False)
                    input_note = gr.Markdown("")
                with gr.Row():
                    png_file = gr.File(label="Annotated image (PNG)", interactive=False)
                    csv_file = gr.File(label="Results table (CSV)", interactive=False)
                    pdf_file = gr.File(label="PDF report", interactive=False)
                report = gr.Button("Create PDF report")
                report_note = gr.Markdown("")
            with gr.Tab("History", id="history"):
                gr.Markdown(HISTORY_NOTE)
                refresh = gr.Button("Refresh list")
                gallery = gr.Gallery(label="Analyses in this session", columns=4)
                chosen = gr.Dropdown(choices=[], label="Analysis to reopen")
                reopen = gr.Button("Open in Analyse tab", variant="primary")
            with gr.Tab("Dashboard", id="dashboard") as dashboard_tab:
                dashboard_text = gr.Markdown("")
                dashboard_refresh = gr.Button("Refresh charts")
                by_class = gr.Plot(label="Defects per class")
                by_image = gr.Plot(label="Defects per image")
                confidence = gr.Plot(label="Confidence distribution")
            with gr.Tab("About", id="about"):
                gr.Markdown(about(workspace))

        components = {
            "session": session,
            "entry": entry,
            "original": original,
            "annotated": annotated,
            "status": status,
            "table": table,
            "model_input": model_input,
            "input_note": input_note,
            "png": png_file,
            "csv": csv_file,
            "pdf": pdf_file,
        }
        full = [components[name] for name in views.FULL]
        light = [components[name] for name in views.LIGHT]
        demo.load(on_load, inputs=session, outputs=session)
        model.change(on_model, inputs=model, outputs=[model_info, threshold])
        analyse.click(
            on_analyse,
            inputs=[photo, model, threshold, classes, stretch, session, entry],
            outputs=full,
        ).then(on_listing, inputs=session, outputs=[session, chosen, gallery])
        threshold.change(
            on_filter,
            inputs=[threshold, classes, model, session, entry],
            outputs=light,
            trigger_mode="always_last",
        )
        classes.change(
            on_filter,
            inputs=[threshold, classes, model, session, entry],
            outputs=light,
            trigger_mode="always_last",
        )
        report.click(on_report, inputs=[session, entry], outputs=[session, pdf_file, report_note])
        refresh.click(on_listing, inputs=session, outputs=[session, chosen, gallery])
        reopen.click(
            on_open,
            inputs=[session, chosen, entry, threshold, classes],
            outputs=[*full, threshold, classes, tabs],
        )
        dashboard_outputs = [session, dashboard_text, by_class, by_image, confidence]
        dashboard_tab.select(on_dashboard, inputs=session, outputs=dashboard_outputs)
        dashboard_refresh.click(on_dashboard, inputs=session, outputs=dashboard_outputs)
    return demo
