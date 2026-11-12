from conftest import SESSION
from defects import charts
from defects.labels import CLASSES

YOLO = "yolov8m_seg_aug"


def test_pdf_report_is_written(workspace, photo):
    first = workspace.analyse(SESSION, str(photo), YOLO, 0.25, list(CLASSES))
    path, message = workspace.report(SESSION, first.entry)
    assert path is not None, message
    data = path.read_bytes()
    assert data.startswith(b"%PDF")
    assert data.count(b"/Type /Page") - data.count(b"/Type /Pages") == 2


def test_pdf_report_of_empty_result(workspace, photo):
    workspace.fake.found = []
    first = workspace.analyse(SESSION, str(photo), YOLO, 0.25, list(CLASSES))
    path, _ = workspace.report(SESSION, first.entry)
    assert path.read_bytes().startswith(b"%PDF")


def test_report_before_analysis(workspace):
    path, message = workspace.report(SESSION, None)
    assert path is None
    assert "Analyse a photo first" in message


def test_dashboard_counts(workspace, photo):
    first = workspace.analyse(SESSION, str(photo), YOLO, 0.25, list(CLASSES))
    workspace.analyse(SESSION, str(photo), YOLO, 0.25, list(CLASSES))
    workspace.refilter(SESSION, first.entry, 0.05, list(CLASSES))
    records = charts.collect(workspace.history, SESSION)
    assert [len(r.shown) for r in records] == [4, 3]
    assert charts.per_class(records) == {"needle mark": 1, "hole": 4, "missing yarn": 0, "stain": 2}
    assert sorted(charts.scores(records)) == sorted([0.9, 0.6, 0.3, 0.1, 0.9, 0.6, 0.3])
    text, by_class, by_image, confidence = workspace.dashboard(SESSION)
    assert text.startswith("2 analyses")
    heights = [patch.get_height() for patch in by_class.axes[0].patches]
    assert heights == [1, 4, 0, 2]
    assert [p.get_height() for p in by_image.axes[0].patches] == [4, 3]
    assert sum(p.get_height() for p in confidence.axes[0].patches) == 7


def test_dashboard_without_history(workspace):
    text, *figures = workspace.dashboard(SESSION)
    assert text.startswith("0 analyses")
    assert len(figures) == 3


def test_history_listing(workspace, photo):
    workspace.analyse(SESSION, str(photo), YOLO, 0.25, list(CLASSES))
    choices, gallery = workspace.listing(SESSION)
    assert len(choices) == 1 and len(gallery) == 1
    assert "YOLOv8m-seg (augmented)" in choices[0][0]
    assert "3 shown" in choices[0][0]
