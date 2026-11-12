import csv

from defects.prediction import select
from defects.table import COLUMNS, counts, rows, write_csv
from fakes import square


def test_counts_include_every_class():
    found = [square("hole", 0.9, 0, 0, 4, 4), square("hole", 0.8, 5, 5, 9, 9)]
    assert counts(found) == {"needle mark": 0, "hole": 2, "missing yarn": 0, "stain": 0}


def test_rows_hold_box_and_area():
    item = square("stain", 0.876, 10, 20, 30, 25, size=(100, 50))
    assert rows([item], (100, 50)) == [[1, "stain", 0.876, 10, 20, 30, 25, 100, 2.0]]


def test_select_threshold_is_inclusive_and_sorted():
    found = [square("hole", 0.3, 0, 0, 2, 2), square("stain", 0.7, 0, 0, 2, 2)]
    assert [i.score for i in select(found, 0.3, ["hole", "stain"])] == [0.7, 0.3]
    assert select(found, 0.31, ["hole"]) == []


def test_csv_export(tmp_path):
    found = [square("hole", 0.9, 0, 0, 4, 4, size=(8, 8))]
    path = write_csv(found, (8, 8), tmp_path / "out" / "results.csv")
    with path.open(encoding="utf-8") as handle:
        lines = list(csv.reader(handle))
    assert lines[0] == list(COLUMNS)
    assert lines[1] == ["1", "hole", "0.9", "0", "0", "4", "4", "16", "25.0"]


def test_csv_export_of_empty_result(tmp_path):
    path = write_csv([], (8, 8), tmp_path / "empty.csv")
    assert path.read_text(encoding="utf-8").strip().split(",")[0] == "number"
