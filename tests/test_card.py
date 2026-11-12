import json
from pathlib import Path

import pytest

from defects import card
from defects.registry import MODELS

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT.parent / "fabrid-benchmark"


def field(data, dotted):
    for part in dotted.split("."):
        data = data[part]
    return data


def test_card_has_entry_per_model_with_source():
    cards = card.read(ROOT / "model_card.json")
    for spec in MODELS:
        entry = cards[spec.key]
        assert entry["metric"] == "mask AP50"
        assert entry["source"].startswith("fabrid-benchmark/results/")
        assert "not seen in training" in entry["split"]


def test_headline_wording():
    entry = card.read(ROOT / "model_card.json")["fabrimssa_net"]
    text = card.headline(entry)
    assert "0.182" in text
    assert "0.127 to 0.254" in text
    assert "not seen in training" in text
    assert "single seed" in text


def test_missing_entry_says_not_evaluated():
    assert "not evaluated yet" in card.headline(None)
    assert "not evaluated yet" in card.headline({"value": None})


def test_missing_card_file_reads_empty(tmp_path):
    assert card.read(tmp_path / "none.json") == {}


@pytest.mark.skipif(not BENCHMARK.is_dir(), reason="benchmark repository not next to the app")
def test_card_values_match_benchmark_files():
    cards = card.read(ROOT / "model_card.json")
    for entry in cards.values():
        source = BENCHMARK / entry["source"].removeprefix("fabrid-benchmark/")
        data = json.loads(source.read_text(encoding="utf-8"))
        found = field(data, entry["field"])
        if isinstance(found, dict):
            assert found["value"] == entry["value"]
            assert [found["low"], found["high"]] == entry["interval"]
        else:
            assert found == entry["value"]
