from defects.models import load_all
from defects.registry import MODELS
from fakes import FakePredictor


def test_missing_weights_make_model_unavailable(tmp_path):
    slots = load_all(MODELS, tmp_path, builders={})
    for spec in MODELS:
        assert not slots[spec.key].ready
        assert "not found" in slots[spec.key].reason


def test_present_weights_are_loaded_and_warmed(tmp_path):
    (tmp_path / "weights").mkdir()
    (tmp_path / MODELS[0].weights).write_bytes(b"x")
    fake = FakePredictor()
    slots = load_all(MODELS, tmp_path, builders={"ultralytics": lambda spec, path: fake})
    assert slots[MODELS[0].key].ready
    assert fake.calls == 1
    assert not slots[MODELS[1].key].ready


def test_loader_error_is_reported(tmp_path):
    (tmp_path / "weights").mkdir()
    (tmp_path / MODELS[1].weights).write_bytes(b"x")

    def broken(spec, path):
        raise ValueError("does not match the model architecture")

    slots = load_all(MODELS, tmp_path, builders={"fabrimssa": broken})
    assert "does not match" in slots[MODELS[1].key].reason


def test_missing_package_is_reported(tmp_path):
    (tmp_path / "weights").mkdir()
    (tmp_path / MODELS[0].weights).write_bytes(b"x")

    def needs_torch(spec, path):
        raise ModuleNotFoundError("No module named 'torch'", name="torch")

    slots = load_all(MODELS, tmp_path, builders={"ultralytics": needs_torch})
    assert slots[MODELS[0].key].reason == "required package missing: torch"
