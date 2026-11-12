import pytest

from defects.checkpoint import CheckpointError, check_keys, load_strict


class StandIn:
    def __init__(self):
        self.state = {"layer.weight": [1.0, 2.0], "layer.bias": [0.0]}
        self.loaded = None

    def state_dict(self):
        return dict(self.state)

    def load_state_dict(self, state, strict=True):
        self.loaded = dict(state)


def reader_of(state):
    return lambda path: state


def test_missing_key_is_refused(tmp_path):
    model = StandIn()
    with pytest.raises(CheckpointError, match="1 missing keys") as error:
        load_strict(model, tmp_path / "best.pth", reader_of({"layer.weight": [3.0]}))
    assert "layer.bias" in str(error.value)
    assert "not loaded" in str(error.value)
    assert model.loaded is None


def test_unexpected_key_is_refused(tmp_path):
    model = StandIn()
    state = {"layer.weight": [3.0], "layer.bias": [1.0], "extra.weight": [0.0]}
    with pytest.raises(CheckpointError, match="unexpected keys"):
        load_strict(model, tmp_path / "best.pth", reader_of(state))
    assert model.loaded is None


def test_matching_checkpoint_is_loaded(tmp_path):
    model = StandIn()
    state = {"layer.weight": [3.0, 4.0], "layer.bias": [1.0]}
    load_strict(model, tmp_path / "best.pth", reader_of(state))
    assert model.loaded == state


def test_training_state_file_is_refused(tmp_path):
    state = {"model": {}, "optimizer": {}, "epoch": 3}
    with pytest.raises(CheckpointError, match="last.pth"):
        load_strict(StandIn(), tmp_path / "last.pth", reader_of(state))


def test_non_mapping_is_refused(tmp_path):
    with pytest.raises(CheckpointError, match="state_dict"):
        load_strict(StandIn(), tmp_path / "best.pth", reader_of([1, 2]))


def test_unreadable_file_is_refused(tmp_path):
    def broken(path):
        raise OSError("bad zip")

    with pytest.raises(CheckpointError, match="could not be read"):
        load_strict(StandIn(), tmp_path / "best.pth", broken)


def test_check_keys_passes_on_equal_sets():
    check_keys(["a", "b"], ["b", "a"], "x")


def test_real_torch_module_with_missing_key(tmp_path):
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    state = model.state_dict()
    state.pop("0.bias")
    path = tmp_path / "best.pth"
    torch.save(state, path)
    with pytest.raises(CheckpointError, match="missing"):
        load_strict(torch.nn.Sequential(torch.nn.Linear(2, 2)), path)
