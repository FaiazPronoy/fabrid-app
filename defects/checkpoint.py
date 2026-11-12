from collections.abc import Mapping
from pathlib import Path

SHOWN = 5


class CheckpointError(Exception):
    pass


def torch_reader(path: Path):
    import torch

    return torch.load(path, map_location="cpu", weights_only=True)


def read_state(path: Path, reader=torch_reader) -> Mapping:
    try:
        state = reader(path)
    except Exception as error:
        raise CheckpointError(
            f"{path.name} could not be read as a PyTorch file: {error}"
        ) from error
    if not isinstance(state, Mapping):
        raise CheckpointError(
            f"{path.name} does not hold a state_dict (found {type(state).__name__})"
        )
    if "model" in state and "optimizer" in state:
        raise CheckpointError(
            f"{path.name} is a training state file (last.pth); use checkpoints/best.pth instead"
        )
    return state


def check_keys(expected, found, name: str) -> None:
    expected, found = set(expected), set(found)
    missing, unexpected = sorted(expected - found), sorted(found - expected)
    if not missing and not unexpected:
        return
    parts = []
    if missing:
        parts.append(f"{len(missing)} missing keys, first {missing[:SHOWN]}")
    if unexpected:
        parts.append(f"{len(unexpected)} unexpected keys, first {unexpected[:SHOWN]}")
    raise CheckpointError(
        f"{name} does not match the model architecture: {'; '.join(parts)}. "
        "The model was not loaded."
    )


def load_strict(model, path: Path, reader=torch_reader):
    state = read_state(path, reader)
    check_keys(model.state_dict().keys(), state.keys(), path.name)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise CheckpointError(f"{path.name} does not fit the model: {error}") from error
    return model
