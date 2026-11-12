import argparse
import os
from pathlib import Path

WEIGHTS_ENV = "FABRID_APP_WEIGHTS"
DATA_ENV = "FABRID_APP_DATA"


def add_folders(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--weights", type=Path, default=None, help=f"weights folder, or set {WEIGHTS_ENV}"
    )
    parser.add_argument("--data", type=Path, default=None, help=f"data folder, or set {DATA_ENV}")


def folders(args: argparse.Namespace, root: Path, env=None) -> tuple[Path | None, Path]:
    env = os.environ if env is None else env
    weights = args.weights or (Path(env[WEIGHTS_ENV]) if env.get(WEIGHTS_ENV) else None)
    data = args.data or (Path(env[DATA_ENV]) if env.get(DATA_ENV) else Path(root) / "data")
    return weights, data
