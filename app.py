import argparse
import sys
from pathlib import Path

from defects import card, examples, settings, verify
from defects.history import History
from defects.models import load_all
from defects.registry import MODELS
from defects.service import Workspace, cleanup

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "reference"


def options(argv=None, env=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fabric defect inspection web app")
    parser.add_argument("--lan", action="store_true", help="listen on the local network")
    parser.add_argument("--share", action="store_true", help="create a public Gradio link")
    parser.add_argument("--port", type=int, default=None)
    settings.add_folders(parser)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    args.weights, args.data = settings.folders(args, ROOT, env)
    return args


def workspace(weights: Path | None, data: Path) -> Workspace:
    history = History(data / "history")
    print(cleanup(history), flush=True)
    slots = load_all(MODELS, ROOT, folder=weights)
    verify.apply(slots, ROOT, REFERENCE, folder=weights)
    return Workspace(slots, history, card.read(ROOT / "model_card.json"))


def public_line(share: bool, share_url: str | None, local_url: str | None) -> str:
    if share_url:
        return f"PUBLIC URL: {share_url}"
    if share:
        return f"WARNING: no public link could be created; local address {local_url}"
    return f"LOCAL URL: {local_url}"


def main(argv=None) -> None:
    args = options(argv)
    from defects.ui import build

    args.data.mkdir(parents=True, exist_ok=True)
    print(f"weights folder: {args.weights or ROOT / 'weights'}", flush=True)
    print(f"data folder: {args.data}", flush=True)
    demo = build(workspace(args.weights, args.data), examples.listing(REFERENCE))
    demo.queue(default_concurrency_limit=1, max_size=32).launch(
        server_name="0.0.0.0" if args.lan else None,
        server_port=args.port,
        share=args.share,
        allowed_paths=[str(args.data), str(REFERENCE)],
        prevent_thread_lock=True,
    )
    share_url = getattr(demo, "share_url", None)
    local_url = getattr(demo, "local_url", None)
    print(public_line(args.share, share_url, local_url), flush=True)
    demo.block_thread()


if __name__ == "__main__":
    main()
