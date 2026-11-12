import os
import platform
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
DATA = WORKING / "data"
PREINSTALLED = ("torch", "torchvision")
PIN = re.compile(r"^([A-Za-z0-9_.-]+)==(\S+)$")


def pins(lines: list[str]) -> dict[str, str]:
    found = {}
    for line in lines:
        match = PIN.match(line.strip())
        if match:
            found[match.group(1).lower()] = match.group(2)
    return found


def base(version: str) -> str:
    return version.split("+", 1)[0]


def installed(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def plan(lines: list[str], present: dict[str, str | None]) -> tuple[list[str], list[str]]:
    kept, notes = [], []
    for line in lines:
        match = PIN.match(line.strip())
        name = match.group(1).lower() if match else None
        if name in PREINSTALLED:
            wanted, found = match.group(2), present.get(name)
            if found and base(found) == base(wanted):
                notes.append(f"keeping preinstalled {name} {found} (pin {wanted})")
                continue
            notes.append(f"installing {name}=={wanted} (found {found or 'none'})")
        kept.append(line)
    return kept, notes


def find_weights(root: Path, names: list[str]) -> Path | None:
    if not root.is_dir():
        return None
    for folder, _, files in sorted(os.walk(root, followlinks=True)):
        if all(name in files for name in names):
            return Path(folder)
    return None


def versions(wanted: dict[str, str], present: dict[str, str | None]) -> list[str]:
    lines = []
    for name, pin in wanted.items():
        found = present.get(name)
        if found is None:
            state = "MISSING"
        elif found == pin or (name in PREINSTALLED and base(found) == base(pin)):
            state = "ok"
        else:
            state = "differs from pin"
        lines.append(f"  {name}: {found or 'not installed'} (pin {pin}) {state}")
    return lines


def stream(command: list[str], cwd: Path) -> int:
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    process = subprocess.Popen(
        command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    for line in process.stdout:
        print(line, end="", flush=True)
    return process.wait()


def commands(folder: Path | None, data: Path) -> tuple[list[str], list[str]]:
    paths = ["--data", str(data)]
    if folder is not None:
        paths = ["--weights", str(folder), *paths]
    check = [sys.executable, "-u", "-m", "defects.check", *paths]
    app = [sys.executable, "-u", "app.py", "--share", *paths]
    return check, app


def ui_check() -> list[str]:
    return [sys.executable, "-u", "-m", "defects.uicheck"]


def choose(app: list[str], passed: bool) -> list[str]:
    return app if passed else [*app, "--classic"]


def banner(text: str) -> str:
    line = "=" * 72
    return f"{line}\n{text}\n{line}"


def install(repo: Path, work: Path) -> dict[str, str]:
    lines = (repo / "requirements.txt").read_text(encoding="utf-8").splitlines()
    wanted = pins(lines)
    kept, notes = plan(lines, {name: installed(name) for name in PREINSTALLED})
    for note in notes:
        print(note, flush=True)
    filtered = work / "requirements-kaggle.txt"
    filtered.write_text("\n".join(kept) + "\n", encoding="utf-8")
    code = stream([sys.executable, "-m", "pip", "install", "-q", "-r", str(filtered)], repo)
    if code != 0:
        raise SystemExit(f"pip install failed with code {code}")
    return wanted


def main() -> None:
    sys.path.insert(0, str(REPO))
    from defects.registry import MODELS

    print(f"python {platform.python_version()}", flush=True)
    WORKING.mkdir(parents=True, exist_ok=True)
    wanted = install(REPO, WORKING)
    print("installed versions:", flush=True)
    for line in versions(wanted, {name: installed(name) for name in wanted}):
        print(line, flush=True)
    names = [Path(spec.weights).name for spec in MODELS]
    folder = find_weights(INPUT, names)
    if folder is None:
        print(banner(f"WARNING: no folder under {INPUT} holds {', '.join(names)}"), flush=True)
    else:
        print(f"weights folder: {folder}", flush=True)
    check, app = commands(folder, DATA)
    if stream(check, REPO) != 0:
        print(
            banner("WARNING: the readiness check reports NOT READY. Starting the app anyway."),
            flush=True,
        )
    passed = stream(ui_check(), REPO) == 0
    if not passed:
        print(
            banner("WARNING: the interface check failed. Starting the classic interface instead."),
            flush=True,
        )
    code = stream(choose(app, passed), REPO)
    raise SystemExit(f"app stopped with code {code}")


if __name__ == "__main__":
    main()
