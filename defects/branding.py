import json
from dataclasses import dataclass
from pathlib import Path

TITLE = "Fabric defect inspection"
SUBTITLE = "Instance segmentation of needle marks, holes, missing yarn and stains on a photo"


@dataclass(frozen=True)
class Branding:
    title: str = TITLE
    subtitle: str = SUBTITLE
    institution: str = ""
    team: tuple[str, ...] = ()
    supervisor: str = ""

    def lines(self) -> list[str]:
        lines = []
        if self.institution:
            lines.append(self.institution)
        if self.team:
            lines.append(f"Team: {', '.join(self.team)}")
        if self.supervisor:
            lines.append(f"Supervisor: {self.supervisor}")
        return lines


def text(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def names(value) -> tuple[str, ...]:
    if isinstance(value, str):
        value = value.split(",")
    if not isinstance(value, list | tuple):
        return ()
    return tuple(name for name in (text(item) for item in value) if name)


def read(path: Path) -> Branding:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return Branding()
    if not isinstance(data, dict):
        return Branding()
    return Branding(
        title=text(data.get("title")) or TITLE,
        subtitle=text(data.get("subtitle")),
        institution=text(data.get("institution")),
        team=names(data.get("team")),
        supervisor=text(data.get("supervisor")),
    )
