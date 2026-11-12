import json
from pathlib import Path

NOT_EVALUATED = "not evaluated yet"


def read(path: Path) -> dict:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must hold one object per model")
    return data


def headline(entry: dict | None) -> str:
    if not entry or entry.get("value") is None:
        return f"Validation mask AP50: {NOT_EVALUATED}."
    text = f"Validation {entry['metric']}: {entry['value']:.3f}"
    if entry.get("interval"):
        low, high = entry["interval"]
        text += f" (95% interval {low:.3f} to {high:.3f})"
    return (
        f"{text}, measured on fabric from capture sessions not seen in training, "
        f"single seed ({entry['seed']})."
    )


def details(entry: dict | None) -> str:
    if not entry or entry.get("value") is None:
        return NOT_EVALUATED
    return (
        f"Scored by: {entry['evaluator']}. Source: `{entry['source']}`, field `{entry['field']}`."
    )


def threshold(entry: dict | None) -> float | None:
    chosen = (entry or {}).get("default_threshold") or {}
    value = chosen.get("value")
    return float(value) if value is not None else None
