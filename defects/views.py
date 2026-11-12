import pandas as pd

from defects.service import Result
from defects.table import COLUMNS

FULL = (
    "session",
    "entry",
    "original",
    "annotated",
    "status",
    "table",
    "model_input",
    "input_note",
    "png",
    "csv",
    "pdf",
)
LIGHT = ("session", "entry", "annotated", "status", "table", "png", "csv", "pdf")


def frame(listed: list[list]) -> pd.DataFrame:
    return pd.DataFrame(listed, columns=list(COLUMNS))


def values(result: Result, session: str, entry: str, clear: bool, keep) -> dict:
    if result.ok:
        return {
            "session": session,
            "entry": result.entry,
            "original": result.display,
            "annotated": result.annotated,
            "status": result.message,
            "table": frame(result.rows),
            "model_input": result.model_input,
            "input_note": result.input_note,
            "png": str(result.png),
            "csv": str(result.csv),
            "pdf": None,
        }
    message = f"**Could not complete:** {result.message}"
    if clear:
        cleared = dict.fromkeys(FULL)
        cleared.update(session=session, entry=entry, status=message, table=frame([]), input_note="")
        return cleared
    kept = dict.fromkeys(FULL, keep)
    kept.update(session=session, entry=entry, status=message)
    return kept


def pick(chosen: dict, names: tuple[str, ...]) -> tuple:
    return tuple(chosen[name] for name in names)


def unchanged(session: str, entry: str, names: tuple[str, ...], keep) -> tuple:
    chosen = dict.fromkeys(names, keep)
    chosen.update(session=session, entry=entry)
    return pick(chosen, names)
