import itertools

import numpy as np

from defects.labels import COLORS

BACKGROUND = (244, 246, 250)
CARD = (255, 255, 255)
BORDER = (226, 231, 238)
TEXT = (17, 24, 39)
MUTED = (75, 85, 99)
ACCENT = (67, 56, 202)
ACCENT_HOVER = (55, 48, 163)
FAINT = (238, 241, 245)
FAINT_TEXT = (95, 104, 117)
WHITE = (255, 255, 255)
VERDICTS = {
    "clear": {"fill": (231, 246, 236), "edge": (46, 125, 79), "text": (20, 83, 45)},
    "found": {"fill": (255, 244, 224), "edge": (183, 121, 31), "text": (110, 64, 0)},
    "error": {"fill": (253, 236, 236), "edge": (185, 28, 28), "text": (127, 29, 29)},
    "idle": {"fill": (238, 241, 245), "edge": (148, 163, 184), "text": (51, 65, 85)},
}
SIMULATIONS = {
    "protanopia": (
        (0.152286, 1.052583, -0.204868),
        (0.114503, 0.786281, 0.099216),
        (-0.003882, -0.048116, 1.051998),
    ),
    "deuteranopia": (
        (0.367322, 0.860646, -0.227968),
        (0.280085, 0.672501, 0.047413),
        (-0.011820, 0.042940, 0.968881),
    ),
    "tritanopia": (
        (1.255528, -0.076749, -0.178779),
        (-0.078411, 0.930809, 0.147602),
        (0.004733, 0.691367, 0.303900),
    ),
}
MODES = {
    "light": {
        "bg": BACKGROUND,
        "card": CARD,
        "border": BORDER,
        "text": TEXT,
        "muted": MUTED,
        "accent": ACCENT,
        "accent-fill": ACCENT,
        "accent-hover": ACCENT_HOVER,
        "faint": FAINT,
        "faint-text": FAINT_TEXT,
        "hover": FAINT,
        "clear-fill": VERDICTS["clear"]["fill"],
        "clear-edge": VERDICTS["clear"]["edge"],
        "clear-text": VERDICTS["clear"]["text"],
        "found-fill": VERDICTS["found"]["fill"],
        "found-edge": VERDICTS["found"]["edge"],
        "found-text": VERDICTS["found"]["text"],
        "error-fill": VERDICTS["error"]["fill"],
        "error-edge": VERDICTS["error"]["edge"],
        "error-text": VERDICTS["error"]["text"],
    },
    "dark": {
        "bg": (15, 23, 42),
        "card": (23, 32, 51),
        "border": (51, 65, 85),
        "text": (241, 245, 249),
        "muted": (176, 186, 200),
        "accent": (165, 180, 252),
        "accent-fill": (79, 70, 229),
        "accent-hover": (67, 56, 202),
        "faint": (36, 48, 70),
        "faint-text": (176, 186, 200),
        "hover": (36, 48, 70),
        "clear-fill": (17, 51, 35),
        "clear-edge": (52, 211, 153),
        "clear-text": (187, 247, 208),
        "found-fill": (61, 43, 10),
        "found-edge": (245, 158, 11),
        "found-text": (253, 230, 138),
        "error-fill": (64, 20, 20),
        "error-edge": (248, 113, 113),
        "error-text": (254, 202, 202),
    },
}
MIN_TEXT_CONTRAST = 4.5
MIN_DISTANCE = 20.0


def hexcode(color) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(int(v) for v in color))


def linear(color) -> np.ndarray:
    values = np.asarray(color, dtype=np.float64) / 255.0
    return np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)


def luminance(color) -> float:
    red, green, blue = linear(color)
    return float(0.2126 * red + 0.7152 * green + 0.0722 * blue)


def contrast(first, second) -> float:
    high, low = sorted((luminance(first), luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def text_on(color) -> tuple[int, int, int]:
    return WHITE if contrast(color, WHITE) >= contrast(color, TEXT) else TEXT


def lab(rgb_linear: np.ndarray) -> np.ndarray:
    matrix = np.array(
        [[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]
    )
    xyz = matrix @ np.clip(rgb_linear, 0, 1) / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def distance(first, second, simulation: str | None = None) -> float:
    a, b = linear(first), linear(second)
    if simulation is not None:
        matrix = np.array(SIMULATIONS[simulation])
        a, b = matrix @ a, matrix @ b
    return float(np.linalg.norm(lab(a) - lab(b)))


def separation(colors=None) -> dict[str, float]:
    colors = list((colors or COLORS).values())
    result = {}
    for simulation in (None, *SIMULATIONS):
        pairs = itertools.combinations(colors, 2)
        result[simulation or "normal"] = min(distance(a, b, simulation) for a, b in pairs)
    return result


def text_pairs() -> dict[str, float]:
    pairs = {
        "text on background": contrast(TEXT, BACKGROUND),
        "text on card": contrast(TEXT, CARD),
        "muted on card": contrast(MUTED, CARD),
        "muted on background": contrast(MUTED, BACKGROUND),
        "faint text on faint tile": contrast(FAINT_TEXT, FAINT),
        "white on accent": contrast(WHITE, ACCENT),
    }
    for name, color in COLORS.items():
        pairs[f"label on {name}"] = contrast(text_on(color), color)
    for kind, colors in VERDICTS.items():
        pairs[f"{kind} verdict"] = contrast(colors["text"], colors["fill"])
    return pairs


def mode_pairs(mode: str) -> dict[str, float]:
    tokens = MODES[mode]
    pairs = {
        "text on background": contrast(tokens["text"], tokens["bg"]),
        "text on card": contrast(tokens["text"], tokens["card"]),
        "muted on card": contrast(tokens["muted"], tokens["card"]),
        "muted on background": contrast(tokens["muted"], tokens["bg"]),
        "faint text on faint tile": contrast(tokens["faint-text"], tokens["faint"]),
        "text on hover": contrast(tokens["text"], tokens["hover"]),
        "accent on card": contrast(tokens["accent"], tokens["card"]),
        "white on accent button": contrast(WHITE, tokens["accent-fill"]),
        "white on accent hover": contrast(WHITE, tokens["accent-hover"]),
    }
    for kind in ("clear", "found", "error"):
        pairs[f"{kind} verdict"] = contrast(tokens[f"{kind}-text"], tokens[f"{kind}-fill"])
    return pairs


def css_variables(mode: str) -> dict[str, str]:
    return {f"--fd-{name}": hexcode(value) for name, value in MODES[mode].items()}
