CLASSES = ("needle mark", "hole", "missing yarn", "stain")
COLORS = {
    "needle mark": (230, 25, 75),
    "hole": (0, 130, 200),
    "missing yarn": (245, 160, 0),
    "stain": (60, 180, 75),
}


def class_name(index: int, offset: int) -> str:
    position = int(index) - offset
    if not 0 <= position < len(CLASSES):
        raise ValueError(
            f"model class index {index} has no defect class (offset {offset}, "
            f"expected {offset} to {offset + len(CLASSES) - 1})"
        )
    return CLASSES[position]
