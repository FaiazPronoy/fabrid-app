CLASSES = ("needle mark", "hole", "missing yarn", "stain")
COLORS = {
    "needle mark": (136, 34, 85),
    "hole": (51, 136, 238),
    "missing yarn": (221, 170, 51),
    "stain": (17, 119, 51),
}


def class_name(index: int, offset: int) -> str:
    position = int(index) - offset
    if not 0 <= position < len(CLASSES):
        raise ValueError(
            f"model class index {index} has no defect class (offset {offset}, "
            f"expected {offset} to {offset + len(CLASSES) - 1})"
        )
    return CLASSES[position]
