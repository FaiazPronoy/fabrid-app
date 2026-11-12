from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    kind: str
    weights: str
    class_offset: int
    default_threshold: float
    settings: dict = field(default_factory=dict)


MODELS = (
    ModelSpec(
        key="yolov8m_seg_aug",
        name="YOLOv8m-seg (augmented)",
        kind="ultralytics",
        weights="weights/yolov8m_seg_aug.pt",
        class_offset=0,
        default_threshold=0.25,
        settings={"imgsz": 640, "conf": 0.05, "iou": 0.7, "max_det": 100},
    ),
    ModelSpec(
        key="fabrimssa_net",
        name="FabriMSSA-Net",
        kind="fabrimssa",
        weights="weights/fabrimssa_net.pth",
        class_offset=1,
        default_threshold=0.5,
        settings={
            "imgsz": 640,
            "mask_threshold": 0.5,
            "components": {"anchors": True, "mssa": True, "aux": True, "preprocess": False},
            "aux": {"focal": 1.0, "dice": 1.0, "boundary": 0.5, "gamma": 2.0},
        },
    ),
)


def find(key: str, specs=MODELS) -> ModelSpec:
    for spec in specs:
        if spec.key == key:
            return spec
    raise KeyError(f"unknown model {key}")
