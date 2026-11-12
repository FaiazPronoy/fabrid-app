from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet50
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.ops.misc import FrozenBatchNorm2d

from defects.labels import CLASSES

STRIPS = ((1, 3), (3, 1), (1, 5), (5, 1), (1, 7), (7, 1))
LEVELS = ("0", "1", "2", "3")
ANCHOR_SIZES = ((8, 16, 24), (16, 32, 48), (32, 64, 96), (64, 128, 192))
ANCHOR_RATIOS = (0.25, 0.5, 1.0, 2.0, 4.0)
N_OUTPUTS = len(CLASSES) + 1


def conv_bn_silu(inputs: int, outputs: int, kernel, padding) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, kernel, padding=padding, bias=False),
        nn.BatchNorm2d(outputs),
        nn.SiLU(),
    )


class MSSA(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(16, channels // reduction)
        self.branches = nn.ModuleList(
            conv_bn_silu(channels, mid, size, (size[0] // 2, size[1] // 2)) for size in STRIPS
        )
        self.fuse = conv_bn_silu(mid * len(STRIPS), channels, 1, 0)
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1),
            nn.SiLU(),
            nn.Conv2d(mid, channels, 1),
            nn.Sigmoid(),
        )
        self.spatial = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid())
        self.out = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False), nn.BatchNorm2d(channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fuse(torch.cat([branch(x) for branch in self.branches], dim=1))
        y = y * self.channel(y)
        pooled = torch.cat([y.mean(dim=1, keepdim=True), y.amax(dim=1, keepdim=True)], dim=1)
        y = y * self.spatial(pooled)
        return F.silu(x + self.out(y))


class Neck(nn.Module):
    def __init__(self, base: nn.Module, levels: tuple[str, ...], mssa: bool):
        super().__init__()
        self.base = base
        self.out_channels = base.out_channels
        self.levels = levels
        blocks = {level: MSSA(base.out_channels) for level in levels}
        self.blocks = nn.ModuleDict(blocks) if mssa else None

    def forward(self, x: torch.Tensor) -> OrderedDict:
        features = self.base(x)
        kept = OrderedDict((k, v) for k, v in features.items() if k in self.levels)
        if self.blocks is None:
            return kept
        return OrderedDict((k, self.blocks[k](v)) for k, v in kept.items())


class AuxHeads(nn.Module):
    def __init__(self, channels: int, settings: dict):
        super().__init__()
        self.settings = dict(settings)
        self.semantic = nn.Sequential(
            conv_bn_silu(channels, 128, 3, 1), nn.Conv2d(128, N_OUTPUTS, 1)
        )
        self.boundary = nn.Sequential(conv_bn_silu(channels, 64, 3, 1), nn.Conv2d(64, 1, 1))


class Hybrid(MaskRCNN):
    def __init__(self, backbone, num_classes: int, aux=None, **kwargs):
        super().__init__(backbone, num_classes=num_classes, **kwargs)
        self.aux = aux

    def forward(self, images, targets=None):
        if self.training:
            raise RuntimeError("this copy of the model is for inference only")
        return super().forward(images, targets)


def build_hybrid(config: dict) -> Hybrid:
    parts = config["components"]
    if parts.get("preprocess"):
        raise ValueError("the preprocess component is not part of the inference copy")
    base = _resnet_fpn_extractor(resnet50(norm_layer=FrozenBatchNorm2d), 3)
    levels = LEVELS if parts["anchors"] else (*LEVELS, "pool")
    options = {"min_size": config["imgsz"], "max_size": config["imgsz"]}
    if parts["anchors"]:
        ratios = (ANCHOR_RATIOS,) * len(ANCHOR_SIZES)
        options["rpn_anchor_generator"] = AnchorGenerator(ANCHOR_SIZES, ratios)
    aux = AuxHeads(base.out_channels, config["aux"]) if parts["aux"] else None
    return Hybrid(Neck(base, levels, parts["mssa"]), N_OUTPUTS, aux, **options)


def build_maskrcnn(config: dict) -> MaskRCNN:
    base = _resnet_fpn_extractor(resnet50(norm_layer=FrozenBatchNorm2d), 3)
    options = {"min_size": config["imgsz"], "max_size": config["imgsz"]}
    return MaskRCNN(base, num_classes=N_OUTPUTS, **options)
