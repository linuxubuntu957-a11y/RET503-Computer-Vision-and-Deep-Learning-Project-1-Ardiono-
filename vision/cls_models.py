"""Model klasifikasi transfer learning (butuh torch, torchvision)."""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch.nn as nn
from torchvision import models

ARCHS = ("resnet18", "mobilenet_v3_small")
MODES = ("feature", "partial", "scratch")
LR_HEAD, LR_MID, LR_SCRATCH = 1e-3, 1e-4, 1e-3


def build(arch: str, mode: str, n_cls: int) -> Tuple[nn.Module, List[Dict]]:
    """Kembalikan (model, grup parameter untuk optimizer).
    feature = hanya head | partial = blok akhir + head | scratch = semua, bobot acak."""
    if arch not in ARCHS or mode not in MODES:
        raise ValueError(f"arch harus salah satu {ARCHS}, mode salah satu {MODES}")
    pre = mode != "scratch"
    if arch == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pre else None)
        m.fc = nn.Linear(m.fc.in_features, n_cls)
        head_p, mid_p = ("fc.",), ("layer4.",)
    else:
        m = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pre else None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, n_cls)
        head_p, mid_p = ("classifier.",), ("features.11.", "features.12.")
    for n, p in m.named_parameters():
        if mode == "scratch":
            p.requires_grad = True
        elif mode == "feature":
            p.requires_grad = n.startswith(head_p)
        else:
            p.requires_grad = n.startswith(head_p + mid_p)
    head = [p for n, p in m.named_parameters() if n.startswith(head_p)]
    mid = [p for n, p in m.named_parameters() if n.startswith(mid_p)]
    if mode == "scratch":
        groups = [{"params": list(m.parameters()), "lr": LR_SCRATCH}]
    elif mode == "feature":
        groups = [{"params": head, "lr": LR_HEAD}]
    else:
        groups = [{"params": mid, "lr": LR_MID}, {"params": head, "lr": LR_HEAD}]
    return m, groups


def set_train(m: nn.Module) -> None:
    """train(), tetapi BatchNorm pada lapisan beku tetap eval() (statistik ImageNet tidak berubah)."""
    m.train()
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d) and not any(p.requires_grad for p in mod.parameters()):
            mod.eval()


def count_trainable(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
