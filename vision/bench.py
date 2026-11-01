"""Pengukuran waktu. `bench` murni; `bench_torch_models` mengimpor torch hanya saat dipanggil."""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


def bench(fn: Callable[[], object], n: int = 100, warm: int = 20,
          sync: Optional[Callable[[], None]] = None) -> Tuple[float, float]:
    """(rata-rata ms, p95 ms) untuk fn() setelah pemanasan."""
    for _ in range(warm):
        fn()
    if sync:
        sync()
    ts: List[float] = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        if sync:
            sync()
        ts.append((time.perf_counter() - t) * 1000)
    return float(np.mean(ts)), float(np.percentile(ts, 95))


def bench_torch_models(names: List[str], n_classes: int, img_size: int) -> List[Dict]:
    """Latensi inferensi model torchvision (bobot acak) pada CPU dan (jika ada) CUDA."""
    import torch
    from torchvision import models
    makers = {"resnet18": models.resnet18, "mobilenet_v3_small": models.mobilenet_v3_small}
    rows = []
    devs = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for name in names:
        net = makers[name](weights=None, num_classes=n_classes)
        params = sum(p.numel() for p in net.parameters()) / 1e6
        for d in devs:
            net = net.to(d).eval()
            x = torch.randn(1, 3, img_size, img_size, device=d)
            sync = torch.cuda.synchronize if d == "cuda" else None
            with torch.inference_mode():
                mean, p95 = bench(lambda: net(x), sync=sync)
            rows.append({"model": name, "device": d, "params_m": params, "mean_ms": mean, "p95_ms": p95})
    return rows
