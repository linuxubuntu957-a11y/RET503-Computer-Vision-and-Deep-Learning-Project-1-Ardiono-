"""Pelatihan klasifikasi: satu epoch, satu mode penuh, dan penulisan hasil (butuh torch)."""
from __future__ import annotations

import csv
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .cls_models import build, count_trainable, set_train
from .config import IMG_SIZE


@dataclass
class FitResult:
    arch: str
    mode: str
    best_val: float = 0.0
    best_epoch: int = 0
    first_90: Optional[int] = None
    seconds: float = 0.0
    rows: List[list] = field(default_factory=list)   # epoch, tl, ta, vl, va, detik


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def run_epoch(model, loader, loss_fn, opt, device, train: bool):
    """Satu epoch. Mengembalikan (loss rata-rata, akurasi)."""
    set_train(model) if train else model.eval()
    tot, correct, ls = 0, 0, 0.0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            ls += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            tot += len(y)
    return ls / tot, correct / tot


def fit_mode(arch: str, mode: str, classes: List[str], tr_dl, va_dl, epochs: int, device,
             seed: int = 0, ckpt_dir=".", log=print) -> FitResult:
    """Latih satu mode; simpan bobot terbaik ke best_<arch>_<mode>.pt."""
    seed_all(seed)
    model, groups = build(arch, mode, len(classes))
    model = model.to(device)
    opt = torch.optim.Adam(groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()
    log(f"\n=== {arch} | {mode} | parameter dilatih: {count_trainable(model)/1e6:.2f} jt ===")
    res, t0 = FitResult(arch, mode), time.time()
    for ep in range(1, epochs + 1):
        tl, ta = run_epoch(model, tr_dl, loss_fn, opt, device, True)
        vl, va = run_epoch(model, va_dl, loss_fn, None, device, False)
        sched.step()
        res.rows.append([ep, tl, ta, vl, va, time.time() - t0])
        if va > res.best_val:
            res.best_val, res.best_epoch = va, ep
            torch.save({"state": model.state_dict(), "classes": classes, "arch": arch, "mode": mode,
                        "img_size": IMG_SIZE}, Path(ckpt_dir) / f"best_{arch}_{mode}.pt")
        if res.first_90 is None and va >= 0.90:
            res.first_90 = ep
        log(f"ep {ep:02d} | train {tl:.3f}/{ta:.3f} | val {vl:.3f}/{va:.3f} | {time.time()-t0:.0f}s")
        if ep <= 2 and va == 1.0:
            log("  ! val 100% di epoch awal -> curigai DATA LEAKAGE (cek split per sesi)")
    res.seconds = time.time() - t0
    return res


def write_logs(results: List[FitResult], out_dir="results") -> None:
    out = Path(out_dir); out.mkdir(exist_ok=True)
    for r in results:
        with open(out / f"log_{r.arch}_{r.mode}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "detik"])
            w.writerows(r.rows)
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arch", "mode", "val_acc_terbaik", "epoch_terbaik", "epoch_acc>=90%", "waktu_latih_detik"])
        for r in results:
            w.writerow([r.arch, r.mode, round(r.best_val, 4), r.best_epoch, r.first_90 or "-", round(r.seconds, 1)])


def plot_curves(results: List[FitResult], out_png="results/curve.png") -> bool:
    """Grafik akurasi val per epoch. False jika matplotlib tidak terpasang."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    for r in results:
        plt.plot(range(1, len(r.rows) + 1), [row[4] for row in r.rows], marker="o", label=r.mode)
    plt.xlabel("epoch"); plt.ylabel("akurasi val"); plt.title(results[0].arch if results else "")
    plt.legend(); plt.grid(alpha=.3)
    plt.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close()
    return True
