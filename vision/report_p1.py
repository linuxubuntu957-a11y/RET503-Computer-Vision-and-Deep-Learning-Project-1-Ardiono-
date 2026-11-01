"""Pembuat bahan laporan P1 (kalibrasi kamera): statistik, gambar, dan ringkasan markdown."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np

from . import board, calib_eval
from .calib import Calibration
from .files import list_images


@dataclass
class PhotoSet:
    names: List[str] = field(default_factory=list)
    images: List[np.ndarray] = field(default_factory=list)
    corners: List[np.ndarray] = field(default_factory=list)
    stats: List[calib_eval.PoseStat] = field(default_factory=list)
    sharp: List[float] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)

    @property
    def errors(self) -> np.ndarray:
        return np.array([s.error for s in self.stats])


def collect(photo_dir, calib: Calibration, cols: int, rows: int, square: float) -> PhotoSet:
    """Baca foto, deteksi papan, dan hitung statistik pose terhadap kalibrasi yang tersimpan."""
    ps = PhotoSet()
    objp = board.object_points(cols, rows, square)
    for p in list_images(photo_dir, recursive=False):
        im = cv2.imread(str(p))
        if im is None or (calib.size and (im.shape[1], im.shape[0]) != tuple(calib.size)):
            ps.skipped.append(f"{p.name} (tak terbaca / ukuran beda)"); continue
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        c = board.detect(g, cols, rows)
        if c is None:
            ps.skipped.append(f"{p.name} (papan tidak terdeteksi)"); continue
        ps.names.append(p.name); ps.images.append(im); ps.corners.append(c)
        ps.stats.append(calib_eval.pose_stat(objp, c, calib))
        ps.sharp.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
    return ps


def corner_tiles(ps: PhotoSet, cols: int, rows: int, n_show: int = 6) -> np.ndarray:
    idx = np.linspace(0, len(ps.images) - 1, min(n_show, len(ps.images))).astype(int)
    tiles = []
    for i in idx:
        v = ps.images[i].copy()
        cv2.drawChessboardCorners(v, (cols, rows), ps.corners[i], True)
        v = cv2.resize(v, (640, 360))
        cv2.putText(v, f"foto {i + 1}: {len(ps.corners[i])} sudut", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        tiles.append(v)
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    return np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])


def before_after(ps: PhotoSet, calib: Calibration) -> np.ndarray:
    """Foto dengan papan paling jauh dari pusat (distorsi terbesar), sebelum dan sesudah undistort."""
    W, H = calib.size or (ps.images[0].shape[1], ps.images[0].shape[0])
    rad = [np.linalg.norm(c.reshape(-1, 2) - [W / 2, H / 2], axis=1).max() for c in ps.corners]
    im = ps.images[int(np.argmax(rad))]
    und = calib.undistorter((W, H))
    pair = []
    for img in (im.copy(), und(im)):
        for y in range(60, H, 120):
            cv2.line(img, (0, y), (W, y), (0, 255, 255), 1)
        for x in range(80, W, 160):
            cv2.line(img, (x, 0), (x, H), (0, 255, 255), 1)
        pair.append(cv2.resize(img, (640, 360)))
    out = np.hstack(pair)
    cv2.putText(out, "SEBELUM undistort", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(out, "SESUDAH undistort", (650, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return out


def plot_errors(errors: Sequence[float], rms: float, out_png) -> bool:
    plt = _plt()
    if plt is None:
        return False
    fig, ax = plt.subplots(figsize=(10, 4))
    cols = ["tab:red" if e > 1.0 else "tab:orange" if e > 0.5 else "tab:green" for e in errors]
    ax.bar(range(1, len(errors) + 1), errors, color=cols)
    ax.axhline(0.5, ls="--", c="gray"); ax.axhline(1.0, ls="--", c="k")
    ax.axhline(rms, c="tab:blue", label=f"RMS = {rms:.3f} px")
    ax.set_xlabel("foto ke-"); ax.set_ylabel("reprojection error (px)")
    ax.set_title("Error per foto (garis putus: 0,5 dan 1,0 px)"); ax.legend()
    fig.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close(fig)
    return True


def plot_coverage(points: np.ndarray, cov: np.ndarray, size, out_png) -> bool:
    plt = _plt()
    if plt is None:
        return False
    W, H = size
    gy, gx = cov.shape
    fig, ax = plt.subplots(figsize=(8, 4.7))
    ax.scatter(points[:, 0], points[:, 1], s=3, alpha=.5); ax.set_xlim(0, W); ax.set_ylim(H, 0)
    for k in range(1, gx):
        ax.axvline(W * k / gx, c="gray", lw=.8)
    for k in range(1, gy):
        ax.axhline(H * k / gy, c="gray", lw=.8)
    for r in range(gy):
        for c in range(gx):
            ax.text(W * (c + .5) / gx, H * (r + .5) / gy, str(cov[r, c]), ha="center", va="center", c="red", fontsize=14)
    ax.set_title("Cakupan titik sudut di citra (angka = jumlah titik per sel)")
    fig.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close(fig)
    return True


def summary_markdown(ps: PhotoSet, calib: Calibration, cols: int, rows: int, square: float) -> str:
    n, W_H = len(ps.names), calib.size
    errs = ps.errors
    rms = calib_eval.overall_rms(errs)
    allc = np.vstack([c.reshape(-1, 2) for c in ps.corners])
    cov = calib_eval.coverage_grid(allc, W_H)
    sb = [calib_eval.straightness(c, calib, cols, rows) for c in ps.corners]
    mb, xb = np.mean([s[0][0] for s in sb]), np.max([s[0][1] for s in sb])
    ma, xa = np.mean([s[1][0] for s in sb]), np.max([s[1][1] for s in sb])
    worst = np.argsort(-errs)[:3]
    K, d = calib.K, calib.dist.ravel()
    tilts, tz = [s.tilt for s in ps.stats], [s.tz for s in ps.stats]
    return "\n".join([
        "# Ringkasan Kalibrasi Kamera (P1)\n",
        f"- Foto dipakai: **{n}**, resolusi **{W_H[0]}x{W_H[1]}**, papan **{cols}x{rows} sudut dalam**, sisi kotak **{square} mm**",
        f"- **RMS reprojection error: {rms:.3f} px** ({calib_eval.grade_rms(rms)}; tabel slide 21)",
        f"- Error per foto: min {errs.min():.2f}, median {np.median(errs):.2f}, maks {errs.max():.2f} px; terburuk: "
        + ", ".join(f"foto {i + 1} ({errs[i]:.2f})" for i in worst),
        f"- Kemiringan papan: {min(tilts):.0f}-{max(tilts):.0f} derajat; jarak papan: {min(tz):.0f}-{max(tz):.0f} mm",
        f"- Ketajaman (Laplacian) terendah: {min(ps.sharp):.0f}\n",
        "## Matriks kamera K", "```", np.array2string(K, precision=2, suppress_small=True), "```",
        f"fx={K[0, 0]:.2f}, fy={K[1, 1]:.2f}, cx={K[0, 2]:.2f}, cy={K[1, 2]:.2f}  "
        f"(FOV horizontal ~ {2 * np.degrees(np.arctan(W_H[0] / 2 / K[0, 0])):.1f} derajat)\n",
        "## Koefisien distorsi (k1, k2, p1, p2, k3)", "```", np.array2string(d, precision=5, suppress_small=True), "```\n",
        "## Kelurusan garis papan (px, rata-rata / maks penyimpangan dari garis lurus)",
        f"- Sebelum undistort: {mb:.2f} / {xb:.2f}", f"- Sesudah undistort: {ma:.2f} / {xa:.2f}\n",
        "## Cakupan titik sudut per sel (4 kolom x 3 baris)", "```", str(cov), "```"])


def write_all(ps: PhotoSet, calib: Calibration, cols: int, rows: int, square: float, out_dir) -> List[str]:
    """Tulis semua keluaran ke out_dir. Mengembalikan daftar nama berkas yang dibuat."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    made = []
    cv2.imwrite(str(out / "P1_sudut_terdeteksi.jpg"), corner_tiles(ps, cols, rows)); made.append("P1_sudut_terdeteksi.jpg")
    cv2.imwrite(str(out / "P1_sebelum_sesudah.jpg"), before_after(ps, calib)); made.append("P1_sebelum_sesudah.jpg")
    rms = calib_eval.overall_rms(ps.errors)
    if plot_errors(ps.errors, rms, out / "P1_error_per_foto.png"):
        made.append("P1_error_per_foto.png")
    allc = np.vstack([c.reshape(-1, 2) for c in ps.corners])
    if plot_coverage(allc, calib_eval.coverage_grid(allc, calib.size), calib.size, out / "P1_cakupan.png"):
        made.append("P1_cakupan.png")
    (out / "P1_ringkasan.md").write_text(summary_markdown(ps, calib, cols, rows, square), encoding="utf-8")
    made.append("P1_ringkasan.md")
    return made


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None
