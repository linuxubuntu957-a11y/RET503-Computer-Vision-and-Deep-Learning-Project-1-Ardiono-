"""Evaluasi kualitas kalibrasi (fungsi murni; tanpa pembacaan berkas)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from .calib import Calibration


@dataclass
class PoseStat:
    error: float     # reprojection error RMS (px)
    tilt: float      # kemiringan papan (derajat)
    tz: float        # jarak papan ke kamera (satuan papan)


def grade_rms(rms: float) -> str:
    """Kategori menurut tabel slide 21 (pedoman, bukan standar baku)."""
    return "Sangat baik" if rms < 0.3 else "Baik" if rms < 0.5 else "Cukup" if rms < 1.0 else "Kurang baik"


def pose_stat(objp: np.ndarray, corners: np.ndarray, calib: Calibration) -> PoseStat:
    ok, rv, tv = cv2.solvePnP(objp, corners, calib.K, calib.dist)
    p, _ = cv2.projectPoints(objp, rv, tv, calib.K, calib.dist)
    err = float(np.sqrt(np.mean(np.sum((p.reshape(-1, 2) - corners.reshape(-1, 2)) ** 2, axis=1))))
    R = cv2.Rodrigues(rv)[0]
    tilt = float(np.degrees(np.arccos(min(1.0, abs(R[2, 2])))))
    return PoseStat(err, tilt, float(tv[2, 0]))


def overall_rms(errors: Sequence[float]) -> float:
    return float(np.sqrt(np.mean(np.square(errors))))


def line_deviation(pts: np.ndarray) -> float:
    """Penyimpangan maksimum titik dari garis lurus terbaik (px)."""
    c = pts - pts.mean(0)
    n = np.linalg.svd(c)[2][-1]
    return float(np.abs(c @ n).max())


def straightness(corners: np.ndarray, calib: Calibration, cols: int, rows: int
                 ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """((rata2, maks) sebelum undistort, (rata2, maks) sesudah) untuk baris dan kolom papan."""
    def dev(p):
        g = p.reshape(rows, cols, 2)
        v = [line_deviation(g[r]) for r in range(rows)] + [line_deviation(g[:, k]) for k in range(cols)]
        return float(np.mean(v)), float(np.max(v))
    raw = corners.reshape(-1, 2)
    und = cv2.undistortPoints(corners.reshape(-1, 1, 2), calib.K, calib.dist, P=calib.K).reshape(-1, 2)
    return dev(raw), dev(und)


def coverage_grid(points: np.ndarray, size: Tuple[int, int], gx: int = 4, gy: int = 3) -> np.ndarray:
    """Jumlah titik per sel grid (gy x gx) untuk melihat cakupan citra."""
    W, H = size
    ix = np.clip((points[:, 0] / W * gx).astype(int), 0, gx - 1)
    iy = np.clip((points[:, 1] / H * gy).astype(int), 0, gy - 1)
    cov = np.zeros((gy, gx), int)
    for x, y in zip(ix, iy):
        cov[y, x] += 1
    return cov


def corner_shift(calib: Calibration, size: Tuple[int, int]) -> List[float]:
    """Pergeseran titik pojok frame akibat koreksi distorsi (px)."""
    W, H = size
    pts = np.array([[[0, 0]], [[W - 1, 0]], [[0, H - 1]], [[W - 1, H - 1]]], np.float32)
    und = cv2.undistortPoints(pts, calib.K, calib.dist, P=calib.K)
    return [float(np.linalg.norm(a - b)) for a, b in zip(pts.reshape(-1, 2), und.reshape(-1, 2))]
