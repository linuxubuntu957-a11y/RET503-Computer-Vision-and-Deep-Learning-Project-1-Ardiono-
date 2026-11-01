"""Papan catur kalibrasi: titik objek dan deteksi sudut."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

_CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def object_points(cols: int, rows: int, square: float) -> np.ndarray:
    """Titik papan pada bidang Z=0, satuan = `square` (mis. mm). cols x rows = SUDUT DALAM."""
    p = np.zeros((cols * rows, 3), np.float32)
    p[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    return p


def detect(gray: np.ndarray, cols: int, rows: int) -> Optional[np.ndarray]:
    """Sudut papan dengan presisi sub-piksel (N,1,2), atau None bila tidak terdeteksi."""
    ok, c = cv2.findChessboardCorners(gray, (cols, rows))
    if not ok:
        return None
    return cv2.cornerSubPix(gray, c, (11, 11), (-1, -1), _CRIT)
