"""Preprocessing model: HARUS identik saat melatih dan saat deploy."""
from __future__ import annotations

import cv2
import numpy as np

from .config import IMG_SIZE, MEAN, STD

_MEAN = np.array(MEAN, np.float32)
_STD = np.array(STD, np.float32)


def normalize_rgb(rgb_uint8: np.ndarray) -> np.ndarray:
    """RGB uint8 HxWx3 -> float32 CHW ternormalisasi ImageNet."""
    f = (rgb_uint8.astype(np.float32) / 255.0 - _MEAN) / _STD
    return np.ascontiguousarray(f.transpose(2, 0, 1))


def to_model_input(img_bgr: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """Citra BGR (sudah di-undistort) -> CHW float32: resize size x size, BGR->RGB, normalisasi."""
    f = cv2.resize(img_bgr, (size, size))
    return normalize_rgb(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
