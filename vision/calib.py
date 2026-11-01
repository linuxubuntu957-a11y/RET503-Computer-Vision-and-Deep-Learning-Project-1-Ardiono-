"""Muat hasil kalibrasi (.npz / .json) dan koreksi distorsi."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

Size = Tuple[int, int]   # (lebar, tinggi)


@dataclass(frozen=True)
class Calibration:
    K: np.ndarray
    dist: np.ndarray
    size: Optional[Size] = None

    @classmethod
    def load(cls, path) -> "Calibration":
        """Key yang dikenali: K / mtx, dist, size / image_size."""
        path = Path(path)
        if path.suffix.lower() == ".json":
            d = json.loads(path.read_text(encoding="utf-8"))
            K = np.array(d["K"] if "K" in d else d["mtx"], np.float64)
            dist = np.array(d["dist"], np.float64).reshape(1, -1)
            raw = d.get("size", d.get("image_size"))
        else:
            d = np.load(path)
            K = np.array(d["K"] if "K" in d.files else d["mtx"], np.float64)
            dist = np.array(d["dist"], np.float64).reshape(1, -1)
            key = "size" if "size" in d.files else ("image_size" if "image_size" in d.files else None)
            raw = d[key] if key else None
        size = (int(raw[0]), int(raw[1])) if raw is not None else None
        return cls(K, dist, size)

    def check_frame_size(self, frame_size: Size) -> None:
        """Error jika resolusi frame berbeda dari resolusi kalibrasi (K jadi tidak berlaku)."""
        if self.size is not None and tuple(self.size) != tuple(frame_size):
            raise ValueError(f"Resolusi frame {frame_size[0]}x{frame_size[1]} != resolusi kalibrasi "
                             f"{self.size[0]}x{self.size[1]}. Samakan dulu.")

    def undistorter(self, size: Size, alpha: float = 0.0) -> "Undistorter":
        return Undistorter(self.K, self.dist, size, alpha)


class Undistorter:
    """Peta koreksi dihitung sekali; dipanggil per frame: `frame_u = undistorter(frame)`."""

    def __init__(self, K, dist, size: Size, alpha: float = 0.0):
        w, h = size
        self.new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha, (w, h))
        self._mx, self._my = cv2.initUndistortRectifyMap(K, dist, None, self.new_K, (w, h), cv2.CV_32FC1)

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        return cv2.remap(frame_bgr, self._mx, self._my, cv2.INTER_LINEAR)
