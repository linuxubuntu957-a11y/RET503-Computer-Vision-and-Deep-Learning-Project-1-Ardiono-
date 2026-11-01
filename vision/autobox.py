"""Usulan kotak balok berwarna lewat segmentasi warna HSV (bukan detektor belajar).
Hasilnya USULAN untuk dicek (draw_labels.py), bukan label final. Profil warna hanya untuk kelas
yang sudah diuji pada foto nyata (merah, hijau=hijau-kebiruan/teal, hitam); 'biru' belum diuji.
Kelas tanpa profil (mis. putih di kertas putih) -> None: beri label manual."""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]   # x1, y1, x2, y2 (piksel)


def _merah(h, s, v): return ((h < 10) | (h > 170)) & (s > 110) & (v > 60)
def _hijau(h, s, v): return (h > 70) & (h < 105) & (s > 70) & (v > 50)
def _hitam(h, s, v): return v < 55
def _biru(h, s, v): return (h >= 105) & (h <= 130) & (s > 90) & (v > 50)    # BELUM diuji


PROFILES: Dict[str, Callable] = {"merah": _merah, "hijau": _hijau, "hitam": _hitam, "biru": _biru}


def has_profile(kelas: str) -> bool:
    return kelas in PROFILES


def propose_box(bgr: np.ndarray, kelas: str, min_area: int = 400, max_frac: float = 0.4,
                pad: int = 3) -> Optional[Box]:
    """Kotak komponen warna terbesar yang TIDAK menyentuh tepi frame (bayangan/benda gelap di tepi
    diabaikan). None bila tidak ada profil, tidak ada kandidat, atau kandidat terlalu besar."""
    fn = PROFILES.get(kelas)
    if fn is None:
        return None
    H, W = bgr.shape[:2]
    h, s, v = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV))
    m = fn(h, s, v).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    best, best_area = None, 0
    for i in range(1, n):
        x, y, w, hh, a = (int(t) for t in st[i])
        if a < min_area or a > max_frac * W * H:
            continue
        if x <= 0 or y <= 0 or x + w >= W or y + hh >= H:      # menyentuh tepi -> bukan balok utuh
            continue
        if a > best_area:
            best, best_area = (x, y, w, hh), a
    if best is None:
        return None
    x, y, w, hh = best
    return (max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + hh + pad))
