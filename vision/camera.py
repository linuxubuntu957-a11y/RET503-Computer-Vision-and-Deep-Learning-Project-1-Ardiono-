"""Membuka kamera dengan pengaturan yang konsisten (resolusi tetap, autofocus dimatikan)."""
from __future__ import annotations

from typing import Tuple

import cv2


def open_camera(index: int = 0, width: int = 1280, height: int = 720,
                dshow: bool = False, autofocus_off: bool = True):
    """Buka kamera dan baca satu frame uji. Mengembalikan (cap, (lebar, tinggi) sebenarnya)."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if dshow else cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if autofocus_off:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)       # tidak semua kamera mendukung
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Kamera tidak terbaca (cek indeks kamera).")
    h, w = frame.shape[:2]
    return cap, (w, h)
