"""Menggambar kotak label pada gambar (pemeriksaan visual)."""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from .yolo_labels import Box, yolo_to_xyxy

PALETTE = [(255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 128, 255), (0, 255, 0)]   # BGR


def draw_boxes(img: np.ndarray, boxes: Sequence[Box], names: Sequence[str]) -> np.ndarray:
    """Kembalikan salinan `img` dengan kotak + 'id:nama' tergambar."""
    out = img.copy()
    H, W = out.shape[:2]
    for b in boxes:
        x1, y1, x2, y2 = (int(v) for v in yolo_to_xyxy(b.cx, b.cy, b.w, b.h, W, H))
        col = PALETTE[b.cls % len(PALETTE)]
        name = names[b.cls] if b.cls < len(names) else f"?{b.cls}"
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        cv2.putText(out, f"{b.cls}:{name}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    return out
