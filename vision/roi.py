"""Area minat (ROI) persegi panjang: simpan/muat dan potong gambar."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

Roi = Tuple[int, int, int, int]   # x, y, lebar, tinggi


def save_roi(path, roi: Roi) -> None:
    x, y, w, h = roi
    Path(path).write_text(json.dumps({"x": x, "y": y, "w": w, "h": h}), encoding="utf-8")


def load_roi(path) -> Optional[Roi]:
    p = Path(path)
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return (d["x"], d["y"], d["w"], d["h"])


def crop(img, roi: Optional[Roi]):
    """Potong gambar (numpy HxWxC) ke ROI; roi=None -> gambar utuh."""
    if roi is None:
        return img
    x, y, w, h = roi
    return img[y:y + h, x:x + w]
