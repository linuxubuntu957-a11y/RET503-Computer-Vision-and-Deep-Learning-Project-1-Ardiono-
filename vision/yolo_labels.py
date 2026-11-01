"""Format label YOLO: parsing, validasi, dan konversi kotak. Murni Python (tanpa cv2/numpy)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


@dataclass(frozen=True)
class Box:
    cls: int
    cx: float
    cy: float
    w: float
    h: float


def parse_line(line: str) -> Box:
    """'<kelas> <cx> <cy> <w> <h>' -> Box. ValueError dengan pesan jelas jika salah format."""
    t = line.split()
    if len(t) != 5:
        raise ValueError(f"harus 5 nilai, ditemukan {len(t)}")
    try:
        return Box(int(t[0]), float(t[1]), float(t[2]), float(t[3]), float(t[4]))
    except ValueError:
        raise ValueError("ada nilai bukan angka (class_id harus bilangan bulat)")


def check_box(b: Box, nc: int, tol: float = 1e-3) -> Tuple[List[str], List[str]]:
    """(error, peringatan) untuk satu kotak."""
    errs, warns = [], []
    if not 0 <= b.cls < nc:
        errs.append(f"class_id {b.cls} di luar 0..{nc - 1}")
    if not all(0 <= v <= 1 for v in (b.cx, b.cy, b.w, b.h)):
        errs.append("koordinat di luar 0-1 (mungkin masih piksel / format COCO?)")
    elif b.w <= 0 or b.h <= 0:
        errs.append("lebar/tinggi <= 0")
    elif (b.cx - b.w / 2 < -tol or b.cx + b.w / 2 > 1 + tol or
          b.cy - b.h / 2 < -tol or b.cy + b.h / 2 > 1 + tol):
        warns.append("kotak keluar batas gambar")
    return errs, warns


def read_label_file(path, nc: int) -> Tuple[List[Box], List[str], List[str]]:
    """(kotak valid, error, peringatan) dari satu file label. File tidak ada -> daftar kosong."""
    path = Path(path)
    boxes, errs, warns = [], [], []
    if not path.exists():
        return boxes, errs, warns
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            b = parse_line(line)
        except ValueError as e:
            errs.append(f"{path.name}:{i}: {e}")
            continue
        e2, w2 = check_box(b, nc)
        errs += [f"{path.name}:{i}: {m}" for m in e2]
        warns += [f"{path.name}:{i}: {m}" for m in w2]
        if not e2:
            boxes.append(b)
    return boxes, errs, warns


def xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float, W: int, H: int
                 ) -> Tuple[float, float, float, float]:
    """Sudut piksel -> (cx, cy, w, h) ternormalisasi."""
    return ((x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H)


def yolo_to_xyxy(cx: float, cy: float, w: float, h: float, W: int, H: int
                 ) -> Tuple[float, float, float, float]:
    """(cx, cy, w, h) ternormalisasi -> sudut piksel (x1, y1, x2, y2)."""
    return ((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H)


def format_line(b: Box) -> str:
    return f"{b.cls} {b.cx:.6f} {b.cy:.6f} {b.w:.6f} {b.h:.6f}"


def count_per_class(boxes: Iterable[Box], nc: int) -> List[int]:
    c = [0] * nc
    for b in boxes:
        c[b.cls] += 1
    return c
