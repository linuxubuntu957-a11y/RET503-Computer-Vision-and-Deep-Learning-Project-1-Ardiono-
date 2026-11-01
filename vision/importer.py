"""Impor foto yang sudah ada (mis. dari aplikasi Kamera Windows, 'WIN_YYYYMMDD_HH_MM_SS_Pro.jpg')
ke format kit: citra di-undistort, dinamai sesuai pola kit, label YOLO (usulan) + crop klasifikasi."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import autobox, files
from .yolo_labels import Box as YoloBox, format_line, xyxy_to_yolo

TIME_RE = re.compile(r"(\d{8})_(\d{2})_(\d{2})_(\d{2})")


def photo_datetime(name: str) -> Optional[datetime]:
    """'WIN_20261026_22_59_08_Pro.jpg' -> datetime; None bila nama tidak berpola."""
    m = TIME_RE.search(name)
    if not m:
        return None
    try:
        return datetime.strptime("".join(m.groups()), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def sort_by_time(paths: Sequence[Path]) -> List[Path]:
    """Urut menurut waktu pada nama; yang tak berpola di belakang, urut nama."""
    return sorted(paths, key=lambda p: (photo_datetime(p.name) or datetime.max, p.name))


def split_sessions(n_items: int, n_sessions: int) -> List[int]:
    """Indeks sesi (0..) per item untuk n_items berurutan: blok berurutan berukuran hampir sama.
    Foto bertetangga mirip, jadi satu blok = satu sesi agar tidak terbelah antara train dan val."""
    k = max(1, min(n_sessions, n_items))
    return [i * k // n_items for i in range(n_items)]


def match_class(folder_name: str, names: Sequence[str]) -> str:
    """'balok merah' / 'balok_merah.zip' -> 'merah' (tepat satu nama kelas harus cocok sebagai kata)."""
    tokens = re.split(r"[^a-z0-9]+", Path(folder_name).stem.lower() if folder_name.lower().endswith(".zip")
                      else folder_name.lower())
    hits = [n for n in names if n.lower() in tokens]
    if len(hits) != 1:
        raise ValueError(f"'{folder_name}': kelas tidak ketemu/ambigu di nama (cocok: {hits}; pilihan: {list(names)})")
    return hits[0]


def square_crop(img: np.ndarray, box: Tuple[int, int, int, int], margin: float = 0.3,
                min_side: int = 64) -> np.ndarray:
    """Potong persegi berpusat di kotak, sisi = max(w,h)*(1+margin), digeser agar tetap di dalam frame."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box
    side = int(round(max(x2 - x1, y2 - y1) * (1 + margin)))
    side = min(max(side, min_side), H, W)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    x0 = min(max(cx - side // 2, 0), W - side)
    y0 = min(max(cy - side // 2, 0), H - side)
    return img[y0:y0 + side, x0:x0 + side]


@dataclass
class ImportReport:
    kelas: str
    n_total: int = 0
    n_labeled: int = 0
    no_box: List[str] = field(default_factory=list)     # tak ada usulan kotak -> label manual
    skipped: List[str] = field(default_factory=list)    # tak terbaca / resolusi beda
    sessions: List[str] = field(default_factory=list)


def import_class(kelas: str, cls_id: int, paths: Sequence[Path], out_cls, out_det, lokasi: str = "lab",
                 n_sessions: int = 5, undistort: Optional[Callable] = None,
                 frame_size: Optional[Tuple[int, int]] = None, day: Optional[str] = None) -> ImportReport:
    """Proses satu kelas. Keluaran:
      out_det/<nama>.png + .txt   frame penuh + label YOLO usulan (untuk deteksi)
      out_det + '_perlu_label'/   frame tanpa usulan kotak (JANGAN dimasukkan ke out_det: dianggap background)
      out_cls/<kelas>/<nama>.png  crop persegi (untuk klasifikasi) + out_cls/metadata.csv"""
    out_cls, out_det = Path(out_cls), Path(out_det)
    cdir, manual = out_cls / kelas, Path(str(out_det) + "_perlu_label")
    if cdir.exists() and any(cdir.glob(f"{kelas}_*.png")):
        raise FileExistsError(f"'{cdir}' sudah berisi citra kelas '{kelas}'. Hapus dulu bila ingin impor ulang.")
    for d in (cdir, out_det):
        d.mkdir(parents=True, exist_ok=True)
    ordered = sort_by_time(paths)
    stamps = [photo_datetime(p.name) for p in ordered]
    day = day or next((t.strftime("%Y%m%d") for t in stamps if t), date.today().strftime("%Y%m%d"))
    sess_idx = split_sessions(len(ordered), n_sessions)
    rep, counters = ImportReport(kelas, n_total=len(ordered)), {}
    for p, si in zip(ordered, sess_idx):
        img = cv2.imread(str(p))
        if img is None or (frame_size and (img.shape[1], img.shape[0]) != tuple(frame_size)):
            rep.skipped.append(p.name); continue
        if undistort is not None:
            img = undistort(img)
        kondisi = f"s{si + 1}"
        counters[kondisi] = counters.get(kondisi, 0) + 1
        prefix = files.make_prefix(kelas, day, lokasi, kondisi)
        name = f"{prefix}{counters[kondisi]:03d}"
        if kondisi not in rep.sessions:
            rep.sessions.append(kondisi)
        box = autobox.propose_box(img, kelas)
        if box is None:
            manual.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(manual / f"{name}.png"), img); rep.no_box.append(f"{name}.png"); continue
        H, W = img.shape[:2]
        cv2.imwrite(str(out_det / f"{name}.png"), img)
        (out_det / f"{name}.txt").write_text(format_line(YoloBox(cls_id, *xyxy_to_yolo(*box, W, H))) + "\n")
        cv2.imwrite(str(cdir / f"{name}.png"), square_crop(img, box))
        files.append_metadata(out_cls / "metadata.csv", [f"{name}.png", kelas, day, kondisi])
        rep.n_labeled += 1
    return rep
