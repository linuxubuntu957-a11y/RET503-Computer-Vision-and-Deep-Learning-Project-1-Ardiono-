"""Utilitas berkas dan penamaan: daftar gambar, penamaan sesi, metadata.csv."""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from .config import IMAGE_EXT

SCENE_RE = re.compile(r"^(.+_\d{8}_.+)_\d+$")   # <apa>_<TANGGAL>_<lokasi>_<kondisi>_<nomor>
META_HEADER = ("nama_file", "kelas", "tanggal", "kondisi_cahaya")


def list_images(folder, recursive: bool = True) -> List[Path]:
    """Semua gambar di folder, terurut menurut nama."""
    folder = Path(folder)
    it = folder.rglob("*") if recursive else folder.glob("*")
    return sorted(p for p in it if p.suffix.lower() in IMAGE_EXT)


def session_of_class_image(kelas: str, filename: str) -> str:
    """'<kelas>_<TANGGAL>_<lokasi>_<kondisi>_<nomor>.png' -> 'TANGGAL_lokasi_kondisi'."""
    stem = Path(filename).stem
    rest = stem[len(kelas) + 1:] if stem.startswith(kelas + "_") else stem
    return rest.rsplit("_", 1)[0] if "_" in rest else rest


def session_of_scene(stem: str) -> Optional[str]:
    """Sesi untuk gambar deteksi; None jika nama tidak mengikuti pola."""
    m = SCENE_RE.match(stem)
    return m.group(1) if m else None


def make_prefix(kelas: str, tanggal: str, lokasi: str, kondisi: str) -> str:
    return f"{kelas}_{tanggal}_{lokasi}_{kondisi}_"


def count_existing(folder, prefix: str) -> int:
    return len(list(Path(folder).glob(prefix + "*.png")))


def append_metadata(meta_path, row: Sequence[str]) -> None:
    """Tambah satu baris ke metadata.csv (header ditulis otomatis jika file baru)."""
    meta_path = Path(meta_path)
    new = not meta_path.exists()
    with open(meta_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(META_HEADER)
        w.writerow(list(row))


def list_class_dirs(root) -> List[str]:
    return sorted(d.name for d in Path(root).iterdir() if d.is_dir())
