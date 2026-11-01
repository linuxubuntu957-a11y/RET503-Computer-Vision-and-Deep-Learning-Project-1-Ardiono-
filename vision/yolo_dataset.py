"""Menyusun folder dataset YOLO (images/labels per split) dan data.yaml."""
from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

from .yolo_labels import parse_line


def write_data_yaml(out, names: Sequence[str]) -> Path:
    out = Path(out)
    lines = [f"path: {out.resolve().as_posix()}", "train: images/train", "val: images/val",
             "test: images/test", "names:"] + [f"  {i}: {n}" for i, n in enumerate(names)]
    p = out / "data.yaml"
    p.write_text("\n".join(lines) + "\n")
    return p


def materialize(splits: Dict[str, List[Path]], lab_dir, out, names: Sequence[str],
                force: bool = False) -> Dict[str, Counter]:
    """Salin gambar + label ke `out`. Gambar tanpa label dibuatkan label kosong (= background).
    Menolak jika `out` sudah berisi images/labels (kecuali force=True) agar tidak ada sisa file lama."""
    out, lab_dir = Path(out), Path(lab_dir)
    if (out / "images").exists() or (out / "labels").exists():
        if not force:
            raise FileExistsError(f"Folder '{out}' sudah berisi images/labels. Hapus dulu atau pakai force "
                                  "(menghapus images/ dan labels/ di sana).")
        shutil.rmtree(out / "images", ignore_errors=True)
        shutil.rmtree(out / "labels", ignore_errors=True)
    stats: Dict[str, Counter] = {}
    for s, items in splits.items():
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)
        cnt: Counter = Counter()
        for p in items:
            shutil.copy2(p, out / "images" / s / p.name)
            lp, dst = lab_dir / (p.stem + ".txt"), out / "labels" / s / (p.stem + ".txt")
            if lp.exists():
                shutil.copy2(lp, dst)
                for line in lp.read_text().splitlines():
                    if line.strip():
                        cnt[parse_line(line).cls] += 1
            else:
                dst.write_text("")
        stats[s] = cnt
    write_data_yaml(out, names)
    return stats


def read_names(yaml_path) -> List[str]:
    """Nama kelas dari data.yaml buatan write_data_yaml (baris '  <id>: <nama>')."""
    names = {}
    in_names = False
    for line in Path(yaml_path).read_text().splitlines():
        if line.startswith("names:"):
            in_names = True
        elif in_names and line.startswith("  ") and ":" in line:
            k, v = line.strip().split(":", 1)
            names[int(k)] = v.strip()
    return [names[i] for i in sorted(names)]
