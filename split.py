"""Bagi dataset KLASIFIKASI train/val PER SESI (bukan acak per gambar) -> split.json
Contoh: python split.py --data dataset_raw --val 0.2"""
import argparse
import json
from pathlib import Path

from vision import files, splitting


def build_split(root, val_frac: float, seed: int):
    """Mengembalikan (split dict, baris laporan)."""
    root = Path(root)
    classes = files.list_class_dirs(root)
    if len(classes) < 2:
        raise SystemExit("Butuh minimal 2 kelas di " + str(root))
    train, val, report = [], [], []
    for ci, c in enumerate(classes):
        names = [p.name for p in files.list_images(root / c, recursive=False)]
        tr, va, note = splitting.split_class_files(
            names, lambda f, c=c: files.session_of_class_image(c, f), val_frac, seed)
        if len(names) < 50:
            note += " | <50 citra (target 50/kelas)"
        train += [[f"{c}/{f}", ci] for f in tr]; val += [[f"{c}/{f}", ci] for f in va]
        report.append(f"{c:<14}{len(names):>6}{len(tr):>7}{len(va):>6}  {note}")
    return {"classes": classes, "train": train, "val": val}, report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", default="dataset_raw"); ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--out", default="split.json")
    a = ap.parse_args(argv)
    split, report = build_split(a.data, a.val, a.seed)
    print(f"{'kelas':<14}{'total':>6}{'train':>7}{'val':>6}  catatan"); print("\n".join(report))
    Path(a.out).write_text(json.dumps(split, indent=1))
    print(f"\nDisimpan: {a.out}  (train={len(split['train'])}, val={len(split['val'])})")


if __name__ == "__main__":
    main()
