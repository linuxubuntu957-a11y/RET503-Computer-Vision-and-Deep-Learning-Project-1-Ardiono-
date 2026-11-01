"""Bagi dataset DETEKSI YOLO menjadi train/val/test PER SESI + buat data.yaml.
Nama gambar yang dikenali: <apa>_<TANGGAL>_<lokasi>_<kondisi>_<nomor>.png (hasil capture.py)
Contoh: python split_yolo.py --images raw --out dataset_v1 --names biru hijau hitam merah putih"""
import argparse
from pathlib import Path

from vision import files, splitting, yolo_dataset


def class_of_image(p: Path) -> str:
    """Awalan nama (<apa> pada <apa>_<TANGGAL>_...): 'merah', atau 'scene' untuk hasil capture.py scene."""
    return p.stem.split("_")[0]


def plan(images, val: float, test: float, seed: int, per_class: bool = True):
    """(dict split -> daftar Path, pesan). per_class: tiap kelas dibagi sendiri (disarankan)."""
    sess = lambda p: files.session_of_scene(p.stem)
    if per_class:
        return splitting.split_scenes_by_class(images, sess, class_of_image, val, test, seed)
    return splitting.split_scenes(images, sess, val, test, seed)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--images", required=True); ap.add_argument("--labels", default=None)
    ap.add_argument("--out", default="dataset_v1"); ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--val", type=float, default=0.2); ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-per-class", action="store_true", help="bagi semua gambar sekaligus (kelas bisa timpang antar split)")
    ap.add_argument("--force", action="store_true", help="hapus images/ dan labels/ lama di folder keluaran")
    a = ap.parse_args(argv)
    imgs = files.list_images(a.images)
    if not imgs:
        raise SystemExit("Tidak ada gambar.")
    splits, msg = plan(imgs, a.val, a.test, a.seed, not a.no_per_class)
    print(msg)
    try:
        stats = yolo_dataset.materialize(splits, Path(a.labels or a.images), a.out, a.names, a.force)
    except FileExistsError as e:
        raise SystemExit(str(e))
    for s, cnt in stats.items():
        print(f"{s:<6}: {len(splits[s]):>4} gambar | kotak per kelas: " +
              ", ".join(f"{n}={cnt[i]}" for i, n in enumerate(a.names)))
    print("Tulis:", Path(a.out) / "data.yaml")


if __name__ == "__main__":
    main()
