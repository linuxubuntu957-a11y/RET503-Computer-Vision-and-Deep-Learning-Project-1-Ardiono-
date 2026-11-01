"""Ubah dataset YOLO (hasil split_yolo.py) menjadi anotasi COCO per split.
Contoh: python yolo_to_coco.py --dataset dataset_v1
Keluaran: dataset_v1/annotations/instances_train.json, instances_val.json, instances_test.json
Gambar tetap di dataset_v1/images/<split>/ (file_name relatif ke folder itu). category_id = class_id + 1."""
import argparse
from pathlib import Path

from vision import coco_export, yolo_dataset


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", default="dataset_v1")
    ap.add_argument("--names", nargs="+", default=None, help="bawaan: dibaca dari <dataset>/data.yaml")
    ap.add_argument("--cat-offset", type=int, default=1, help="1 = id kategori mulai dari 1 (gaya COCO); 0 = sama dgn YOLO")
    a = ap.parse_args(argv)
    root = Path(a.dataset)
    if not (root / "images").exists():
        raise SystemExit(f"'{root}/images' tidak ada. Jalankan split_yolo.py dulu.")
    names = a.names or yolo_dataset.read_names(root / "data.yaml")
    for split, path in coco_export.export_dataset(root, names, a.cat_offset).items():
        print(f"{split:<6}: {path}")


if __name__ == "__main__":
    main()
