"""Gambar label YOLO di atas gambar untuk PEMERIKSAAN VISUAL sebelum training.
Contoh: python draw_labels.py --images raw --names biru hijau hitam merah putih --n 12 --out preview"""
import argparse
import random
from pathlib import Path

import cv2

from vision import files, yolo_labels as yl
from vision.draw import draw_boxes


def render(image_path: Path, lab_dir: Path, names):
    """(gambar bergambar kotak, jumlah kotak, ada_file_label)."""
    im = cv2.imread(str(image_path))
    lp = lab_dir / (image_path.stem + ".txt")
    boxes, _, _ = yl.read_label_file(lp, len(names))
    return draw_boxes(im, boxes, names), len(boxes), lp.exists()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--images", required=True); ap.add_argument("--labels", default=None)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=12); ap.add_argument("--out", default="preview")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    imgs = files.list_images(a.images)
    random.Random(a.seed).shuffle(imgs)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for p in imgs[:a.n]:
        img, n, has = render(p, Path(a.labels or a.images), a.names)
        cv2.imwrite(str(out / p.name), img)
        print(f"{p.name}: {n} kotak" + ("" if has else "  (TANPA file label = gambar background)"))
    print(f"\nHasil di folder '{out}'. Pastikan kotak menempel pada objek dan nama kelas benar.")


if __name__ == "__main__":
    main()
