"""Periksa label YOLO sebelum dibagi/dilatih.
Contoh: python check_labels.py --images raw --labels raw --nc 5 --names biru hijau hitam merah putih
Memeriksa: label hilang, 5 nilai per baris, class_id valid, koordinat 0-1, kotak di dalam gambar,
jumlah kotak per kelas, ukuran rata-rata kotak (px), dan ketidakseimbangan kelas."""
import argparse
from pathlib import Path

import cv2

from vision import files, yolo_labels as yl


def analyze(images, lab_dir: Path, nc: int):
    """Kembalikan dict: errors, warns, count[nc], sizes[nc] (px), n_images, n_empty, last_size."""
    errors, warns, n_empty, last = [], [], 0, (0, 0)
    count = [0] * nc
    sizes = [[] for _ in range(nc)]
    for p in images:
        im = cv2.imread(str(p))
        if im is None:
            errors.append(f"{p.name}: gambar tidak terbaca"); continue
        H, W = im.shape[:2]; last = (W, H)
        lp = lab_dir / (p.stem + ".txt")
        if not lp.exists():
            warns.append(f"{p.name}: tidak ada file label (dianggap tanpa objek)"); n_empty += 1; continue
        boxes, e, w = yl.read_label_file(lp, nc)
        errors += e; warns += w
        n_empty += (not boxes and not e)
        for b in boxes:
            count[b.cls] += 1; sizes[b.cls].append((b.w * W, b.h * H))
    vals = count
    if vals and min(vals) == 0:
        warns.append("Ada kelas tanpa kotak sama sekali!")
    elif vals and max(vals) / min(vals) > 3:
        warns.append(f"Kelas tidak seimbang (max/min = {max(vals) / min(vals):.1f}x): tambah data kelas yang sedikit")
    return {"errors": errors, "warns": warns, "count": count, "sizes": sizes,
            "n_images": len(images), "n_empty": n_empty, "size": last}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--images", required=True); ap.add_argument("--labels", default=None)
    ap.add_argument("--nc", type=int, required=True); ap.add_argument("--names", nargs="*", default=None)
    a = ap.parse_args(argv)
    imgs = files.list_images(a.images)
    if not imgs:
        raise SystemExit("Tidak ada gambar di " + a.images)
    names = a.names or [str(i) for i in range(a.nc)]
    r = analyze(imgs, Path(a.labels or a.images), a.nc)
    print(f"Gambar: {r['n_images']} | tanpa objek: {r['n_empty']} | resolusi contoh: {r['size'][0]}x{r['size'][1]}")
    print(f"\n{'kelas':<10}{'kotak':>7}{'rata2 lebar px':>16}{'rata2 tinggi px':>17}")
    for c in range(a.nc):
        s = r["sizes"][c]
        mw = sum(x for x, _ in s) / len(s) if s else 0; mh = sum(y for _, y in s) / len(s) if s else 0
        print(f"{c}:{names[c]:<8}{r['count'][c]:>7}{mw:>16.1f}{mh:>17.1f}")
    for w in r["warns"][:20]:
        print("PERINGATAN:", w)
    for e in r["errors"][:30]:
        print("ERROR:", e)
    print(f"\nHasil: {len(r['errors'])} error, {len(r['warns'])} peringatan.", "SIAP" if not r["errors"] else "PERBAIKI DULU")


if __name__ == "__main__":
    main()
