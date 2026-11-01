"""Buat gambar + ringkasan untuk LAPORAN P1 (kalibrasi kamera) dari foto dan file hasil kalibrasimu.
Tidak mengkalibrasi ulang: yang dievaluasi adalah K dan distorsi yang tersimpan di file.
Bawaan: foto dan hasil kalibrasi dibaca dari folder kalibrasi/.
Contoh: python laporan_p1.py   (atau: --foto kalibrasi/kalibrasi_foto --calib kalibrasi/kalibrasi_kamera.npz --cols 6 --rows 9 --square 25)"""
import argparse

from vision import report_p1
from vision.calib import Calibration
from vision.config import DEFAULT_CALIB, DEFAULT_CALIB_PHOTOS


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--foto", default=DEFAULT_CALIB_PHOTOS); ap.add_argument("--calib", default=DEFAULT_CALIB)
    ap.add_argument("--cols", type=int, default=6); ap.add_argument("--rows", type=int, default=9)
    ap.add_argument("--square", type=float, default=25.0); ap.add_argument("--out", default="laporan_p1")
    a = ap.parse_args(argv)
    cal = Calibration.load(a.calib)
    if cal.size is None:
        raise SystemExit("Resolusi kalibrasi tidak tersimpan di file (key 'size' / 'image_size').")
    ps = report_p1.collect(a.foto, cal, a.cols, a.rows, a.square)
    for s in ps.skipped:
        print("dilewati:", s)
    if not ps.names:
        raise SystemExit("Tidak ada foto yang valid.")
    made = report_p1.write_all(ps, cal, a.cols, a.rows, a.square, a.out)
    print(report_p1.summary_markdown(ps, cal, a.cols, a.rows, a.square))
    print(f"\nBerkas di folder '{a.out}': {', '.join(made)}")


if __name__ == "__main__":
    main()
