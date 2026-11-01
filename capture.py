"""Ambil citra TERKOREKSI (undistort) dari kamera -> dataset_raw/<kelas>/ + metadata.csv
Contoh : python capture.py baut terang-s1 [--calib kalibrasi/kalibrasi_kamera.npz] [--roi]
Tombol : s = simpan | q = keluar
Tips   : beri nama kondisi berbeda untuk tiap sesi (terang-s1, terang-s2, redup-s1) agar split bisa per sesi.
Untuk data DETEKSI (frame penuh): jalankan TANPA --roi, nama kelas bebas (mis. 'scene')."""
import argparse
import datetime
import sys
from pathlib import Path

import cv2

from vision import camera, files
from vision.calib import Calibration
from vision.config import DEFAULT_CALIB
from vision.roi import crop, load_roi, save_roi


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("kelas"); ap.add_argument("kondisi")
    ap.add_argument("--lokasi", default="lab")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280); ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--calib", default=DEFAULT_CALIB)
    ap.add_argument("--out", default="dataset_raw")
    ap.add_argument("--roi", action="store_true", help="simpan hanya area ROI (disarankan persegi)")
    ap.add_argument("--dshow", action="store_true", help="backend DirectShow (Windows)")
    return ap.parse_args(argv)


def make_undistorter(calib_path, frame_size):
    """Undistorter dari file kalibrasi, atau None (dengan peringatan) bila file tidak ada."""
    if not Path(calib_path).exists():
        print("PERINGATAN: file kalibrasi tidak ditemukan -> citra disimpan TANPA undistort.")
        return None
    cal = Calibration.load(calib_path)
    if cal.size is None:
        print("PERINGATAN: resolusi kalibrasi tidak tersimpan; pastikan sama dengan saat kalibrasi.")
    cal.check_frame_size(frame_size)
    return cal.undistorter(frame_size)


def pick_roi(out_dir: Path, first_frame, undistort):
    """ROI dari roi.json bila ada, jika tidak pilih interaktif lalu simpan."""
    path = out_dir / "roi.json"
    roi = load_roi(path)
    if roi:
        print("ROI dipakai dari", path, "(hapus file itu untuk memilih ulang)")
        return roi
    view = undistort(first_frame) if undistort else first_frame
    x, y, w, h = cv2.selectROI("Pilih ROI (ENTER = ok)", view, False)
    cv2.destroyWindow("Pilih ROI (ENTER = ok)")
    if w > 0 and h > 0:
        roi = (int(x), int(y), int(w), int(h)); save_roi(path, roi)
    return roi


def main(argv=None):
    a = parse_args(argv)
    out = Path(a.out); cdir = out / a.kelas; cdir.mkdir(parents=True, exist_ok=True)
    try:
        cap, size = camera.open_camera(a.cam, a.width, a.height, a.dshow)
        undistort = make_undistorter(a.calib, size)
    except (RuntimeError, ValueError) as e:
        sys.exit(str(e))
    print(f"Resolusi kamera: {size[0]}x{size[1]}")
    ok, frame = cap.read()
    roi = pick_roi(out, frame, undistort) if a.roi else None

    today = datetime.date.today().strftime("%Y%m%d")
    prefix = files.make_prefix(a.kelas, today, a.lokasi, a.kondisi)
    n = files.count_existing(cdir, prefix)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        img = crop(undistort(frame) if undistort else frame, roi)
        view = img.copy()
        cv2.putText(view, f"{a.kelas}/{a.kondisi} tersimpan: {n}  [s]=simpan [q]=keluar", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("capture", view)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("s"):
            n += 1
            name = f"{prefix}{n:03d}.png"
            cv2.imwrite(str(cdir / name), img)
            files.append_metadata(out / "metadata.csv", [name, a.kelas, today, a.kondisi])
        elif k == ord("q"):
            break
    cap.release(); cv2.destroyAllWindows()
    print(f"Selesai. {n} citra untuk kelas '{a.kelas}' (kondisi {a.kondisi}).")


if __name__ == "__main__":
    main()
