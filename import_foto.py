"""Impor foto balok yang SUDAH ADA (zip atau folder) ke format kit.
Per kelas: undistort (kalibrasi/), beri nama pola kit, bagi jadi --sesi blok waktu, buat label YOLO USULAN
(segmentasi warna) dan crop persegi untuk klasifikasi.
Contoh: python import_foto.py balok_merah.zip balok_hijau.zip balok_hitam.zip --names merah hijau hitam
Urutan --names = class_id YOLO (0,1,2,...). Saat menambah kelas baru, TAMBAHKAN di belakang agar id lama tetap.
Keluaran: raw/ (+ label)   raw_perlu_label/   dataset_raw/<kelas>/ + metadata.csv"""
import argparse
import tempfile
import zipfile
from pathlib import Path

from vision import autobox, files, importer
from vision.calib import Calibration
from vision.config import DEFAULT_CALIB


def make_undistort(calib_path, use: bool):
    """(undistort atau None, ukuran frame atau None, pesan)."""
    if not use:
        return None, None, "TANPA undistort (--no-undistort)"
    if not Path(calib_path).exists():
        return None, None, "PERINGATAN: file kalibrasi tidak ada -> TANPA undistort"
    cal = Calibration.load(calib_path)
    if cal.size is None:
        return None, None, "PERINGATAN: resolusi kalibrasi tidak tersimpan -> TANPA undistort"
    return cal.undistorter(cal.size), cal.size, f"undistort dengan {calib_path} ({cal.size[0]}x{cal.size[1]})"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("src", nargs="+", help="zip atau folder per kelas; nama harus memuat nama kelas (mis. 'balok merah')")
    ap.add_argument("--names", nargs="+", required=True, help="nama kelas, urutan = class_id YOLO")
    ap.add_argument("--calib", default=DEFAULT_CALIB); ap.add_argument("--no-undistort", action="store_true")
    ap.add_argument("--sesi", type=int, default=5, help="jumlah sesi (blok waktu berurutan) per kelas; min 3, disarankan 5")
    ap.add_argument("--lokasi", default="lab")
    ap.add_argument("--out-cls", default="dataset_raw"); ap.add_argument("--out-det", default="raw")
    a = ap.parse_args(argv)

    und, size, msg = make_undistort(a.calib, not a.no_undistort)
    print(msg)
    with tempfile.TemporaryDirectory() as tmp:
        for s in a.src:
            try:
                kelas = importer.match_class(Path(s).name, a.names)
            except ValueError as e:
                raise SystemExit(str(e))
            folder = Path(s)
            if folder.suffix.lower() == ".zip":
                folder = Path(tmp) / kelas
                with zipfile.ZipFile(s) as z:
                    z.extractall(folder)
            imgs = files.list_images(folder)
            if not imgs:
                raise SystemExit(f"Tidak ada gambar di {s}")
            try:
                r = importer.import_class(kelas, a.names.index(kelas), imgs, a.out_cls, a.out_det, a.lokasi,
                                          a.sesi, und, size)
            except FileExistsError as e:
                raise SystemExit(str(e))
            print(f"\n[{kelas}] id={a.names.index(kelas)}: {r.n_total} foto -> {r.n_labeled} berlabel usulan, "
                  f"{len(r.no_box)} perlu label manual, {len(r.skipped)} dilewati | sesi: {', '.join(r.sessions)}")
            if not autobox.has_profile(kelas):
                print(f"  CATATAN: kelas '{kelas}' tak punya profil warna -> semua foto masuk {a.out_det}_perlu_label/")
            for n in r.no_box[:5]:
                print("  perlu label:", n)
            for n in r.skipped[:5]:
                print("  dilewati   :", n)
    print(f"\nLangkah berikut: python draw_labels.py --images {a.out_det} --names {' '.join(a.names)} --n 24")
    print("Periksa kotaknya; perbaiki lewat tool anotasi (mis. LabelImg) bila ada yang meleset.")


if __name__ == "__main__":
    main()
