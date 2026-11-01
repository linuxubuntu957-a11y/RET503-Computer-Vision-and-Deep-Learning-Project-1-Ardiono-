"""Kalibrasi kamera (metode Zhang, papan catur 6x9, kotak 25 mm) -> folder kalibrasi/
Pembungkus tipis untuk kalibrasi/kalibrasi_kamera_live.py: hasil (.json + .npz) dan foto
disimpan di folder kalibrasi/, tempat yang dibaca capture.py, latency.py, dan laporan_p1.py.
Contoh : python kalibrasi.py                          # live dari kamera 0 (SPASI foto, C hitung, S simpan)
         python kalibrasi.py --from-folder kalibrasi/kalibrasi_foto   # hitung ulang dari foto tersimpan
         python kalibrasi.py --camera 1 --backend dshow                # Windows
Semua opsi lain sama dengan kalibrasi_kamera_live.py (lihat: python kalibrasi.py -h)."""
import importlib.util
import sys

from vision.config import DEFAULT_CALIB_PHOTOS, KALIBRASI_DIR


def _load_live():
    spec = importlib.util.spec_from_file_location("kalibrasi_kamera_live", KALIBRASI_DIR / "kalibrasi_kamera_live.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod        # dataclass di modul tujuan butuh entri di sys.modules
    spec.loader.exec_module(mod)
    return mod


def with_defaults(argv):
    """Tambahkan --output dan --save-images ke folder kalibrasi/ bila belum ditentukan."""
    argv = list(argv)
    if "--output" not in argv:
        argv += ["--output", str(KALIBRASI_DIR / "kalibrasi_kamera")]
    if "--save-images" not in argv:
        argv += ["--save-images", DEFAULT_CALIB_PHOTOS]
    return argv


def main(argv=None):
    live = _load_live()
    return live.main(with_defaults(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())
