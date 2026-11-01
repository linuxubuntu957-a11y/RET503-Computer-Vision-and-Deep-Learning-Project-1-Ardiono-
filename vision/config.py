"""Konstanta bersama. Ubah di sini, bukan di banyak tempat."""
from pathlib import Path as _Path

IMG_SIZE = 224                       # ukuran input model klasifikasi
MEAN = (0.485, 0.456, 0.406)         # statistik ImageNet (RGB)
STD = (0.229, 0.224, 0.225)
IMAGE_EXT = frozenset({".png", ".jpg", ".jpeg", ".bmp"})
KALIBRASI_DIR = _Path(__file__).resolve().parent.parent / "kalibrasi"   # folder hasil kalibrasi (di samping paket vision/)
DEFAULT_CALIB = str(KALIBRASI_DIR / "kalibrasi_kamera.npz")
DEFAULT_CALIB_PHOTOS = str(KALIBRASI_DIR / "kalibrasi_foto")
DEFAULT_CLASSES = ("biru", "hijau", "hitam", "merah", "putih")   # urutan = class_id 0..4
BUDGET_MS_15FPS = 1000.0 / 15        # anggaran satu frame pada 15 FPS (~66,7 ms)
