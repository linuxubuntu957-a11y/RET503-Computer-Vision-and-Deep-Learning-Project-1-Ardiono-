"""Ukur latensi: preprocessing + ResNet-18 vs MobileNetV3-Small, bandingkan dengan anggaran 15 FPS (~67 ms).
Contoh: python latency.py   (bawaan: kalibrasi/kalibrasi_kamera.npz)"""
import argparse
from pathlib import Path

import numpy as np

from vision import bench
from vision.calib import Calibration
from vision.config import BUDGET_MS_15FPS, DEFAULT_CALIB, IMG_SIZE
from vision.preprocess import to_model_input


def preprocessing_ms(calib_path, width: int, height: int):
    """(rata-rata ms, p95 ms, memakai_undistort) untuk undistort + resize + normalisasi satu frame."""
    frame = np.random.randint(0, 255, (height, width, 3), np.uint8)
    und = Calibration.load(calib_path).undistorter((width, height)) if Path(calib_path).exists() else None
    mean, p95 = bench.bench(lambda: to_model_input(und(frame) if und else frame))
    return mean, p95, und is not None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calib", default=DEFAULT_CALIB); ap.add_argument("--classes", type=int, default=5)
    ap.add_argument("--width", type=int, default=1280); ap.add_argument("--height", type=int, default=720)
    a = ap.parse_args(argv)

    pre, pre95, with_und = preprocessing_ms(a.calib, a.width, a.height)
    print(f"Preprocessing ({'dengan' if with_und else 'TANPA'} undistort): {pre:.1f} ms (p95 {pre95:.1f})")
    rows = bench.bench_torch_models(["resnet18", "mobilenet_v3_small"], a.classes, IMG_SIZE)
    print(f"\n{'model':<20}{'device':<7}{'param(jt)':>10}{'mean ms':>9}{'p95 ms':>8}{'FPS':>7}{'pre+inf ms':>12}")
    for r in rows:
        tot = r["mean_ms"] + pre
        flag = "OK" if tot <= BUDGET_MS_15FPS else f"> {BUDGET_MS_15FPS:.0f} ms"
        print(f"{r['model']:<20}{r['device']:<7}{r['params_m']:>10.2f}{r['mean_ms']:>9.1f}{r['p95_ms']:>8.1f}"
              f"{1000 / r['mean_ms']:>7.1f}{tot:>10.1f}  {flag}")
    print(f"\nAnggaran 15 FPS = {BUDGET_MS_15FPS:.1f} ms untuk SELURUH pipeline (akuisisi, preprocess, inferensi, postprocess, ROS2, margin).")


if __name__ == "__main__":
    main()
