"""Deteksi objek langsung (real-time webcam) menggunakan model terbaik YOLOv8.
Menggunakan DirectShow (Windows) dengan resolusi asli 1280x720 tanpa lag/stutter.

Cara pakai:
    python live_detect.py
    python live_detect.py --cam 0 --conf 0.5
Tekan tombol 'q' atau ESC pada jendela video untuk keluar.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from vision.calib import Calibration
from vision.config import DEFAULT_CALIB


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="hasil_praktik/E3_tuning/weights/best.pt",
                        help="Path ke file bobot model (.pt)")
    parser.add_argument("--cam", type=int, default=2, help="Index kamera (default: 2 untuk RealSense RGB, 0 untuk laptop)")
    parser.add_argument("--width", type=int, default=1280, help="Lebar frame (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Tinggi frame (default: 720)")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Ukuran resolusi inferensi YOLO")
    parser.add_argument("--calib", default=DEFAULT_CALIB, help="File kalibrasi kamera (.npz)")
    parser.add_argument("--no-undistort", action="store_true", help="Jangan lakukan koreksi distorsi lensa")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        sys.exit(f"Error: Model '{model_path}' tidak ditemukan.")

    print(f"[1/3] Memuat model YOLO: {model_path}")
    model = YOLO(str(model_path))

    # Cek kalibrasi
    undistorter = None
    if not args.no_undistort and Path(args.calib).exists():
        try:
            cal = Calibration.load(args.calib)
            undistorter = cal.undistorter((args.width, args.height))
            print(f"[2/3] Koreksi distorsi lensa aktif ({args.calib})")
        except Exception as e:
            print(f"[2/3] Peringatan: Gagal memuat kalibrasi ({e}), frame mentah akan dipakai.")
    else:
        print("[2/3] Koreksi distorsi lensa dimatikan.")

    # Buka kamera dengan DirectShow (CAP_DSHOW) agar mulus di Windows & set 1280x720
    print(f"[3/3] Membuka kamera {args.cam} (target {args.width}x{args.height}) via DirectShow...")
    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        # Fallback bila DSHOW gagal
        cap = cv2.VideoCapture(args.cam)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"-> Kamera aktif: {actual_w}x{actual_h}")
    print("-> Tekan 'q' atau ESC pada jendela untuk keluar.\n")

    prev_time = time.time()
    fps_smooth = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Peringatan: Gagal membaca frame kamera.")
                time.sleep(0.01)
                continue

            # Undistort jika kalibrasi tersedia
            if undistorter is not None:
                frame = undistorter(frame)

            # Inferensi YOLO
            t0 = time.time()
            results = model(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)
            inf_time_ms = (time.time() - t0) * 1000.0

            # Gambar visualisasi kotak deteksi
            annotated_frame = results[0].plot()

            # Hitung FPS
            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-5)
            prev_time = curr_time
            fps_smooth = 0.9 * fps_smooth + 0.1 * fps if fps_smooth > 0 else fps

            # Overlay info FPS & latensi
            info_text = f"FPS: {fps_smooth:.1f} | Inferensi: {inf_time_ms:.1f}ms | Res: {actual_w}x{actual_h}"
            cv2.putText(annotated_frame, info_text, (16, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("Vision Kit - Live YOLOv8 Detection", annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' atau ESC
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Kamera ditutup. Selesai.")


if __name__ == "__main__":
    main()
