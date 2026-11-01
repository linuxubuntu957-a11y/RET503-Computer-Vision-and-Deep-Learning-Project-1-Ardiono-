"""Latih YOLO (transfer learning dari bobot pralatih COCO) pada dataset_v1, lalu uji di split test.
Jalankan dari folder vision_kit:  python train_yolo.py
Uji cepat (beberapa menit di CPU):  python train_yolo.py --epochs 3
Butuh: pip install ultralytics   (bobot yolov8n.pt diunduh otomatis saat pertama jalan)"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", default="dataset_v1/data.yaml")
    ap.add_argument("--weights", default="yolov8n.pt", help="bobot awal pralatih COCO")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()

    model = YOLO(a.weights)
    model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz)

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print(f"\nBobot terbaik: {best}")
    print("Evaluasi pada split TEST (data yang tidak dipakai untuk memilih model):")
    YOLO(str(best)).val(data=a.data, split="test")


if __name__ == "__main__":      # wajib di Windows
    main()
