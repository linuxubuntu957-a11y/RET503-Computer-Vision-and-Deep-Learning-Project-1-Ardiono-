"""Ekspor dataset YOLO (images/<split>, labels/<split>) ke format COCO (instances_<split>.json).
Murni berkas: tanpa print dan tanpa membaca argumen."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence

import cv2

from .files import list_images
from .yolo_labels import read_label_file, yolo_to_xyxy


def split_to_coco(images_dir, labels_dir, names: Sequence[str], cat_offset: int = 1) -> dict:
    """Satu split -> dict COCO. category_id = class_id YOLO + cat_offset (1 = gaya COCO/torchvision,
    0 dipakai latar). bbox = [x, y, lebar, tinggi] dalam piksel. Gambar tanpa label tetap dicantumkan."""
    images, anns = [], []
    for img_id, p in enumerate(list_images(images_dir, recursive=False), 1):
        im = cv2.imread(str(p))
        if im is None:
            raise ValueError(f"gambar tidak terbaca: {p}")
        H, W = im.shape[:2]
        images.append({"id": img_id, "file_name": p.name, "width": W, "height": H})
        boxes, errs, _ = read_label_file(Path(labels_dir) / (p.stem + ".txt"), len(names))
        if errs:
            raise ValueError("label bermasalah: " + "; ".join(errs[:3]))
        for b in boxes:
            x1, y1, x2, y2 = yolo_to_xyxy(b.cx, b.cy, b.w, b.h, W, H)
            w, h = x2 - x1, y2 - y1
            anns.append({"id": len(anns) + 1, "image_id": img_id, "category_id": b.cls + cat_offset,
                         "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                         "area": round(w * h, 2), "iscrowd": 0})
    cats = [{"id": i + cat_offset, "name": n} for i, n in enumerate(names)]
    return {"info": {"description": "diekspor dari dataset YOLO (vision_kit)"}, "images": images,
            "annotations": anns, "categories": cats}


def export_dataset(dataset_dir, names: Sequence[str], cat_offset: int = 1) -> Dict[str, Path]:
    """Tulis <dataset>/annotations/instances_<split>.json untuk setiap split di images/. {split: path}."""
    root = Path(dataset_dir)
    out_dir = root / "annotations"
    out_dir.mkdir(exist_ok=True)
    written = {}
    for sdir in sorted(d for d in (root / "images").iterdir() if d.is_dir()):
        coco = split_to_coco(sdir, root / "labels" / sdir.name, names, cat_offset)
        path = out_dir / f"instances_{sdir.name}.json"
        path.write_text(json.dumps(coco, indent=1), encoding="utf-8")
        written[sdir.name] = path
    return written
