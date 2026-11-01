"""Paket `vision`: pipeline vision PBL (kalibrasi -> data -> klasifikasi / deteksi) dalam modul kecil.

Lapisan (ketergantungan hanya boleh mengarah ke bawah):
  CLI (skrip di folder utama)  ->  modul tugas  ->  modul dasar
  Modul dasar (tanpa torch)   : config, files, roi, preprocess, camera, calib, board, calib_eval,
                                yolo_labels, splitting, yolo_dataset, draw, bench (bagian dasar)
  Modul torch (diimpor hanya oleh train/latency): cls_data, cls_models, cls_train
"""
__version__ = "1.0"
