# RET503: Computer Vision & Deep Learning — Project 1 (PBL)
## Politeknik Negeri Batam — Program Studi Teknologi Rekayasa Robotika
### Repositori Praktikum Terpadu Minggu 4–7 & Pj1: Dari Data ke Model YOLO

---

### IDENTITAS MAHASISWA
- **Nama Mahasiswa** : Ardiono
- **NIM**            : 422411022
- **Waktu Eksekusi**  : 31-10-2026 22:57:02 WIB
- **Komputer / Host** : LAPTOP-G84B67B2 / User
- **Dataset Proyek**  : `dataset_v1` (Sidik Jari Label: `8E25699C0A30`)
- **Model**           : YOLOv8n (Nano)

---

## 📌 PEMENUHAN ASESMEN RPS & KETENTUAN TUGAS (PPT MINGGU 4–7)

Berdasarkan ketentuan pada **Slide 38 PPT Minggu 4–7**, berikut adalah matriks bukti pemenuhan tugas yang telah diselesaikan di repositori ini:

| Asesmen RPS | Ketentuan PPT Dosen | Bukti di Repositori Ini | Status |
|---|---|---|:---:|
| **P3 (CLO-3)** | **Tahap 1–2: Data projek mahasiswa** (YOLO/COCO, train/val/test, data.yaml) | Folder [`dataset_v1/`](dataset_v1/) lengkap dengan `data.yaml`, anotasi COCO [`instances_*.json`](dataset_v1/annotations/), label YOLO [`labels/`](dataset_v1/labels/), pembagian per sesi anti-leakage via [`split_yolo.py`](split_yolo.py), dan validasi [`check_labels.py`](check_labels.py). | ✅ **LENGKAP** |
| **P4 (CLO-4)** | **Tahap 3: Hasil E1 & E3, `results.png` kedua eksperimen** | Grafik kurva pelatihan [`results.png`](hasil_praktik/E1_pretrained/results.png), confusion matrix, dan kurva PR untuk **E1_pretrained**, **E2_scratch**, dan **E3_tuning** di folder [`hasil_praktik/`](hasil_praktik/). | ✅ **LENGKAP** |
| **Pj1 (CLO-4)** | **Tahap 4–6: `hasil_praktik_<NIM>.txt/.csv`, analisis & hasil test (README)** | Berkas resmi [`hasil_praktik_422411022.txt`](hasil_praktik_422411022.txt), [`hasil_praktik_422411022.csv`](hasil_praktik_422411022.csv), tabel hasil, analisis test set, dan konsep sistem persepsi di README ini serta dokumen [`LAPORAN_PRAKTIK_RET503_MINGGU_4-7.md`](LAPORAN_PRAKTIK_RET503_MINGGU_4-7.md). | ✅ **LENGKAP** |

---

## 📊 HASIL PRAKTIK TERPADU RESMI (Slide 37)

Tabel berikut diekstrak langsung dari berkas resmi `hasil_praktik_422411022.txt`:

```text
==============================================================================
  PRAKTIK TERPADU MINGGU 4-7 - RET503
==============================================================================
  NIM / Nama    : 422411022 / Ardiono
  Waktu         : 31-10-2026 22:57:02
  Komputer/user : LAPTOP-G84B67B2 / User
  Dataset       : dataset_v1  (sidik jari label: 8E25699C0A30)
  Pengaturan    : model yolov8n, epochs 30, imgsz 416, batch 16
==============================================================================
  Eksperimen     Perubahan        mAP50  mAP50-95      P      R     ms   menit
  E1_pretrained  -                0.995     0.917  0.990  1.000   14.1    13.8
  E2_scratch     -                0.969     0.679  0.853  0.911    8.2    13.7
  E3_tuning      imgsz=640        0.995     0.919  0.993  0.989   24.9    90.0
==============================================================================
  Model terpilih (mAP50-95 val tertinggi): E3_tuning
  Hasil di data TEST (sekali saja)       : mAP50 0.995 | mAP50-95 0.915
==============================================================================
```

### Rangkuman Metrik Utama

| Eksperimen | Perubahan | mAP50 | mAP50-95 | Precision | Recall | Inferensi (ms) | Waktu Latih (menit) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **E1_pretrained** | *Baseline (Transfer Learning 416 px)* | **0.995** | **0.917** | 0.990 | 1.000 | 14.1 ms | 13.8 m |
| **E2_scratch** | *Bobot Acak (Tanpa Transfer)* | **0.969** | **0.679** | 0.853 | 0.911 | 8.2 ms | 13.7 m |
| **E3_tuning** | `imgsz=640` *(Resolusi Input Tinggi)* | **0.995** | **0.919** | 0.993 | 0.989 | 24.9 ms | 90.0 m |

---

## 🔬 ANALISIS EKSPERIMEN & HASIL TEST (Pj1 - CLO-4)

### 1. Pembuktian Transfer Learning (E1 vs E2)
- Pada metrik ketat **`mAP50-95`**, model **E1_pretrained (0.917)** mengungguli **E2_scratch (0.679)** secara signifikan (+23.8%).
- **Analisis Fisik/Komputasi**: Backbone yang telah dilatih pada COCO membawa filter spasial kaya (tepi, tekstur, sudut). Model hanya perlu menyesuaikan bobot head untuk kelas proyek balok. Sebaliknya, model dari nol (*scratch*) memerlukan ribuan epoch dan data lebih besar agar lokalisasi kotaknya presisi.

### 2. Evaluasi Hyperparameter Tuning (E1 vs E3)
- Menaikkan resolusi input dari 416 ke 640 pada **E3_tuning** meningkatkan `mAP50-95` menjadi **0.919** (tertinggi di data validasi).
- **Trade-off Komputasi**: Peningkatan akurasi membutuhkan waktu komputasi inferensi lebih besar (14.1 ms menjadi 24.9 ms) dan waktu latih CPU melonjak dari 13.8 menit ke 90 menit.

### 3. Hasil Pengujian pada Data TEST (Unseen Data - 1 Kali Saja)
- Model terpilih (**E3_tuning**) diuji pada split `test` (81 citra) yang belum pernah dilihat saat pelatihan:
  - **mAP50** = **0.995 (99.5%)**
  - **mAP50-95** = **0.915 (91.5%)**
  - **Precision** = **0.997** | **Recall** = **1.000**
- **Kesimpulan Generalisasi**: Performa data uji konsisten dengan data validasi (0.915 vs 0.919). Hal ini membuktikan bahwa strategi pemisahan dataset per sesi (*anti data leakage*) berhasil mencegah model dari menghafal (*overfitting*).

### 4. Analisis Kelayakan Sistem Robotika (Latensi vs Target 15 FPS)
- Sesuai standar rancangan sistem robotika (misi manipulasi Universal Robots / UR), batas total anggaran satu siklus penglihatan pada 15 FPS adalah **66.7 ms**.
- Model E3 membutuhkan **24.9 ms** untuk inferensi (dan E1 hanya **14.1 ms**). Dengan tambahan preprocessing & postprocessing (~2 ms), sistem berjalan pada **~40 FPS**, sehingga **sangat aman dan layak untuk integrasi kontrol robot real-time**.

---

## 🛠️ KONSEP SISTEM PERSEPSI (Pj1 - Slide 32)

1. **Lingkup Persepsi**: Mendeteksi 3 balok objek (`merah`, `hijau`, `hitam`) pada meja kerja robot di bawah variasi pencahayaan laboratorium.
2. **Target Kinerja**:
   - Akurasi: mAP50 > 90% (Tercapai: **99.5%**).
   - Kecepatan: Latensi < 66.7 ms / > 15 FPS (Tercapai: **24.9 ms / ~40 FPS**).
3. **Hardware & Pipeline**:
   - Kamera: **Intel RealSense D455** (Webcam eksternal, resolusi asli 1280x720, DirectShow).
   - Kalibrasi Lensa: Metode Zhang ([`kalibrasi/kalibrasi_kamera.npz`](kalibrasi/kalibrasi_kamera.npz), RMS error **0.541 px**).
   - Inferensi: Model PyTorch YOLOv8n terpilih ([`hasil_praktik/E3_tuning/weights/best.pt`](hasil_praktik/E3_tuning/weights/best.pt)) atau format ekspor ONNX ([`export_eval.py`](export_eval.py)).

---

## 📁 STRUKTUR FOLDER REPOSITORI

```
vision_kit/
├── vision/                 # Framework modular vision (calib, camera, autobox, splitting, dll.)
├── tests/                  # 62 Unit tests (100% lulus)
├── kalibrasi/              # Hasil kalibrasi kamera (RMS 0.541 px, K, dist, foto catur)
├── dataset_v1/             # Dataset YOLO & COCO (data.yaml, labels/, annotations/)
├── hasil_praktik/          # Output lengkap 3 eksperimen (E1, E2, E3, kurva loss, PR, CF matrix)
│   ├── E1_pretrained/      # results.png, BoxPR_curve.png, confusion_matrix.png, weights/
│   ├── E2_scratch/         # results.png, BoxPR_curve.png, confusion_matrix.png, weights/
│   └── E3_tuning/          # results.png, BoxPR_curve.png, confusion_matrix.png, weights/best.pt
├── hasil_praktik_422411022.txt   # Bukti numerik resmi teks (sidik jari: 8E25699C0A30)
├── hasil_praktik_422411022.csv   # Bukti numerik resmi CSV
├── LAPORAN_PRAKTIK_RET503_MINGGU_4-7.md # Laporan praktikum formal lengkap
├── live_detect.py          # Skrip inferensi live webcam RealSense D455 (1280x720)
├── export_eval.py          # Ekspor model ke format ONNX & evaluasi
├── 06_latih_bandingkan.py  # Skrip resmi pelatihan 3 eksperimen dosen
└── README.md               # Dokumentasi utama proyek & laporan Pj1
```

---

## 🚀 CARA MENJALANKAN

### 1. Inferensi Kamera Langsung (Live RealSense D455)
```powershell
python live_detect.py
```
*(Menjalankan webcam RealSense pada resolusi 1280x720 terkoreksi distorsi secara real-time. Tekan 'q' untuk keluar).*

### 2. Pelatihan & Evaluasi Ulang (Skrip Dosen)
```powershell
python 06_latih_bandingkan.py --nim 422411022 --nama "Ardiono" --model yolov8n --epochs 30
```

### 3. Ekspor & Evaluasi Model ONNX
```powershell
python export_eval.py
```
