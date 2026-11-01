# LAPORAN PRAKTIKUM TERPADU MINGGU 4–7
## MATA KULIAH: SISTEM VISI ROBOTIKA (RET503 / RET501)
### Topik: Pelatihan, Tuning, Pemilihan, dan Pengujian Model Deteksi Objek YOLO

---

### IDENTITAS MAHASISWA & PRAKTIKUM
- **Nama Mahasiswa** : Ardiono
- **NIM**            : 422411022
- **Waktu Eksekusi**  : 31-10-2026 22:57:02 WIB
- **Komputer / Host** : LAPTOP-G84B67B2 / User
- **Dataset**         : `dataset_v1` (Sidik Jari Label: `8E25699C0A30`)
- **Arsitektur Model**: YOLOv8n (Nano)
- **Parameter Dasar** : Epochs = 30, Batch Size = 16, Optimizer = AdamW, Device = CPU

---

## I. PENDAHULUAN & TUJUAN PRAKTIKUM

Praktikum terpadu minggu 4–7 ini berfokus pada tahapan kritis dalam siklus pengembangan sistem *computer vision* untuk aplikasi robotika industri:
1. **Memahami Alur Kerja Deteksi Objek:** Menerapkan pipeline deteksi objek berbasis *deep learning* mulai dari dataset beranotasi format YOLO, augmentasi citra (*mosaic*), hingga pelatihan model.
2. **Evaluasi Transfer Learning vs Scratch:** Membandingkan konvergensi model yang memanfaatkan bobot pralatih (*pretrained weights*) COCO terhadap model yang dilatih dari bobot acak (*random initialization / from scratch*).
3. **Eksperimen Hyperparameter Tuning:** Menguji pengaruh resolusi input citra (*image size*) terhadap ketelitian lokalisasi kotak (*bounding box*) dan kecepatan inferensi.
4. **Metodologi Evaluasi Bebas Kebocoran Data (*Anti Data Leakage*):** Memilih model terbaik secara objektif hanya berdasarkan metrik validasi, lalu menguji performa finalnya satu kali pada data uji (*test set*).

---

## II. DESAIN & METODOLOGI EKSPERIMEN

### 1. Struktur Dataset & Anti-Leakage
Dataset yang digunakan adalah `dataset_v1` yang mencakup 3 kelas objek balok industri: `0: merah`, `1: hijau`, `2: hitam`.
- **Jumlah Citra**:
  - Train: 236 citra
  - Validation: 81 citra
  - Test: 81 citra
- **Integritas Pembagian**: Citra dibagi berdasarkan sesi pengambilan waktu (*per-session splitting*), sehingga variasi pencahayaan dan sudut pada satu sesi tidak bocor secara bersamaan di train dan val/test.
- **Sidik Jari Dataset**: `8E25699C0A30` (SHA-256 seluruh label teks) untuk menjamin validitas dan reproduksibilitas.

### 2. Tiga Skenario Eksperimen
Sesuai rancangan pada skrip `06_latih_bandingkan.py`, dilakukan tiga eksperimen independen dengan parameter dasar yang dikontrol secara ketat:
- **E1 (Pretrained Baseline):** Menggunakan bobot awal `yolov8n.pt` (pralatih COCO), resolusi input `imgsz=416`.
- **E2 (Scratch Pembanding):** Arsitektur identik (`yolov8n.yaml`) dengan bobot diinisialisasi secara acak, resolusi input `imgsz=416`.
- **E3 (Tuning Eksperimen):** Baseline E1 dengan **satu modifikasi terukur**, yaitu menaikkan resolusi citra input menjadi `imgsz=640`.

---

## III. HASIL PENGUJIAN & DATA NUMERIK

Tabel berikut merupakan hasil eksekusi resmi yang tersimpan di berkas `hasil_praktik_422411022.txt` dan `hasil_praktik_422411022.csv`:

| Eksperimen | Perubahan Konfigurasi | mAP50 | mAP50-95 | Precision (P) | Recall (R) | Inferensi (ms) | Waktu Latih (menit) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **E1_pretrained** | *None (Baseline 416 px)* | **0.995** | **0.917** | 0.990 | 1.000 | 14.1 ms | 13.8 m |
| **E2_scratch** | *Bobot Acak (Tanpa Transfer)* | **0.969** | **0.679** | 0.853 | 0.911 | 8.2 ms | 13.7 m |
| **E3_tuning** | `imgsz=640` *(Resolusi Tinggi)* | **0.995** | **0.919** | 0.993 | 0.989 | 24.9 ms | 90.0 m |

### Pemilihan Model Terbaik & Hasil Uji Final
- **Kriteria Pemilihan**: Nilai `mAP50-95` tertinggi pada split **Validation**.
- **Model Terpilih**: **`E3_tuning`** (mAP50-95 = **0.919**).
- **Pengujian pada Data TEST (Unseen Data - 1 Kali Eksekusi)**:
  - **mAP50 (IoU 0.50)** : **0.995 (99.5%)**
  - **mAP50-95 (Rata-rata COCO)** : **0.915 (91.5%)**
  - **Precision Final** : **0.997**
  - **Recall Final** : **1.000**

---

## IV. ANALISIS & PEMBAHASAN MENDALAM

### 1. Dampak Nyata Transfer Learning (E1 vs E2)
Perbandingan antara **E1_pretrained** dan **E2_scratch** menunjukkan signifikansi penggunaan bobot pralatih (*Transfer Learning*):
- Pada metrik longgar `mAP50`, E2 masih mampu mencapai 0.969. Namun, pada metrik ketat **`mAP50-95`** (yang mengevaluasi ketepatan koordinat batas objek dari IoU 0.50 hingga 0.95), performa E2 anjlok ke **0.679**, sedangkan E1 mencapai **0.917** (terdapat peningkatan absolut sebesar **+23.8%**).
- **Alasan Teknis**: Layer konvolusi awal (*backbone*) pada model E1 telah dilatih pada jutaan citra COCO untuk mengenali representasi visual umum (gradien, sudut, tepi objek, kontur bentuk). Model hanya perlu menyesuaikan bobot klasifikasi dan regresi *head* untuk balok. Sebaliknya, model E2 harus mempelajari filter spasial dari nol hanya dengan 236 citra latih, sehingga dalam 30 epoch koordinat lokalisasi kotak belum terkalibrasi secara optimal.

### 2. Pengaruh Resolusi Input & Trade-off Komputasi (E1 vs E3)
Peningkatan resolusi citra input dari 416 ke 640 piksel pada eksperimen **E3_tuning** memberikan evaluasi *trade-off* nyata dalam visi komputer:
- **Peningkatan Akurasi**: `mAP50-95` naik tipis menjadi **0.919** dan Precision mencapai **0.993**. Resolusi lebih tinggi mempertahankan detail piksel batas objek balok terhadap latar belakang meja kerja.
- **Beban Komputasi (*Computational Cost*)**:
  - Waktu pelatihan melonjak dari **13.8 menit** (E1) menjadi **90.0 menit** (E3) pada CPU (sekitar 6.5x lebih lama).
  - Waktu inferensi per frame meningkat dari **14.1 ms** (~71 FPS) menjadi **24.9 ms** (~40 FPS).

### 3. Evaluasi Kelayakan untuk Sistem Robotika & ROS 2
Dalam perancangan sistem robot (seperti manipulasi Universal Robots / UR), pipeline penglihatan harus memenuhi anggaran latensi *real-time*:
- Target standar 15 FPS mengalokasikan total anggaran **66.7 ms** untuk seluruh alur (akuisisi frame kamera, undistort kalibrasi lensa, inferensi model, post-processing NMS, dan komunikasi perintah gerak URScript).
- Model **E3_tuning** membutuhkan **24.9 ms** untuk inferensi, ditambah preprocessing dan postprocessing (~1.8 ms), total waktu komputasi adalah **~26.7 ms**. Nilai ini berada jauh di bawah anggaran 66.7 ms, sehingga **sangat layak dan aman digunakan dalam sistem kontrol loop tertutup robot**.
- Jika sistem robot membutuhkan *frame rate* lebih tinggi (misal 60 FPS untuk pelacakan objek bergerak cepat), maka model **E1_pretrained** (14.1 ms) menjadi alternatif terbaik dengan penurunan akurasi yang sangat minimal.

### 4. Generalisasi Model pada Data Uji (Test Set)
Salah satu prinsip utama pembelajaran mesin adalah mencegah *overfitting*:
- Model `E3_tuning` yang dipilih dari data validasi kemudian diuji pada 81 citra *test set* yang sama sekali belum pernah dilihat oleh algoritma pemilihan bobot.
- Hasil uji menunjukkan **mAP50-95 sebesar 0.915** dan **Recall 1.000** (seluruh balok terdeteksi tanpa *false negative*). Performa data uji yang hampir persis sama dengan data validasi (0.919 vs 0.915) membuktikan bahwa dataset dan model memiliki **kemampuan generalisasi luar biasa** dan terbebas dari bias kebocoran data (*data leakage*).

---

## V. KESIMPULAN

1. Praktikum terpadu minggu 4–7 telah berhasil dilaksanakan secara tuntas dengan seluruh metrik terekam secara deterministik (sidik jari label `8E25699C0A30`).
2. **Transfer Learning terbukti sangat krusial**: meningkatkan mAP50-95 sebesar 23.8% dibandingkan inisialisasi bobot acak (*scratch*).
3. **Model Terpilih adalah `E3_tuning`** dengan akurasi validasi tertinggi (`mAP50` = 0.995, `mAP50-95` = 0.919) dan akurasi uji final `mAP50-95` = **0.915**.
4. Model memiliki latensi inferensi **24.9 ms**, memenuhi kriteria *real-time* untuk integrasi sistem visi lengan robotika (di bawah batas anggaran 15 FPS / 66.7 ms).

---

## VI. DAFTAR BERKAS LUARAN & LAMPIRAN

| Berkas | Lokasi / Path | Deskripsi |
|---|---|---|
| **Laporan Teks Resmi** | `vision_kit/hasil_praktik_422411022.txt` | Output ringkasan tabular hasil praktikum |
| **Data Numerik CSV** | `vision_kit/hasil_praktik_422411022.csv` | Data metrik lengkap semua eksperimen |
| **Bobot Model Terbaik** | `vision_kit/hasil_praktik/E3_tuning/weights/best.pt` | Model PyTorch terpilih untuk inferensi |
| **Grafik Kurva Loss & mAP** | `vision_kit/hasil_praktik/E3_tuning/results.png` | Grafik 30 epoch konvergensi loss & metrik |
| **Confusion Matrix** | `vision_kit/hasil_praktik/E3_tuning/confusion_matrix.png` | Matriks prediksi vs ground truth |
| **Kurva Precision-Recall** | `vision_kit/hasil_praktik/E3_tuning/BoxPR_curve.png` | Kurva PR per kelas objek (merah, hijau, hitam) |
