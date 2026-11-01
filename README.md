# vision_kit: pipeline vision PBL (RET503 / RET501)

Kit modular: logika ada di paket `vision/` (modul kecil, bisa diimpor ulang),
skrip di folder utama hanya "tipis" (baca argumen -> panggil modul -> cetak hasil).

## Arsitektur (ketergantungan hanya mengarah ke bawah)
```
Skrip CLI (tipis)   kalibrasi  import_foto  yolo_to_coco  capture  split  train  latency  check_labels  draw_labels  split_yolo  laporan_p1
        |
Modul tugas         report_p1  cls_train  cls_data  cls_models  yolo_dataset  bench
        |
Modul dasar         config  files  roi  preprocess  camera  calib  board  calib_eval
(tanpa torch)       yolo_labels  splitting  draw
```
Hanya `cls_*` (dan `bench.bench_torch_models`) yang memakai torch; sisanya jalan tanpa torch.

| Modul | Satu tanggung jawab |
|---|---|
| `config` | Konstanta bersama (ukuran input, mean/std, nama berkas bawaan, kelas) |
| `files` | Daftar gambar, penamaan sesi, metadata.csv |
| `roi` | Simpan/muat/potong ROI |
| `preprocess` | Preprocessing model (SAMA saat latih dan deploy) |
| `camera` | Membuka kamera (resolusi tetap, autofocus mati) |
| `calib` | Muat kalibrasi (.npz/.json), `Undistorter` |
| `board` | Titik papan catur dan deteksi sudut |
| `calib_eval` | Metrik kalibrasi (error, kemiringan, kelurusan, cakupan) |
| `yolo_labels` | Parse/validasi/konversi label YOLO |
| `splitting` | Pembagian PER SESI (anti data leakage) |
| `yolo_dataset` | Menyusun folder dataset YOLO + data.yaml |
| `draw` | Menggambar kotak label |
| `bench` | Pengukuran waktu |
| `autobox` | Usulan kotak balok lewat segmentasi warna (merah, hijau, hitam teruji) |
| `importer` | Impor foto yang sudah ada: nama, sesi, label usulan, crop |
| `coco_export` | Ekspor dataset YOLO ke COCO (`instances_<split>.json`) |
| `report_p1` | Gambar dan ringkasan laporan P1 |
| `cls_data` / `cls_models` / `cls_train` | Dataset, model, dan pelatihan klasifikasi |

## Folder `kalibrasi/` (hasil kalibrasi RET503)
```
kalibrasi/
  kalibrasi_kamera.npz / .json   K, dist, resolusi 1280x720 (RMS 0,541 px, 28 foto, papan 6x9, kotak 25 mm)
  kalibrasi_foto/                foto papan catur mentah
  kalibrasi_kamera_live.py       program kalibrasi live (metode Zhang)
```
Folder ini dibaca otomatis (`config.DEFAULT_CALIB`) oleh `capture.py`, `latency.py`, dan `laporan_p1.py`,
dari direktori mana pun skrip dijalankan. Kalibrasi ulang: `python kalibrasi.py` (SPASI foto, C hitung, S simpan)
atau `python kalibrasi.py --from-folder kalibrasi/kalibrasi_foto`. Hasilnya menimpa berkas di folder `kalibrasi/`.
Kalibrasi hanya berlaku untuk resolusi yang sama (1280x720); jika kamera atau resolusi berganti, kalibrasi ulang.

## Alur kerja
```
python kalibrasi.py                                   # (sekali / bila kamera berganti) buat kalibrasi/
python laporan_p1.py                                  # P1 (bawaan: kalibrasi/kalibrasi_foto + kalibrasi/kalibrasi_kamera.npz)
python capture.py biru terang-s1 --roi          # P2: ulangi 5 kelas x >=3 sesi (kelas: biru hijau hitam merah putih)
python split.py
python train.py --mode all --arch resnet18 --epochs 10
python latency.py
python capture.py scene terang-s1               # Pj1: frame PENUH (tanpa --roi), lalu anotasi YOLO
python check_labels.py --images raw --nc 5 --names biru hijau hitam merah putih
python draw_labels.py  --images raw --names biru hijau hitam merah putih --n 12
python split_yolo.py   --images raw --out dataset_v1 --names biru hijau hitam merah putih
# lalu program dosen: python 06_latih_bandingkan.py --nim <NIM> --nama "<Nama>" --data dataset_v1/data.yaml
```
Praktik COCO (P3 dan P4) memakai program dosen, bukan kit ini.

## Memakai foto yang sudah ada (mis. `balok_merah.zip`, `balok_hijau.zip`, `balok_hitam.zip`)
Satu sumber foto dipakai untuk tiga keluaran: klasifikasi (transfer learning), YOLO, dan COCO.
```
python import_foto.py balok_merah.zip balok_hijau.zip balok_hitam.zip --names merah hijau hitam
python draw_labels.py --images raw --names merah hijau hitam --n 40      # PERIKSA kotak usulan di folder preview/
python check_labels.py --images raw --nc 3 --names merah hijau hitam
# koreksi kotak yang meleset + beri label foto di raw_perlu_label/ (LabelImg dsb., format YOLO), lalu pindahkan ke raw/

# klasifikasi (transfer learning) : crop persegi di dataset_raw/<kelas>/
python split.py
python train.py --mode all --arch resnet18 --epochs 10

# deteksi YOLO + COCO : frame penuh di raw/
python split_yolo.py --images raw --out dataset_v1 --names merah hijau hitam
python yolo_to_coco.py --dataset dataset_v1          # -> dataset_v1/annotations/instances_{train,val,test}.json
```
Catatan:
- `import_foto.py` meng-undistort foto dengan `kalibrasi/` (sama seperti `capture.py`), membagi tiap kelas menjadi
  `--sesi` blok waktu berurutan (bawaan 5) sebagai "sesi", dan menamai `<kelas>_<tanggal>_lab_s<n>_<nomor>.png`.
- Kotak dibuat otomatis dari warna: USULAN, bukan label final. Kelas tanpa profil warna (mis. putih) dan foto yang
  gagal terdeteksi masuk `raw_perlu_label/` dan harus dilabel manual. Foto tanpa label di `raw/` dianggap background.
- Urutan `--names` = class_id YOLO. Saat menambah kelas baru, tambahkan DI BELAKANG (id lama tidak boleh berubah),
  lalu impor kelas baru saja: `python import_foto.py balok_X.zip --names merah hijau hitam X`.
- `split_yolo.py` kini membagi PER KELAS (tiap kelas ada di train/val/test); `--no-per-class` untuk perilaku lama.
- COCO: `category_id` = class_id + 1, bbox `[x, y, w, h]` piksel, `file_name` relatif ke `images/<split>/`.
  YOLO yang dilatih dari bobot pralatih COCO (transfer learning) memakai `dataset_v1/data.yaml`, bukan JSON COCO.

## Tes
```
python -m unittest discover -s tests -t .
```
Tes torch (`test_torch_modules.py`) dilewati otomatis bila torch belum terpasang.

## Menambah fitur
1. Taruh logika di modul `vision/` yang sesuai (atau modul baru), tanpa `print` dan tanpa membaca argumen.
2. Buat skrip CLI tipis yang memanggilnya.
3. Tambahkan tes di `tests/`.
