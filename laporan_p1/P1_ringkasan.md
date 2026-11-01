# Ringkasan Kalibrasi Kamera (P1)

- Foto dipakai: **28**, resolusi **1280x720**, papan **6x9 sudut dalam**, sisi kotak **25.0 mm**
- **RMS reprojection error: 0.541 px** (Cukup; tabel slide 21)
- Error per foto: min 0.26, median 0.55, maks 0.81 px; terburuk: foto 8 (0.81), foto 9 (0.73), foto 19 (0.70)
- Kemiringan papan: 3-33 derajat; jarak papan: 165-287 mm
- Ketajaman (Laplacian) terendah: 198

## Matriks kamera K
```
[[642.03   0.   634.09]
 [  0.   639.78 358.05]
 [  0.     0.     1.  ]]
```
fx=642.03, fy=639.78, cx=634.09, cy=358.05  (FOV horizontal ~ 89.8 derajat)

## Koefisien distorsi (k1, k2, p1, p2, k3)
```
[-0.02192  0.01471 -0.00101  0.00013  0.01228]
```

## Kelurusan garis papan (px, rata-rata / maks penyimpangan dari garis lurus)
- Sebelum undistort: 0.47 / 3.38
- Sesudah undistort: 0.40 / 1.63

## Cakupan titik sudut per sel (4 kolom x 3 baris)
```
[[ 45 231 187  12]
 [ 85 409 337  16]
 [ 36  94  54   6]]
```