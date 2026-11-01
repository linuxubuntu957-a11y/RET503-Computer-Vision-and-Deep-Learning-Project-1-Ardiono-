#!/usr/bin/env python3
"""Kalibrasi kamera langsung dari webcam (metode Zhang) - RET503 Pertemuan 2.

Alur mengikuti materi:
  foto papan catur -> deteksi sudut (findChessboardCorners + cornerSubPix)
  -> calibrateCamera (homografi, K awal, optimasi LM) -> reprojection error
  -> simpan K dan distorsi ke .json dan .npz -> uji koreksi distorsi (undistort).

Contoh:
  python kalibrasi_kamera_live.py                        # kamera 0, 1280x720, papan 6x9 sudut dalam
  python kalibrasi_kamera_live.py --camera 1 --square-mm 25
  python kalibrasi_kamera_live.py --from-folder kalibrasi_foto   # hitung ulang dari foto tersimpan
  python kalibrasi_kamera_live.py --format npz                  # simpan .npz saja (atau json / both)
  python kalibrasi_kamera_live.py --load kalibrasi_kamera.npz    # hanya uji undistort

Pengambilan foto MANUAL: tekan SPASI saat papan terdeteksi; layar menampilkan jumlah foto yang sudah diambil.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------- konstanta
# Sama dengan calibration_utils.py: sub-piksel berhenti di 30 iterasi / 0,001 px.
CORNER_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
FLAGS_FULL = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
FLAGS_FAST = FLAGS_FULL | cv2.CALIB_CB_FAST_CHECK  # menolak frame tanpa papan dengan cepat

GRID_COLS, GRID_ROWS = 4, 3  # peta cakupan: 4 x 3 sel

BACKENDS = {
    "auto": cv2.CAP_ANY,
    "dshow": getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY),
    "msmf": getattr(cv2, "CAP_MSMF", cv2.CAP_ANY),
    "v4l2": getattr(cv2, "CAP_V4L2", cv2.CAP_ANY),
    "avfoundation": getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY),
}

GREEN, RED, YELLOW, WHITE, CYAN = (0, 200, 0), (0, 0, 255), (0, 220, 255), (255, 255, 255), (255, 200, 0)


# ------------------------------------------------------------- struktur data
@dataclass
class Shot:
    corners: np.ndarray  # (N,1,2) float32, sudut hasil cornerSubPix
    path: Path | None = None  # file foto mentah (jika disimpan)


@dataclass
class CalResult:
    rms: float
    rmse: float
    K: np.ndarray
    dist: np.ndarray
    rvecs: list
    tvecs: list
    per_image_error: np.ndarray
    image_size: tuple[int, int]


# --------------------------------------------------------------- fungsi inti
def make_object_points(board: tuple[int, int], square: float) -> np.ndarray:
    """Titik dunia papan di bidang Z=0, satuan = ukuran kotak (square)."""
    obj = np.zeros((board[0] * board[1], 3), np.float32)
    obj[:, :2] = np.mgrid[0 : board[0], 0 : board[1]].T.reshape(-1, 2)
    return obj * float(square)


def detect_board(gray: np.ndarray, board: tuple[int, int], fast: bool):
    """Deteksi 6x9 sudut dalam lalu haluskan ke sub-piksel. Mengembalikan (ok, corners)."""
    found, corners = cv2.findChessboardCorners(gray, board, FLAGS_FAST if fast else FLAGS_FULL)
    if not found:
        return False, None
    corners = np.asarray(corners, np.float32).reshape(-1, 1, 2)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CORNER_CRITERIA)
    return True, corners


def calibrate(shots: list[Shot], image_size, board, square) -> CalResult:
    """calibrateCamera + reprojection error per foto (rumus slide 21 dan 29)."""
    obj = make_object_points(board, square)
    obj_pts = [obj.copy() for _ in shots]
    img_pts = [s.corners for s in shots]
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, image_size, None, None)

    per_image, sq_total, n_total = [], 0.0, 0
    for o, c, r, t in zip(obj_pts, img_pts, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, r, t, K, dist)
        d = c.reshape(-1, 2).astype(np.float64) - proj.reshape(-1, 2).astype(np.float64)
        sq = float(np.square(d).sum())
        per_image.append(np.sqrt(sq / len(o)))
        sq_total += sq
        n_total += len(o)
    return CalResult(float(rms), float(np.sqrt(sq_total / n_total)), K, dist,
                     list(rvecs), list(tvecs), np.array(per_image), tuple(image_size))


def grade(rms: float) -> tuple[str, tuple[int, int, int]]:
    """Tabel interpretasi RMS dari slide 21 (pedoman praktis, bukan standar baku)."""
    if rms < 0.3:
        return "Sangat baik", GREEN
    if rms < 0.5:
        return "Baik", GREEN
    if rms < 1.0:
        return "Cukup - periksa foto buram / sudut salah deteksi", YELLOW
    return "Kurang baik - ulangi pengambilan data", RED


def board_tilts(res: CalResult) -> np.ndarray:
    """Sudut kemiringan papan terhadap sumbu optik (derajat) untuk tiap foto."""
    out = []
    for r in res.rvecs:
        R, _ = cv2.Rodrigues(r)
        out.append(np.degrees(np.arccos(np.clip(abs(R[2, 2]), 0.0, 1.0))))
    return np.array(out)


def quality_warnings(res: CalResult, shots: list[Shot], size, min_shots: int) -> list[str]:
    """Peringatan berdasarkan tips pengambilan foto (slide 21-22)."""
    w = []
    n = len(shots)
    if n < min_shots:
        w.append(f"Foto baru {n}; disarankan {min_shots}-25 foto")
    tilts = board_tilts(res)
    if tilts.max() < 15:
        w.append("Papan kurang dimiringkan (maks %.0f derajat); variasikan pose" % tilts.max())
    cov = np.zeros((GRID_ROWS, GRID_COLS), bool)
    for s in shots:
        cov |= coverage_of(s.corners, size)
    if not cov.all():
        w.append(f"Cakupan citra {cov.sum()}/{cov.size} sel; papan belum menjangkau tepi/pojok")
    mean, std = res.per_image_error.mean(), res.per_image_error.std()
    bad = np.where(res.per_image_error > mean + 2 * std)[0]
    if len(bad):
        w.append("Foto error tinggi (indeks): " + ", ".join(str(i + 1) for i in bad))
    cx, cy = res.K[0, 2], res.K[1, 2]
    if abs(cx - size[0] / 2) > 0.1 * size[0] or abs(cy - size[1] / 2) > 0.1 * size[1]:
        w.append("Principal point jauh dari pusat citra; periksa pose/foto")
    return w


def coverage_of(corners: np.ndarray, size) -> np.ndarray:
    """Sel mana (4x3) yang disentuh sudut-sudut sebuah foto."""
    cov = np.zeros((GRID_ROWS, GRID_COLS), bool)
    pts = corners.reshape(-1, 2)
    gx = np.clip((pts[:, 0] / size[0] * GRID_COLS).astype(int), 0, GRID_COLS - 1)
    gy = np.clip((pts[:, 1] / size[1] * GRID_ROWS).astype(int), 0, GRID_ROWS - 1)
    cov[gy, gx] = True
    return cov


def output_paths(output: str, fmt: str) -> list[Path]:
    """--output boleh tanpa ekstensi (kalibrasi_kamera) atau berekstensi .json/.npz."""
    base = Path(output)
    if base.suffix.lower() in (".json", ".npz"):
        base = base.with_suffix("")
    paths = []
    if fmt in ("both", "json"):
        paths.append(base.with_name(base.name + ".json"))
    if fmt in ("both", "npz"):
        paths.append(base.with_name(base.name + ".npz"))
    return paths


def save_results(paths: list[Path], res: CalResult, n_shots: int, board, square, unit: str) -> None:
    """Simpan hasil kalibrasi ke .json (terbaca manusia) dan/atau .npz (numpy)."""
    tanggal = datetime.now().isoformat(timespec="seconds")
    for path in paths:
        if path.suffix == ".json":
            data = {
                "image_size": list(res.image_size),  # [lebar, tinggi]
                "K": res.K.tolist(),
                "dist": res.dist.ravel().tolist(),  # k1, k2, p1, p2, k3
                "rms": res.rms,
                "rmse": res.rmse,
                "num_images": n_shots,
                "board_inner_corners": list(board),
                "square_size": square,
                "satuan_dunia": unit,
                "tanggal": tanggal,
            }
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        else:
            np.savez(
                path,
                K=res.K,
                dist=res.dist.reshape(1, -1),
                image_size=np.array(res.image_size),  # (lebar, tinggi)
                rms=res.rms,
                rmse=res.rmse,
                num_images=n_shots,
                board_inner_corners=np.array(board),
                square_size=square,
                satuan_dunia=unit,
                tanggal=tanggal,
            )


def load_calibration(path: Path):
    """Muat K, dist, image_size dari file .json atau .npz."""
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as d:
            K = np.array(d["K"], np.float64)
            dist = np.array(d["dist"], np.float64).reshape(1, -1)
            size = tuple(int(v) for v in d["image_size"])
        return K, dist, size
    d = json.loads(path.read_text(encoding="utf-8"))
    return (np.array(d["K"], np.float64), np.array(d["dist"], np.float64).reshape(1, -1),
            tuple(d["image_size"]))


def print_report(res: CalResult, warns: list[str], unit: str) -> None:
    label, _ = grade(res.rms)
    print("\n" + "=" * 60)
    print(f"RMS OpenCV : {res.rms:.6f} px   RMSE manual : {res.rmse:.6f} px  ->  {label}")
    print("Camera matrix K:\n", np.array2string(res.K, precision=4, suppress_small=True))
    print("Distorsi (k1, k2, p1, p2, k3):\n", np.array2string(res.dist.ravel(), precision=6))
    fx, fy = res.K[0, 0], res.K[1, 1]
    w, h = res.image_size
    fov = (2 * np.degrees(np.arctan(w / 2 / fx)), 2 * np.degrees(np.arctan(h / 2 / fy)))
    print(f"FOV kira-kira: {fov[0]:.1f} x {fov[1]:.1f} derajat   (translasi dalam satuan: {unit})")
    print("Error per foto (px): " + " ".join(f"{e:.2f}" for e in res.per_image_error))
    for m in warns:
        print("  [!]", m)
    print("=" * 60)


def build_undistort_maps(K, dist, size, alpha: float = 0.0):
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, size, alpha, size)
    return cv2.initUndistortRectifyMap(K, dist, None, newK, size, cv2.CV_32FC1)


# ------------------------------------------------------------------- kamera
def open_camera(source: str, width: int, height: int, backend: str, fourcc: str, autofocus: bool):
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src, BACKENDS[backend]) if isinstance(src, int) else cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Kamera/sumber '{source}' tidak bisa dibuka. Coba --camera 1 atau --backend dshow.")
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not autofocus:
        accepted = cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # fokus harus terkunci (slide 22)
        print(f"[info] Autofocus: perintah {'diterima' if accepted else 'ditolak'} driver, "
              f"nilai terbaca = {cap.get(cv2.CAP_PROP_AUTOFOCUS)}. Hasil ini tidak selalu akurat; "
              "pastikan lewat pengaturan kamera (tombol F di program atau aplikasi kamera).")
    for _ in range(5):  # buang frame awal (auto-exposure belum stabil)
        cap.read()
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Kamera terbuka tetapi tidak mengirim gambar.")
    h, w = frame.shape[:2]
    if (w, h) != (width, height):
        cap.release()
        raise RuntimeError(
            f"Kamera memberi {w}x{h}, bukan {width}x{height}. Kalibrasi harus memakai resolusi yang "
            "sama dengan saat kamera dipakai. Coba --fourcc MJPG, --backend dshow, atau kamera lain."
        )
    return cap


# ------------------------------------------------------------ gambar overlay
def put(img, text, org, color=WHITE, scale=0.55, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def panel(img, lines, x, y, w=None):
    """Kotak semi-transparan berisi teks [(teks, warna), ...]."""
    lh = 20
    w = w or max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0] for t, _ in lines) + 16
    h = lh * len(lines) + 10
    roi = img[y : y + h, x : x + w]
    if roi.size:
        img[y : y + h, x : x + w] = (roi * 0.35).astype(np.uint8)
    for i, (t, c) in enumerate(lines):
        put(img, t, (x + 8, y + 20 + i * lh), c)


def draw_minimap(img, cov: np.ndarray):
    """Peta cakupan 4x3 di pojok kanan atas: hijau = sel sudah dijangkau papan."""
    H, W = img.shape[:2]
    cw, ch = 40, 30
    x0, y0 = W - GRID_COLS * cw - 12, 10
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            p1 = (x0 + c * cw, y0 + r * ch)
            p2 = (p1[0] + cw - 2, p1[1] + ch - 2)
            cv2.rectangle(img, p1, p2, (0, 160, 0) if cov[r, c] else (60, 60, 60), -1)
    put(img, "cakupan", (x0, y0 + GRID_ROWS * ch + 14), WHITE, 0.45)


# ----------------------------------------------------------------- mode live
def run_live(args) -> int:
    board = (args.cols, args.rows)
    size = (args.width, args.height)
    cap = open_camera(args.camera, args.width, args.height, args.backend, args.fourcc, args.keep_autofocus)
    out_dir = Path(args.save_images) if args.save_images else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    win = "Kalibrasi Kamera (RET503)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    shots: list[Shot] = []
    result: CalResult | None = None
    saved = False
    undistort_on = False
    maps = None
    msg, msg_until, flash_until = "", 0.0, 0.0
    novelty_px = args.novelty * args.width

    if args.load:
        K, dist, lsize = load_calibration(Path(args.load))
        if tuple(lsize) != size:
            print(f"[!] Kalibrasi lama untuk {lsize}, kamera sekarang {size}.")
        else:
            maps = build_undistort_maps(K, dist, size)
            undistort_on = True
            msg, msg_until = "Kalibrasi lama dimuat (tekan U untuk bandingkan)", time.time() + 4

    def say(text: str, secs: float = 3.0):
        nonlocal msg, msg_until
        msg, msg_until = text, time.time() + secs

    def coverage_total() -> np.ndarray:
        cov = np.zeros((GRID_ROWS, GRID_COLS), bool)
        for s in shots:
            cov |= coverage_of(s.corners, size)
        return cov

    def add_shot(frame, corners) -> None:
        """Dipanggil hanya saat SPASI ditekan (tidak ada pengambilan otomatis)."""
        nonlocal result, saved, flash_until
        similar = ""
        if shots:
            d = min(float(np.linalg.norm(corners.reshape(-1, 2) - s.corners.reshape(-1, 2), axis=1).mean())
                    for s in shots)
            if d <= novelty_px:
                similar = "  (pose mirip foto sebelumnya, variasikan)"
        path = None
        if out_dir:
            path = out_dir / f"foto_{datetime.now():%Y%m%d_%H%M%S}_{len(shots) + 1:03d}.png"
            cv2.imwrite(str(path), frame)
        shots.append(Shot(corners.copy(), path))
        result, saved = None, False  # hasil lama tidak berlaku lagi
        flash_until = time.time() + 0.35
        print(f"Foto ke-{len(shots)} diambil")
        say(f"Foto ke-{len(shots)} berhasil diambil{similar}", 2.5)

    def do_calibrate() -> None:
        nonlocal result, maps, saved
        if len(shots) < 3:
            say("Minimal 3 foto (idealnya 15-25)", 3)
            return
        result = calibrate(shots, size, board, args.square_mm)
        saved = False
        print_report(result, quality_warnings(result, shots, size, args.min_shots), args.unit)
        maps = build_undistort_maps(result.K, result.dist, size)
        say(f"Kalibrasi selesai, RMS {result.rms:.3f} px. Tekan S untuk menyimpan", 4)

    print(__doc__)
    print("Tombol: SPASI ambil foto | C kalibrasi | U undistort | X buang foto terburuk | "
          "Z batalkan foto terakhir | S simpan hasil | F pengaturan kamera | Q keluar\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame gagal dibaca; berhenti.")
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = detect_board(gray, board, fast=True)

            sharp = None
            if found:
                pts = corners.reshape(-1, 2)
                x0, y0 = np.maximum(pts.min(0).astype(int), 0)
                x1, y1 = pts.max(0).astype(int)
                roi = gray[y0:y1, x0:x1]
                sharp = float(cv2.Laplacian(roi, cv2.CV_64F).var()) if roi.size else 0.0
            blurry = sharp is not None and sharp < args.min_sharpness

            # ---- tampilan
            if undistort_on and maps is not None:
                show = cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR)
                panel(show, [("UNDISTORT (alpha=0)  |  tekan U untuk kembali", YELLOW)], 10, 10)
            else:
                show = frame.copy()
                for s in shots:  # titik foto lama: bantu melihat area yang belum tercakup
                    for p in s.corners.reshape(-1, 2)[::3]:
                        cv2.circle(show, (int(p[0]), int(p[1])), 2, (0, 140, 255), -1)
                if found:
                    cv2.drawChessboardCorners(show, board, corners, found)
                if not found:
                    status = ("Papan tidak terdeteksi (6x9 sudut dalam harus terlihat utuh)", RED)
                elif blurry:
                    status = (f"Buram (ketajaman {sharp:.0f} < {args.min_sharpness:.0f}); tahan papan diam", YELLOW)
                else:
                    status = (f"Papan terdeteksi (ketajaman {sharp:.0f})  ->  tekan SPASI", GREEN)
                n = len(shots)
                count_color = GREEN if n >= args.min_shots else WHITE
                lines = [
                    (f"Foto diambil: {n}   (target {args.min_shots}-{args.max_shots})", count_color),
                    status,
                ]
                if result is not None:
                    g, gc = grade(result.rms)
                    lines.append((f"RMS {result.rms:.3f} px: {g}" + ("" if saved else "  [belum disimpan]"), gc))
                panel(show, lines, 10, 10)
                put(show, f"FOTO: {n}", (14, 150), count_color, 1.4, 3)
                draw_minimap(show, coverage_total())
                if time.time() < flash_until:  # bingkai hijau sesaat = foto berhasil diambil
                    cv2.rectangle(show, (0, 0), (show.shape[1] - 1, show.shape[0] - 1), GREEN, 14)

            if time.time() < msg_until:
                put(show, msg, (10, show.shape[0] - 40), CYAN, 0.7, 2)
            put(show, "SPASI ambil | C kalibrasi | U undistort | X buang terburuk | Z batal | S simpan | F kamera | Q keluar",
                (10, show.shape[0] - 12), WHITE, 0.5)
            cv2.imshow(win, show)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                if undistort_on:
                    say("Matikan mode undistort (U) dulu")
                elif not found:
                    say("Papan belum terdeteksi")
                elif blurry:
                    say("Foto buram, tahan papan diam lalu tekan SPASI lagi")
                elif len(shots) >= args.max_shots:
                    say(f"Sudah {args.max_shots} foto; tekan C untuk kalibrasi")
                else:
                    add_shot(frame, corners)
            elif key == ord("u"):
                if maps is None:
                    say("Belum ada kalibrasi (tekan C atau pakai --load)")
                else:
                    undistort_on = not undistort_on
            elif key == ord("c"):
                do_calibrate()
            elif key == ord("z"):
                if shots:
                    s = shots.pop()
                    if s.path and s.path.exists():
                        s.path.unlink()
                    result, saved = None, False
                    say(f"Foto terakhir dibatalkan. Sekarang {len(shots)} foto")
            elif key == ord("x"):
                if result is None:
                    say("Tekan C dulu")
                elif len(shots) <= 3:
                    say("Foto terlalu sedikit untuk dibuang")
                else:
                    worst = int(np.argmax(result.per_image_error))
                    print(f"Membuang foto #{worst + 1} (error {result.per_image_error[worst]:.2f} px)")
                    s = shots.pop(worst)
                    if s.path and s.path.exists():
                        s.path.unlink()
                    do_calibrate()
            elif key == ord("s"):
                if result is None:
                    say("Tekan C dulu sebelum menyimpan")
                else:
                    paths = output_paths(args.output, args.format)
                    save_results(paths, result, len(shots), board, args.square_mm, args.unit)
                    saved = True
                    names = " + ".join(p.name for p in paths)
                    say(f"Tersimpan: {names}", 4)
                    print("Tersimpan: " + ", ".join(str(p) for p in paths))
            elif key == ord("f"):
                # Dialog properti driver (fokus, eksposur). Umumnya hanya di Windows DirectShow.
                cap.set(getattr(cv2, "CAP_PROP_SETTINGS", 37), 1)
                say("Jika dialog tidak muncul, pakai aplikasi bawaan kamera (lihat panduan)", 4)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if result is not None and not saved:
        print("[!] Hasil kalibrasi terakhir belum disimpan (tekan S sebelum keluar).")
    return 0


# ------------------------------------------------------------ mode dari foto
def run_folder(args) -> int:
    board = (args.cols, args.rows)
    paths = sorted(p for ext in ("png", "jpg", "jpeg", "bmp") for p in glob.glob(str(Path(args.from_folder) / f"*.{ext}")))
    if not paths:
        raise FileNotFoundError(f"Tidak ada foto di {args.from_folder}")
    shots, size = [], None
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"[!] Lewati (tak terbaca): {p}")
            continue
        h, w = img.shape[:2]
        if size is None:
            size = (w, h)
        elif (w, h) != size:
            raise ValueError(f"Ukuran {p} = {(w, h)}, seharusnya {size}")
        ok, c = detect_board(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), board, fast=False)
        if ok:
            shots.append(Shot(c, Path(p)))
        else:
            print(f"[!] Papan tidak terdeteksi: {Path(p).name}")
    print(f"{len(paths)} foto, {len(shots)} terdeteksi, ukuran {size}")
    if size != (args.width, args.height):
        print(f"[!] Ukuran foto {size} berbeda dari --width/--height; hasil berlaku untuk {size}.")
    if len(shots) < 3:
        raise RuntimeError("Butuh minimal 3 foto yang terdeteksi.")
    res = calibrate(shots, size, board, args.square_mm)
    print_report(res, quality_warnings(res, shots, size, args.min_shots), args.unit)
    paths = output_paths(args.output, args.format)
    save_results(paths, res, len(shots), board, args.square_mm, args.unit)
    print("Tersimpan: " + ", ".join(str(p) for p in paths))
    return 0


# --------------------------------------------------------------------- main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", default="0", help="indeks kamera (0,1,..) atau path video (default 0)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--backend", choices=list(BACKENDS), default="auto",
                   help="Windows: dshow atau msmf; Linux: v4l2")
    p.add_argument("--fourcc", default="MJPG", help="format kamera; MJPG membantu 720p lancar. Kosongkan: ''")
    p.add_argument("--keep-autofocus", action="store_true", help="jangan matikan autofocus (tidak disarankan)")
    p.add_argument("--cols", type=int, default=6, help="sudut dalam ke samping (default 6)")
    p.add_argument("--rows", type=int, default=9, help="sudut dalam ke bawah (default 9)")
    p.add_argument("--square-mm", type=float, default=25.0, help="sisi satu kotak papan, hasil ukur sendiri")
    p.add_argument("--unit", default="mm", help="nama satuan untuk --square-mm (default mm)")
    p.add_argument("--min-shots", type=int, default=15)
    p.add_argument("--max-shots", type=int, default=30)
    p.add_argument("--min-sharpness", type=float, default=50.0, help="batas ketajaman (0 = nonaktif)")
    p.add_argument("--novelty", type=float, default=0.04,
                   help="pose dianggap mirip jika selisih < nilai ini x lebar citra (hanya peringatan)")
    p.add_argument("--save-images", default="kalibrasi_foto", help="folder foto mentah ('' = jangan simpan)")
    p.add_argument("--output", default="kalibrasi_kamera",
                   help="nama dasar file hasil; ekstensi .json/.npz ditambahkan otomatis")
    p.add_argument("--format", choices=["both", "json", "npz"], default="both",
                   help="format file hasil (default both: .json dan .npz)")
    p.add_argument("--load", help="muat file .json atau .npz lama untuk uji undistort")
    p.add_argument("--from-folder", help="hitung dari folder foto tanpa membuka kamera")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        return run_folder(args) if args.from_folder else run_live(args)
    except Exception as e:  # noqa: BLE001
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
