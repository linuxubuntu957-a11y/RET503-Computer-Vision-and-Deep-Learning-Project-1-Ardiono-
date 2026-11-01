"""Alur capture.py dengan kamera PALSU (tanpa perangkat dan tanpa jendela)."""
import csv, tempfile, unittest
from pathlib import Path
from unittest import mock
import numpy as np
import capture


class FakeCap:
    def __init__(self, n=50, size=(1280, 720)):
        self.n, self.size = n, size
    def read(self):
        self.n -= 1
        return (self.n > 0), np.random.randint(0, 255, (self.size[1], self.size[0], 3), np.uint8)
    def release(self): pass


class TestCaptureFlow(unittest.TestCase):
    def run_capture(self, extra, keys, calib=True):
        tmp = Path(tempfile.mkdtemp())
        cal = tmp / "kal.npz"
        if calib:
            np.savez(cal, K=np.array([[640., 0, 640], [0, 640, 360], [0, 0, 1]]),
                     dist=np.array([[-0.02, 0, 0, 0, 0]]), image_size=np.array([1280, 720]))
        keys = iter(keys)
        argv = ["merah", "terang-s1", "--calib", str(cal), "--out", str(tmp / "ds")] + extra
        with mock.patch("capture.camera.open_camera", return_value=(FakeCap(), (1280, 720))), \
             mock.patch("capture.cv2.imshow"), mock.patch("capture.cv2.destroyAllWindows"), \
             mock.patch("capture.cv2.destroyWindow"), \
             mock.patch("capture.cv2.waitKey", side_effect=lambda _=0: next(keys, ord("q"))):
            capture.main(argv)
        return tmp / "ds"

    def test_saves_images_and_metadata(self):
        ds = self.run_capture([], [ord("s"), -1, ord("s"), ord("q")])
        pngs = sorted((ds / "merah").glob("*.png"))
        self.assertEqual(len(pngs), 2)
        self.assertRegex(pngs[0].name, r"^merah_\d{8}_lab_terang-s1_001\.png$")
        with open(ds / "metadata.csv", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        self.assertEqual(len(rows), 3); self.assertEqual(rows[1][1], "merah")

    def test_roi_crop_is_saved(self):
        with mock.patch("capture.cv2.selectROI", return_value=(100, 50, 200, 200)):
            ds = self.run_capture(["--roi"], [ord("s"), ord("q")])
        import cv2
        im = cv2.imread(str(next((ds / "merah").glob("*.png"))))
        self.assertEqual(im.shape[:2], (200, 200)); self.assertTrue((ds / "roi.json").exists())

    def test_without_calibration_still_saves(self):
        ds = self.run_capture([], [ord("s"), ord("q")], calib=False)
        self.assertEqual(len(list((ds / "merah").glob("*.png"))), 1)

    def test_resolution_mismatch_aborts(self):
        tmp = Path(tempfile.mkdtemp()); cal = tmp / "k.npz"
        np.savez(cal, K=np.eye(3), dist=np.zeros((1, 5)), image_size=np.array([640, 480]))
        with mock.patch("capture.camera.open_camera", return_value=(FakeCap(), (1280, 720))):
            with self.assertRaises(SystemExit):
                capture.main(["merah", "t", "--calib", str(cal), "--out", str(tmp / "ds")])


if __name__ == "__main__":
    unittest.main()
