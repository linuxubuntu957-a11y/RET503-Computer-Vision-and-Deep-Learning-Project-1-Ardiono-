import json, tempfile, unittest
from pathlib import Path
import numpy as np
from vision.calib import Calibration

K = np.array([[642.4, 0, 634.0], [0, 640.1, 357.0], [0, 0, 1]])
DIST = np.array([[-0.02, 0.01, 0.0, 0.0, 0.0]])


class TestCalibration(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_load_npz_image_size_key(self):
        p = self.tmp / "a.npz"; np.savez(p, K=K, dist=DIST, image_size=np.array([1280, 720]), tanggal="x")
        c = Calibration.load(p)
        self.assertEqual(c.size, (1280, 720)); self.assertAlmostEqual(c.K[0, 0], 642.4)

    def test_load_npz_mtx_and_size_keys(self):
        p = self.tmp / "b.npz"; np.savez(p, mtx=K, dist=DIST, size=np.array([640, 480]))
        self.assertEqual(Calibration.load(p).size, (640, 480))

    def test_load_json(self):
        p = self.tmp / "c.json"
        p.write_text(json.dumps({"K": K.tolist(), "dist": DIST.tolist(), "image_size": [1280, 720]}))
        c = Calibration.load(p)
        self.assertEqual(c.size, (1280, 720)); self.assertEqual(c.dist.shape, (1, 5))

    def test_size_optional(self):
        p = self.tmp / "d.npz"; np.savez(p, K=K, dist=DIST)
        self.assertIsNone(Calibration.load(p).size)

    def test_check_frame_size(self):
        c = Calibration(K, DIST, (1280, 720))
        c.check_frame_size((1280, 720))
        with self.assertRaises(ValueError):
            c.check_frame_size((640, 480))

    def test_undistorter_shape_and_identity_for_zero_distortion(self):
        c = Calibration(K, np.zeros((1, 5)), (1280, 720))
        img = np.random.randint(0, 255, (720, 1280, 3), np.uint8)
        out = c.undistorter((1280, 720))(img)
        self.assertEqual(out.shape, img.shape)
        self.assertLessEqual(int(np.abs(out.astype(int) - img.astype(int)).max()), 1)


if __name__ == "__main__":
    unittest.main()
