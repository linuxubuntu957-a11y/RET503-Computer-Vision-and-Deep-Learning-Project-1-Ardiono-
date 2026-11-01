import unittest
import cv2, numpy as np
from vision import board, calib_eval, bench
from vision.calib import Calibration
from vision.config import IMG_SIZE
from vision.preprocess import to_model_input


class TestPreprocess(unittest.TestCase):
    def test_shape_dtype(self):
        x = to_model_input(np.random.randint(0, 255, (720, 1280, 3), np.uint8))
        self.assertEqual(x.shape, (3, IMG_SIZE, IMG_SIZE)); self.assertEqual(x.dtype, np.float32)

    def test_bgr_to_rgb_order(self):
        red_bgr = np.zeros((50, 50, 3), np.uint8); red_bgr[:, :, 2] = 255
        x = to_model_input(red_bgr)
        self.assertAlmostEqual(float(x[0, 0, 0]), (1 - 0.485) / 0.229, places=3)    # kanal R
        self.assertAlmostEqual(float(x[2, 0, 0]), (0 - 0.406) / 0.225, places=3)    # kanal B


def synthetic_board(cols=6, rows=9, sq=60, margin=60):
    h, w = (rows + 1) * sq + 2 * margin, (cols + 1) * sq + 2 * margin
    im = np.full((h, w), 255, np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                y, x = margin + r * sq, margin + c * sq
                im[y:y + sq, x:x + sq] = 0
    return im


class TestBoard(unittest.TestCase):
    def test_object_points(self):
        p = board.object_points(6, 9, 25.0)
        self.assertEqual(p.shape, (54, 3)); self.assertEqual(tuple(p[1]), (25.0, 0.0, 0.0))
        self.assertEqual(tuple(p[6]), (0.0, 25.0, 0.0)); self.assertTrue((p[:, 2] == 0).all())

    def test_detect_synthetic_board(self):
        c = board.detect(synthetic_board(), 6, 9)
        self.assertIsNotNone(c); self.assertEqual(len(c), 54)
        self.assertIsNone(board.detect(np.full((200, 200), 128, np.uint8), 6, 9))


class TestCalibEval(unittest.TestCase):
    K = np.array([[640.0, 0, 640], [0, 640, 360], [0, 0, 1]])

    def test_pose_stat_recovers_known_pose(self):
        cal = Calibration(self.K, np.zeros((1, 5)), (1280, 720))
        objp = board.object_points(6, 9, 25.0)
        rvec = np.array([[np.radians(30)], [0.0], [0.0]]); tvec = np.array([[-60.0], [-100.0], [400.0]])
        pts, _ = cv2.projectPoints(objp, rvec, tvec, self.K, None)
        st = calib_eval.pose_stat(objp, pts.astype(np.float32), cal)
        self.assertLess(st.error, 1e-2); self.assertAlmostEqual(st.tilt, 30.0, delta=0.2)
        self.assertAlmostEqual(st.tz, 400.0, delta=1.0)

    def test_line_deviation(self):
        line = np.stack([np.arange(10.0), 2 * np.arange(10.0)], 1)
        self.assertLess(calib_eval.line_deviation(line), 1e-9)
        bent = line.copy(); bent[5, 1] += 3
        self.assertGreater(calib_eval.line_deviation(bent), 1.0)

    def test_grade_and_rms(self):
        self.assertEqual([calib_eval.grade_rms(v) for v in (0.2, 0.4, 0.59, 1.5)],
                         ["Sangat baik", "Baik", "Cukup", "Kurang baik"])
        self.assertAlmostEqual(calib_eval.overall_rms([3.0, 4.0]), np.sqrt(12.5))

    def test_coverage_grid(self):
        pts = np.array([[10, 10], [1270, 710], [640, 360], [650, 365]], float)
        cov = calib_eval.coverage_grid(pts, (1280, 720))
        self.assertEqual(cov.shape, (3, 4)); self.assertEqual(cov.sum(), 4); self.assertEqual(cov[0, 0], 1)

    def test_corner_shift_zero_without_distortion(self):
        cal = Calibration(self.K, np.zeros((1, 5)), (1280, 720))
        self.assertTrue(all(s < 1e-3 for s in calib_eval.corner_shift(cal, (1280, 720))))
        cal2 = Calibration(self.K, np.array([[-0.2, 0.05, 0, 0, 0]]), (1280, 720))
        self.assertTrue(all(s > 1.0 for s in calib_eval.corner_shift(cal2, (1280, 720))))


class TestBench(unittest.TestCase):
    def test_bench_calls_and_returns(self):
        calls = []
        mean, p95 = bench.bench(lambda: calls.append(1), n=10, warm=3)
        self.assertEqual(len(calls), 13); self.assertGreaterEqual(p95, 0.0); self.assertGreaterEqual(mean, 0.0)


if __name__ == "__main__":
    unittest.main()
