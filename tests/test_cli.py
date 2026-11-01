"""Tes fungsi inti skrip CLI (tanpa kamera dan tanpa torch)."""
import tempfile, unittest
from pathlib import Path
import cv2, numpy as np
import check_labels, split, split_yolo
from vision import files


def dataset_cls(root: Path):
    for c in ("biru", "merah"):
        (root / c).mkdir(parents=True)
        for s in ("terang-s1", "terang-s2", "redup-s1"):
            for i in range(1, 6):
                cv2.imwrite(str(root / c / f"{c}_20261018_lab_{s}_{i:03d}.png"), np.zeros((8, 8, 3), np.uint8))


class TestCli(unittest.TestCase):
    def test_split_build_split_by_session(self):
        root = Path(tempfile.mkdtemp()); dataset_cls(root)
        sp, report = split.build_split(root, 0.2, 42)
        self.assertEqual(sp["classes"], ["biru", "merah"]); self.assertEqual(len(report), 2)
        sess = lambda it: files.session_of_class_image(it[0].split("/")[0], it[0].split("/")[1])
        tr = {(it[1], sess(it)) for it in sp["train"]}; va = {(it[1], sess(it)) for it in sp["val"]}
        self.assertTrue(tr.isdisjoint(va)); self.assertEqual(len(sp["train"]) + len(sp["val"]), 30)

    def test_check_labels_analyze(self):
        d = Path(tempfile.mkdtemp())
        for i in range(3):
            cv2.imwrite(str(d / f"a{i}.png"), np.zeros((720, 1280, 3), np.uint8))
        (d / "a0.txt").write_text("3 0.5 0.5 0.0586 0.0347\n3 0.2 0.2 0.0586 0.0347\n")
        (d / "a1.txt").write_text("9 0.5 0.5 0.1 0.1\n")                         # class_id salah
        r = check_labels.analyze(files.list_images(d), d, 5)                     # a2 tanpa label
        self.assertEqual(r["count"][3], 2); self.assertEqual(len(r["errors"]), 1)
        self.assertTrue(any("tidak ada file label" in w for w in r["warns"]))
        self.assertAlmostEqual(r["sizes"][3][0][0], 75.0, delta=0.1)               # lebar kotak ~75 px

    def test_split_yolo_plan_by_session(self):
        d = Path(tempfile.mkdtemp())
        paths = []
        for s in ("terang-s1", "terang-s2", "redup-s1", "redup-s2"):
            for i in range(1, 4):
                p = d / f"scene_20261018_lab_{s}_{i:03d}.png"; p.write_bytes(b"x"); paths.append(p)
        splits, msg = split_yolo.plan(sorted(paths), 0.2, 0.1, 42)
        self.assertIn("per sesi", msg); self.assertEqual(sum(len(v) for v in splits.values()), 12)


if __name__ == "__main__":
    unittest.main()
