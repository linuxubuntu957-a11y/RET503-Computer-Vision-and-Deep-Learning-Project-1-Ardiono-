import tempfile, unittest
from pathlib import Path
import cv2, numpy as np
from vision import yolo_dataset as yd

NAMES = ["biru", "hijau", "hitam", "merah", "putih"]


def make_src(n=4):
    src = Path(tempfile.mkdtemp())
    imgs = []
    for i in range(n):
        p = src / f"scene_{i}.png"; cv2.imwrite(str(p), np.zeros((20, 20, 3), np.uint8)); imgs.append(p)
        if i != n - 1:                                   # gambar terakhir tanpa label (= background)
            (src / f"scene_{i}.txt").write_text(f"{i % 5} 0.5 0.5 0.1 0.1\n")
    return src, imgs


class TestYoloDataset(unittest.TestCase):
    def test_materialize_structure_yaml_and_background(self):
        src, imgs = make_src()
        out = Path(tempfile.mkdtemp()) / "ds"
        stats = yd.materialize({"train": imgs[:2], "val": imgs[2:3], "test": imgs[3:]}, src, out, NAMES)
        self.assertTrue((out / "images/train/scene_0.png").exists())
        self.assertTrue((out / "labels/test/scene_3.txt").exists())
        self.assertEqual((out / "labels/test/scene_3.txt").read_text(), "")
        self.assertEqual(stats["train"][0], 1)
        yaml = (out / "data.yaml").read_text()
        self.assertIn("  0: biru", yaml); self.assertIn("  4: putih", yaml); self.assertIn("val: images/val", yaml)

    def test_refuses_existing_unless_force(self):
        src, imgs = make_src()
        out = Path(tempfile.mkdtemp()) / "ds"
        yd.materialize({"train": imgs[:3], "val": imgs[3:]}, src, out, NAMES)
        with self.assertRaises(FileExistsError):
            yd.materialize({"train": imgs[:3], "val": imgs[3:]}, src, out, NAMES)
        yd.materialize({"train": imgs[:1], "val": imgs[1:2]}, src, out, NAMES, force=True)
        self.assertFalse((out / "images/train/scene_2.png").exists())    # sisa lama sudah hilang


if __name__ == "__main__":
    unittest.main()
