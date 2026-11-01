import json, tempfile, unittest
from pathlib import Path
import cv2, numpy as np
from vision import coco_export, splitting, yolo_dataset

NAMES = ["merah", "hijau", "hitam"]


def make_dataset():
    root = Path(tempfile.mkdtemp()) / "ds"
    for s in ("train", "val"):
        (root / "images" / s).mkdir(parents=True); (root / "labels" / s).mkdir(parents=True)
    cv2.imwrite(str(root / "images/train/a.png"), np.zeros((100, 200, 3), np.uint8))
    (root / "labels/train/a.txt").write_text("1 0.5 0.5 0.5 0.5\n")
    cv2.imwrite(str(root / "images/val/b.png"), np.zeros((100, 200, 3), np.uint8))
    (root / "labels/val/b.txt").write_text("")
    yolo_dataset.write_data_yaml(root, NAMES)
    return root


class TestCocoExport(unittest.TestCase):
    def test_bbox_category_and_empty_image(self):
        root = make_dataset()
        written = coco_export.export_dataset(root, NAMES)
        tr = json.loads(written["train"].read_text()); va = json.loads(written["val"].read_text())
        a = tr["annotations"][0]
        self.assertEqual(a["bbox"], [50.0, 25.0, 100.0, 50.0]); self.assertEqual(a["category_id"], 2)
        self.assertEqual(a["area"], 5000.0)
        self.assertEqual(tr["images"][0]["width"], 200); self.assertEqual(tr["images"][0]["height"], 100)
        self.assertEqual([c["name"] for c in tr["categories"]], NAMES)
        self.assertEqual((len(va["images"]), len(va["annotations"])), (1, 0))

    def test_cat_offset_zero_and_read_names(self):
        root = make_dataset()
        tr = json.loads(coco_export.export_dataset(root, NAMES, cat_offset=0)["train"].read_text())
        self.assertEqual(tr["annotations"][0]["category_id"], 1)
        self.assertEqual(yolo_dataset.read_names(root / "data.yaml"), NAMES)


class TestSplitByClass(unittest.TestCase):
    def test_every_class_in_every_split(self):
        items = [f"{c}_20261026_lab_s{s}_{n:03d}" for c in ("merah", "hijau") for s in range(1, 6) for n in range(1, 4)]
        sess = lambda x: x.rsplit("_", 1)[0]
        out, msg = splitting.split_scenes_by_class(items, sess, lambda x: x.split("_")[0], 0.2, 0.1)
        for split in ("train", "val", "test"):
            self.assertEqual({x.split("_")[0] for x in out[split]}, {"merah", "hijau"}, split)
        self.assertEqual(sum(len(v) for v in out.values()), len(items))
        self.assertEqual(len({s for v in out.values() for s in map(sess, v)}), 10)


if __name__ == "__main__":
    unittest.main()
