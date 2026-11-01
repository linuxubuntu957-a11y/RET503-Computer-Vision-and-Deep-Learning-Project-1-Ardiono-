import csv, tempfile, unittest
from pathlib import Path
import cv2, numpy as np
from vision import files


class TestFiles(unittest.TestCase):
    def test_session_of_class_image(self):
        self.assertEqual(files.session_of_class_image("merah", "merah_20261018_lab_terang-s1_003.png"),
                         "20261018_lab_terang-s1")
        self.assertEqual(files.session_of_class_image("plat_besi", "plat_besi_20261018_lab_redup-s1_010.png"),
                         "20261018_lab_redup-s1")

    def test_session_of_scene(self):
        self.assertEqual(files.session_of_scene("scene_20261018_lab_terang-s1_001"), "scene_20261018_lab_terang-s1")
        self.assertIsNone(files.session_of_scene("img_0001"))

    def test_prefix_and_count(self):
        d = Path(tempfile.mkdtemp())
        pre = files.make_prefix("biru", "20261018", "lab", "terang-s1")
        self.assertEqual(pre, "biru_20261018_lab_terang-s1_")
        (d / (pre + "001.png")).write_bytes(b"x"); (d / (pre + "002.png")).write_bytes(b"x")
        self.assertEqual(files.count_existing(d, pre), 2)

    def test_metadata_header_written_once(self):
        m = Path(tempfile.mkdtemp()) / "metadata.csv"
        files.append_metadata(m, ["a.png", "biru", "20261018", "terang"])
        files.append_metadata(m, ["b.png", "biru", "20261018", "terang"])
        with open(m, encoding="utf-8") as f:
            rows = list(csv.reader(f))
        self.assertEqual(rows[0], list(files.META_HEADER)); self.assertEqual(len(rows), 3)

    def test_list_images_sorted_and_filtered(self):
        d = Path(tempfile.mkdtemp())
        for n in ["b.png", "a.jpg", "c.txt"]:
            (d / n).write_bytes(b"x")
        self.assertEqual([p.name for p in files.list_images(d)], ["a.jpg", "b.png"])


if __name__ == "__main__":
    unittest.main()
