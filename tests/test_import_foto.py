import tempfile, unittest
from datetime import datetime
from pathlib import Path
import cv2, numpy as np
from vision import autobox, files, importer, yolo_labels


def scene(color_bgr, box=(100, 80, 160, 200), size=(320, 240)):
    im = np.full((size[1], size[0], 3), 190, np.uint8)
    x1, y1, x2, y2 = box
    im[y1:y2, x1:x2] = color_bgr
    return im


class TestAutobox(unittest.TestCase):
    def test_merah_box_found(self):
        b = autobox.propose_box(scene((30, 30, 200)), "merah")
        self.assertIsNotNone(b)
        x1, y1, x2, y2 = b
        self.assertTrue(abs(x1 - 100) <= 5 and abs(y1 - 80) <= 5 and abs(x2 - 160) <= 5 and abs(y2 - 200) <= 5)

    def test_hitam_box_found(self):
        self.assertIsNotNone(autobox.propose_box(scene((20, 20, 20)), "hitam"))

    def test_none_without_profile_or_object_or_on_border(self):
        self.assertIsNone(autobox.propose_box(scene((30, 30, 200)), "putih"))
        self.assertIsNone(autobox.propose_box(np.full((240, 320, 3), 190, np.uint8), "merah"))
        self.assertIsNone(autobox.propose_box(scene((30, 30, 200), box=(0, 80, 60, 200)), "merah"))


class TestImporterHelpers(unittest.TestCase):
    def test_photo_datetime_and_sort(self):
        self.assertEqual(importer.photo_datetime("WIN_20261026_22_59_08_Pro.jpg"), datetime(2026, 10, 26, 22, 59, 8))
        self.assertIsNone(importer.photo_datetime("foto.jpg"))
        ps = [Path("WIN_20261026_23_00_00_Pro.jpg"), Path("x.jpg"), Path("WIN_20261026_22_59_08_Pro.jpg")]
        self.assertEqual([p.name for p in importer.sort_by_time(ps)],
                         ["WIN_20261026_22_59_08_Pro.jpg", "WIN_20261026_23_00_00_Pro.jpg", "x.jpg"])

    def test_split_sessions_contiguous_and_balanced(self):
        idx = importer.split_sessions(10, 5)
        self.assertEqual(idx, sorted(idx)); self.assertEqual(set(idx), {0, 1, 2, 3, 4})
        self.assertEqual(importer.split_sessions(2, 5), [0, 1])

    def test_match_class(self):
        names = ["merah", "hijau", "hitam"]
        self.assertEqual(importer.match_class("balok merah", names), "merah")
        self.assertEqual(importer.match_class("balok_hijau.zip", names), "hijau")
        with self.assertRaises(ValueError): importer.match_class("balok biru", names)
        with self.assertRaises(ValueError): importer.match_class("merah hijau", names)

    def test_square_crop_inside_frame(self):
        img = np.zeros((240, 320, 3), np.uint8)
        for box in [(0, 0, 10, 10), (300, 220, 320, 240), (100, 80, 160, 200)]:
            c = importer.square_crop(img, box)
            self.assertEqual(c.shape[0], c.shape[1]); self.assertGreaterEqual(c.shape[0], 64)


class TestImportClass(unittest.TestCase):
    def test_end_to_end(self):
        src, out = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
        paths = []
        for i in range(6):
            p = src / f"WIN_20261026_22_59_{i:02d}_Pro.jpg"
            cv2.imwrite(str(p), scene((30, 30, 200))); paths.append(p)
        p = src / "WIN_20261026_23_00_00_Pro.jpg"
        cv2.imwrite(str(p), np.full((240, 320, 3), 190, np.uint8)); paths.append(p)   # tanpa balok
        r = importer.import_class("merah", 2, paths, out / "cls", out / "raw", n_sessions=3, frame_size=(320, 240))
        self.assertEqual((r.n_total, r.n_labeled, len(r.no_box)), (7, 6, 1))
        names = sorted(q.name for q in (out / "raw").glob("*.png"))
        self.assertEqual(len(names), 6)
        self.assertTrue(all(files.session_of_scene(Path(n).stem) for n in names))
        self.assertEqual(files.session_of_class_image("merah", names[0]), "20261026_lab_s1")
        boxes, errs, _ = yolo_labels.read_label_file(out / "raw" / (Path(names[0]).stem + ".txt"), 3)
        self.assertEqual((len(boxes), errs, boxes[0].cls), (1, [], 2))
        self.assertEqual(len(list((out / "cls" / "merah").glob("*.png"))), 6)
        self.assertEqual(len(list((out / "raw_perlu_label").glob("*.png"))), 1)   # tidak masuk raw/
        self.assertEqual(len((out / "cls" / "metadata.csv").read_text().splitlines()), 7)
        with self.assertRaises(FileExistsError):
            importer.import_class("merah", 2, paths, out / "cls", out / "raw")

    def test_wrong_resolution_skipped(self):
        src, out = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
        p = src / "WIN_20261026_22_59_00_Pro.jpg"; cv2.imwrite(str(p), scene((30, 30, 200)))
        r = importer.import_class("merah", 0, [p], out / "c", out / "r", frame_size=(1280, 720))
        self.assertEqual(len(r.skipped), 1); self.assertEqual(r.n_labeled, 0)


if __name__ == "__main__":
    unittest.main()
