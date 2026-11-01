import tempfile, unittest
from pathlib import Path
from vision import yolo_labels as yl


class TestYoloLabels(unittest.TestCase):
    def test_parse_ok_and_errors(self):
        b = yl.parse_line("3 0.4199 0.4340 0.0586 0.0347")
        self.assertEqual((b.cls, round(b.cx, 4)), (3, 0.4199))
        for bad in ("1 2 3", "x 0.1 0.1 0.1 0.1", "0 a b c d"):
            with self.assertRaises(ValueError):
                yl.parse_line(bad)

    def test_check_box(self):
        e, w = yl.check_box(yl.Box(9, 0.5, 0.5, 0.1, 0.1), 5); self.assertTrue(e)
        e, w = yl.check_box(yl.Box(1, 640, 360, 75, 25), 5); self.assertTrue(any("0-1" in m for m in e))
        e, w = yl.check_box(yl.Box(1, 0.99, 0.5, 0.1, 0.1), 5); self.assertEqual((e, bool(w)), ([], True))
        e, w = yl.check_box(yl.Box(1, 0.5, 0.5, 0.1, 0.1), 5); self.assertEqual((e, w), ([], []))

    def test_xyxy_conversion_matches_academy_example(self):
        # contoh pelajaran Ultralytics Academy: bus.jpg 810x1080, orang (49,398)-(240,902)
        cx, cy, w, h = yl.xyxy_to_yolo(49, 398, 240, 902, 810, 1080)
        for got, want in zip((cx, cy, w, h), (0.178395, 0.601852, 0.235802, 0.466667)):
            self.assertAlmostEqual(got, want, places=6)

    def test_roundtrip(self):
        back = yl.yolo_to_xyxy(*yl.xyxy_to_yolo(500, 300, 575, 325, 1280, 720), 1280, 720)
        for g, w in zip(back, (500, 300, 575, 325)):
            self.assertAlmostEqual(g, w, places=6)

    def test_read_label_file(self):
        p = Path(tempfile.mkdtemp()) / "a.txt"
        p.write_text("0 0.5 0.5 0.1 0.1\n\n7 0.5 0.5 0.1 0.1\n1 0.5 0.5\n2 0.5 0.5 0.2 0.2\n")
        boxes, errs, warns = yl.read_label_file(p, 5)
        self.assertEqual([b.cls for b in boxes], [0, 2]); self.assertEqual(len(errs), 2)
        self.assertEqual(yl.read_label_file(p.parent / "tidak_ada.txt", 5), ([], [], []))

    def test_count_and_format(self):
        bs = [yl.Box(0, .5, .5, .1, .1), yl.Box(0, .5, .5, .1, .1), yl.Box(3, .5, .5, .1, .1)]
        self.assertEqual(yl.count_per_class(bs, 5), [2, 0, 0, 1, 0])
        self.assertEqual(yl.format_line(bs[0]), "0 0.500000 0.500000 0.100000 0.100000")


if __name__ == "__main__":
    unittest.main()
