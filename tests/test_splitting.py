import unittest
from vision import splitting as sp


def groups(n_groups, per=5):
    return {f"s{i}": [f"s{i}_{j}" for j in range(per)] for i in range(n_groups)}


class TestSplitting(unittest.TestCase):
    def test_no_group_split_across_sets(self):
        r = sp.split_groups(groups(6), 0.2, 0.1, seed=1)
        owner = {}
        for s, items in r.items.items():
            for it in items:
                owner.setdefault(it.split("_")[0], set()).add(s)
        self.assertTrue(all(len(v) == 1 for v in owner.values()))
        self.assertEqual(sum(len(v) for v in r.items.values()), 30)
        self.assertTrue(r.items["val"] and r.items["test"] and r.items["train"])

    def test_deterministic_seed(self):
        a = sp.split_groups(groups(6), 0.2, 0.1, seed=3).items
        b = sp.split_groups(groups(6), 0.2, 0.1, seed=3).items
        self.assertEqual(a, b)

    def test_too_few_groups_raises(self):
        with self.assertRaises(ValueError):
            sp.split_groups(groups(2), 0.2, 0.1)
        with self.assertRaises(ValueError):
            sp.split_groups(groups(1), 0.2, 0.0)

    def test_contiguous_sizes(self):
        r = sp.split_contiguous(list(range(100)), 0.2, 0.1, gap=3)
        self.assertEqual(len(r["test"]), 10); self.assertEqual(len(r["val"]), 20)
        self.assertEqual(r["train"][-1], 100 - 10 - 20 - 6 - 1)
        self.assertTrue(set(r["train"]).isdisjoint(r["val"]) and set(r["val"]).isdisjoint(r["test"]))

    def test_split_class_files_by_session(self):
        fs = [f"merah_20261018_lab_{k}_{i:03d}.png" for k in ("a", "b", "c") for i in range(1, 6)]
        sess = lambda f: f.rsplit("_", 1)[0]
        tr, va, note = sp.split_class_files(fs, sess, 0.2, seed=1)
        self.assertTrue({sess(f) for f in tr}.isdisjoint({sess(f) for f in va}))
        self.assertIn("sesi", note)

    def test_split_class_files_single_session_fallback(self):
        fs = [f"x_{i:03d}.png" for i in range(30)]
        tr, va, note = sp.split_class_files(fs, lambda f: "satu", 0.2)
        self.assertIn("HANYA 1 SESI", note); self.assertEqual(len(va), 6)
        self.assertTrue(set(tr).isdisjoint(va))

    def test_split_scenes_warns_without_sessions(self):
        items = [f"img{i}" for i in range(40)]
        out, msg = sp.split_scenes(items, lambda x: None, 0.2, 0.1)
        self.assertIn("PERINGATAN", msg)
        out2, msg2 = sp.split_scenes([f"s{i % 5}_{i}" for i in range(50)], lambda x: x.split("_")[0], 0.2, 0.1)
        self.assertIn("per sesi", msg2); self.assertEqual(sum(len(v) for v in out2.values()), 50)


if __name__ == "__main__":
    unittest.main()
