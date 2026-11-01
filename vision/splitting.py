"""Pembagian dataset PER SESI (anti data leakage). Murni Python: tidak membaca/menulis berkas."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, TypeVar

T = TypeVar("T")


@dataclass
class GroupSplit:
    """Hasil pembagian per kelompok: item per split + nama kelompok per split."""
    items: Dict[str, List]
    groups: Dict[str, List[str]]


def split_groups(groups: Dict[str, List[T]], val_frac: float, test_frac: float = 0.0,
                 seed: int = 42) -> GroupSplit:
    """Bagi per KELOMPOK (sesi): satu sesi tidak pernah terbelah antar split.
    Syarat: test_frac=0 -> minimal 2 kelompok; test_frac>0 -> minimal 3 kelompok."""
    keys = sorted(groups)
    need = 3 if test_frac > 0 else 2
    if len(keys) < need:
        raise ValueError(f"butuh minimal {need} kelompok, ada {len(keys)}")
    random.Random(seed).shuffle(keys)
    n_te = max(1, round(test_frac * len(keys))) if test_frac > 0 else 0
    n_va = max(1, round(val_frac * len(keys)))
    n_va = min(n_va, len(keys) - n_te - 1)
    items: Dict[str, List] = {"train": [], "val": [], "test": []}
    for i, k in enumerate(keys):
        s = "test" if i < n_te else "val" if i < n_te + n_va else "train"
        items[s] += groups[k]
    return GroupSplit(items, {"test": keys[:n_te], "val": keys[n_te:n_te + n_va], "train": keys[n_te + n_va:]})


def split_contiguous(items: Sequence[T], val_frac: float, test_frac: float = 0.0,
                     gap: int = 3) -> Dict[str, List[T]]:
    """Cadangan bila sesi tidak cukup: potong berurutan (train | jeda | val | jeda | test)."""
    n = len(items)
    n_te = max(1, round(test_frac * n)) if test_frac > 0 else 0
    n_va = max(1, round(val_frac * n))
    n_tr = max(0, n - n_te - n_va - (2 * gap if n_te else gap))
    out = {"train": list(items[:n_tr]), "test": list(items[n - n_te:]) if n_te else []}
    start = n_tr + gap
    out["val"] = list(items[start:start + n_va]) if n_te else list(items[n - n_va:])
    return out


def split_class_files(files: Sequence[str], session_of, val_frac: float,
                      seed: int = 42) -> Tuple[List[str], List[str], str]:
    """Klasifikasi, satu kelas: (train, val, catatan). `session_of(nama_file)` -> id sesi."""
    sess: Dict[str, List[str]] = {}
    for f in files:
        sess.setdefault(session_of(f), []).append(f)
    if len(sess) >= 2:
        r = split_groups(sess, val_frac, 0.0, seed)
        return r.items["train"], r.items["val"], f"{len(sess)} sesi, val = {sorted(r.groups['val'])}"
    n = len(files)
    n_val = max(1, round(val_frac * n))
    return list(files[:max(0, n - n_val - 3)]), list(files[n - n_val:]), \
        "HANYA 1 SESI -> split berurutan (tambah sesi/kondisi baru!)"


def split_scenes(items: Sequence[T], session_of, val_frac: float, test_frac: float,
                 seed: int = 42) -> Tuple[Dict[str, List[T]], str]:
    """Deteksi: bagi per sesi; jika sesi < 3 atau ada nama tak dikenali -> berurutan + peringatan."""
    sess: Dict[Optional[str], List[T]] = {}
    for it in items:
        sess.setdefault(session_of(it), []).append(it)
    if None not in sess and len(sess) >= 3:
        r = split_groups(sess, val_frac, test_frac, seed)  # type: ignore[arg-type]
        return r.items, f"Dibagi per sesi ({len(sess)} sesi): test={r.groups['test']}, val={r.groups['val']}"
    return split_contiguous(items, val_frac, test_frac), \
        "PERINGATAN: sesi < 3 atau nama file tidak dikenali -> split berurutan (bukan per sesi)."


def split_scenes_by_class(items: Sequence[T], session_of, class_of, val_frac: float, test_frac: float,
                          seed: int = 42) -> Tuple[Dict[str, List[T]], str]:
    """Deteksi dengan banyak kelas: split_scenes dijalankan PER KELAS lalu digabung, sehingga tiap kelas
    muncul di train/val/test (tanpa ini, satu kelas bisa seluruhnya jatuh ke val atau test)."""
    by_cls: Dict[str, List[T]] = {}
    for it in items:
        by_cls.setdefault(class_of(it), []).append(it)
    out: Dict[str, List[T]] = {"train": [], "val": [], "test": []}
    notes = []
    for c in sorted(by_cls):
        sp, note = split_scenes(by_cls[c], session_of, val_frac, test_frac, seed)
        for k in out:
            out[k] += sp[k]
        notes.append(f"[{c}] {note}")
    return out, "\n".join(notes)
