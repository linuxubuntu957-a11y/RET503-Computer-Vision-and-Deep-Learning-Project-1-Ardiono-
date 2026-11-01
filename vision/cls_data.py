"""Dataset klasifikasi (butuh torch, torchvision, PIL)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import IMG_SIZE
from .preprocess import normalize_rgb, to_model_input


def load_split(path) -> dict:
    """Baca split.json: {'classes': [...], 'train': [[path, label], ...], 'val': [...]}."""
    return json.loads(Path(path).read_text())


def make_train_augment(size: int = IMG_SIZE):
    """Augmentasi train. Hue TIDAK diubah karena kelas ditentukan warna; saturasi hanya sedikit."""
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.1, hue=0.0),
    ])


class ClsDataset(Dataset):
    def __init__(self, root, items: Sequence[Tuple[str, int]], train: bool, size: int = IMG_SIZE):
        self.root, self.items, self.train, self.size = Path(root), list(items), train, size
        self.aug = make_train_augment(size)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        rel, y = self.items[i]
        bgr = cv2.imread(str(self.root / rel))
        if bgr is None:
            raise FileNotFoundError(rel)
        if self.train:      # resize dulu (sama seperti saat deploy), baru augmentasi
            small = cv2.resize(bgr, (self.size, self.size))
            pil = self.aug(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
            x = normalize_rgb(np.asarray(pil))
        else:
            x = to_model_input(bgr, self.size)
        return torch.from_numpy(x), y


def make_loaders(root, split: dict, batch: int = 16, size: int = IMG_SIZE) -> Tuple[DataLoader, DataLoader]:
    tr = DataLoader(ClsDataset(root, split["train"], True, size), batch_size=batch, shuffle=True, num_workers=0)
    va = DataLoader(ClsDataset(root, split["val"], False, size), batch_size=batch, shuffle=False, num_workers=0)
    return tr, va
