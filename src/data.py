from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


def make_surface(label: int, size: int, rng: np.random.Generator) -> np.ndarray:
    """Create a synthetic grayscale surface; label 1 contains a visible defect."""
    base = rng.normal(0.5, 0.08, (size, size)).astype(np.float32)
    gradient = np.linspace(-0.08, 0.08, size, dtype=np.float32)[None, :]
    image = base + gradient
    if label:
        if rng.random() < 0.5:
            row = int(rng.integers(4, size - 4))
            image[max(0, row - 1): min(size, row + 2), 3:-3] += rng.uniform(0.3, 0.5)
        else:
            cx, cy = rng.integers(5, size - 5, 2)
            radius = int(rng.integers(2, 5))
            yy, xx = np.ogrid[:size, :size]
            image[(xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2] -= rng.uniform(0.35, 0.55)
    return np.clip(image, 0, 1)


class SyntheticDefectDataset(Dataset):
    def __init__(self, samples: int = 1_000, size: int = 32, seed: int = 42):
        rng = np.random.default_rng(seed)
        labels = rng.integers(0, 2, samples)
        self.images = torch.tensor(
            np.stack([make_surface(int(label), size, rng) for label in labels]), dtype=torch.float32
        ).unsqueeze(1)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.images[index], self.labels[index]

