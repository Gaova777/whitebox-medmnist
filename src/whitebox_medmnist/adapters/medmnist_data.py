"""MedMNIST v2 (Yang et al., 2023; CC BY 4.0) through the official `medmnist` package, downloaded from Zenodo."""
from pathlib import Path

import medmnist
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

INFO = medmnist.INFO


def arrays(name: str, split: str, size: int, root: Path = Path("data/raw")) -> tuple[np.ndarray, np.ndarray]:
    """Images as float32 N x C x H x W in [0, 1] and integer labels."""
    root.mkdir(parents=True, exist_ok=True)
    cls = getattr(medmnist, INFO[name]["python_class"])
    ds = cls(split=split, download=True, size=size, root=str(root))
    x = ds.imgs.astype(np.float32) / 255.0
    x = x[:, None] if x.ndim == 3 else x.transpose(0, 3, 1, 2)
    return x, ds.labels.reshape(-1).astype(np.int64)


def loader(name: str, split: str, size: int, batch: int = 128, shuffle: bool = False) -> DataLoader:
    x, y = arrays(name, split, size)
    return DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=batch, shuffle=shuffle)


def classes(name: str) -> list[str]:
    labels = INFO[name]["label"]
    return [labels[str(i)] for i in range(len(labels))]
