"""
src/data/datasets.py
────────────────────
Dataset and DataLoader factories for PreserveNet.

Usage
-----
    from src.data.datasets import get_cifar10_loaders

    train_loader, val_loader, meta = get_cifar10_loaders(
        data_dir='./data',
        batch_size=128,
        image_size=32,      # or 224 for pretrained ViT
        num_workers=2,
    )
    print(meta)
    # {
    #   'classes': ['airplane', ...],
    #   'n_classes': 10,
    #   'n_train': 50000,
    #   'n_val': 10000,
    #   'image_size': 32,
    #   'mean': (0.4914, 0.4822, 0.4465),
    #   'std':  (0.2470, 0.2435, 0.2616),
    # }
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from src.data.transforms import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    IMAGENET_MEAN,
    IMAGENET_STD,
    cifar10_train_transform,
    cifar10_val_transform,
)


# ── Metadata container ─────────────────────────────────────────────────────────

@dataclass
class DatasetMeta:
    """Lightweight metadata bundle returned alongside every DataLoader pair."""

    classes:    list[str]
    n_classes:  int
    n_train:    int
    n_val:      int
    image_size: int
    mean:       tuple[float, float, float]
    std:        tuple[float, float, float]
    name:       str = ''

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"DatasetMeta(name='{self.name}', classes={self.n_classes}, "
            f"train={self.n_train:,}, val={self.n_val:,}, "
            f"image_size={self.image_size})"
        )


# ── CIFAR-10 ───────────────────────────────────────────────────────────────────

#: Human-readable class names in the canonical torchvision label order.
CIFAR10_CLASSES: list[str] = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck',
]


def get_cifar10_loaders(
    data_dir:    str | Path = './data',
    batch_size:  int = 128,
    image_size:  int = 32,
    num_workers: int | None = None,
    augment:     bool = True,
    pin_memory:  bool = True,
    download:    bool = True,
) -> tuple[DataLoader, DataLoader, DatasetMeta]:
    """
    Build train and validation DataLoaders for CIFAR-10.

    Args:
        data_dir:    Root directory where the dataset will be downloaded/cached.
        batch_size:  Mini-batch size for both loaders.
        image_size:  Spatial resolution fed to the model.
                     • 32  — native CIFAR-10 resolution; uses CIFAR statistics.
                     • 224 — resize for a pretrained ViT; uses ImageNet statistics.
        num_workers: Number of DataLoader worker processes.
                     Defaults to ``min(os.cpu_count(), 4)`` on Linux/Mac
                     and ``0`` on Windows (avoids multiprocessing spawn issues).
        augment:     Whether to apply random crop + horizontal flip to the
                     training split.
        pin_memory:  Pin tensors to page-locked memory for faster GPU transfer.

    Returns:
        (train_loader, val_loader, meta) where *meta* is a :class:`DatasetMeta`.

    Example:
        >>> train_loader, val_loader, meta = get_cifar10_loaders(
        ...     data_dir='./data', batch_size=128, image_size=224
        ... )
        >>> imgs, labels = next(iter(train_loader))
        >>> imgs.shape   # torch.Size([128, 3, 224, 224])
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # ── Resolve num_workers ──────────────────────────────────────────────────
    if num_workers is None:
        # Windows multiprocessing with DataLoader can deadlock with workers > 0
        # unless the calling script guards with `if __name__ == '__main__'`.
        # Default to 0 on Windows to be safe.
        import platform
        num_workers = 0 if platform.system() == 'Windows' else min(os.cpu_count() or 1, 4)

    # ── Transforms ────────────────────────────────────────────────────────────
    train_tf = cifar10_train_transform(image_size) if augment else cifar10_val_transform(image_size)
    val_tf   = cifar10_val_transform(image_size)

    # ── Datasets ──────────────────────────────────────────────────────────────
    cifar_root = str(data_dir / 'cifar10')
    train_ds = datasets.CIFAR10(
        root=cifar_root, train=True, download=download, transform=train_tf
    )
    val_ds = datasets.CIFAR10(
        root=cifar_root, train=False, download=download, transform=val_tf
    )

    # ── DataLoaders ───────────────────────────────────────────────────────────
    _loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **_loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **_loader_kwargs)

    # ── Metadata ──────────────────────────────────────────────────────────────
    mean = CIFAR10_MEAN if image_size == 32 else IMAGENET_MEAN
    std  = CIFAR10_STD  if image_size == 32 else IMAGENET_STD

    meta = DatasetMeta(
        name='CIFAR-10',
        classes=CIFAR10_CLASSES,
        n_classes=10,
        n_train=len(train_ds),
        n_val=len(val_ds),
        image_size=image_size,
        mean=mean,
        std=std,
    )

    return train_loader, val_loader, meta


# ── Quick smoke-test (run this file directly) ──────────────────────────────────

if __name__ == '__main__':
    import sys
    import time
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')          # headless — save PNG instead of showing
    import matplotlib.pyplot as plt

    print('=' * 55)
    print('  PreserveNet — CIFAR-10 DataLoader smoke test')
    print('=' * 55)

    # ── Build loaders ────────────────────────────────────────────────────────
    IMAGE_SIZE   = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    BATCH_SIZE   = 64
    DATA_DIR     = Path(__file__).resolve().parents[3] / 'data'

    print(f'\nimage_size : {IMAGE_SIZE}')
    print(f'batch_size : {BATCH_SIZE}')
    print(f'data_dir   : {DATA_DIR}')

    t0 = time.perf_counter()
    train_loader, val_loader, meta = get_cifar10_loaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
    )
    print(f'\nLoader built in {time.perf_counter() - t0:.2f}s')
    print(f'Meta       : {meta}')
    print(f'Train batches : {len(train_loader)}')
    print(f'Val   batches : {len(val_loader)}')

    # ── First batch ──────────────────────────────────────────────────────────
    t1 = time.perf_counter()
    imgs, labels = next(iter(train_loader))
    print(f'\nFirst batch fetched in {time.perf_counter() - t1:.3f}s')
    print(f'imgs shape    : {tuple(imgs.shape)}   dtype={imgs.dtype}')
    print(f'labels shape  : {tuple(labels.shape)}  dtype={labels.dtype}')
    print(f'pixel range   : [{imgs.min():.4f}, {imgs.max():.4f}]')
    print(f'label counts  : {dict(zip(*np.unique(labels.numpy(), return_counts=True)))}')

    # ── Visualise one image per class ─────────────────────────────────────────
    from src.data.transforms import denormalise

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle(
        f'CIFAR-10 smoke test  |  image_size={IMAGE_SIZE}  |  batch_size={BATCH_SIZE}',
        fontsize=12, fontweight='bold'
    )

    seen = {}
    for img, lbl in zip(imgs, labels):
        lbl_int = int(lbl)
        if lbl_int not in seen:
            seen[lbl_int] = img

    for idx in range(10):
        ax  = axes[idx // 5, idx % 5]
        img = denormalise(
            seen[idx],
            mean=meta.mean,
            std=meta.std,
        )
        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.set_title(meta.classes[idx], fontsize=9)
        ax.axis('off')

    out_png = Path(__file__).parent / 'cifar10_smoke_test.png'
    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches='tight')
    print(f'\nSample image grid saved to: {out_png}')
    print('\nSmoke test PASSED.')
