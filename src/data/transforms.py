"""
src/data/transforms.py
─────────────────────
Transform factories for PreserveNet datasets.

All normalisation constants live here so every other module can import
from a single source of truth rather than repeating magic numbers.
"""

from __future__ import annotations

import torchvision.transforms as T

# ── Normalisation constants ────────────────────────────────────────────────────

# CIFAR-10 dataset statistics (computed over the full 50K train split)
CIFAR10_MEAN: tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
CIFAR10_STD:  tuple[float, float, float] = (0.2470, 0.2435, 0.2616)

# ImageNet statistics — used when a pretrained ViT encoder is the backbone
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD:  tuple[float, float, float] = (0.229, 0.224, 0.225)


# ── CIFAR-10 transforms ────────────────────────────────────────────────────────

def cifar10_train_transform(image_size: int = 32) -> T.Compose:
    """
    Training transform for CIFAR-10.

    Args:
        image_size: Output spatial resolution.
                    32  → native CIFAR-10 with CIFAR statistics.
                    224 → resize for a pretrained ViT; uses ImageNet stats.

    Returns:
        A ``torchvision.transforms.Compose`` pipeline.
    """
    if image_size == 32:
        mean, std = CIFAR10_MEAN, CIFAR10_STD
        ops = [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    else:
        mean, std = IMAGENET_MEAN, IMAGENET_STD
        pad = max(4, image_size // 8)
        ops = [
            T.Resize(image_size),
            T.RandomCrop(image_size, padding=pad),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    return T.Compose(ops)


def cifar10_val_transform(image_size: int = 32) -> T.Compose:
    """
    Deterministic validation/test transform for CIFAR-10.

    Args:
        image_size: Output spatial resolution (32 or 224).

    Returns:
        A ``torchvision.transforms.Compose`` pipeline.
    """
    if image_size == 32:
        mean, std = CIFAR10_MEAN, CIFAR10_STD
        ops = [T.ToTensor(), T.Normalize(mean, std)]
    else:
        mean, std = IMAGENET_MEAN, IMAGENET_STD
        ops = [T.Resize(image_size), T.ToTensor(), T.Normalize(mean, std)]
    return T.Compose(ops)


def cifar10_raw_transform(image_size: int = 32) -> T.Compose:
    """
    Raw transform — converts to [0, 1] tensor with NO normalisation.
    Use this for visualisation only (not for model input).

    Args:
        image_size: If != 32, images are resized before conversion.

    Returns:
        A ``torchvision.transforms.Compose`` pipeline.
    """
    if image_size == 32:
        return T.Compose([T.ToTensor()])
    return T.Compose([T.Resize(image_size), T.ToTensor()])


def denormalise(
    tensor,  # noqa: ANN001
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std:  tuple[float, float, float] = IMAGENET_STD,
):
    """
    Undo ImageNet/CIFAR normalisation for display purposes.

    Args:
        tensor: Normalised image tensor of shape (C, H, W) or (B, C, H, W).
        mean:   Per-channel mean used during normalisation.
        std:    Per-channel std used during normalisation.

    Returns:
        Tensor clamped to [0, 1].
    """
    import torch
    m = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)
    s = torch.tensor(std,  dtype=tensor.dtype, device=tensor.device)
    # Broadcast over spatial dims
    if tensor.dim() == 3:          # C × H × W
        m, s = m.view(3, 1, 1), s.view(3, 1, 1)
    elif tensor.dim() == 4:        # B × C × H × W
        m, s = m.view(1, 3, 1, 1), s.view(1, 3, 1, 1)
    return (tensor * s + m).clamp(0.0, 1.0)
