"""
src/baselines/uniform.py
────────────────────────
Uniform patch-drop and pixel-drop baselines for PreserveNet.
Supports both patch-level (e.g. 16×16) and pixel-level granularity.

UniformRandomReducer  (primary)
--------------------------------
Drops (1 - r) fraction of 16×16 spatial patches (or pixels) by sampling
a per-image random binary mask. Because the mask is sampled independently
per patch/pixel, the actual retention rate matches the target at every value.

UniformGridReducer  (reference)
--------------------------------
Keeps patches/pixels on a regular spatial grid using linspace-spaced indices.

Usage
-----
    from src.baselines.uniform import UniformRandomReducer, UniformGridReducer

    # 16x16 patch-level reduction on 224x224 image:
    reducer = UniformRandomReducer(retention_rate=0.75, patch_size=16)
    x_out   = reducer(x)               # x: Tensor[B, C, H, W]
    print(reducer.actual_retention())  # -> 0.75
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F


class UniformRandomReducer:
    """
    Zero out a random (1 - retention_rate) fraction of patches or pixels per image.

    Args:
        retention_rate: Fraction of spatial units to *keep*, e.g. 0.75.
        patch_size:     Spatial patch dimension (default 16 for 16×16 patch drop).
                        Set to 1 for pixel-level drop.
        seed:           Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        retention_rate: float,
        patch_size: int = 16,
        seed: int | None = None,
    ) -> None:
        if not (0.0 < retention_rate <= 1.0):
            raise ValueError(f'retention_rate must be in (0, 1], got {retention_rate}')
        self.retention_rate = retention_rate
        self.patch_size = patch_size
        self._rng = torch.Generator()
        if seed is not None:
            self._rng.manual_seed(seed)

    def actual_retention(self, H: int = 224, W: int = 224) -> float:
        """Always equals the requested retention_rate (no rounding)."""
        return self.retention_rate

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply a fresh random patch mask to a batch of images.

        Args:
            x: Float tensor of shape (B, C, H, W).

        Returns:
            Masked tensor of the same shape; dropped patches/pixels are set to 0.
        """
        B, C, H, W = x.shape
        P = self.patch_size

        if P <= 1:
            mask = torch.zeros(B, 1, H, W, device=x.device)
            torch.bernoulli(
                torch.full((B, 1, H, W), self.retention_rate, device=x.device),
                generator=self._rng if x.device.type == 'cpu' else None,
                out=mask,
            )
            return x * mask

        N_H = max(1, H // P)
        N_W = max(1, W // P)
        patch_mask = torch.zeros(B, 1, N_H, N_W, device=x.device)
        torch.bernoulli(
            torch.full((B, 1, N_H, N_W), self.retention_rate, device=x.device),
            generator=self._rng if x.device.type == 'cpu' else None,
            out=patch_mask,
        )
        mask = F.interpolate(patch_mask, size=(H, W), mode='nearest')
        return x * mask

    def __repr__(self) -> str:
        return f'UniformRandomReducer(target_r={self.retention_rate:.0%}, patch_size={self.patch_size})'


class UniformGridReducer:
    """
    Retain patches or pixels on a regular spatial grid using linspace-spaced indices.

    Args:
        retention_rate: Target fraction of units to keep (0 < r <= 1).
        patch_size:     Spatial patch dimension (default 16 for 16×16 patches).
                        Set to 1 for pixel-level grid.
    """

    def __init__(self, retention_rate: float, patch_size: int = 16) -> None:
        if not (0.0 < retention_rate <= 1.0):
            raise ValueError(f'retention_rate must be in (0, 1], got {retention_rate}')
        self.retention_rate = retention_rate
        self.patch_size = patch_size

    def _make_mask(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Return a (1, 1, H, W) binary mask with ones at evenly-spaced grid units."""
        P = self.patch_size
        if P <= 1:
            keep_h = max(1, round(H * math.sqrt(self.retention_rate)))
            keep_w = max(1, round(W * math.sqrt(self.retention_rate)))
            idx_h = torch.linspace(0, H - 1, keep_h, device=device).long()
            idx_w = torch.linspace(0, W - 1, keep_w, device=device).long()
            mask = torch.zeros(H, W, device=device)
            mask[idx_h.unsqueeze(1), idx_w.unsqueeze(0)] = 1.0
            return mask.unsqueeze(0).unsqueeze(0)

        N_H = max(1, H // P)
        N_W = max(1, W // P)
        keep_h = max(1, round(N_H * math.sqrt(self.retention_rate)))
        keep_w = max(1, round(N_W * math.sqrt(self.retention_rate)))
        idx_h = torch.linspace(0, N_H - 1, keep_h, device=device).long()
        idx_w = torch.linspace(0, N_W - 1, keep_w, device=device).long()
        patch_mask = torch.zeros(N_H, N_W, device=device)
        patch_mask[idx_h.unsqueeze(1), idx_w.unsqueeze(0)] = 1.0
        mask = patch_mask.unsqueeze(0).unsqueeze(0)
        return F.interpolate(mask, size=(H, W), mode='nearest')

    def actual_retention(self, H: int = 224, W: int = 224) -> float:
        """Fraction of units actually kept for an H x W image."""
        P = self.patch_size
        N_H = max(1, H // P) if P > 1 else H
        N_W = max(1, W // P) if P > 1 else W
        keep_h = max(1, round(N_H * math.sqrt(self.retention_rate)))
        keep_w = max(1, round(N_W * math.sqrt(self.retention_rate)))
        return (keep_h * keep_w) / (N_H * N_W)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply uniform grid mask to a batch of images.

        Args:
            x: Float tensor of shape (B, C, H, W).

        Returns:
            Masked tensor of the same shape; dropped units are set to 0.
        """
        _, _, H, W = x.shape
        mask = self._make_mask(H, W, x.device)
        return x * mask

    def __repr__(self) -> str:
        return f'UniformGridReducer(target_r={self.retention_rate:.0%}, patch_size={self.patch_size})'
