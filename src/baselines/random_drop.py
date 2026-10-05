"""
src/baselines/random_drop.py
────────────────────────────
Random patch-drop reduction for PreserveNet.
Supports both patch-level (default 16×16) and pixel-level granularity.

Strategy
--------
For each image in the batch, independently retain each spatial patch (16×16)
with probability ``retention_rate``; dropped patches are set to zero.

The mask is shared across channels (the whole patch is either kept or
dropped) to mimic block/sensor-level missing data.

With ``seed`` set, evaluation is reproducible across runs.

Usage
-----
    from src.baselines.random_drop import RandomDropReducer

    # 16x16 patch drop on 224x224 image:
    reducer = RandomDropReducer(retention_rate=0.5, patch_size=16, seed=42)
    x_reduced = reducer(x)   # x: Tensor[B, C, H, W]
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class RandomDropReducer:
    """
    Randomly zero out (1 - retention_rate) fraction of patches or pixels.

    Args:
        retention_rate: Probability each patch/pixel is *kept* (0 < r <= 1).
        patch_size:     Spatial patch dimension (default 16 for 16×16 patch drop).
                        Set to 1 for pixel-level drop.
        seed:           Optional RNG seed for reproducible evaluation.
    """

    def __init__(
        self,
        retention_rate: float,
        patch_size: int = 16,
        seed: int | None = 42,
    ) -> None:
        if not (0.0 < retention_rate <= 1.0):
            raise ValueError(f'retention_rate must be in (0, 1], got {retention_rate}')

        self.retention_rate = retention_rate
        self.patch_size = patch_size
        self.seed = seed
        self._rng: torch.Generator | None = None

        if seed is not None:
            self._rng = torch.Generator()
            self._rng.manual_seed(seed)

    def actual_retention(self, H: int = 224, W: int = 224) -> float:
        """Expected retention rate."""
        return self.retention_rate

    # ── Mask factory ──────────────────────────────────────────────────────────

    def _make_mask(self, B: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        """
        Return a (B, 1, H, W) binary mask.
        Each spatial patch is independently 1 with probability ``retention_rate``.
        """
        P = self.patch_size

        if P <= 1:
            mask = torch.bernoulli(
                torch.full((B, 1, H, W), self.retention_rate),
                generator=self._rng,
            )
            return mask.to(device)

        N_H = max(1, H // P)
        N_W = max(1, W // P)
        patch_mask = torch.bernoulli(
            torch.full((B, 1, N_H, N_W), self.retention_rate),
            generator=self._rng,
        )
        mask = F.interpolate(patch_mask, size=(H, W), mode='nearest')
        return mask.to(device)

    # ── Forward pass ──────────────────────────────────────────────────────────

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply random patch-drop mask to a batch of images.

        Args:
            x: Float tensor of shape (B, C, H, W), already normalised.

        Returns:
            Masked tensor of the same shape; dropped patches are set to 0.
        """
        B, _, H, W = x.shape
        mask = self._make_mask(B, H, W, x.device)
        return x * mask   # broadcasts over C

    def __repr__(self) -> str:
        return (
            f'RandomDropReducer('
            f'retention_rate={self.retention_rate:.0%}, '
            f'patch_size={self.patch_size}, '
            f'seed={self.seed})'
        )
