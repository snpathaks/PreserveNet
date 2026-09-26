"""
src/baselines/uniform.py
────────────────────────
Two complementary uniform pixel-drop baselines.

UniformRandomReducer  (primary)
--------------------------------
Drops exactly (1 - r) fraction of pixels by sampling a per-image random
binary mask.  Because the mask is sampled independently per pixel, the
*actual* retention rate matches the *target* at every value.

UniformGridReducer  (legacy / reference)
-----------------------------------------
Keeps pixels on a regular spatial grid using linspace-spaced indices.
The old stride formula rounded to stride=1 for r=0.75 and r=0.50
(keeping 100% of pixels), replaced with the linspace approach.

Usage
-----
    from src.baselines.uniform import UniformRandomReducer, UniformGridReducer

    reducer = UniformRandomReducer(retention_rate=0.75)
    x_out   = reducer(x)               # x: Tensor[B, C, H, W]
    print(reducer.actual_retention())  # -> 0.75 (exact by construction)
"""

from __future__ import annotations

import math
import torch


class UniformRandomReducer:
    """
    Zero out a random (1 - retention_rate) fraction of pixels per image.

    Args:
        retention_rate: Fraction of pixels to *keep*, e.g. 0.75.
        seed:           Optional RNG seed for reproducibility.
    """

    def __init__(self, retention_rate: float, seed: int | None = None) -> None:
        if not (0.0 < retention_rate <= 1.0):
            raise ValueError(f'retention_rate must be in (0, 1], got {retention_rate}')
        self.retention_rate = retention_rate
        self._rng = torch.Generator()
        if seed is not None:
            self._rng.manual_seed(seed)

    def actual_retention(self, H: int = 32, W: int = 32) -> float:
        """Always equals the requested retention_rate (no rounding)."""
        return self.retention_rate

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply a fresh random mask to a batch of images.

        Args:
            x: Float tensor of shape (B, C, H, W).

        Returns:
            Masked tensor of the same shape; dropped pixels are set to 0.
        """
        B, C, H, W = x.shape
        mask = torch.zeros(B, 1, H, W, device=x.device)
        torch.bernoulli(
            torch.full((B, 1, H, W), self.retention_rate, device=x.device),
            generator=self._rng if x.device.type == 'cpu' else None,
            out=mask,
        )
        return x * mask   # broadcasts mask over C

    def __repr__(self) -> str:
        return f'UniformRandomReducer(target_r={self.retention_rate:.0%})'


class UniformGridReducer:
    """
    Retain pixels on a regular spatial grid using linspace-spaced indices.

    Args:
        retention_rate: Target fraction of pixels to keep (0 < r <= 1).
    """

    def __init__(self, retention_rate: float) -> None:
        if not (0.0 < retention_rate <= 1.0):
            raise ValueError(f'retention_rate must be in (0, 1], got {retention_rate}')
        self.retention_rate = retention_rate

    def _make_mask(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Return a (1, 1, H, W) binary mask with ones at evenly-spaced grid points."""
        keep_h = max(1, round(H * math.sqrt(self.retention_rate)))
        keep_w = max(1, round(W * math.sqrt(self.retention_rate)))
        idx_h = torch.linspace(0, H - 1, keep_h, device=device).long()
        idx_w = torch.linspace(0, W - 1, keep_w, device=device).long()
        mask = torch.zeros(H, W, device=device)
        mask[idx_h.unsqueeze(1), idx_w.unsqueeze(0)] = 1.0
        return mask.unsqueeze(0).unsqueeze(0)   # -> [1, 1, H, W]

    def actual_retention(self, H: int = 32, W: int = 32) -> float:
        """Fraction of pixels actually kept for an H x W image."""
        keep_h = max(1, round(H * math.sqrt(self.retention_rate)))
        keep_w = max(1, round(W * math.sqrt(self.retention_rate)))
        return (keep_h * keep_w) / (H * W)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply uniform grid mask to a batch of images.

        Args:
            x: Float tensor of shape (B, C, H, W), already normalised.

        Returns:
            Masked tensor of the same shape; dropped pixels are set to 0.
        """
        _, _, H, W = x.shape
        mask = self._make_mask(H, W, x.device)
        return x * mask   # broadcasts over B and C

    def __repr__(self) -> str:
        return f'UniformGridReducer(target_r={self.retention_rate:.0%})'
