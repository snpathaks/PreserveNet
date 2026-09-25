"""
src/baselines/uniform.py
────────────────────────
Uniform spatial grid reduction.

Strategy
--------
Keep pixels at regular grid positions; zero out everything else.
Given a target retention rate ``r``, the grid stride is computed as:

    stride = max(1, round(1 / sqrt(r)))

so that approximately ``r`` of all pixels are retained.

Example — 32×32 image:
    stride=1  →  all 1024 pixels  (r ≈ 1.00)
    stride=2  →  16×16 = 256 px   (r ≈ 0.25)
    stride=3  →  11×11 = 121 px   (r ≈ 0.12)
    stride=4  →   8×8  =  64 px   (r ≈ 0.06)

The mask is identical for every image in the batch and every channel,
making this a fully deterministic, parameter-free baseline.

Usage
-----
    from src.baselines.uniform import UniformGridReducer

    reducer = UniformGridReducer(retention_rate=0.25)
    x_reduced = reducer(x)          # x: Tensor[B, C, H, W]
    print(reducer.actual_retention)  # real fraction of pixels kept
"""

from __future__ import annotations

import math
import torch


class UniformGridReducer:
    """
    Retain pixels on a regular spatial grid; zero out the rest.

    Args:
        retention_rate: Target fraction of pixels to keep (0 < r <= 1).
                        The actual rate may differ slightly due to integer
                        stride rounding — check ``actual_retention``.
    """

    def __init__(self, retention_rate: float) -> None:
        if not (0.0 < retention_rate <= 1.0):
            raise ValueError(f'retention_rate must be in (0, 1], got {retention_rate}')

        self.retention_rate = retention_rate
        # Compute grid stride so that kept pixels ≈ retention_rate * total pixels
        self.stride = max(1, round(1.0 / math.sqrt(retention_rate)))

    # ── Mask factory ──────────────────────────────────────────────────────────

    def _make_mask(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Return a (1, 1, H, W) binary mask with ones at grid positions."""
        mask = torch.zeros(H, W, device=device)
        mask[:: self.stride, :: self.stride] = 1.0
        return mask.unsqueeze(0).unsqueeze(0)   # → [1, 1, H, W]

    # ── Actual retention (after rounding) ─────────────────────────────────────

    def actual_retention(self, H: int = 32, W: int = 32) -> float:
        """Fraction of pixels actually kept for an H×W image."""
        kept = math.ceil(H / self.stride) * math.ceil(W / self.stride)
        return kept / (H * W)

    # ── Forward pass ──────────────────────────────────────────────────────────

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
        return (
            f'UniformGridReducer('
            f'target_r={self.retention_rate:.0%}, '
            f'stride={self.stride})'
        )
