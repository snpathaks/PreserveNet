"""
src/baselines/random_drop.py
────────────────────────────
Random pixel drop reduction.

Strategy
--------
For each image in the batch, independently retain each spatial pixel
with probability ``retention_rate``; the dropped pixels are set to zero.

The mask is shared across channels (either the whole pixel is kept or
dropped) to mimic realistic sensor-level missing data.

With ``seed`` set, evaluation is reproducible across runs.

Usage
-----
    from src.baselines.random_drop import RandomDropReducer

    reducer = RandomDropReducer(retention_rate=0.5, seed=42)
    x_reduced = reducer(x)   # x: Tensor[B, C, H, W]

Notes
-----
* Unlike UniformGridReducer the actual retention is stochastic —
  it is a Binomial(H*W, r) random variable and converges to ``r``
  as image resolution grows.
* Pass ``seed`` for deterministic evaluation; omit for stochastic
  augmentation-style usage during training experiments.
"""

from __future__ import annotations

import torch


class RandomDropReducer:
    """
    Randomly zero out (1 - retention_rate) fraction of pixels.

    Args:
        retention_rate: Probability each pixel is *kept* (0 < r <= 1).
        seed:           Optional RNG seed for reproducible evaluation.
    """

    def __init__(
        self,
        retention_rate: float,
        seed: int | None = 42,
    ) -> None:
        if not (0.0 < retention_rate <= 1.0):
            raise ValueError(f'retention_rate must be in (0, 1], got {retention_rate}')

        self.retention_rate = retention_rate
        self.seed = seed
        self._rng: torch.Generator | None = None

        if seed is not None:
            self._rng = torch.Generator()
            self._rng.manual_seed(seed)

    # ── Mask factory ──────────────────────────────────────────────────────────

    def _make_mask(self, B: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        """
        Return a (B, 1, H, W) binary mask.

        Each spatial location is independently 1 with prob ``retention_rate``.
        """
        # Generate on CPU then move — Generator doesn't support non-CPU devices
        mask = torch.bernoulli(
            torch.full((B, 1, H, W), self.retention_rate),
            generator=self._rng,
        )
        return mask.to(device)

    # ── Forward pass ──────────────────────────────────────────────────────────

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply random pixel-drop mask to a batch of images.

        Args:
            x: Float tensor of shape (B, C, H, W), already normalised.

        Returns:
            Masked tensor of the same shape; dropped pixels are set to 0.
        """
        B, _, H, W = x.shape
        mask = self._make_mask(B, H, W, x.device)
        return x * mask   # broadcasts over C

    def __repr__(self) -> str:
        return (
            f'RandomDropReducer('
            f'retention_rate={self.retention_rate:.0%}, '
            f'seed={self.seed})'
        )
