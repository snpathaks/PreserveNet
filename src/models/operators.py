"""
src/models/operators.py
-----------------------
Differentiable and hard masking operators for PreserveNet -- Step 6.

MaskOperator  (primary export)
-------------------------------
Takes a (B, 1, N_H, N_W) score map from PatchScorer and converts it
into a binary (B, 1, H, W) pixel mask at 16x16 patch granularity,
matching the exact format used by every baseline reducer.

Two modes
---------
hard (default, inference)
    Top-k thresholding: retain the top `retention_rate` fraction of
    patches by raw logit score.  The result is a {0, 1} tensor -- the
    same binary mask format produced by GradCAMReducer, UniformRandom-
    Reducer, etc.

soft (training)
    Differentiable relaxation via Gumbel-Softmax (or plain Sigmoid).
    Enables gradient flow from a classification loss back through the
    mask into PatchScorer weights.

Format compatibility
---------------------
Output of MaskOperator.hard_mask() is pixel-level (B, 1, H, W) float
tensor with values in {0.0, 1.0} and nearest-neighbour upsampling from
(N_H, N_W) to (H, W).  This is bit-for-bit identical to what

    F.interpolate(patch_mask_BN_H_N_W, size=(H, W), mode='nearest')

produces in gradcam.py and uniform.py, so all downstream comparison
code reuses cleanly.

Usage
-----
    from src.models.operators import MaskOperator

    op = MaskOperator(retention_rate=0.5, patch_size=16)

    scores = scorer(x)                   # (B, 1, 14, 14) from PatchScorer
    mask   = op.hard_mask(scores, (224, 224))  # (B, 1, 224, 224)
    x_out  = x * mask                    # zero out low-score patches

    # Differentiable path (training):
    soft = op.soft_mask(scores)          # (B, 1, 14, 14) in [0, 1]
    x_soft = F.interpolate(soft, (224,224), mode='nearest') * x
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# MaskOperator
# =============================================================================

class MaskOperator(nn.Module):
    """
    Converts PatchScorer logits into binary or soft patch masks.

    The two modes are intentionally separated so the training loop can
    call soft_mask() with gradients enabled, while inference always
    calls hard_mask() for a crisp {0, 1} result compatible with every
    baseline.

    Args:
        retention_rate: Fraction of patches to keep (0 < r <= 1).
        patch_size:     Pixel dimension of each spatial patch (default 16).
        gumbel_tau:     Temperature for Gumbel-Softmax relaxation (soft mode).
        gumbel_hard:    If True, straight-through Gumbel gives a hard sample
                        during training while remaining differentiable.

    Shape conventions
    -----------------
    Input  scores : (B, 1, N_H, N_W)  -- raw logits from PatchScorer
    Output mask   : (B, 1, H,   W  )  -- binary float at pixel resolution
    """

    def __init__(
        self,
        retention_rate: float,
        patch_size:     int   = 16,
        gumbel_tau:     float = 1.0,
        gumbel_hard:    bool  = True,
    ) -> None:
        super().__init__()

        if not (0.0 < retention_rate <= 1.0):
            raise ValueError(
                f'retention_rate must be in (0, 1], got {retention_rate}'
            )

        self.retention_rate = retention_rate
        self.patch_size     = patch_size
        self.gumbel_tau     = gumbel_tau
        self.gumbel_hard    = gumbel_hard

    # -- Hard mask (inference) ------------------------------------------------

    def hard_mask(
        self,
        scores:   torch.Tensor,
        out_size: tuple,
    ) -> torch.Tensor:
        """
        Top-k binary mask.  No gradients.

        Selects the top ``retention_rate`` fraction of patches by raw logit
        and upsample to pixel resolution with nearest-neighbour, exactly
        matching the format of GradCAMReducer, UniformRandomReducer, etc.

        Args:
            scores:   (B, 1, N_H, N_W) raw logits from PatchScorer.
            out_size: (H, W) target pixel resolution.

        Returns:
            mask: (B, 1, H, W) binary float tensor in {0.0, 1.0}.
        """
        with torch.no_grad():
            return self._topk_mask(scores, out_size)

    def _topk_mask(
        self,
        scores:   torch.Tensor,
        out_size: tuple,
    ) -> torch.Tensor:
        """Core top-k logic (reused by hard_mask and apply())."""
        B, _, N_H, N_W = scores.shape
        n_keep = max(1, round(N_H * N_W * self.retention_rate))

        flat   = scores.view(B, -1)
        _, idx = torch.topk(flat, n_keep, dim=1, largest=True, sorted=False)

        mask_flat = torch.zeros_like(flat)
        mask_flat.scatter_(1, idx, 1.0)

        patch_mask = mask_flat.view(B, 1, N_H, N_W)
        return F.interpolate(patch_mask, size=out_size, mode='nearest')

    # -- Soft mask (training) -------------------------------------------------

    def soft_mask(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Differentiable relaxation of the binary mask.

        Uses sigmoid activation so each patch gets a continuous weight in
        [0, 1].  Gradient flows back into PatchScorer during training.

        For Gumbel-Softmax training (pair-wise binary), see gumbel_mask().

        Args:
            scores: (B, 1, N_H, N_W) raw logits.

        Returns:
            soft:   (B, 1, N_H, N_W) weights in [0, 1].
        """
        return torch.sigmoid(scores)

    def gumbel_mask(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Straight-through Gumbel-Softmax binary mask for training.

        Treats each patch as an independent Bernoulli variable.  The
        Gumbel noise provides stochastic exploration; the hard=True flag
        gives a {0, 1} forward pass with an approximate gradient.

        Args:
            scores: (B, 1, N_H, N_W) raw logits.

        Returns:
            mask:   (B, 1, N_H, N_W) in {0, 1} (hard) or [0, 1] (soft).
        """
        # Stack (keep_logit, drop_logit=0) so gumbel_softmax picks one
        drop_logit = torch.zeros_like(scores)
        logits_2   = torch.cat([scores, drop_logit], dim=1)   # (B, 2, N_H, N_W)
        probs_2    = F.gumbel_softmax(
            logits_2,
            tau=self.gumbel_tau,
            hard=self.gumbel_hard,
            dim=1,
        )
        return probs_2[:, :1, :, :]  # "keep" channel

    # -- Combined apply() convenience -----------------------------------------

    def apply(
        self,
        x:      torch.Tensor,
        scores: torch.Tensor,
        *,
        soft:   bool = False,
    ) -> torch.Tensor:
        """
        Apply patch mask to a batch of images.

        This is the single entry-point that covers both training and
        inference, keeping the signature consistent with baseline reducers.

        Args:
            x:      (B, C, H, W) input image tensor (normalised).
            scores: (B, 1, N_H, N_W) raw logits from PatchScorer.
            soft:   If True, apply sigmoid soft mask (training path).
                    If False, apply top-k hard mask (inference path).

        Returns:
            x_masked: (B, C, H, W) -- same shape, dropped patches are 0.
        """
        _, _, H, W = x.shape

        if soft:
            patch_weights = self.soft_mask(scores)             # (B, 1, N_H, N_W)
            pixel_weights = F.interpolate(
                patch_weights, size=(H, W), mode='nearest'
            )                                                  # (B, 1, H, W)
            return x * pixel_weights
        else:
            mask = self._topk_mask(scores, (H, W))             # (B, 1, H, W) binary
            return (x * mask).detach()

    # -- Metadata helpers -----------------------------------------------------

    def actual_retention(self, H: int = 224, W: int = 224) -> float:
        """
        Return the exact fraction of patches kept for an H x W image.
        Equals retention_rate except for very small grids where rounding
        causes a slight deviation.
        """
        P    = self.patch_size
        N_H  = max(1, H // P)
        N_W  = max(1, W // P)
        kept = max(1, round(N_H * N_W * self.retention_rate))
        return kept / (N_H * N_W)

    def __repr__(self) -> str:
        return (
            f'MaskOperator('
            f'retention_rate={self.retention_rate:.0%}, '
            f'patch_size={self.patch_size}, '
            f'gumbel_tau={self.gumbel_tau})'
        )

    def forward(self, x: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """nn.Module forward -- delegates to apply() in hard mode."""
        return self.apply(x, scores, soft=False)


# =============================================================================
# Convenience factory
# =============================================================================

def build_mask_operator(
    retention_rate: float,
    patch_size:     int   = 16,
    gumbel_tau:     float = 1.0,
) -> MaskOperator:
    """
    Convenience factory for MaskOperator.

    Args:
        retention_rate: Fraction of patches to keep (0 < r <= 1).
        patch_size:     Pixel size of each patch block (default 16).
        gumbel_tau:     Gumbel-Softmax temperature (default 1.0).

    Returns:
        Configured MaskOperator.
    """
    return MaskOperator(
        retention_rate=retention_rate,
        patch_size=patch_size,
        gumbel_tau=gumbel_tau,
    )
