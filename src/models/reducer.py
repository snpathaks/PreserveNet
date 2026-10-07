"""
src/models/reducer.py
---------------------
Dynamic patch-selection module for PreserveNet — Step 6.

Architecture overview
─────────────────────
PatchScorer
    A lightweight CNN that maps a (B, 3, 224, 224) image to a
    (B, 1, 14, 14) score map.  Each of the 196 spatial positions
    gets a scalar importance score.  The network is intentionally
    shallow so it can be trained end-to-end alongside a frozen
    classifier without blowing up memory/compute.

    Layers
    ------
    stem  : Conv2d(3 → 32, 3×3, stride 2) + BN + ReLU  → (B, 32, 112, 112)
    block1: Conv2d(32 → 64, 3×3, stride 2) + BN + ReLU  → (B, 64,  56,  56)
    block2: Conv2d(64 → 64, 3×3, stride 2) + BN + ReLU  → (B, 64,  28,  28)
    block3: Conv2d(64 → 32, 3×3, stride 2) + BN + ReLU  → (B, 32,  14,  14)
    head  : Conv2d(32 → 1, 1×1)                          → (B,  1,  14,  14)
    (no activation on head — raw logits for Gumbel-Softmax / sigmoid)

    Output contract
    ---------------
    * Shape  : (B, 1, 14, 14)  — one score per 16×16 spatial patch
    * Range  : R (raw logits); pass through sigmoid for values in [0, 1]
    * The caller (MaskOperator / training loop) decides the threshold.

PatchScoreReducer (public high-level class)
    Wraps PatchScorer and applies a top-k binary mask to produce a
    reduced image with the *same shape* as the input.  Matches the
    reducer(x) -> x_masked callable contract used by every baseline,
    so the existing evaluation code can swap it in with zero changes.

Usage
-----
    from src.models.reducer import PatchScorer, PatchScoreReducer

    scorer  = PatchScorer()
    reducer = PatchScoreReducer(scorer, retention_rate=0.5)

    x_in  = torch.randn(4, 3, 224, 224)
    x_out = reducer(x_in)                          # (4, 3, 224, 224)

    # Raw scores only:
    scores = scorer(x_in)                          # (4, 1, 14, 14)
    probs  = torch.sigmoid(scores)                 # in [0, 1]
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Internal building blocks
# =============================================================================

def _conv_bn_relu(
    in_ch:   int,
    out_ch:  int,
    kernel:  int = 3,
    stride:  int = 1,
    padding: int = 1,
) -> nn.Sequential:
    """Conv -> BN -> ReLU block (spatial dims halved when stride=2)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


# =============================================================================
# PatchScorer -- the core CNN
# =============================================================================

class PatchScorer(nn.Module):
    """
    Lightweight CNN that produces a per-patch importance score map.

    For a 224x224 input the network applies four stride-2 convolutions,
    reducing the spatial dimensions:
        224 -> 112 -> 56 -> 28 -> 14
    matching the 14x14 = 196 patch grid that corresponds to 16x16 blocks.

    Args:
        in_channels:   Number of input channels (default: 3 for RGB).
        base_channels: Width multiplier for hidden channels (default: 32).

    Input:
        x : (B, in_channels, H, W)  -- recommended H = W = 224

    Output:
        scores : (B, 1, H//16, W//16)  raw logits per patch
                 Apply sigmoid() to get values in [0, 1].
    """

    def __init__(
        self,
        in_channels:   int = 3,
        base_channels: int = 32,
    ) -> None:
        super().__init__()

        C = base_channels

        # Four stride-2 blocks: 224->112->56->28->14
        self.stem   = _conv_bn_relu(in_channels, C,     stride=2)  # /2
        self.block1 = _conv_bn_relu(C,           C * 2, stride=2)  # /4
        self.block2 = _conv_bn_relu(C * 2,       C * 2, stride=2)  # /8
        self.block3 = _conv_bn_relu(C * 2,       C,     stride=2)  # /16

        # 1x1 conv -- projects to single score channel (raw logit)
        self.head = nn.Conv2d(C, 1, kernel_size=1, bias=True)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) -- expected H = W = 224.

        Returns:
            scores: (B, 1, H//16, W//16) raw importance logits.
                    Shape is (B, 1, 14, 14) for 224x224 inputs.
        """
        z = self.stem(x)     # (B,  C, H/2,  W/2)
        z = self.block1(z)   # (B, 2C, H/4,  W/4)
        z = self.block2(z)   # (B, 2C, H/8,  W/8)
        z = self.block3(z)   # (B,  C, H/16, W/16)
        return self.head(z)  # (B,  1, H/16, W/16)

    def extra_repr(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        return f'params={total:,}'


# =============================================================================
# PatchScoreReducer -- drop-in replacement for baseline reducers
# =============================================================================

class PatchScoreReducer:
    """
    Wraps PatchScorer and applies a top-k binary patch mask, producing a
    reduced image with *exactly* the same shape as the input.

    Callable contract (matches all existing baselines):
        reducer(x: Tensor[B, C, H, W]) -> Tensor[B, C, H, W]

    Masking strategy
    ----------------
    1. Run PatchScorer(x) -> (B, 1, N_H, N_W) raw logits.
    2. Select top ceil(N_H * N_W * retention_rate) patches per image.
    3. Upsample binary patch mask to (B, 1, H, W) via nearest-neighbour.
    4. Return x * mask  (dropped patches -> 0).

    Args:
        scorer:         A PatchScorer instance (trained or untrained).
        retention_rate: Fraction of 16x16 patches to keep (0 < r <= 1).
        patch_size:     Spatial patch dimension in pixels (default 16).
        device:         Torch device for inference (defaults to CPU).
        eval_mode:      If True, call scorer.eval() on construction.
    """

    def __init__(
        self,
        scorer:         PatchScorer,
        retention_rate: float,
        patch_size:     int = 16,
        device:         Optional[torch.device] = None,
        eval_mode:      bool = True,
    ) -> None:
        if not (0.0 < retention_rate <= 1.0):
            raise ValueError(
                f'retention_rate must be in (0, 1], got {retention_rate}'
            )

        self.scorer         = scorer
        self.retention_rate = retention_rate
        self.patch_size     = patch_size
        self.device         = device or torch.device('cpu')

        self.scorer.to(self.device)
        if eval_mode:
            self.scorer.eval()

    # -- Public interface -----------------------------------------------------

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply learned patch-importance mask to a batch of images.

        Args:
            x: Float tensor (B, C, H, W), already normalised.

        Returns:
            Masked tensor of the same shape; dropped 16x16 patches are 0.
        """
        x_dev = x.to(self.device)
        with torch.no_grad():
            scores = self.scorer(x_dev)                              # (B, 1, N_H, N_W)
        mask = self._topk_patch_mask(scores, x_dev.shape[-2:])      # (B, 1, H, W)
        return (x_dev * mask).detach()

    def score_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return sigmoid-normalised scores without applying the mask.

        Args:
            x: Float tensor (B, C, H, W).

        Returns:
            probs: Float tensor (B, 1, N_H, N_W) in [0, 1].
        """
        x_dev = x.to(self.device)
        with torch.no_grad():
            return torch.sigmoid(self.scorer(x_dev))

    # -- Internal helpers -----------------------------------------------------

    def _topk_patch_mask(
        self,
        scores:   torch.Tensor,
        out_size: tuple,
    ) -> torch.Tensor:
        """
        Convert raw logit score map to a binary (B, 1, H, W) pixel mask.

        Selects top retention_rate fraction of patches per image.

        Args:
            scores:   (B, 1, N_H, N_W) raw logits from PatchScorer.
            out_size: (H, W) target spatial resolution for the mask.

        Returns:
            Binary float mask (B, 1, H, W).
        """
        B, _, N_H, N_W = scores.shape
        n_keep = max(1, round(N_H * N_W * self.retention_rate))

        flat   = scores.view(B, -1)                               # (B, N_H*N_W)
        _, idx = torch.topk(flat, n_keep, dim=1, largest=True, sorted=False)

        mask_flat = torch.zeros_like(flat)
        mask_flat.scatter_(1, idx, 1.0)

        patch_mask = mask_flat.view(B, 1, N_H, N_W)              # (B, 1, 14, 14)
        return F.interpolate(patch_mask, size=out_size, mode='nearest')

    def actual_retention(self, H: int = 224, W: int = 224) -> float:
        """Exact fraction kept for an H x W image."""
        P    = self.patch_size
        N_H  = max(1, H // P)
        N_W  = max(1, W // P)
        kept = max(1, round(N_H * N_W * self.retention_rate))
        return kept / (N_H * N_W)

    def __repr__(self) -> str:
        return (
            f'PatchScoreReducer('
            f'retention_rate={self.retention_rate:.0%}, '
            f'patch_size={self.patch_size}, '
            f'scorer={self.scorer.__class__.__name__})'
        )


# =============================================================================
# Factory helper
# =============================================================================

def build_patch_score_reducer(
    retention_rate: float,
    base_channels:  int = 32,
    patch_size:     int = 16,
    device:         Optional[torch.device] = None,
    checkpoint:     Optional[str] = None,
) -> PatchScoreReducer:
    """
    Convenience factory: build a PatchScorer + PatchScoreReducer pair.

    Args:
        retention_rate: Fraction of patches to keep (0 < r <= 1).
        base_channels:  PatchScorer hidden width (default 32).
        patch_size:     Patch pixel size (default 16).
        device:         Torch device (defaults to CUDA if available, else CPU).
        checkpoint:     Optional path to a saved PatchScorer state_dict.

    Returns:
        A ready-to-use PatchScoreReducer.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    scorer = PatchScorer(base_channels=base_channels)

    if checkpoint is not None:
        state = torch.load(checkpoint, map_location=device)
        scorer.load_state_dict(state)

    return PatchScoreReducer(
        scorer=scorer,
        retention_rate=retention_rate,
        patch_size=patch_size,
        device=device,
    )
