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
# TokenDropOperator (Step 8: Physical ViT Token Pruning)
# =============================================================================

class TokenDropOperator(nn.Module):
    """
    Physically removes ViT tokens based on patch scores, yielding true FLOP
    reduction and acceleration in Vision Transformers.

    Unlike pixel-level masking (which sets uninformative patches to zero but
    still processes the full 196+1 tokens through all 12 transformer layers),
    TokenDropOperator physically slices the sequence length from (1 + N) down to
    (1 + K) tokens. Because self-attention complexity scales quadratically as
    O(L^2), dropping 90% of tokens (r=0.10) reduces attention FLOPs by ~98.8%.

    Args:
        retention_rate: Fraction of spatial patch tokens to keep (0 < r <= 1.0).
        patch_size:     Pixel dimension of each spatial patch (default 16).
        keep_cls_token: If True, always retains the prepended CLS token (index 0).

    Shape conventions:
        - Image input x      : (B, 3, H, W)
        - Scores             : (B, 1, N_H, N_W) or (B, N_patches) from PatchScorer
        - Pruned ViT tokens  : (B, 1 + K, D) where K = round(N * retention_rate)
    """

    def __init__(
        self,
        retention_rate: float = 0.5,
        patch_size:     int   = 16,
        keep_cls_token: bool  = True,
    ) -> None:
        super().__init__()
        if not (0.0 < retention_rate <= 1.0):
            raise ValueError(
                f"retention_rate must be in (0, 1], got {retention_rate}"
            )
        self.retention_rate = retention_rate
        self.patch_size     = patch_size
        self.keep_cls_token = keep_cls_token

    def select_indices(
        self,
        scores: torch.Tensor,
        k:      Optional[int] = None,
    ) -> torch.Tensor:
        """
        Extract the top-k spatial patch indices from raw score maps.

        Args:
            scores: (B, 1, N_H, N_W) or (B, N) raw logits from PatchScorer.
            k:      Exact number of tokens to keep. If None, derived from
                    retention_rate * total_patches.

        Returns:
            indices: (B, K) long tensor containing selected patch indices in [0, N-1].
        """
        if scores.dim() == 4:
            B, _, N_H, N_W = scores.shape
            scores_flat = scores.view(B, N_H * N_W)
        elif scores.dim() == 3:
            B = scores.shape[0]
            scores_flat = scores.view(B, -1)
        else:
            B = scores.shape[0]
            scores_flat = scores

        n_total = scores_flat.shape[1]
        if k is None:
            k = max(1, round(n_total * self.retention_rate))
        k = min(k, n_total)

        # Sort indices to preserve spatial ordering for positional stability
        _, topk_idx = torch.topk(scores_flat, k, dim=1, largest=True, sorted=False)
        topk_idx, _ = torch.sort(topk_idx, dim=1)
        return topk_idx

    def prune_tokens(
        self,
        tokens:         torch.Tensor,
        scores:         Optional[torch.Tensor] = None,
        indices:        Optional[torch.Tensor] = None,
        has_cls_token:  bool = True,
    ) -> torch.Tensor:
        """
        Physically prune patch tokens from a ViT sequence embedding.

        Args:
            tokens:        (B, N+1, D) if has_cls_token else (B, N, D).
            scores:        (B, 1, N_H, N_W) or (B, N) logits to compute top-k indices.
            indices:       (B, K) precomputed indices (optional if scores provided).
            has_cls_token: Whether index 0 is the CLS token.

        Returns:
            pruned_tokens: (B, K+1, D) if has_cls_token else (B, K, D).
        """
        B, seq_len, D = tokens.shape

        if indices is None:
            if scores is None:
                raise ValueError("Either scores or indices must be provided.")
            indices = self.select_indices(scores)

        if has_cls_token:
            cls_token    = tokens[:, :1, :]          # (B, 1, D)
            patch_tokens = tokens[:, 1:, :]          # (B, N, D)
            idx_expanded = indices.unsqueeze(-1).expand(-1, -1, D)  # (B, K, D)
            kept_patches = torch.gather(patch_tokens, dim=1, index=idx_expanded)
            return torch.cat([cls_token, kept_patches], dim=1)      # (B, K+1, D)
        else:
            idx_expanded = indices.unsqueeze(-1).expand(-1, -1, D)  # (B, K, D)
            return torch.gather(tokens, dim=1, index=idx_expanded)  # (B, K, D)

    def forward_vit(
        self,
        vit_model: nn.Module,
        x:         torch.Tensor,
        scores:    Optional[torch.Tensor] = None,
        indices:   Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Execute an end-to-end token-pruned forward pass through a ViT model.

        Performs:
        1. Patch embedding: x -> tokens (B, N, D)
        2. Positional embedding addition: tokens -> (B, N+1, D)
        3. Physical token dropping: (B, N+1, D) -> (B, K+1, D)
        4. Transformer blocks execution on reduced sequence length
        5. LayerNorm + Head projection -> (B, num_classes)

        Args:
            vit_model: ImagenetteZeroShotClassifier wrapping a ViT or a raw timm ViT.
            x:         (B, 3, H, W) input image batch.
            scores:    (B, 1, N_H, N_W) logits from PatchScorer.
            indices:   (B, K) precomputed top-k patch indices.

        Returns:
            logits:    (B, num_classes) output predictions.
        """
        # Unwrap ImagenetteZeroShotClassifier if needed
        is_zero_shot_wrapper = hasattr(vit_model, 'backbone') and hasattr(vit_model, '_class_idx')
        backbone = vit_model.backbone if is_zero_shot_wrapper else vit_model

        # 1. Patch embedding
        tokens = backbone.patch_embed(x)                  # (B, N, D)

        # 2. Add positional embeddings (CLS token prepended inside _pos_embed)
        tokens_with_pos = backbone._pos_embed(tokens)      # (B, N+1, D)

        # 3. Physically drop patch tokens
        pruned_tokens = self.prune_tokens(
            tokens_with_pos, scores=scores, indices=indices, has_cls_token=True
        )                                                 # (B, K+1, D)

        # 4. Optional pre-norm & drop
        if hasattr(backbone, 'patch_drop') and backbone.patch_drop is not None:
            pruned_tokens = backbone.patch_drop(pruned_tokens)
        if hasattr(backbone, 'norm_pre') and backbone.norm_pre is not None:
            pruned_tokens = backbone.norm_pre(pruned_tokens)

        # 5. Transformer blocks on pruned token sequence
        out_tokens = backbone.blocks(pruned_tokens)       # (B, K+1, D)
        out_tokens = backbone.norm(out_tokens)            # (B, K+1, D)

        # 6. Classifier head projection
        if hasattr(backbone, 'forward_head'):
            logits_1000 = backbone.forward_head(out_tokens)
        else:
            # Fallback head projection on CLS token
            cls_out = out_tokens[:, 0]
            if hasattr(backbone, 'fc_norm') and backbone.fc_norm is not None:
                cls_out = backbone.fc_norm(cls_out)
            if hasattr(backbone, 'head_drop') and backbone.head_drop is not None:
                cls_out = backbone.head_drop(cls_out)
            logits_1000 = backbone.head(cls_out)

        # 7. Slicing for Imagenette zero-shot head if applicable
        if is_zero_shot_wrapper:
            return logits_1000[:, vit_model._class_idx]
        return logits_1000

    def compute_theoretical_savings(
        self,
        H:          int = 224,
        W:          int = 224,
        embed_dim:  int = 384,
        depth:      int = 12,
    ) -> dict[str, float]:
        """
        Compute exact theoretical sequence length reduction and FLOP savings.

        Args:
            H, W:      Input image dimensions.
            embed_dim: Transformer hidden dimension (384 for ViT-Small).
            depth:     Number of transformer layers (12 for ViT-Small).

        Returns:
            dict containing token counts, reduction ratios, and FLOP speedup estimates.
        """
        n_patches = (H // self.patch_size) * (W // self.patch_size)  # e.g. 196
        n_kept = max(1, round(n_patches * self.retention_rate))      # e.g. 20 for r=0.10

        orig_seq_len = n_patches + 1  # 197
        pruned_seq_len = n_kept + 1   # 21

        # Attention FLOPs per layer: 4 * L * D^2 (QKV + Proj) + 2 * L^2 * D (Scores + Values)
        orig_attn_flops_per_layer = (4 * orig_seq_len * (embed_dim ** 2)) + (2 * (orig_seq_len ** 2) * embed_dim)
        pruned_attn_flops_per_layer = (4 * pruned_seq_len * (embed_dim ** 2)) + (2 * (pruned_seq_len ** 2) * embed_dim)

        # MLP FLOPs per layer: 8 * L * D^2 (fc1: D->4D, fc2: 4D->D)
        orig_mlp_flops_per_layer = 8 * orig_seq_len * (embed_dim ** 2)
        pruned_mlp_flops_per_layer = 8 * pruned_seq_len * (embed_dim ** 2)

        orig_total_layer_flops = depth * (orig_attn_flops_per_layer + orig_mlp_flops_per_layer)
        pruned_total_layer_flops = depth * (pruned_attn_flops_per_layer + pruned_mlp_flops_per_layer)

        token_savings_pct = (1.0 - (pruned_seq_len / orig_seq_len)) * 100.0
        attn_quadratic_ratio = (pruned_seq_len / orig_seq_len) ** 2
        attn_flops_savings_pct = (1.0 - (pruned_attn_flops_per_layer / orig_attn_flops_per_layer)) * 100.0
        total_flops_savings_pct = (1.0 - (pruned_total_layer_flops / orig_total_layer_flops)) * 100.0

        return {
            'n_patches_total': n_patches,
            'n_patches_kept': n_kept,
            'orig_seq_len': orig_seq_len,
            'pruned_seq_len': pruned_seq_len,
            'token_savings_pct': token_savings_pct,
            'attn_quadratic_ratio': attn_quadratic_ratio,
            'attn_flops_savings_pct': attn_flops_savings_pct,
            'total_transformer_flops_savings_pct': total_flops_savings_pct,
            'theoretical_speedup': orig_total_layer_flops / pruned_total_layer_flops,
        }

    def __repr__(self) -> str:
        return (
            f"TokenDropOperator("
            f"retention_rate={self.retention_rate:.0%}, "
            f"patch_size={self.patch_size}, "
            f"keep_cls_token={self.keep_cls_token})"
        )


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


def build_token_drop_operator(
    retention_rate: float,
    patch_size:     int  = 16,
    keep_cls_token: bool = True,
) -> TokenDropOperator:
    """
    Convenience factory for TokenDropOperator (Step 8).

    Args:
        retention_rate: Fraction of spatial patch tokens to keep (0 < r <= 1.0).
        patch_size:     Pixel size of each patch block (default 16).
        keep_cls_token: If True, preserves the ViT CLS token.

    Returns:
        Configured TokenDropOperator.
    """
    return TokenDropOperator(
        retention_rate=retention_rate,
        patch_size=patch_size,
        keep_cls_token=keep_cls_token,
    )

