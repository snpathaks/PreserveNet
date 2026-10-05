"""
src/baselines/saliency.py
─────────────────────────
Saliency-guided patch-retention reducers (default 16×16 patch granularity).

Two complementary strategies are provided:

GradientSaliencyReducer  (primary — vanilla input-gradient)
────────────────────────
Computes ∂(logit_pred) / ∂(input pixel) for each image in the batch.
The channel-wise norm gives spatial importance, which is pooled across
16×16 patches. The top-k patches are retained; the rest are zeroed.

GradCAMReducer  (secondary — class activation map)
───────────────
Hooks the gradient and activation of the target convolutional layer to
build a GradCAM heatmap. The heatmap is pooled across 16×16 patches,
and the top-k patches are retained.

Both reducers:
  * Accept a classifier (e.g. ImagenetteZeroShotClassifier or standard PyTorch model).
  * Support patch-level (default 16×16) and pixel-level granularity.
  * Produce an output tensor of the SAME shape as the input.
"""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn

# Re-export GradCAMReducer and _topk_patch_mask from gradcam.py
from src.baselines.gradcam import GradCAMReducer, _topk_patch_mask, _find_target_layer


class GradientSaliencyReducer:
    """
    Vanilla input-gradient saliency reducer with patch-level granularity.

    For each image the signed input gradient (d max-logit / d pixel) is
    computed in a single forward+backward pass. Channel L2-norm gives
    pixel importance, which is pooled into 16×16 spatial patches.
    Top-k patches are kept.

    Args:
        model:          A classifier (timm / torch.nn.Module).
        retention_rate: Fraction of patches to keep (0 < r <= 1).
        patch_size:     Spatial patch dimension (default 16 for 16×16 patches).
        device:         Torch device to run the saliency computation on.
        abs_grad:       If True, use |d/dx| (unsigned magnitude).
    """

    def __init__(
        self,
        model:          nn.Module,
        retention_rate: float,
        patch_size:     int = 16,
        device:         Optional[torch.device] = None,
        abs_grad:       bool = True,
    ) -> None:
        if not (0.0 < retention_rate <= 1.0):
            raise ValueError(f'retention_rate must be in (0, 1], got {retention_rate}')

        self.model          = model
        self.retention_rate = retention_rate
        self.patch_size     = patch_size
        self.device         = device or torch.device('cpu')
        self.abs_grad       = abs_grad

        self.model.eval().to(self.device)

    # ── Saliency computation ──────────────────────────────────────────────────

    def _compute_saliency(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute (B, H, W) saliency from input-gradient.

        Args:
            x: Normalised float tensor (B, C, H, W) on self.device.

        Returns:
            Saliency map tensor (B, H, W) on the same device as *x*.
        """
        x_in = x.detach().requires_grad_(True)

        logits     = self.model(x_in)               # (B, num_classes)
        pred_class = logits.argmax(dim=1)            # (B,)

        selected = logits[torch.arange(logits.size(0)), pred_class]
        self.model.zero_grad(set_to_none=True)
        selected.sum().backward()

        grad = x_in.grad                             # (B, C, H, W)
        if self.abs_grad:
            grad = grad.abs()

        saliency = grad.norm(dim=1)                  # (B, H, W)
        return saliency.detach()

    # ── Public interface ──────────────────────────────────────────────────────

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply patch-level gradient-saliency mask to a batch of images.

        Args:
            x: Float tensor (B, C, H, W), already normalised.

        Returns:
            Masked tensor of the same shape; dropped patches set to 0.
        """
        x_dev    = x.to(self.device)
        saliency = self._compute_saliency(x_dev)    # (B, H, W)
        mask     = _topk_patch_mask(saliency, self.retention_rate, patch_size=self.patch_size)
        return (x_dev * mask).detach()

    def __repr__(self) -> str:
        return (
            f'GradientSaliencyReducer('
            f'retention_rate={self.retention_rate:.0%}, '
            f'patch_size={self.patch_size}, '
            f'abs={self.abs_grad})'
        )


__all__ = [
    'GradientSaliencyReducer',
    'GradCAMReducer',
    '_topk_patch_mask',
    '_find_target_layer',
]
