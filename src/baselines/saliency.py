"""
src/baselines/saliency.py
─────────────────────────
Step 5 — Saliency-guided pixel-retention reducer.

Two complementary strategies are provided:

GradientSaliencyReducer  (primary — vanilla input-gradient)
────────────────────────
Computes ∂(logit_pred) / ∂(input pixel) for each image in the batch.
The L2-norm over the channel dimension gives a per-pixel importance
score.  The top-k pixels (by score) are retained; the rest are zeroed.

GradCAMReducer  (secondary — class activation map)
───────────────
Hooks the gradient and activation of the last convolutional layer to
build a GradCAM heatmap.  The heatmap is bilinearly upsampled to the
input resolution and used as the saliency signal.

Both reducers:
  * Accept a *frozen* classifier and call it internally.
  * Produce an output tensor of the SAME shape as the input.
  * Can be used as drop-in replacements for UniformGridReducer /
    RandomDropReducer in eval_dumb_baselines.py.

Usage
-----
    from src.baselines.saliency import GradientSaliencyReducer, GradCAMReducer

    reducer = GradientSaliencyReducer(model, retention_rate=0.5, device=device)
    x_out   = reducer(x)   # x: Tensor[B, C, H, W]  (already normalised)

Notes
-----
* These reducers are *not* trained — they are wrappers around an
  existing classifier.  The model's weights are never updated here.
* Because a forward+backward pass is required per batch, throughput is
  ~2-3x lower than the dumb baselines.  This is expected.
* GradCAM hooks the LAST convolutional layer (``model.layer4`` for
  timm ResNet-18).  Other architectures may need a different layer name.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Helper: top-k binary mask
# ─────────────────────────────────────────────────────────────────────────────

def _topk_mask(saliency: torch.Tensor, retention_rate: float) -> torch.Tensor:
    """
    Given a (B, H, W) saliency map, return a (B, 1, H, W) binary mask
    that is 1 for the top ``retention_rate`` fraction of pixels per image.

    Args:
        saliency:       Float tensor (B, H, W) — higher = more important.
        retention_rate: Fraction of pixels to *keep* (0 < r <= 1).

    Returns:
        Binary float tensor of shape (B, 1, H, W).
    """
    B, H, W = saliency.shape
    n_keep  = max(1, int(H * W * retention_rate))

    # Flatten spatial dims -> (B, H*W), find top-k indices
    flat       = saliency.view(B, -1)                           # (B, H*W)
    _, indices = torch.topk(flat, n_keep, dim=1, largest=True, sorted=False)

    # Scatter 1s into mask
    mask_flat = torch.zeros_like(flat)
    mask_flat.scatter_(1, indices, 1.0)

    return mask_flat.view(B, 1, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# Gradient Saliency Reducer
# ─────────────────────────────────────────────────────────────────────────────

class GradientSaliencyReducer:
    """
    Vanilla input-gradient saliency reducer.

    For each image the signed input gradient (d max-logit / d pixel) is
    computed in a single forward+backward pass.  The channel-wise L2 norm
    gives a (B, H, W) importance map; the top-k pixels are kept.

    Args:
        model:          A trained classifier (timm / torch.nn.Module).
                        Its weights are *not* modified.
        retention_rate: Fraction of pixels to keep (0 < r <= 1).
        device:         Torch device to run the saliency computation on.
        abs_grad:       If True, use |d/dx| (unsigned magnitude).
                        If False, use the raw signed gradient.
                        Unsigned is almost always better for masking.
    """

    def __init__(
        self,
        model:          nn.Module,
        retention_rate: float,
        device:         Optional[torch.device] = None,
        abs_grad:       bool = True,
    ) -> None:
        if not (0.0 < retention_rate <= 1.0):
            raise ValueError(f'retention_rate must be in (0, 1], got {retention_rate}')

        self.model          = model
        self.retention_rate = retention_rate
        self.device         = device or torch.device('cpu')
        self.abs_grad       = abs_grad

        # Ensure model is in eval mode and on the right device
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

        # Forward pass — take score of predicted class
        logits     = self.model(x_in)               # (B, num_classes)
        pred_class = logits.argmax(dim=1)            # (B,)

        # Aggregate selected logits -> scalar, then backprop
        selected = logits[torch.arange(logits.size(0)), pred_class]
        selected.sum().backward()

        grad = x_in.grad                             # (B, C, H, W)
        if self.abs_grad:
            grad = grad.abs()

        # Reduce channel dim -> (B, H, W) via L2 norm
        saliency = grad.norm(dim=1)                  # (B, H, W)
        return saliency.detach()

    # ── Public interface ──────────────────────────────────────────────────────

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply gradient-saliency mask to a batch of images.

        Args:
            x: Float tensor (B, C, H, W), already normalised.

        Returns:
            Masked tensor of the same shape; dropped pixels set to 0.
        """
        x_dev    = x.to(self.device)
        saliency = self._compute_saliency(x_dev)    # (B, H, W)
        mask     = _topk_mask(saliency, self.retention_rate)  # (B, 1, H, W)
        return (x_dev * mask).detach()

    def __repr__(self) -> str:
        return (
            f'GradientSaliencyReducer('
            f'retention_rate={self.retention_rate:.0%}, '
            f'abs={self.abs_grad})'
        )


# ─────────────────────────────────────────────────────────────────────────────
# GradCAM Reducer
# ─────────────────────────────────────────────────────────────────────────────

class GradCAMReducer:
    """
    GradCAM-based saliency reducer.

    Hooks the gradient and activation of a target convolutional layer
    (default: ``layer4`` of timm ResNet-18) to compute a class activation
    map.  The CAM is upsampled to input resolution and used as the
    importance signal for top-k pixel selection.

    Args:
        model:          A trained timm / torch.nn.Module classifier.
        retention_rate: Fraction of pixels to keep (0 < r <= 1).
        device:         Torch device to run on.
        target_layer:   Name of the timm attribute to hook.  For ResNet-18
                        this is ``'layer4'``; for ViT it might be the last
                        attention block.
    """

    def __init__(
        self,
        model:          nn.Module,
        retention_rate: float,
        device:         Optional[torch.device] = None,
        target_layer:   str = 'layer4',
    ) -> None:
        if not (0.0 < retention_rate <= 1.0):
            raise ValueError(f'retention_rate must be in (0, 1], got {retention_rate}')

        self.model          = model
        self.retention_rate = retention_rate
        self.device         = device or torch.device('cpu')
        self.target_layer   = target_layer

        self.model.eval().to(self.device)

        # Internal state updated by hooks
        self._activations: torch.Tensor | None = None
        self._gradients:   torch.Tensor | None = None

        # Register forward + backward hooks
        self._register_hooks()

    # ── Hook registration ─────────────────────────────────────────────────────

    def _register_hooks(self) -> None:
        """Attach forward and backward hooks to the target layer."""
        layer = dict(self.model.named_modules()).get(self.target_layer)
        if layer is None:
            raise AttributeError(
                f"Layer '{self.target_layer}' not found in model.  "
                f"Available top-level modules: "
                f"{list(dict(self.model.named_children()).keys())}"
            )

        def _save_activation(module, input, output) -> None:  # noqa: ARG001
            self._activations = output

        def _save_gradient(module, grad_input, grad_output) -> None:  # noqa: ARG001
            # grad_output[0]: gradient w.r.t. the layer's output feature map
            self._gradients = grad_output[0]

        layer.register_forward_hook(_save_activation)
        layer.register_full_backward_hook(_save_gradient)

    # ── CAM computation ───────────────────────────────────────────────────────

    def _compute_cam(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute (B, H_in, W_in) GradCAM for a batch.

        Args:
            x: Float tensor (B, C, H_in, W_in) on self.device.

        Returns:
            Upsampled, ReLU'd saliency map (B, H_in, W_in).
        """
        _, _, H_in, W_in = x.shape

        x_in = x.detach().requires_grad_(True)

        # Forward
        logits     = self.model(x_in)
        pred_class = logits.argmax(dim=1)
        selected   = logits[torch.arange(logits.size(0)), pred_class]
        selected.sum().backward()

        # Both hooks should have fired by now
        acts  = self._activations                   # (B, C_feat, H_f, W_f)
        grads = self._gradients                     # (B, C_feat, H_f, W_f)

        # Global-average-pool gradients -> channel weights  (B, C_feat)
        weights = grads.mean(dim=(2, 3))

        # Weighted sum over channels  (B, H_f, W_f)
        cam = torch.einsum('bc,bchw->bhw', weights, acts)

        # ReLU: only features that *support* the prediction
        cam = F.relu(cam)

        # Upsample to input resolution  (B, 1, H_in, W_in) -> (B, H_in, W_in)
        cam_up = F.interpolate(
            cam.unsqueeze(1).float(),
            size=(H_in, W_in),
            mode='bilinear',
            align_corners=False,
        ).squeeze(1)

        return cam_up.detach()

    # ── Public interface ──────────────────────────────────────────────────────

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply GradCAM mask to a batch of images.

        Args:
            x: Float tensor (B, C, H, W), already normalised.

        Returns:
            Masked tensor of the same shape; dropped pixels set to 0.
        """
        x_dev = x.to(self.device)
        cam   = self._compute_cam(x_dev)                       # (B, H, W)
        mask  = _topk_mask(cam, self.retention_rate)           # (B, 1, H, W)
        return (x_dev * mask).detach()

    def __repr__(self) -> str:
        return (
            f'GradCAMReducer('
            f'retention_rate={self.retention_rate:.0%}, '
            f'layer={self.target_layer!r})'
        )
