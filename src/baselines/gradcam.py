"""
src/baselines/gradcam.py
────────────────────────
GradCAM-based saliency reducer with patch-level granularity (default: 16×16).

Strategy
--------
1. Hooks the gradient and activation of a target layer (e.g. 'layer4' for ResNet-18)
   to compute a Class Activation Map (CAM).
2. Pools the CAM over 16×16 spatial patches (using adaptive average pooling).
3. Selects the top-k most important patches according to ``retention_rate``.
4. Produces a binary mask where the top-k 16×16 patches are preserved (1)
   and all other patches are zeroed out (0).

Usage
-----
    from src.baselines.gradcam import GradCAMReducer

    reducer = GradCAMReducer(model, retention_rate=0.5, patch_size=16, device=device)
    x_out   = reducer(x)  # x: Tensor[B, 3, 224, 224]
"""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def _topk_patch_mask(
    saliency: torch.Tensor,
    retention_rate: float,
    patch_size: int = 16,
) -> torch.Tensor:
    """
    Given a (B, H, W) saliency map, return a (B, 1, H, W) binary mask
    that retains the top ``retention_rate`` fraction of spatial patches
    (or pixels if patch_size <= 1).

    Args:
        saliency:       Float tensor (B, H, W) -- higher = more important.
        retention_rate: Fraction of spatial units to *keep* (0 < r <= 1).
        patch_size:     Spatial patch dimension (default: 16 for 16×16 patches).

    Returns:
        Binary float tensor of shape (B, 1, H, W).
    """
    B, H, W = saliency.shape
    P = patch_size

    if P <= 1:
        n_keep = max(1, int(H * W * retention_rate))
        flat = saliency.view(B, -1)
        _, indices = torch.topk(flat, n_keep, dim=1, largest=True, sorted=False)
        mask_flat = torch.zeros_like(flat)
        mask_flat.scatter_(1, indices, 1.0)
        return mask_flat.view(B, 1, H, W)

    # Patch-level granularity: pool saliency per patch
    N_H = max(1, H // P)
    N_W = max(1, W // P)
    n_keep = max(1, round(N_H * N_W * retention_rate))

    # Adaptive avg pool over patches: (B, 1, H, W) -> (B, N_H, N_W)
    sal_4d = saliency.unsqueeze(1)
    patch_sal = F.adaptive_avg_pool2d(sal_4d, (N_H, N_W)).squeeze(1)

    flat = patch_sal.view(B, -1)
    _, indices = torch.topk(flat, n_keep, dim=1, largest=True, sorted=False)
    mask_flat = torch.zeros_like(flat)
    mask_flat.scatter_(1, indices, 1.0)
    patch_mask = mask_flat.view(B, 1, N_H, N_W)

    return F.interpolate(patch_mask, size=(H, W), mode='nearest')


def _find_target_layer(model: nn.Module, target_layer: str) -> nn.Module:
    """
    Find module by name, supporting both direct module names,
    'backbone.' prefixes (e.g. for wrapped zero-shot classifiers),
    and suffix matching.
    """
    modules = dict(model.named_modules())
    if target_layer in modules:
        return modules[target_layer]

    # Check model.backbone if present
    if hasattr(model, 'backbone'):
        backbone_modules = dict(model.backbone.named_modules())
        if target_layer in backbone_modules:
            return backbone_modules[target_layer]
        if f'backbone.{target_layer}' in modules:
            return modules[f'backbone.{target_layer}']

    # Suffix match (e.g. 'layer4' matching 'backbone.layer4')
    for name, mod in modules.items():
        if name == target_layer or name.endswith('.' + target_layer):
            return mod

    raise AttributeError(
        f"Layer '{target_layer}' not found in model. Available modules: {list(modules.keys())[:15]}"
    )


class GradCAMReducer:
    """
    GradCAM-based saliency reducer with patch-level (default 16×16) granularity.

    Hooks the gradient and activation of a target layer (default: 'layer4' for ResNet-18)
    to compute a class activation map. Then ranks spatial patches (e.g. 16×16 blocks)
    and retains only the top ``retention_rate`` fraction.

    Args:
        model:          A trained or pretrained classifier (e.g. ImagenetteZeroShotClassifier).
        retention_rate: Fraction of patches to keep (0 < r <= 1).
        patch_size:     Spatial patch dimension (default 16 for 16×16 patch granularity).
        device:         Torch device to run on.
        target_layer:   Name of layer to hook (default: 'layer4').
    """

    def __init__(
        self,
        model: nn.Module,
        retention_rate: float,
        patch_size: int = 16,
        device: Optional[torch.device] = None,
        target_layer: str = 'layer4',
    ) -> None:
        if not (0.0 < retention_rate <= 1.0):
            raise ValueError(f'retention_rate must be in (0, 1], got {retention_rate}')

        self.model = model
        self.retention_rate = retention_rate
        self.patch_size = patch_size
        self.device = device or torch.device('cpu')
        self.target_layer = target_layer

        self.model.eval().to(self.device)

        # Internal state updated by hooks
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        self._register_hooks()

    def _register_hooks(self) -> None:
        """Attach forward and backward hooks to the target layer."""
        layer = _find_target_layer(self.model, self.target_layer)

        def _save_activation(module, input, output) -> None:
            self._activations = output

        def _save_gradient(module, grad_input, grad_output) -> None:
            self._gradients = grad_output[0]

        layer.register_forward_hook(_save_activation)
        layer.register_full_backward_hook(_save_gradient)

    def _compute_cam(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute (B, H_in, W_in) GradCAM heatmap for a batch.

        Args:
            x: Float tensor (B, C, H_in, W_in) on self.device.

        Returns:
            Upsampled, ReLU'd saliency map (B, H_in, W_in).
        """
        _, _, H_in, W_in = x.shape
        x_in = x.detach().requires_grad_(True)

        logits = self.model(x_in)
        pred_class = logits.argmax(dim=1)
        selected = logits[torch.arange(logits.size(0)), pred_class]

        self.model.zero_grad(set_to_none=True)
        selected.sum().backward()

        acts = self._activations
        grads = self._gradients

        # Global average pool gradients -> channel weights (B, C_feat)
        weights = grads.mean(dim=(2, 3))

        # Weighted sum over channels (B, H_f, W_f)
        cam = torch.einsum('bc,bchw->bhw', weights, acts)
        cam = F.relu(cam)

        # Upsample to input resolution
        cam_up = F.interpolate(
            cam.unsqueeze(1).float(),
            size=(H_in, W_in),
            mode='bilinear',
            align_corners=False,
        ).squeeze(1)

        return cam_up.detach()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply patch-level GradCAM mask to a batch of images.

        Args:
            x: Float tensor (B, C, H, W), already normalised.

        Returns:
            Masked tensor of the same shape; dropped 16×16 patches are set to 0.
        """
        x_dev = x.to(self.device)
        cam = self._compute_cam(x_dev)
        mask = _topk_patch_mask(cam, self.retention_rate, patch_size=self.patch_size)
        return (x_dev * mask).detach()

    def __repr__(self) -> str:
        return (
            f'GradCAMReducer('
            f'retention_rate={self.retention_rate:.0%}, '
            f'patch_size={self.patch_size}, '
            f'layer={self.target_layer!r})'
        )
