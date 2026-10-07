"""
src/models/pipeline.py
-----------------------
PreserveNet end-to-end forward pipeline -- Step 7.

Wires PatchScorer -> MaskOperator -> frozen Classifier into a single
differentiable forward pass for training.

Pipeline (one forward call)
---------------------------
1.  full_logits  = classifier(x)               -- no grad, frozen
2.  scores       = scorer(x)                   -- grad flows here
3.  soft_mask    = gumbel_softmax(scores, tau)  -- differentiable
    OR
    pixel_mask   = topk(scores)                -- hard, inference only
4.  x_masked     = x * upsampled_mask
5.  masked_logits = classifier(x_masked)       -- no grad through clf

Losses (computed externally by trainer):
    L_task      = CrossEntropy(masked_logits, labels)
    L_budget    = L1Budget(scores, target_rate)
    L_agree     = KL(softmax(full_logits), softmax(masked_logits))
    L_total     = lam_task * L_task + lam_budget * L_budget + lam_agree * L_agree

Only PatchScorer parameters receive gradient.

Usage
-----
    from src.models.pipeline import PreserveNetPipeline

    pipeline = PreserveNetPipeline(
        scorer=PatchScorer(),
        classifier=build_imagenette_classifier(),
        operator=MaskOperator(retention_rate=0.5),
    )
    out = pipeline(x, tau=2.0, hard=False)
    # out.full_logits   -- (B, C)
    # out.masked_logits -- (B, C)
    # out.scores        -- (B, 1, 14, 14)
    # out.soft_mask     -- (B, 1, 14, 14) in [0,1]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.reducer   import PatchScorer
from src.models.operators import MaskOperator


# ---------------------------------------------------------------------------
# Output bundle
# ---------------------------------------------------------------------------

@dataclass
class PipelineOutput:
    """
    All tensors produced by one forward pass of PreserveNetPipeline.

    Attributes
    ----------
    full_logits:    (B, C) -- classifier on original (full) image.
    masked_logits:  (B, C) -- classifier on masked image.
    scores:         (B, 1, N_H, N_W) -- raw patch logits from PatchScorer.
    soft_mask:      (B, 1, N_H, N_W) -- continuous mask (sigmoid or Gumbel).
    pixel_mask:     (B, 1, H, W)     -- upsampled pixel-level mask.
    x_masked:       (B, C, H, W)     -- masked input image.
    """
    full_logits:   torch.Tensor
    masked_logits: torch.Tensor
    scores:        torch.Tensor
    soft_mask:     torch.Tensor
    pixel_mask:    torch.Tensor
    x_masked:      torch.Tensor


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class PreserveNetPipeline(nn.Module):
    """
    End-to-end PreserveNet pipeline.

    Args
    ----
    scorer:     PatchScorer (trainable).
    classifier: Any nn.Module with a (B,C,H,W) -> (B,K) forward;
                it is immediately frozen on construction.
    operator:   MaskOperator (no parameters; handles masking logic).
    """

    def __init__(
        self,
        scorer:     PatchScorer,
        classifier: nn.Module,
        operator:   MaskOperator,
    ) -> None:
        super().__init__()
        self.scorer     = scorer
        self.classifier = classifier
        self.operator   = operator

        # Freeze classifier -- only scorer trains
        for p in self.classifier.parameters():
            p.requires_grad_(False)
        self.classifier.eval()

    # ------------------------------------------------------------------

    @torch.no_grad()
    def _full_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Run frozen classifier on original image. No gradient."""
        was_training = self.classifier.training
        self.classifier.eval()
        out = self.classifier(x)
        if was_training:
            self.classifier.train()
        return out.detach()

    # ------------------------------------------------------------------

    def forward(
        self,
        x:    torch.Tensor,
        tau:  float = 1.0,
        hard: bool  = False,
    ) -> PipelineOutput:
        """
        Full pipeline forward pass.

        Args
        ----
        x:    (B, 3, H, W) normalised input batch.
        tau:  Gumbel-Softmax temperature (annealed during training).
        hard: If True, straight-through hard Gumbel sample
              (better for training; use False for pure soft).

        Returns
        -------
        PipelineOutput dataclass with all intermediate tensors.
        """
        B, C, H, W = x.shape

        # 1. Full-image reference (frozen, no grad)
        full_logits = self._full_logits(x)                        # (B, K)

        # 2. PatchScorer: produce patch-level logits (grad flows here)
        scores = self.scorer(x)                                   # (B, 1, N_H, N_W)

        # 3. Differentiable mask
        soft_mask = self.operator.gumbel_mask(scores) if tau > 0  \
                    else torch.sigmoid(scores)
        # Override tau by re-running gumbel with correct temp
        if tau > 0:
            drop_logit = torch.zeros_like(scores)
            logits_2   = torch.cat([scores, drop_logit], dim=1)   # (B, 2, N_H, N_W)
            probs_2    = F.gumbel_softmax(logits_2, tau=tau, hard=hard, dim=1)
            soft_mask  = probs_2[:, :1, :, :]                     # (B, 1, N_H, N_W)

        # 4. Upsample to pixel resolution
        pixel_mask = F.interpolate(soft_mask, size=(H, W), mode='nearest')  # (B,1,H,W)
        x_masked   = x * pixel_mask                               # (B, C, H, W)

        # 5. Masked logits (frozen classifier, grad stops at classifier boundary)
        with torch.no_grad():
            self.classifier.eval()
        # We need grad to flow through x_masked INTO the classifier input
        # (so it arrives at scores via the mask).  But we must NOT let
        # classifier weights accumulate grad.  torch.no_grad on the params
        # is guaranteed by requires_grad_(False) set in __init__.
        masked_logits = self.classifier(x_masked)                 # (B, K)

        return PipelineOutput(
            full_logits   = full_logits,
            masked_logits = masked_logits,
            scores        = scores,
            soft_mask     = soft_mask,
            pixel_mask    = pixel_mask,
            x_masked      = x_masked,
        )

    # ------------------------------------------------------------------

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Hard-mask inference (no gradient, no Gumbel noise).

        Args:
            x: (B, 3, H, W) normalised batch.

        Returns:
            (B, K) logits from classifier on hard-masked image.
        """
        self.eval()
        with torch.no_grad():
            scores     = self.scorer(x)
            mask       = self.operator.hard_mask(scores, x.shape[-2:])
            x_masked   = x * mask
            return self.classifier(x_masked)

    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        scorer_params = sum(p.numel() for p in self.scorer.parameters())
        clf_params    = sum(p.numel() for p in self.classifier.parameters())
        return (
            f"scorer_params={scorer_params:,} (trainable), "
            f"classifier_params={clf_params:,} (frozen)"
        )


__all__ = ["PreserveNetPipeline", "PipelineOutput"]
