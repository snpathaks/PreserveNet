"""
src/losses/agreement.py
------------------------
L_agreement: KL-divergence between masked and full-image predictions.

This is the *real contribution* of PreserveNet.

Motivation
----------
The ceiling-effect problem:
  With a powerful zero-shot classifier (ResNet-18: 98.88%,
  ViT-S: 99.13%), cross-entropy on masked images is nearly saturated
  from epoch 0.  The classifier already gets almost everything right
  even with random masking, so L_task provides almost no gradient signal
  to PatchScorer.

The fix: agreement loss
  Instead of asking "did the masked image produce the correct label?",
  we ask "did the masked image produce the *same distribution* as the
  full image?"

  L_agreement = KL( p_full(x) || p_masked(x_m) )

  where p(x) = softmax(classifier(x)).

  This is strictly harder than L_task:
    - It must match every probability, not just argmax.
    - It penalises confident wrong predictions AND uncertain correct ones.
    - It provides a dense gradient at every token position, even when
      the top-1 label is already correct.
    - It is self-supervised: no ground-truth labels required during
      PatchScorer training.

Architecture note
-----------------
The classifier is always kept frozen (eval mode, no grad).
Only PatchScorer parameters receive gradient.

Both KL directions are available:
  * forward_kl  (default): KL(p_full || p_masked)
    -- punishes missing mass where full-image is confident
  * reverse_kl: KL(p_masked || p_full)
    -- mode-seeking; more aggressive

Usage
-----
    from src.losses.agreement import AgreementLoss

    criterion = AgreementLoss(temperature=1.0, direction='forward')
    loss = criterion(logits_full, logits_masked)
"""

from __future__ import annotations
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class AgreementLoss(nn.Module):
    """
    KL-divergence agreement loss between full-image and masked predictions.

    L_agreement = KL( softmax(logits_full / T) || softmax(logits_masked / T) )

    A temperature T > 1 softens the distributions, preventing the loss
    from vanishing when one distribution is very peaked (which happens
    frequently with a powerful pretrained classifier).

    Args:
        temperature: Softmax temperature (T >= 1; default 2.0 for soft targets).
        direction:   'forward'  -> KL(p_full || p_masked)  [default]
                     'reverse'  -> KL(p_masked || p_full)
                     'symmetric'-> 0.5 * (forward + reverse)
        reduction:   'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        temperature: float = 2.0,
        direction:   Literal['forward', 'reverse', 'symmetric'] = 'forward',
        reduction:   str   = 'mean',
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = temperature
        self.direction   = direction
        self.reduction   = reduction

    # ------------------------------------------------------------------

    def _kl(
        self,
        log_p: torch.Tensor,
        q:     torch.Tensor,
    ) -> torch.Tensor:
        """KL(p || q) = sum_i p_i * (log p_i - log q_i)."""
        return F.kl_div(log_p, q, reduction=self.reduction, log_target=False)

    def forward(
        self,
        logits_full:   torch.Tensor,
        logits_masked: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute agreement loss.

        Args:
            logits_full:   (B, C) classifier output on the original image.
                           Should be detached (frozen classifier, no grad).
            logits_masked: (B, C) classifier output on the masked image.
                           Gradients flow through this to PatchScorer.

        Returns:
            Scalar (or per-sample if reduction='none') loss tensor.
        """
        T = self.temperature

        # Soft probability targets from full image (stop gradient)
        p_full   = F.softmax(logits_full.detach()   / T, dim=-1)
        p_masked = F.softmax(logits_masked           / T, dim=-1)

        # log probabilities for KL computation
        log_p_full   = torch.log(p_full   + 1e-8)
        log_p_masked = torch.log(p_masked + 1e-8)

        if self.direction == 'forward':
            # KL(p_full || p_masked): log_target=full, target=masked... wait:
            # F.kl_div(input=log_q, target=p) = sum p * (log p - input)
            # We want KL(p_full || p_masked) = sum p_full * (log p_full - log p_masked)
            # -> F.kl_div(log_p_masked, p_full)
            return self._kl(log_p_masked, p_full)

        elif self.direction == 'reverse':
            # KL(p_masked || p_full) = sum p_masked * (log p_masked - log p_full)
            return self._kl(log_p_full, p_masked)

        else:  # symmetric
            fwd = self._kl(log_p_masked, p_full)
            rev = self._kl(log_p_full, p_masked)
            return 0.5 * (fwd + rev)

    def __repr__(self) -> str:
        return (
            f"AgreementLoss(T={self.temperature}, "
            f"direction='{self.direction}', reduction='{self.reduction}')"
        )


class TaskLoss(nn.Module):
    """
    Standard cross-entropy on masked image predictions.

    Kept as a thin wrapper so the training loop can always combine:
        L_total = lambda_task * L_task
                + lambda_budget * L_budget
                + lambda_agree * L_agreement

    Args:
        label_smoothing: Label smoothing factor (default 0.0).
    """

    def __init__(self, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        logits_masked: torch.Tensor,
        labels:        torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits_masked: (B, C) logits from classifier(x_masked).
            labels:        (B,) ground-truth integer class indices.

        Returns:
            Scalar cross-entropy loss.
        """
        return self.ce(logits_masked, labels)

    def __repr__(self) -> str:
        return f"TaskLoss(label_smoothing={self.ce.label_smoothing})"


__all__ = ["AgreementLoss", "TaskLoss"]
