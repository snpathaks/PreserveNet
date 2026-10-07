"""
src/losses/budget.py
--------------------
L_budget: sparsity / retention-rate penalty.

Forces PatchScorer to keep exactly the target fraction of patches
instead of lazily keeping all of them (which would minimise L_task
trivially at the cost of zero compression).

Two flavours are provided:

L1BudgetLoss  (default)
    Penalises mean sigmoid score above the target:
        L_budget = max(0,  mean(sigmoid(scores)) - target_rate )
    Smooth, zero when under-budget, linear when over.

L2BudgetLoss
    Squared deviation from target fraction:
        L_budget = ( mean(sigmoid(scores)) - target_rate )^2
    Differentiable everywhere; smoother gradient near target.

Usage
-----
    from src.losses.budget import L1BudgetLoss, L2BudgetLoss

    criterion = L1BudgetLoss(target_rate=0.5)
    loss = criterion(scores)   # scores: (B, 1, N_H, N_W) raw logits
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class L1BudgetLoss(nn.Module):
    """
    One-sided L1 penalty: penalises over-retention only.

    L_budget = ReLU( mean(sigmoid(s)) - target_rate )

    This lets the scorer freely go *below* the budget (sparser is fine),
    but costs it linearly for going over.

    Args:
        target_rate: Desired mean retention fraction (0 < r <= 1).
    """

    def __init__(self, target_rate: float) -> None:
        super().__init__()
        if not (0.0 < target_rate <= 1.0):
            raise ValueError(f"target_rate must be in (0,1], got {target_rate}")
        self.target_rate = target_rate

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scores: (B, 1, N_H, N_W) raw logits from PatchScorer.

        Returns:
            Scalar loss tensor.
        """
        mean_retention = torch.sigmoid(scores).mean()
        return F.relu(mean_retention - self.target_rate)

    def __repr__(self) -> str:
        return f"L1BudgetLoss(target_rate={self.target_rate:.2f})"


class L2BudgetLoss(nn.Module):
    """
    Two-sided L2 penalty: penalises both over- and under-retention.

    L_budget = ( mean(sigmoid(s)) - target_rate )^2

    Use this when you want to hit the budget precisely rather than just
    being under it.

    Args:
        target_rate: Desired mean retention fraction (0 < r <= 1).
    """

    def __init__(self, target_rate: float) -> None:
        super().__init__()
        if not (0.0 < target_rate <= 1.0):
            raise ValueError(f"target_rate must be in (0,1], got {target_rate}")
        self.target_rate = target_rate

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scores: (B, 1, N_H, N_W) raw logits from PatchScorer.

        Returns:
            Scalar loss tensor.
        """
        mean_retention = torch.sigmoid(scores).mean()
        return (mean_retention - self.target_rate) ** 2

    def __repr__(self) -> str:
        return f"L2BudgetLoss(target_rate={self.target_rate:.2f})"


__all__ = ["L1BudgetLoss", "L2BudgetLoss"]
