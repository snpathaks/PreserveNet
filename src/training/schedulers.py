"""
src/training/schedulers.py
--------------------------
Annealing schedules for Gumbel-Softmax temperature and loss coefficients.

Why annealing?
--------------
Gumbel-Softmax temperature (tau)
    High tau (> 1) -> soft, nearly-uniform samples.  PatchScorer can
    explore freely; gradients are dense but biased.
    Low tau (-> 0) -> samples approach discrete {0,1} Bernoulli.
    Gradients vanish near convergence but the mask is crisp.

    Strategy: start warm (tau=5), anneal exponentially to tau=0.5 over
    training.  This gives exploration early and crisp decisions late.

Lambda warmup (lambda_agree, lambda_budget)
    Ramping loss weights up from 0 over the first N steps prevents the
    agreement loss from dominating before PatchScorer has any useful
    signal.  A linear warmup of 500-1000 steps works well in practice.

Usage
-----
    from src.training.schedulers import GumbelTauScheduler, LambdaWarmup

    tau_sched  = GumbelTauScheduler(tau_start=5.0, tau_end=0.5, total_steps=10_000)
    lam_sched  = LambdaWarmup(warmup_steps=500, target=1.0)

    # Inside training loop:
    tau = tau_sched.step()          # call once per batch
    lam = lam_sched.step()
"""

from __future__ import annotations
import math


class GumbelTauScheduler:
    """
    Exponential annealing of the Gumbel-Softmax temperature.

    tau(t) = tau_end * (tau_start / tau_end) ^ (1 - t / T)

    Args:
        tau_start:    Initial temperature (default 5.0  -- very soft).
        tau_end:      Final temperature   (default 0.5  -- near-discrete).
        total_steps:  Total number of training steps over the schedule.
        step_offset:  Starting step index (useful when resuming).
    """

    def __init__(
        self,
        tau_start:   float = 5.0,
        tau_end:     float = 0.5,
        total_steps: int   = 10_000,
        step_offset: int   = 0,
    ) -> None:
        if tau_start <= 0 or tau_end <= 0:
            raise ValueError("tau values must be positive")
        self.tau_start   = tau_start
        self.tau_end     = tau_end
        self.total_steps = total_steps
        self._step       = step_offset

    @property
    def current_tau(self) -> float:
        """Current temperature without advancing the counter."""
        t = min(self._step, self.total_steps)
        frac = t / self.total_steps
        return self.tau_end * (self.tau_start / self.tau_end) ** (1.0 - frac)

    def step(self) -> float:
        """Advance one step and return the new temperature."""
        tau = self.current_tau
        self._step += 1
        return tau

    def state_dict(self) -> dict:
        return {"_step": self._step}

    def load_state_dict(self, sd: dict) -> None:
        self._step = sd["_step"]

    def __repr__(self) -> str:
        return (
            f"GumbelTauScheduler(tau={self.current_tau:.3f}, "
            f"step={self._step}/{self.total_steps})"
        )


class LambdaWarmup:
    """
    Linear warmup for a loss coefficient from 0 to target.

    lambda(t) = min(t / warmup_steps, 1.0) * target

    Args:
        warmup_steps: Number of steps to ramp up over.
        target:       Final coefficient value after warmup (default 1.0).
        step_offset:  Starting step index (for resuming).
    """

    def __init__(
        self,
        warmup_steps: int   = 500,
        target:       float = 1.0,
        step_offset:  int   = 0,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.target       = target
        self._step        = step_offset

    @property
    def current_value(self) -> float:
        """Current coefficient without advancing."""
        return min(self._step / max(self.warmup_steps, 1), 1.0) * self.target

    def step(self) -> float:
        """Advance one step and return the new coefficient."""
        val = self.current_value
        self._step += 1
        return val

    def state_dict(self) -> dict:
        return {"_step": self._step}

    def load_state_dict(self, sd: dict) -> None:
        self._step = sd["_step"]

    def __repr__(self) -> str:
        return (
            f"LambdaWarmup(value={self.current_value:.4f}, "
            f"step={self._step}/{self.warmup_steps}, target={self.target})"
        )


class CosineDecay:
    """
    Cosine decay from start to end value over total_steps.

    value(t) = end + 0.5 * (start - end) * (1 + cos(pi * t / T))

    Args:
        start:       Initial value.
        end:         Final value.
        total_steps: Duration of decay.
    """

    def __init__(
        self,
        start:       float,
        end:         float,
        total_steps: int,
        step_offset: int = 0,
    ) -> None:
        self.start       = start
        self.end         = end
        self.total_steps = total_steps
        self._step       = step_offset

    @property
    def current_value(self) -> float:
        t = min(self._step, self.total_steps)
        return self.end + 0.5 * (self.start - self.end) * (
            1 + math.cos(math.pi * t / self.total_steps)
        )

    def step(self) -> float:
        val = self.current_value
        self._step += 1
        return val

    def state_dict(self) -> dict:
        return {"_step": self._step}

    def load_state_dict(self, sd: dict) -> None:
        self._step = sd["_step"]

    def __repr__(self) -> str:
        return (
            f"CosineDecay(value={self.current_value:.4f}, "
            f"step={self._step}/{self.total_steps})"
        )


__all__ = ["GumbelTauScheduler", "LambdaWarmup", "CosineDecay"]
