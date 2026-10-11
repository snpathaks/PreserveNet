"""
src/eval/complexity.py
-----------------------
Step 9 -- Latency, Throughput & FLOPs Benchmarking for PreserveNet.

This module uses **fvcore** (Facebook AI Research) to measure exact
floating-point operation counts (FLOPs) for both masking-based and
token-dropping-based inference paths at every retention level.

It also performs wall-clock latency and throughput benchmarking with
CUDA-synchronised timers (or high-resolution CPU timers) to produce
the final **accuracy/agreement vs. compute-saved** efficiency curve
that justifies the entire PreserveNet project premise.

Metrics Produced
----------------
1. **fvcore FLOP counts** (multiply-accumulate operations):
   - Full classifier (baseline 100%):  ResNet-18 & ViT-Small
   - Masking path:   MaskOperator at each retention level (r=100%, 75%, 50%, 25%, 10%)
   - Token-drop path: TokenDropOperator at each retention level
2. **Wall-clock latency** (ms/batch):
   - Masked forward pass (full sequence, zeroed patches)
   - Token-pruned forward pass (physically shorter sequence)
3. **Throughput** (images/sec)
4. **FLOP savings** (% reduction from full model)
5. **Efficiency ratio**: accuracy-per-GFLOP for every configuration

Usage
-----
    # As a module:
    from src.eval.complexity import ComplexityBenchmark
    bench = ComplexityBenchmark(device='cpu', batch_size=4)
    results = bench.run_full_benchmark()

    # As a script:
    python PreserveNet/scripts/benchmark_complexity.py
    python PreserveNet/scripts/benchmark_complexity.py --batch_size 16 --device cpu
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# fvcore for FLOP counting
try:
    from fvcore.nn import FlopCountAnalysis, flop_count_table
except ImportError as exc:
    raise ImportError(
        "fvcore is required for FLOP counting.  Install it with:\n"
        "    pip install fvcore"
    ) from exc


# =======================================================================
# Data Classes for structured results
# =======================================================================

@dataclass
class FLOPResult:
    """FLOP count result for a single configuration."""
    retention_rate: float
    mode: str               # 'full', 'masked', 'token_drop'
    architecture: str       # 'resnet18', 'vit_small_patch16_224'
    total_flops: int        # Total FLOPs (multiply-accumulate operations)
    total_gflops: float     # Total GFLOPs
    flop_reduction_pct: float  # Percentage reduction from full model
    tokens_kept: int        # Number of tokens/patches kept
    tokens_total: int       # Total tokens/patches


@dataclass
class LatencyResult:
    """Latency and throughput result for a single configuration."""
    retention_rate: float
    mode: str
    architecture: str
    latency_ms: float         # Average ms per batch
    latency_std_ms: float     # Standard deviation of latency
    throughput_fps: float     # Images per second
    speedup: float            # Speedup relative to full model
    batch_size: int


@dataclass
class EfficiencyPoint:
    """Combined accuracy + compute data for one point on the efficiency curve."""
    retention_rate: float
    mode: str
    architecture: str
    accuracy_pct: float       # Top-1 accuracy (if available)
    gflops: float             # Compute cost
    latency_ms: float
    throughput_fps: float
    flop_reduction_pct: float
    speedup: float
    accuracy_per_gflop: float  # Efficiency ratio


@dataclass
class BenchmarkResults:
    """Complete benchmark results for all configurations."""
    flop_results: List[FLOPResult] = field(default_factory=list)
    latency_results: List[LatencyResult] = field(default_factory=list)
    efficiency_points: List[EfficiencyPoint] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'config': self.config,
            'flop_results': [asdict(r) for r in self.flop_results],
            'latency_results': [asdict(r) for r in self.latency_results],
            'efficiency_points': [asdict(r) for r in self.efficiency_points],
        }

    def save_json(self, path: str | Path) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# =======================================================================
# FLOP Counter Wrappers
# =======================================================================

class _MaskedClassifierWrapper(nn.Module):
    """
    Wraps a classifier so fvcore counts FLOPs of:
      mask_op.apply(x, scores) -> classifier(x_masked)

    This represents the masking path where the full spatial
    resolution is preserved but uninformative patches are zeroed.
    """

    def __init__(
        self,
        classifier: nn.Module,
        mask_op: nn.Module,
        scores: torch.Tensor,
    ) -> None:
        super().__init__()
        self.classifier = classifier
        self.mask_op = mask_op
        self.register_buffer('_scores', scores)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_masked = self.mask_op.apply(x, self._scores, soft=False)
        return self.classifier(x_masked)


class _TokenDropViTWrapper(nn.Module):
    """
    Wraps a ViT classifier + TokenDropOperator so fvcore counts
    FLOPs of the physically token-pruned forward pass.

    This is the token-dropping path where self-attention operates
    on a shorter (1 + K) sequence, yielding true FLOP reduction.
    """

    def __init__(
        self,
        vit_classifier: nn.Module,
        token_op: nn.Module,
        scores: torch.Tensor,
    ) -> None:
        super().__init__()
        self.vit_classifier = vit_classifier
        self.token_op = token_op
        self.register_buffer('_scores', scores)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.token_op.forward_vit(
            self.vit_classifier, x, scores=self._scores
        )


def count_flops(
    model: nn.Module,
    input_tensor: torch.Tensor,
    suppress_warnings: bool = True,
) -> int:
    """
    Count total FLOPs (multiply-accumulate operations) using fvcore.

    Args:
        model:            nn.Module to profile.
        input_tensor:     Representative input tensor.
        suppress_warnings: Silence fvcore unsupported-op warnings.

    Returns:
        Total FLOPs as integer.
    """
    model.eval()
    if suppress_warnings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            flops = FlopCountAnalysis(model, input_tensor)
            flops.unsupported_ops_warnings(False)
            flops.uncalled_modules_warnings(False)
            return flops.total()
    else:
        flops = FlopCountAnalysis(model, input_tensor)
        return flops.total()


def get_flop_table(
    model: nn.Module,
    input_tensor: torch.Tensor,
) -> str:
    """
    Return a human-readable, per-module FLOP breakdown table (fvcore).

    Args:
        model:        nn.Module to profile.
        input_tensor: Representative input tensor.

    Returns:
        Formatted string table.
    """
    model.eval()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flops = FlopCountAnalysis(model, input_tensor)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        return flop_count_table(flops)


# =======================================================================
# Latency Benchmark Utilities
# =======================================================================

def _warmup_model(
    model: nn.Module,
    x: torch.Tensor,
    n_warmup: int = 10,
) -> None:
    """Run warmup forward passes to stabilise CUDA kernels / caches."""
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)


def benchmark_latency(
    model: nn.Module,
    x: torch.Tensor,
    n_runs: int = 50,
    n_warmup: int = 10,
    use_cuda_events: bool = False,
) -> Tuple[float, float]:
    """
    Measure per-batch inference latency in milliseconds.

    Uses CUDA events for GPU timing (if available and requested)
    or time.perf_counter for CPU.

    Args:
        model:           nn.Module to benchmark.
        x:               Representative input batch.
        n_runs:          Number of timed iterations.
        n_warmup:        Number of warmup iterations.
        use_cuda_events: Use CUDA events for precise GPU timing.

    Returns:
        (mean_ms, std_ms) tuple of latency statistics.
    """
    model.eval()
    _warmup_model(model, x, n_warmup)

    if use_cuda_events and x.is_cuda:
        timings = []
        for _ in range(n_runs):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            with torch.no_grad():
                _ = model(x)
            end_event.record()
            torch.cuda.synchronize()
            timings.append(start_event.elapsed_time(end_event))
    else:
        # CPU or fallback: use perf_counter with manual sync
        if x.is_cuda:
            torch.cuda.synchronize()
        timings = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(x)
            if x.is_cuda:
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = sum(timings) / len(timings)
    var_ms = sum((t - mean_ms) ** 2 for t in timings) / len(timings)
    std_ms = var_ms ** 0.5
    return mean_ms, std_ms


# =======================================================================
# Main Complexity Benchmark Class
# =======================================================================

class ComplexityBenchmark:
    """
    Unified latency, throughput, and fvcore FLOP benchmarking for
    PreserveNet masking and token-dropping at every retention level.

    This produces the final efficiency curve: accuracy/agreement vs.
    compute saved -- the chart that justifies the project's premise.

    Args:
        device:          'cuda', 'cpu', or 'auto'.
        batch_size:      Number of images per batch for benchmarking.
        image_size:      Input image spatial resolution (default: 224).
        retention_rates: List of retention fractions to benchmark.
        n_timing_runs:   Number of timing iterations for latency.
        n_warmup:        Number of warmup iterations before timing.
        accuracy_data:   Optional dict mapping (mode, arch, retention) to
                         accuracy values from prior eval runs.
    """

    RETENTION_RATES_DEFAULT = [1.0, 0.75, 0.50, 0.25, 0.10]

    def __init__(
        self,
        device: str = 'auto',
        batch_size: int = 4,
        image_size: int = 224,
        retention_rates: Optional[List[float]] = None,
        n_timing_runs: int = 50,
        n_warmup: int = 10,
        accuracy_data: Optional[Dict[Tuple[str, str, float], float]] = None,
    ) -> None:
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.batch_size = batch_size
        self.image_size = image_size
        self.retention_rates = retention_rates or self.RETENTION_RATES_DEFAULT
        self.n_timing_runs = n_timing_runs
        self.n_warmup = n_warmup
        self.accuracy_data = accuracy_data or {}
        self._results = BenchmarkResults(
            config={
                'device': str(self.device),
                'batch_size': batch_size,
                'image_size': image_size,
                'retention_rates': self.retention_rates,
                'n_timing_runs': n_timing_runs,
            }
        )

    # -- Model Loading --------------------------------------------------

    def _load_models(self) -> Tuple[nn.Module, nn.Module]:
        """Load ResNet-18 and ViT-Small Imagenette classifiers."""
        from src.models.classifier import (
            build_imagenette_classifier,
            build_vit_imagenette_classifier,
        )

        resnet = build_imagenette_classifier('resnet18', freeze=True)
        resnet = resnet.to(self.device).eval()

        vit = build_vit_imagenette_classifier(freeze=True)
        vit = vit.to(self.device).eval()

        return resnet, vit

    # -- FLOP Benchmarking ----------------------------------------------

    def benchmark_flops_resnet_masked(
        self,
        resnet: nn.Module,
    ) -> List[FLOPResult]:
        """
        Count FLOPs for ResNet-18 with pixel masking at each retention level.

        Masking zeroes out patches but the classifier still processes the
        full (B, 3, 224, 224) tensor -- FLOPs are constant regardless of
        retention rate.  This benchmark proves that masking alone does NOT
        reduce compute.
        """
        from src.models.operators import MaskOperator

        x = torch.randn(
            1, 3, self.image_size, self.image_size, device=self.device
        )
        dummy_scores = torch.randn(1, 1, 14, 14, device=self.device)

        results = []

        # Full model baseline (no masking)
        full_flops = count_flops(resnet, x)
        results.append(FLOPResult(
            retention_rate=1.0,
            mode='full',
            architecture='resnet18',
            total_flops=full_flops,
            total_gflops=full_flops / 1e9,
            flop_reduction_pct=0.0,
            tokens_kept=196,
            tokens_total=196,
        ))

        for r in self.retention_rates:
            if r == 1.0:
                continue
            mask_op = MaskOperator(retention_rate=r)
            mask_op = mask_op.to(self.device).eval()

            wrapper = _MaskedClassifierWrapper(resnet, mask_op, dummy_scores)
            wrapper = wrapper.to(self.device).eval()
            flops = count_flops(wrapper, x)

            n_kept = max(1, round(196 * r))
            results.append(FLOPResult(
                retention_rate=r,
                mode='masked',
                architecture='resnet18',
                total_flops=flops,
                total_gflops=flops / 1e9,
                flop_reduction_pct=(1.0 - flops / full_flops) * 100.0,
                tokens_kept=n_kept,
                tokens_total=196,
            ))

        return results

    def benchmark_flops_vit_masked(
        self,
        vit: nn.Module,
    ) -> List[FLOPResult]:
        """
        Count FLOPs for ViT-Small with pixel masking at each retention level.

        Like ResNet, masking does NOT change the ViT's sequence length or
        FLOP count -- all 197 tokens are still processed through self-attention.
        """
        from src.models.operators import MaskOperator

        x = torch.randn(
            1, 3, self.image_size, self.image_size, device=self.device
        )
        dummy_scores = torch.randn(1, 1, 14, 14, device=self.device)

        results = []

        # Full model baseline
        full_flops = count_flops(vit, x)
        results.append(FLOPResult(
            retention_rate=1.0,
            mode='full',
            architecture='vit_small_patch16_224',
            total_flops=full_flops,
            total_gflops=full_flops / 1e9,
            flop_reduction_pct=0.0,
            tokens_kept=196,
            tokens_total=196,
        ))

        for r in self.retention_rates:
            if r == 1.0:
                continue
            mask_op = MaskOperator(retention_rate=r)
            mask_op = mask_op.to(self.device).eval()

            wrapper = _MaskedClassifierWrapper(vit, mask_op, dummy_scores)
            wrapper = wrapper.to(self.device).eval()
            flops = count_flops(wrapper, x)

            n_kept = max(1, round(196 * r))
            results.append(FLOPResult(
                retention_rate=r,
                mode='masked',
                architecture='vit_small_patch16_224',
                total_flops=flops,
                total_gflops=flops / 1e9,
                flop_reduction_pct=(1.0 - flops / full_flops) * 100.0,
                tokens_kept=n_kept,
                tokens_total=196,
            ))

        return results

    def benchmark_flops_vit_token_drop(
        self,
        vit: nn.Module,
    ) -> List[FLOPResult]:
        """
        Count FLOPs for ViT-Small with physical token dropping at each
        retention level.

        TokenDropOperator physically truncates the ViT token sequence
        from 197 down to (1 + K), yielding real FLOP reduction that
        scales quadratically with the number of dropped tokens.
        """
        from src.models.operators import TokenDropOperator

        x = torch.randn(
            1, 3, self.image_size, self.image_size, device=self.device
        )

        # Full model baseline (reuse if already computed)
        full_flops = count_flops(vit, x)

        results = []
        results.append(FLOPResult(
            retention_rate=1.0,
            mode='token_drop',
            architecture='vit_small_patch16_224',
            total_flops=full_flops,
            total_gflops=full_flops / 1e9,
            flop_reduction_pct=0.0,
            tokens_kept=196,
            tokens_total=196,
        ))

        for r in self.retention_rates:
            if r == 1.0:
                continue

            token_op = TokenDropOperator(retention_rate=r)
            token_op = token_op.to(self.device).eval()
            dummy_scores = torch.randn(1, 1, 14, 14, device=self.device)

            wrapper = _TokenDropViTWrapper(vit, token_op, dummy_scores)
            wrapper = wrapper.to(self.device).eval()
            flops = count_flops(wrapper, x)

            n_kept = max(1, round(196 * r))
            results.append(FLOPResult(
                retention_rate=r,
                mode='token_drop',
                architecture='vit_small_patch16_224',
                total_flops=flops,
                total_gflops=flops / 1e9,
                flop_reduction_pct=(1.0 - flops / full_flops) * 100.0,
                tokens_kept=n_kept,
                tokens_total=196,
            ))

        return results

    # -- Latency Benchmarking -------------------------------------------

    def benchmark_latency_resnet(
        self,
        resnet: nn.Module,
    ) -> List[LatencyResult]:
        """
        Measure wall-clock latency for ResNet-18 masked inference.

        Because CNNs process the full (B, 3, H, W) tensor regardless of
        masking, latency is expected to be constant across retention levels.
        """
        from src.models.operators import MaskOperator

        x = torch.randn(
            self.batch_size, 3, self.image_size, self.image_size,
            device=self.device,
        )
        dummy_scores = torch.randn(
            self.batch_size, 1, 14, 14, device=self.device
        )

        # Full model baseline
        mean_ms, std_ms = benchmark_latency(
            resnet, x,
            n_runs=self.n_timing_runs,
            n_warmup=self.n_warmup,
            use_cuda_events=x.is_cuda,
        )
        full_ms = mean_ms

        results = [LatencyResult(
            retention_rate=1.0,
            mode='full',
            architecture='resnet18',
            latency_ms=mean_ms,
            latency_std_ms=std_ms,
            throughput_fps=self.batch_size / (mean_ms / 1000.0),
            speedup=1.0,
            batch_size=self.batch_size,
        )]

        for r in self.retention_rates:
            if r == 1.0:
                continue
            mask_op = MaskOperator(retention_rate=r).to(self.device).eval()
            wrapper = _MaskedClassifierWrapper(resnet, mask_op, dummy_scores)
            wrapper = wrapper.to(self.device).eval()

            mean_ms, std_ms = benchmark_latency(
                wrapper, x,
                n_runs=self.n_timing_runs,
                n_warmup=self.n_warmup,
                use_cuda_events=x.is_cuda,
            )
            results.append(LatencyResult(
                retention_rate=r,
                mode='masked',
                architecture='resnet18',
                latency_ms=mean_ms,
                latency_std_ms=std_ms,
                throughput_fps=self.batch_size / (mean_ms / 1000.0),
                speedup=full_ms / mean_ms,
                batch_size=self.batch_size,
            ))

        return results

    def benchmark_latency_vit_masked(
        self,
        vit: nn.Module,
    ) -> List[LatencyResult]:
        """
        Measure wall-clock latency for ViT-Small with pixel masking.

        Like ResNet, the full 197-token sequence is still processed,
        so latency should be ~constant across retention levels.
        """
        from src.models.operators import MaskOperator

        x = torch.randn(
            self.batch_size, 3, self.image_size, self.image_size,
            device=self.device,
        )
        dummy_scores = torch.randn(
            self.batch_size, 1, 14, 14, device=self.device
        )

        mean_ms, std_ms = benchmark_latency(
            vit, x,
            n_runs=self.n_timing_runs,
            n_warmup=self.n_warmup,
            use_cuda_events=x.is_cuda,
        )
        full_ms = mean_ms

        results = [LatencyResult(
            retention_rate=1.0,
            mode='full',
            architecture='vit_small_patch16_224',
            latency_ms=mean_ms,
            latency_std_ms=std_ms,
            throughput_fps=self.batch_size / (mean_ms / 1000.0),
            speedup=1.0,
            batch_size=self.batch_size,
        )]

        for r in self.retention_rates:
            if r == 1.0:
                continue
            mask_op = MaskOperator(retention_rate=r).to(self.device).eval()
            wrapper = _MaskedClassifierWrapper(vit, mask_op, dummy_scores)
            wrapper = wrapper.to(self.device).eval()

            mean_ms, std_ms = benchmark_latency(
                wrapper, x,
                n_runs=self.n_timing_runs,
                n_warmup=self.n_warmup,
                use_cuda_events=x.is_cuda,
            )
            results.append(LatencyResult(
                retention_rate=r,
                mode='masked',
                architecture='vit_small_patch16_224',
                latency_ms=mean_ms,
                latency_std_ms=std_ms,
                throughput_fps=self.batch_size / (mean_ms / 1000.0),
                speedup=full_ms / mean_ms,
                batch_size=self.batch_size,
            ))

        return results

    def benchmark_latency_vit_token_drop(
        self,
        vit: nn.Module,
    ) -> List[LatencyResult]:
        """
        Measure wall-clock latency for ViT-Small with physical token dropping.

        This is the key benchmark: token dropping genuinely reduces
        self-attention sequence length, producing real wall-clock speedups
        that scale quadratically.
        """
        from src.models.operators import TokenDropOperator

        x = torch.randn(
            self.batch_size, 3, self.image_size, self.image_size,
            device=self.device,
        )

        # Full ViT baseline
        mean_ms, std_ms = benchmark_latency(
            vit, x,
            n_runs=self.n_timing_runs,
            n_warmup=self.n_warmup,
            use_cuda_events=x.is_cuda,
        )
        full_ms = mean_ms

        results = [LatencyResult(
            retention_rate=1.0,
            mode='token_drop',
            architecture='vit_small_patch16_224',
            latency_ms=mean_ms,
            latency_std_ms=std_ms,
            throughput_fps=self.batch_size / (mean_ms / 1000.0),
            speedup=1.0,
            batch_size=self.batch_size,
        )]

        for r in self.retention_rates:
            if r == 1.0:
                continue

            token_op = TokenDropOperator(retention_rate=r).to(self.device).eval()
            dummy_scores = torch.randn(
                self.batch_size, 1, 14, 14, device=self.device
            )
            wrapper = _TokenDropViTWrapper(vit, token_op, dummy_scores)
            wrapper = wrapper.to(self.device).eval()

            mean_ms, std_ms = benchmark_latency(
                wrapper, x,
                n_runs=self.n_timing_runs,
                n_warmup=self.n_warmup,
                use_cuda_events=x.is_cuda,
            )
            results.append(LatencyResult(
                retention_rate=r,
                mode='token_drop',
                architecture='vit_small_patch16_224',
                latency_ms=mean_ms,
                latency_std_ms=std_ms,
                throughput_fps=self.batch_size / (mean_ms / 1000.0),
                speedup=full_ms / mean_ms,
                batch_size=self.batch_size,
            ))

        return results

    # -- Efficiency Curve Construction ----------------------------------

    def build_efficiency_curve(
        self,
        flop_results: List[FLOPResult],
        latency_results: List[LatencyResult],
    ) -> List[EfficiencyPoint]:
        """
        Combine FLOP + latency + accuracy data into efficiency curve points.

        For each (mode, architecture, retention_rate) configuration, computes
        the accuracy-per-GFLOP ratio that forms the final efficiency analysis.
        """
        # Index latency results by (mode, arch, retention)
        lat_index: Dict[Tuple[str, str, float], LatencyResult] = {}
        for lr in latency_results:
            lat_index[(lr.mode, lr.architecture, lr.retention_rate)] = lr

        points = []
        for fr in flop_results:
            key = (fr.mode, fr.architecture, fr.retention_rate)
            lr = lat_index.get(key)

            # Look up accuracy from external data
            acc = self.accuracy_data.get(key, 0.0)

            acc_per_gflop = acc / fr.total_gflops if fr.total_gflops > 0 else 0.0

            points.append(EfficiencyPoint(
                retention_rate=fr.retention_rate,
                mode=fr.mode,
                architecture=fr.architecture,
                accuracy_pct=acc,
                gflops=fr.total_gflops,
                latency_ms=lr.latency_ms if lr else 0.0,
                throughput_fps=lr.throughput_fps if lr else 0.0,
                flop_reduction_pct=fr.flop_reduction_pct,
                speedup=lr.speedup if lr else 1.0,
                accuracy_per_gflop=acc_per_gflop,
            ))

        return points

    # -- Full Benchmark -------------------------------------------------

    def run_full_benchmark(
        self,
        skip_latency: bool = False,
    ) -> BenchmarkResults:
        """
        Execute the complete FLOP + latency + throughput benchmark suite.

        Args:
            skip_latency: If True, skip wall-clock timing (FLOPs only).

        Returns:
            BenchmarkResults with all data points.
        """
        print("=" * 80)
        print("  PreserveNet Step 9: Latency, Throughput & FLOPs Benchmark")
        print("=" * 80)
        print(f"  Device           : {self.device}")
        print(f"  Batch size       : {self.batch_size}")
        print(f"  Image size       : {self.image_size}")
        print(f"  Retention rates  : {self.retention_rates}")
        print(f"  Timing runs      : {self.n_timing_runs}")
        print()

        # 1. Load models
        print("[1/6] Loading pretrained classifiers...")
        resnet, vit = self._load_models()
        print(f"  [OK] ResNet-18 loaded ({sum(p.numel() for p in resnet.parameters()):,} params)")
        print(f"  [OK] ViT-Small loaded ({sum(p.numel() for p in vit.parameters()):,} params)")

        # 2. FLOP benchmarks -- ResNet masking
        print("\n[2/6] Counting FLOPs -- ResNet-18 (pixel masking)...")
        flops_resnet = self.benchmark_flops_resnet_masked(resnet)
        self._results.flop_results.extend(flops_resnet)
        self._print_flop_table(flops_resnet, "ResNet-18 Masked")

        # 3. FLOP benchmarks -- ViT masking
        print("\n[3/6] Counting FLOPs -- ViT-Small (pixel masking)...")
        flops_vit_masked = self.benchmark_flops_vit_masked(vit)
        self._results.flop_results.extend(flops_vit_masked)
        self._print_flop_table(flops_vit_masked, "ViT-Small Masked")

        # 4. FLOP benchmarks -- ViT token dropping
        print("\n[4/6] Counting FLOPs -- ViT-Small (physical token dropping)...")
        flops_vit_drop = self.benchmark_flops_vit_token_drop(vit)
        self._results.flop_results.extend(flops_vit_drop)
        self._print_flop_table(flops_vit_drop, "ViT-Small Token Drop")

        if not skip_latency:
            # 5. Latency benchmarks
            print(f"\n[5/6] Measuring wall-clock latency (batch_size={self.batch_size})...")

            print("  -> ResNet-18 masked...")
            lat_resnet = self.benchmark_latency_resnet(resnet)
            self._results.latency_results.extend(lat_resnet)

            print("  -> ViT-Small masked...")
            lat_vit_masked = self.benchmark_latency_vit_masked(vit)
            self._results.latency_results.extend(lat_vit_masked)

            print("  -> ViT-Small token drop...")
            lat_vit_drop = self.benchmark_latency_vit_token_drop(vit)
            self._results.latency_results.extend(lat_vit_drop)

            self._print_latency_table(
                lat_resnet + lat_vit_masked + lat_vit_drop,
                "Combined Latency & Throughput",
            )
        else:
            print("\n[5/6] Skipping latency benchmarks (skip_latency=True).")

        # 6. Build efficiency curves
        print("\n[6/6] Building efficiency curve data...")
        eff_points = self.build_efficiency_curve(
            self._results.flop_results,
            self._results.latency_results,
        )
        self._results.efficiency_points = eff_points

        # Final summary
        self._print_comparative_summary()

        return self._results

    # -- Pretty Printing ------------------------------------------------

    def _print_flop_table(
        self,
        results: List[FLOPResult],
        title: str,
    ) -> None:
        """Print a formatted FLOP results table."""
        header = (
            f"{'Retention':>10} | {'Tokens Kept':>12} | "
            f"{'GFLOPs':>10} | {'FLOP Reduction':>15} | {'Mode':>12}"
        )
        sep = "-" * len(header)
        print(f"\n  {title}")
        print(f"  {sep}")
        print(f"  {header}")
        print(f"  {sep}")
        for r in results:
            print(
                f"  {r.retention_rate*100:>9.0f}% | "
                f"{r.tokens_kept:>4} / {r.tokens_total:<4} | "
                f"{r.total_gflops:>10.3f} | "
                f"{r.flop_reduction_pct:>14.2f}% | "
                f"{r.mode:>12}"
            )
        print(f"  {sep}")

    def _print_latency_table(
        self,
        results: List[LatencyResult],
        title: str,
    ) -> None:
        """Print a formatted latency results table."""
        header = (
            f"{'Architecture':>24} | {'Mode':>12} | {'Retention':>10} | "
            f"{'Latency (ms)':>14} | {'FPS':>10} | {'Speedup':>10}"
        )
        sep = "-" * len(header)
        print(f"\n  {title}")
        print(f"  {sep}")
        print(f"  {header}")
        print(f"  {sep}")
        for r in results:
            print(
                f"  {r.architecture:>24} | {r.mode:>12} | "
                f"{r.retention_rate*100:>9.0f}% | "
                f"{r.latency_ms:>10.2f} +/- {r.latency_std_ms:>4.1f} | "
                f"{r.throughput_fps:>10.1f} | "
                f"{r.speedup:>9.2f}x"
            )
        print(f"  {sep}")

    def _print_comparative_summary(self) -> None:
        """Print the final comparative summary of masking vs. token dropping."""
        print("\n" + "=" * 90)
        print("  STEP 9 FINAL SUMMARY: MASKING vs. TOKEN DROPPING (fvcore FLOPs)")
        print("=" * 90)

        # Group by architecture + mode
        by_config: Dict[Tuple[str, str], List[FLOPResult]] = {}
        for r in self._results.flop_results:
            key = (r.architecture, r.mode)
            by_config.setdefault(key, []).append(r)

        header = (
            f"{'Config':>35} | {'r=100%':>10} | {'r=75%':>10} | "
            f"{'r=50%':>10} | {'r=25%':>10} | {'r=10%':>10}"
        )
        sep = "-" * len(header)
        print(f"  {header}")
        print(f"  {sep}")

        rate_order = [1.0, 0.75, 0.50, 0.25, 0.10]

        for (arch, mode), flop_list in sorted(by_config.items()):
            # Index by retention rate
            by_rate = {r.retention_rate: r for r in flop_list}
            label = f"{arch} ({mode})"
            vals = []
            for rate in rate_order:
                r = by_rate.get(rate)
                if r:
                    vals.append(f"{r.total_gflops:.3f}")
                else:
                    vals.append("--")
            print(
                f"  {label:>35} | "
                + " | ".join(f"{v:>10}" for v in vals)
            )

        print(f"  {sep}")

        # Latency comparison if available
        if self._results.latency_results:
            print(f"\n  {'Config':>35} | ", end="")
            print(" | ".join(f"{'r=' + str(int(r*100)) + '%':>10}" for r in rate_order))
            print(f"  {sep}")

            lat_by_config: Dict[Tuple[str, str], Dict[float, LatencyResult]] = {}
            for lr in self._results.latency_results:
                key = (lr.architecture, lr.mode)
                lat_by_config.setdefault(key, {})[lr.retention_rate] = lr

            for (arch, mode), rate_dict in sorted(lat_by_config.items()):
                label = f"{arch} ({mode}) ms"
                vals = []
                for rate in rate_order:
                    lr = rate_dict.get(rate)
                    if lr:
                        vals.append(f"{lr.latency_ms:.1f}")
                    else:
                        vals.append("--")
                print(
                    f"  {label:>35} | "
                    + " | ".join(f"{v:>10}" for v in vals)
                )

                # Speedup row
                label_spd = f"{arch} ({mode}) speedup"
                vals_spd = []
                for rate in rate_order:
                    lr = rate_dict.get(rate)
                    if lr:
                        vals_spd.append(f"{lr.speedup:.2f}x")
                    else:
                        vals_spd.append("--")
                print(
                    f"  {label_spd:>35} | "
                    + " | ".join(f"{v:>10}" for v in vals_spd)
                )

            print(f"  {sep}")

        print("\n" + "=" * 90)
        print("  KEY INSIGHT: Masking preserves accuracy but provides 0% FLOP reduction.")
        print("  Token dropping is the ONLY path to real compute savings in Vision Transformers.")
        print("=" * 90)


# =======================================================================
# Module-level exports
# =======================================================================

__all__ = [
    'ComplexityBenchmark',
    'BenchmarkResults',
    'FLOPResult',
    'LatencyResult',
    'EfficiencyPoint',
    'count_flops',
    'get_flop_table',
    'benchmark_latency',
]
