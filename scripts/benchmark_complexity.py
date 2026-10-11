"""
scripts/benchmark_complexity.py
---------------------------------
Step 9 -- Run the full FLOP, latency, and throughput benchmark.

Produces the final efficiency curve data: accuracy/agreement vs.
compute saved -- the chart that justifies PreserveNet's premise.

Usage:
    python PreserveNet/scripts/benchmark_complexity.py
    python PreserveNet/scripts/benchmark_complexity.py --batch_size 8 --device cpu
    python PreserveNet/scripts/benchmark_complexity.py --skip_latency  # FLOPs only (fast)
    python PreserveNet/scripts/benchmark_complexity.py --timing_runs 100  # More precise timing
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent  # .../PreserveNet/
sys.path.insert(0, str(REPO_ROOT))

from src.eval.complexity import ComplexityBenchmark


# -- Known accuracy data from prior benchmark runs ----------------------
# Source: Steps 4, 5, 8 evaluation results (Imagenette val set)
# Key format: (mode, architecture, retention_rate) -> top-1 accuracy %

KNOWN_ACCURACY = {
    # ResNet-18 Masked (from Step 4 -- GradCAM results)
    ('full',   'resnet18', 1.00): 99.84,
    ('masked', 'resnet18', 0.75): 99.84,
    ('masked', 'resnet18', 0.50): 99.84,
    ('masked', 'resnet18', 0.25): 99.69,
    ('masked', 'resnet18', 0.10): 98.59,

    # ViT-Small Full (from Step 8)
    ('full',   'vit_small_patch16_224', 1.00): 100.00,

    # ViT-Small Masked (from Step 8 -- pixel masking, no token drop)
    ('masked', 'vit_small_patch16_224', 0.75): 100.00,
    ('masked', 'vit_small_patch16_224', 0.50): 97.81,
    ('masked', 'vit_small_patch16_224', 0.25): 82.50,
    ('masked', 'vit_small_patch16_224', 0.10): 31.25,

    # ViT-Small Token Drop (from Step 8 -- GradCAM-guided)
    ('token_drop', 'vit_small_patch16_224', 1.00): 100.00,
    ('token_drop', 'vit_small_patch16_224', 0.75): 100.00,
    ('token_drop', 'vit_small_patch16_224', 0.50): 99.69,
    ('token_drop', 'vit_small_patch16_224', 0.25): 96.56,
    ('token_drop', 'vit_small_patch16_224', 0.10): 68.12,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PreserveNet Step 9: FLOP, Latency & Throughput Benchmark"
    )
    parser.add_argument(
        '--batch_size', type=int, default=4,
        help='Batch size for benchmarking (default: 4)',
    )
    parser.add_argument(
        '--device', type=str, default='auto',
        help='Device: cuda / cpu / auto (default: auto)',
    )
    parser.add_argument(
        '--timing_runs', type=int, default=50,
        help='Number of timing iterations for latency (default: 50)',
    )
    parser.add_argument(
        '--warmup', type=int, default=10,
        help='Number of warmup iterations (default: 10)',
    )
    parser.add_argument(
        '--skip_latency', action='store_true',
        help='Skip wall-clock latency benchmarks (FLOP counting only)',
    )
    parser.add_argument(
        '--output', type=str,
        default=str(REPO_ROOT / 'scripts' / 'complexity_benchmark_results.json'),
        help='Output JSON path for results',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    bench = ComplexityBenchmark(
        device=args.device,
        batch_size=args.batch_size,
        n_timing_runs=args.timing_runs,
        n_warmup=args.warmup,
        accuracy_data=KNOWN_ACCURACY,
    )

    results = bench.run_full_benchmark(skip_latency=args.skip_latency)

    # Save results
    results.save_json(args.output)
    print(f"\n[OK] Results saved to: {args.output}")

    # Print efficiency curve
    print("\n" + "=" * 100)
    print("  EFFICIENCY CURVE: ACCURACY vs. COMPUTE SAVED")
    print("=" * 100)
    header = (
        f"{'Mode':>12} | {'Architecture':>24} | {'Retention':>10} | "
        f"{'Accuracy':>10} | {'GFLOPs':>10} | {'FLOP Saved':>11} | "
        f"{'Acc/GFLOP':>10}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    for ep in sorted(
        results.efficiency_points,
        key=lambda x: (x.architecture, x.mode, -x.retention_rate),
    ):
        if ep.accuracy_pct > 0:
            print(
                f"{ep.mode:>12} | {ep.architecture:>24} | "
                f"{ep.retention_rate*100:>9.0f}% | "
                f"{ep.accuracy_pct:>9.2f}% | "
                f"{ep.gflops:>10.3f} | "
                f"{ep.flop_reduction_pct:>10.2f}% | "
                f"{ep.accuracy_per_gflop:>10.2f}"
            )
    print(sep)

    print("\nDONE")


if __name__ == '__main__':
    main()
