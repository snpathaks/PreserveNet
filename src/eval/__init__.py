"""
src/eval/
---------
Evaluation and benchmarking utilities for PreserveNet.

Modules
-------
complexity  — FLOPs / latency benchmarking utilities (Step 9)
pareto      — Full retention sweep + Pareto curve (Step 10)
"""

from src.eval.complexity import (
    BenchmarkResults,
    ComplexityBenchmark,
    EfficiencyPoint,
    FLOPResult,
    LatencyResult,
    benchmark_latency,
    count_flops,
    get_flop_table,
)

from src.eval.pareto import (
    agreement_rate,
    eval_simple_reducer,
    eval_gradcam_all_rates,
    eval_grad_saliency_all_rates,
    plot_pareto,
    run_sweep,
    print_summary,
)

__all__ = [
    # Step 9 — complexity
    "BenchmarkResults",
    "ComplexityBenchmark",
    "EfficiencyPoint",
    "FLOPResult",
    "LatencyResult",
    "benchmark_latency",
    "count_flops",
    "get_flop_table",
    # Step 10 — pareto
    "agreement_rate",
    "eval_simple_reducer",
    "eval_gradcam_all_rates",
    "eval_grad_saliency_all_rates",
    "plot_pareto",
    "run_sweep",
    "print_summary",
]
