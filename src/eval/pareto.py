"""
src/eval/pareto.py
──────────────────
Step 10 — Full Retention Sweep + Pareto Curve for the Learned Reducer.

Evaluates PatchScoreReducer (learned) against five baselines at four
retention levels (10 / 25 / 50 / 75 %) plus the 100 % anchor:

  Baselines
  ---------
  1. Uniform Random   — random Bernoulli patch drop, seed=42
  2. Uniform Grid     — regular spatial grid of patches
  3. Random Drop      — seeded Bernoulli patch drop (same as Uniform Random
                        but named separately to match prior scripts)
  4. GradCAM          — class activation map patch drop (layer4)
  5. Grad Saliency    — input-gradient patch drop

  Learned Reducer
  ---------------
  6. PreserveNet      — PatchScorer loaded from checkpoints/best.pt

Metrics
-------
  Primary y-axis  : Agreement Rate  (fraction of images where the masked
                    prediction matches the full-image prediction)
  Secondary panel : Top-1 Accuracy  (absolute correct / total)

Output artefacts (written to results/step10/ by default)
---------
  pareto_results.json  — raw numbers for every method × retention point
  pareto_curve.png     — two-panel headline figure (agreement + accuracy)

Usage (from repo root  e:\\PreserveNet)
-----
    python PreserveNet/src/eval/pareto.py
    python PreserveNet/src/eval/pareto.py --config PreserveNet/configs/sweep_retention.yaml
    python PreserveNet/src/eval/pareto.py --max_val_batches 25 --batch_size 32
    python PreserveNet/src/eval/pareto.py --full   # all 3,925 val images
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Repo path bootstrap ───────────────────────────────────────────────────────
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]   # …/PreserveNet/
DATA_DIR  = REPO_ROOT.parent / 'data'
sys.path.insert(0, str(REPO_ROOT))

# ── Project imports ───────────────────────────────────────────────────────────
from src.models.classifier import build_imagenette_classifier
from src.models.reducer    import PatchScorer, PatchScoreReducer
from src.data.datasets     import get_imagenette_loaders
from src.baselines.uniform     import UniformRandomReducer, UniformGridReducer
from src.baselines.random_drop import RandomDropReducer
from src.baselines.gradcam     import GradCAMReducer, _topk_patch_mask
from src.baselines.saliency    import GradientSaliencyReducer

# ── Optional YAML support ─────────────────────────────────────────────────────
try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

# ── Optional matplotlib ───────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Step 10: PreserveNet full retention sweep + Pareto curve'
    )
    p.add_argument('--config', type=str, default=None,
                   help='Optional YAML config (configs/sweep_retention.yaml)')
    p.add_argument('--arch', type=str, default='resnet18',
                   help='timm backbone (default: resnet18)')
    p.add_argument('--patch_size', type=int, default=16,
                   help='Patch size in pixels (default: 16)')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--max_val_batches', type=int, default=0,
                   help='Max val batches (0 = full set)')
    p.add_argument('--full', action='store_true',
                   help='Evaluate on complete val set (overrides max_val_batches)')
    p.add_argument('--data_dir', type=str, default=str(DATA_DIR))
    p.add_argument('--reducer_checkpoint', type=str,
                   default=str(REPO_ROOT / 'checkpoints' / 'best.pt'),
                   help='Path to trained PatchScorer checkpoint')
    p.add_argument('--output_dir', type=str,
                   default=str(REPO_ROOT / 'results' / 'step10'))
    p.add_argument('--figure_name', type=str, default='pareto_curve.png')
    p.add_argument('--results_name', type=str, default='pareto_results.json')
    p.add_argument('--gradcam_target_layer', type=str, default='layer4')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no_plot', action='store_true',
                   help='Skip the matplotlib figure (useful on headless servers)')
    return p.parse_args()


def _merge_yaml(args: argparse.Namespace) -> argparse.Namespace:
    """If --config is given and PyYAML is available, merge YAML values."""
    if args.config is None or not _HAS_YAML:
        return args
    cfg_path = pathlib.Path(args.config)
    if not cfg_path.exists():
        print(f'  [warn] Config file not found: {cfg_path}; using CLI defaults.')
        return args
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)
    # Only override if user did NOT pass the flag explicitly
    for key, val in cfg.items():
        if val is None:
            continue
        if key == 'retention_rates':
            # Store as list on namespace
            args.retention_rates_yaml = val
        elif key == 'baselines':
            args.baselines_cfg = val
        elif key == 'data_dir' and val is not None:
            args.data_dir = val
        elif hasattr(args, key):
            setattr(args, key, val)
    return args


# =============================================================================
# Agreement-rate helper
# =============================================================================

def agreement_rate(pred_full: torch.Tensor, pred_masked: torch.Tensor) -> float:
    """
    Fraction of samples where argmax(full) == argmax(masked).

    Args:
        pred_full:   (B,) integer predicted class from full image.
        pred_masked: (B,) integer predicted class from masked image.

    Returns:
        Scalar in [0, 1].
    """
    return (pred_full == pred_masked).float().mean().item()


# =============================================================================
# Per-method evaluation loops
# =============================================================================

@torch.no_grad()
def eval_simple_reducer(
    reducer,
    model:       nn.Module,
    val_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    device:      torch.device,
) -> Dict[str, float]:
    """
    Evaluate a stateless reducer (no gradient required) for one retention rate.

    Returns dict with keys 'agreement', 'top1_acc', 'n_samples'.
    """
    n_agree = n_correct = n_total = 0

    for imgs, labels in val_batches:
        imgs   = imgs.to(device)
        labels = labels.to(device)

        full_logits   = model(imgs)
        imgs_red      = reducer(imgs)
        masked_logits = model(imgs_red)

        pred_full   = full_logits.argmax(dim=1)
        pred_masked = masked_logits.argmax(dim=1)

        n_agree   += (pred_full == pred_masked).sum().item()
        n_correct += (pred_masked == labels).sum().item()
        n_total   += imgs.size(0)

    return {
        'agreement': n_agree   / max(1, n_total),
        'top1_acc':  n_correct / max(1, n_total),
        'n_samples': n_total,
    }


def eval_gradcam_all_rates(
    model:            nn.Module,
    val_batches:      List[Tuple[torch.Tensor, torch.Tensor]],
    device:           torch.device,
    retention_rates:  List[float],
    patch_size:       int = 16,
    target_layer:     str = 'layer4',
) -> Dict[float, Dict[str, float]]:
    """
    Evaluate GradCAM across all retention rates in one pass (compute CAM once).

    Returns {rate: {'agreement': float, 'top1_acc': float, 'n_samples': int}}.
    """
    cam_reducer = GradCAMReducer(
        model=model,
        retention_rate=1.0,
        patch_size=patch_size,
        device=device,
        target_layer=target_layer,
    )

    agree_counts  = {r: 0 for r in retention_rates}
    correct_count = {r: 0 for r in retention_rates}
    n_total       = 0

    for imgs, labels in val_batches:
        imgs   = imgs.to(device)
        labels = labels.to(device)

        # Compute CAM heatmap once per batch (requires grad)
        cam = cam_reducer._compute_cam(imgs)   # (B, H, W) or (B, 1, H, W)

        with torch.no_grad():
            full_logits = model(imgs)
            pred_full   = full_logits.argmax(dim=1)

        for r in retention_rates:
            if r >= 0.999:
                imgs_red = imgs
            else:
                mask     = _topk_patch_mask(cam, retention_rate=r, patch_size=patch_size)
                imgs_red = imgs * mask

            with torch.no_grad():
                masked_logits = model(imgs_red)
                pred_masked   = masked_logits.argmax(dim=1)

            agree_counts[r]  += (pred_full == pred_masked).sum().item()
            correct_count[r] += (pred_masked == labels).sum().item()

        n_total += imgs.size(0)

    return {
        r: {
            'agreement': agree_counts[r]  / max(1, n_total),
            'top1_acc':  correct_count[r] / max(1, n_total),
            'n_samples': n_total,
        }
        for r in retention_rates
    }


def eval_grad_saliency_all_rates(
    model:           nn.Module,
    val_batches:     List[Tuple[torch.Tensor, torch.Tensor]],
    device:          torch.device,
    retention_rates: List[float],
    patch_size:      int = 16,
) -> Dict[float, Dict[str, float]]:
    """
    Evaluate Gradient Saliency across all retention rates (compute saliency once).
    """
    sal_reducer = GradientSaliencyReducer(
        model=model,
        retention_rate=1.0,
        patch_size=patch_size,
        device=device,
    )

    agree_counts  = {r: 0 for r in retention_rates}
    correct_count = {r: 0 for r in retention_rates}
    n_total       = 0

    for imgs, labels in val_batches:
        imgs   = imgs.to(device)
        labels = labels.to(device)

        # Saliency computation requires grad
        saliency = sal_reducer._compute_saliency(imgs)  # (B, H, W)

        with torch.no_grad():
            full_logits = model(imgs)
            pred_full   = full_logits.argmax(dim=1)

        for r in retention_rates:
            if r >= 0.999:
                imgs_red = imgs
            else:
                mask     = _topk_patch_mask(saliency, retention_rate=r, patch_size=patch_size)
                imgs_red = imgs * mask

            with torch.no_grad():
                masked_logits = model(imgs_red)
                pred_masked   = masked_logits.argmax(dim=1)

            agree_counts[r]  += (pred_full == pred_masked).sum().item()
            correct_count[r] += (pred_masked == labels).sum().item()

        n_total += imgs.size(0)

    return {
        r: {
            'agreement': agree_counts[r]  / max(1, n_total),
            'top1_acc':  correct_count[r] / max(1, n_total),
            'n_samples': n_total,
        }
        for r in retention_rates
    }


# =============================================================================
# Plotting
# =============================================================================

# Colour / marker scheme — consistent across both panels
_STYLE: Dict[str, dict] = {
    'Uniform Random':  dict(color='#9B59B6', marker='o',  linestyle='--',  zorder=2),
    'Uniform Grid':    dict(color='#3498DB', marker='s',  linestyle='--',  zorder=2),
    'Random Drop':     dict(color='#1ABC9C', marker='^',  linestyle='--',  zorder=2),
    'GradCAM':         dict(color='#E67E22', marker='D',  linestyle='-.',  zorder=3),
    'Grad Saliency':   dict(color='#E74C3C', marker='v',  linestyle='-.',  zorder=3),
    'PreserveNet':     dict(color='#2ECC71', marker='*',  linestyle='-',   zorder=5,
                            linewidth=2.5, markersize=14),
}


def _plot_panel(
    ax,
    methods:         List[str],
    retention_rates: List[float],
    results:         Dict[str, Dict[float, Dict[str, float]]],
    metric:          str,
    ylabel:          str,
    title:           str,
    annotate_preservenet: bool = False,
) -> None:
    """Draw one panel (agreement or accuracy) on *ax*."""
    pct_rates = [r * 100 for r in retention_rates]

    for name in methods:
        style = _STYLE.get(name, dict(color='grey', marker='o', linestyle='-'))
        ys    = [results[name][r][metric] * 100 for r in retention_rates]

        lw  = style.pop('linewidth', 1.8)
        ms  = style.pop('markersize', 8)

        ax.plot(
            pct_rates, ys,
            label=name,
            linewidth=lw,
            markersize=ms,
            **style,
        )

        # Re-add popped keys so the dict isn't mutated permanently
        style['linewidth'] = lw
        style['markersize'] = ms

    ax.set_xlabel('Retention Rate (%)', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 102)
    ax.set_xticks([10, 25, 50, 75, 100])
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    ax.grid(True, linestyle=':', alpha=0.55)
    ax.legend(fontsize=9, loc='lower right', framealpha=0.85)


def plot_pareto(
    results:         Dict[str, Dict[float, Dict[str, float]]],
    retention_rates: List[float],
    output_path:     pathlib.Path,
) -> None:
    """
    Generate a two-panel publication-quality Pareto figure.

    Top panel    : Agreement Rate vs Retention Rate   (primary metric)
    Bottom panel : Top-1 Accuracy vs Retention Rate   (secondary)
    """
    methods = list(results.keys())

    fig, (ax_agree, ax_acc) = plt.subplots(
        2, 1, figsize=(9, 10), dpi=150, sharex=True,
        gridspec_kw={'hspace': 0.38},
    )

    fig.suptitle(
        'PreserveNet Step 10 — Retention Sweep Pareto Curves\n'
        'Imagenette · ResNet-18 backbone · 16 × 16 patches',
        fontsize=13, fontweight='bold', y=0.98,
    )

    _plot_panel(
        ax_agree, methods, retention_rates, results,
        metric='agreement',
        ylabel='Agreement Rate (%)',
        title='Panel A — Agreement Rate  (primary metric)',
        annotate_preservenet=True,
    )

    _plot_panel(
        ax_acc, methods, retention_rates, results,
        metric='top1_acc',
        ylabel='Top-1 Accuracy (%)',
        title='Panel B — Top-1 Accuracy  (secondary metric)',
    )

    ax_agree.set_xlabel('')   # shared x-axis; only label the bottom panel

    # Watermark / caption
    fig.text(
        0.5, 0.01,
        'Agreement Rate = fraction of images where masked prediction == full-image prediction.\n'
        'PreserveNet learned reducer (★) is trained with λ_agree loss to maximise this metric.',
        ha='center', fontsize=8, color='#555555', style='italic',
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  [plot] Saved Pareto figure → {output_path}')


# =============================================================================
# Main sweep
# =============================================================================

def run_sweep(args: argparse.Namespace) -> Dict:
    """
    Execute the full retention sweep and return the results dict.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Retention rates: prefer explicit YAML list, else CLI
    retention_rates: List[float] = getattr(
        args, 'retention_rates_yaml', [1.0, 0.75, 0.50, 0.25, 0.10]
    )
    retention_rates = sorted(retention_rates, reverse=True)   # high → low

    max_batches = None
    if args.full or args.max_val_batches <= 0:
        max_batches = None
    else:
        max_batches = args.max_val_batches

    # ── Banner ────────────────────────────────────────────────────────────────
    print()
    print('=' * 78)
    print('  PreserveNet  |  Step 10 — Full Retention Sweep + Pareto Curve')
    print('=' * 78)
    print(f'  Device            : {device}')
    print(f'  Backbone          : {args.arch}')
    print(f'  Patch size        : {args.patch_size}×{args.patch_size}')
    print(f'  Batch size        : {args.batch_size}')
    print(f'  Max val batches   : {max_batches or "Full val set"}')
    print(f'  Retention rates   : {[f"{r:.0%}" for r in retention_rates]}')
    print(f'  Reducer ckpt      : {args.reducer_checkpoint}')
    print(f'  Output dir        : {args.output_dir}')
    print('=' * 78)
    print()

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print('[1/4] Loading Imagenette validation set …')
    _, val_loader, meta = get_imagenette_loaders(
        data_dir=pathlib.Path(args.data_dir),
        batch_size=args.batch_size,
        num_workers=0,
        source='fastai',
    )
    print(f'  {meta.n_val:,} val images, {meta.n_classes} classes')

    # Cache batches in RAM for multi-pass evaluation
    print('  Caching validation batches …')
    val_batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for b_idx, (imgs, labels) in enumerate(val_loader):
        if max_batches is not None and b_idx >= max_batches:
            break
        val_batches.append((imgs, labels))
    n_eval = sum(b[0].size(0) for b in val_batches)
    print(f'  Cached {n_eval} images in {len(val_batches)} batches.\n')

    # ── 2. Classifier ─────────────────────────────────────────────────────────
    print(f'[2/4] Loading zero-shot classifier: {args.arch} …')
    model = build_imagenette_classifier(arch=args.arch, freeze=True)
    model.to(device).eval()
    print('  Done.\n')

    # ── 3. Learned reducer ────────────────────────────────────────────────────
    print('[3/4] Loading learned PatchScoreReducer …')
    scorer = PatchScorer().to(device)
    ckpt_path = pathlib.Path(args.reducer_checkpoint)
    reducer_loaded = False
    if ckpt_path.exists():
        try:
            ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
            # Checkpoint saved by trainer has key 'scorer'
            if isinstance(ckpt, dict) and 'scorer' in ckpt:
                scorer.load_state_dict(ckpt['scorer'])
            else:
                scorer.load_state_dict(ckpt)
            reducer_loaded = True
            print(f'  + Loaded PatchScorer from {ckpt_path}')
        except Exception as exc:
            print(f'  [warn] Could not load checkpoint ({exc}); using random scorer.')
    else:
        print(f'  [warn] Checkpoint not found at {ckpt_path}; using random scorer.')
    scorer.eval()
    print()

    # ── 4. Sweep ──────────────────────────────────────────────────────────────
    print('[4/4] Running retention sweep …\n')
    results: Dict[str, Dict[float, Dict[str, float]]] = {}

    # ── A. Uniform Random ────────────────────────────────────────────────────
    print('  [A] Uniform Random …')
    results['Uniform Random'] = {}
    for r in retention_rates:
        t0 = time.perf_counter()
        reducer = UniformRandomReducer(
            retention_rate=r, patch_size=args.patch_size, seed=args.seed
        )
        m = eval_simple_reducer(reducer, model, val_batches, device)
        results['Uniform Random'][r] = m
        print(f'      r={r:.0%}  agree={m["agreement"]*100:.2f}%  '
              f'acc={m["top1_acc"]*100:.2f}%  ({time.perf_counter()-t0:.1f}s, n={m["n_samples"]})')
    print()

    # ── B. Uniform Grid ──────────────────────────────────────────────────────
    print('  [B] Uniform Grid …')
    results['Uniform Grid'] = {}
    for r in retention_rates:
        t0 = time.perf_counter()
        reducer = UniformGridReducer(retention_rate=r, patch_size=args.patch_size)
        m = eval_simple_reducer(reducer, model, val_batches, device)
        results['Uniform Grid'][r] = m
        print(f'      r={r:.0%}  agree={m["agreement"]*100:.2f}%  '
              f'acc={m["top1_acc"]*100:.2f}%  ({time.perf_counter()-t0:.1f}s, n={m["n_samples"]})')
    print()

    # ── C. Random Drop ───────────────────────────────────────────────────────
    print('  [C] Random Drop …')
    results['Random Drop'] = {}
    for r in retention_rates:
        t0 = time.perf_counter()
        reducer = RandomDropReducer(
            retention_rate=r, patch_size=args.patch_size, seed=args.seed
        )
        m = eval_simple_reducer(reducer, model, val_batches, device)
        results['Random Drop'][r] = m
        print(f'      r={r:.0%}  agree={m["agreement"]*100:.2f}%  '
              f'acc={m["top1_acc"]*100:.2f}%  ({time.perf_counter()-t0:.1f}s, n={m["n_samples"]})')
    print()

    # ── D. GradCAM (single-pass over all rates) ───────────────────────────────
    print(f'  [D] GradCAM (target_layer={args.gradcam_target_layer}) …')
    t0_cam = time.perf_counter()
    cam_results = eval_gradcam_all_rates(
        model, val_batches, device, retention_rates,
        patch_size=args.patch_size,
        target_layer=args.gradcam_target_layer,
    )
    results['GradCAM'] = cam_results
    for r in retention_rates:
        m = cam_results[r]
        print(f'      r={r:.0%}  agree={m["agreement"]*100:.2f}%  '
              f'acc={m["top1_acc"]*100:.2f}%  (n={m["n_samples"]})')
    print(f'      GradCAM total: {time.perf_counter()-t0_cam:.1f}s\n')

    # ── E. Gradient Saliency (single-pass over all rates) ────────────────────
    print('  [E] Gradient Saliency (input gradient) …')
    t0_sal = time.perf_counter()
    sal_results = eval_grad_saliency_all_rates(
        model, val_batches, device, retention_rates,
        patch_size=args.patch_size,
    )
    results['Grad Saliency'] = sal_results
    for r in retention_rates:
        m = sal_results[r]
        print(f'      r={r:.0%}  agree={m["agreement"]*100:.2f}%  '
              f'acc={m["top1_acc"]*100:.2f}%  (n={m["n_samples"]})')
    print(f'      Saliency total: {time.perf_counter()-t0_sal:.1f}s\n')

    # ── F. PreserveNet learned reducer ───────────────────────────────────────
    print('  [F] PreserveNet (learned PatchScoreReducer) …')
    results['PreserveNet'] = {}
    for r in retention_rates:
        t0 = time.perf_counter()
        reducer = PatchScoreReducer(
            scorer=scorer,
            retention_rate=r,
            patch_size=args.patch_size,
            device=device,
            eval_mode=False,   # scorer already in eval mode
        )
        m = eval_simple_reducer(reducer, model, val_batches, device)
        results['PreserveNet'][r] = m
        print(f'      r={r:.0%}  agree={m["agreement"]*100:.2f}%  '
              f'acc={m["top1_acc"]*100:.2f}%  ({time.perf_counter()-t0:.1f}s, n={m["n_samples"]})')
    print()

    return {
        'retention_rates': retention_rates,
        'results': {
            method: {
                str(r): v
                for r, v in method_results.items()
            }
            for method, method_results in results.items()
        },
        'meta': {
            'arch':         args.arch,
            'patch_size':   args.patch_size,
            'n_eval':       n_eval,
            'reducer_ckpt': str(args.reducer_checkpoint),
            'reducer_loaded': reducer_loaded,
        },
    }, results


# =============================================================================
# Summary table
# =============================================================================

def print_summary(
    results:         Dict[str, Dict[float, Dict[str, float]]],
    retention_rates: List[float],
) -> None:
    methods = list(results.keys())
    col_w   = 18

    print()
    print('=' * (14 + col_w * len(methods) + 2))
    print('  STEP 10 RESULTS — AGREEMENT RATE (%)')
    print('=' * (14 + col_w * len(methods) + 2))
    header = f'  {"Retention":>10}  ' + '  '.join(f'{m:>{col_w}}' for m in methods)
    print(header)
    print('  ' + '-' * (12 + col_w * len(methods)))
    for r in retention_rates:
        row = f'  {r:>9.0%}  '
        for m in methods:
            val = results[m][r]['agreement'] * 100
            row += f'  {val:>{col_w}.2f}%'
        print(row)
    print('=' * (14 + col_w * len(methods) + 2))

    print()
    print('=' * (14 + col_w * len(methods) + 2))
    print('  STEP 10 RESULTS — TOP-1 ACCURACY (%)')
    print('=' * (14 + col_w * len(methods) + 2))
    print(header)
    print('  ' + '-' * (12 + col_w * len(methods)))
    for r in retention_rates:
        row = f'  {r:>9.0%}  '
        for m in methods:
            val = results[m][r]['top1_acc'] * 100
            row += f'  {val:>{col_w}.2f}%'
        print(row)
    print('=' * (14 + col_w * len(methods) + 2))
    print()


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    args = parse_args()
    args = _merge_yaml(args)

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep_output, results = run_sweep(args)

    # Save JSON
    json_path = out_dir / args.results_name
    with open(json_path, 'w') as fh:
        json.dump(sweep_output, fh, indent=2)
    print(f'  [json] Results saved → {json_path}')

    # Summary table
    retention_rates = sweep_output['retention_rates']
    print_summary(results, retention_rates)

    # Plot
    if not args.no_plot and _HAS_MPL:
        fig_path = out_dir / args.figure_name
        plot_pareto(results, retention_rates, fig_path)
    elif not _HAS_MPL:
        print('  [warn] matplotlib not available; skipping plot.')

    # Highlight PreserveNet vs best baseline at each rate
    print()
    print('  Key Findings (Agreement Rate at each retention level)')
    print('  ' + '-' * 60)
    competitors = [k for k in results if k != 'PreserveNet']
    for r in retention_rates:
        pn_agree = results['PreserveNet'][r]['agreement'] * 100
        best_base_name = max(competitors, key=lambda m: results[m][r]['agreement'])
        best_base_val  = results[best_base_name][r]['agreement'] * 100
        delta = pn_agree - best_base_val
        flag  = '▲' if delta >= 0 else '▼'
        print(f'  r={r:.0%}  PreserveNet={pn_agree:.2f}%  '
              f'Best baseline={best_base_name} ({best_base_val:.2f}%)  '
              f'{flag}{abs(delta):.2f} pp')
    print()
    print('DONE')


if __name__ == '__main__':
    main()
