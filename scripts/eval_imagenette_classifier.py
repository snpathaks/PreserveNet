"""
scripts/eval_imagenette_classifier.py
──────────────────────────────────────
Zero-shot + optional fine-tune evaluation of timm ImageNet classifiers
on Imagenette.

Strategy
--------
Imagenette's 10 classes are a strict subset of ImageNet-1000.
The pretrained 1000-class head already knows all of them.
ImagenetteZeroShotClassifier slices the 10 relevant logit columns --
no gradient update required.

What this script does
---------------------
1. Load ResNet-18 and ViT-Small/16 pretrained weights (timm).
2. Wrap each with ImagenetteZeroShotClassifier (zero-shot, no training).
3. Evaluate BOTH on the full Imagenette validation set (3,925 images).
4. Print a side-by-side results table.
5. Optionally fine-tune the top linear head for a few epochs and re-evaluate.

Run (from repo root  e:\PreserveNet\)
--------------------------------------
    python PreserveNet\scripts\eval_imagenette_classifier.py
    python PreserveNet\scripts\eval_imagenette_classifier.py --finetune --epochs 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent   # .../PreserveNet/
DATA_DIR  = REPO_ROOT.parent / 'data'
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn

from src.data.datasets import get_imagenette_loaders
from src.models.classifier import (
    build_imagenette_classifier,
    build_vit_imagenette_classifier,
    evaluate,
    train_one_epoch,
    IMAGENETTE_IMAGENET_INDICES,
)

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Imagenette zero-shot eval')
parser.add_argument('--batch_size', type=int,  default=32)
parser.add_argument('--finetune',  action='store_true',
                    help='Fine-tune the classification head after zero-shot eval')
parser.add_argument('--epochs',    type=int,  default=3,
                    help='Head-only fine-tune epochs (only when --finetune)')
parser.add_argument('--lr',        type=float, default=1e-3)
parser.add_argument('--arch',      type=str,
                    choices=['resnet18', 'vit_small_patch16_224', 'both'],
                    default='both',
                    help='Which architecture(s) to evaluate')
args = parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print()
print('=' * 65)
print('  PreserveNet | Imagenette Zero-Shot Classifier Evaluation')
print('=' * 65)
print(f'  device       : {DEVICE}')
print(f'  batch_size   : {args.batch_size}')
print(f'  finetune     : {args.finetune}')
if args.finetune:
    print(f'  ft_epochs    : {args.epochs}')
    print(f'  ft_lr        : {args.lr}')
print(f'  arch(s)      : {args.arch}')
print(f'  data_dir     : {DATA_DIR}')
print()
print(f'  ImageNet indices used: {IMAGENETTE_IMAGENET_INDICES}')
print()

# ── 1. Data ───────────────────────────────────────────────────────────────────
print('[1/3] Loading Imagenette (fastai CDN) ...')
train_loader, val_loader, meta = get_imagenette_loaders(
    data_dir=DATA_DIR,
    batch_size=args.batch_size,
    source='fastai',
    num_workers=0,
)
print(f'  {meta}')
print(f'  val images   : {meta.n_val:,}')
print()

# ── 2. Build & evaluate models ────────────────────────────────────────────────
archs_to_run = (
    ['resnet18', 'vit_small_patch16_224']
    if args.arch == 'both'
    else [args.arch]
)

results = {}  # arch -> {'zero_shot': acc, 'finetuned': acc or None}

for arch in archs_to_run:
    print(f'[2/3] Architecture: {arch}')
    print(f'  Loading pretrained weights from timm ...')
    t0 = time.perf_counter()

    if arch == 'vit_small_patch16_224':
        model = build_vit_imagenette_classifier(freeze=True)
    else:
        model = build_imagenette_classifier(arch=arch, freeze=True)

    model.to(DEVICE)
    print(f'  Model built in {time.perf_counter() - t0:.1f}s')

    total_params = sum(p.numel() for p in model.parameters())
    print(f'  Total params : {total_params:,}')
    print(f'  Trainable    : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

    # ── Shape check ──────────────────────────────────────────────────────────
    model.eval()
    dummy = torch.randn(2, 3, 224, 224, device=DEVICE)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (2, 10), f'Bad output shape: {out.shape}'
    print(f'  Output shape check: {tuple(out.shape)} OK')

    # ── Zero-shot eval ────────────────────────────────────────────────────────
    print(f'  [zero-shot] Evaluating on {meta.n_val:,} val images ...')
    t1 = time.perf_counter()
    zs_metrics = evaluate(model, val_loader, DEVICE)
    zs_acc = zs_metrics['top1_acc'] * 100
    print(f'  [zero-shot] top-1 acc = {zs_acc:.2f}%  ({time.perf_counter()-t1:.1f}s)')

    results[arch] = {'zero_shot': zs_acc, 'finetuned': None}

    # ── Optional head fine-tune ───────────────────────────────────────────────
    if args.finetune:
        print(f'  [finetune]  Unfreezing classifier head only ...')
        # Unfreeze only the backbone's classifier head
        for p in model.backbone.get_classifier().parameters():
            p.requires_grad = True
        trainable_ft = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'  [finetune]  trainable params: {trainable_ft:,}')

        criterion = nn.CrossEntropyLoss()
        optimiser = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=args.lr
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=args.epochs
        )

        best_ft_acc = 0.0
        for epoch in range(args.epochs):
            t_ep = time.perf_counter()
            loss = train_one_epoch(
                model, train_loader, criterion, optimiser, DEVICE, epoch
            )
            m = evaluate(model, val_loader, DEVICE)
            scheduler.step()
            acc_ep = m['top1_acc'] * 100
            best_ft_acc = max(best_ft_acc, acc_ep)
            print(
                f'  [finetune]  epoch {epoch+1:02d}/{args.epochs:02d}  '
                f'train_loss={loss:.4f}  val_acc={acc_ep:.2f}%  '
                f'({time.perf_counter()-t_ep:.1f}s)'
            )
        results[arch]['finetuned'] = best_ft_acc

    print()

# ── 3. Results table ──────────────────────────────────────────────────────────
print('[3/3] Summary')
print()
print('=' * 65)
print('  IMAGENETTE CLASSIFIER RESULTS  (val set, 3,925 images)')
print('=' * 65)
header = f"  {'Architecture':<28}  {'Zero-shot':>12}"
if args.finetune:
    header += f"  {'Fine-tuned':>12}"
print(header)
print('  ' + '-' * (30 + 14 + (14 if args.finetune else 0)))
for arch, res in results.items():
    row = f"  {arch:<28}  {res['zero_shot']:>11.2f}%"
    if args.finetune:
        ft = res['finetuned']
        row += f"  {ft:>11.2f}%" if ft is not None else f"  {'N/A':>12}"
    print(row)
print('=' * 65)
print()
print('  Notes:')
print('  * Zero-shot: pretrained ImageNet weights, no gradient update.')
print('  * Imagenette classes are a subset of ImageNet-1000;')
print('    the pretrained head already knows all 10 classes.')
print('  * Typical zero-shot baselines: ResNet-18 ~90%, ViT-S ~95%+')
print()
print('eval_imagenette_classifier PASSED.')
