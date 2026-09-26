"""
scripts/eval_dumb_baselines.py
───────────────────────────────
Evaluate "dumb" pixel-drop baselines against the trained ResNet-18.

Pipeline
--------
1. Train the ResNet-18 classification head for EPOCHS epochs on full images.
2. For each reducer (Uniform, RandomDrop):
   For each retention rate in RETENTION_RATES:
       Apply the reducer to every validation batch and measure top-1 accuracy.
3. Print a comparison table.

Run from the repo root (e:\\PreserveNet):
    e:\\.venv\\Scripts\\python.exe PreserveNet\\scripts\\eval_dumb_baselines.py
"""

import sys, pathlib, time
import torch
import torch.nn as nn

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent   # .../PreserveNet/
DATA_DIR  = REPO_ROOT.parent / 'data'
sys.path.insert(0, str(REPO_ROOT))

from src.models.classifier import build_resnet18, train_one_epoch, evaluate
from src.data.datasets     import get_cifar10_loaders
from src.baselines.uniform      import UniformRandomReducer, UniformGridReducer
from src.baselines.random_drop  import RandomDropReducer

# ── Config ────────────────────────────────────────────────────────────────────
BATCH_SIZE      = 128
IMAGE_SIZE      = 32          # native CIFAR-10 — fast on CPU
NUM_CLASSES     = 10
EPOCHS          = 1           # bump to 5 for a stronger reference model
LR              = 1e-3
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Retention rates to sweep  (1.0 = full image, 0.1 = keep 10 % of pixels)
RETENTION_RATES = [1.0, 0.75, 0.50, 0.25, 0.10]

print()
print('=' * 65)
print('  PreserveNet | Dumb Baseline Sweep')
print('=' * 65)
print(f'  device     : {DEVICE}')
print(f'  image_size : {IMAGE_SIZE}')
print(f'  epochs     : {EPOCHS}  (head-only fine-tuning on full images)')
print(f'  retentions : {RETENTION_RATES}')
print()

# ── 1. Data ───────────────────────────────────────────────────────────────────
print('[1/3] Loading CIFAR-10 ...')
train_loader, val_loader, meta = get_cifar10_loaders(
    data_dir=DATA_DIR,
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    num_workers=0,
    augment=True,
    download=True,
)
print(f'  {meta}')
print()

# ── 2. Train head on full images ──────────────────────────────────────────────
print(f'[2/3] Training ResNet-18 head for {EPOCHS} epoch(s) on FULL images ...')
model = build_resnet18(num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=True)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=LR
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)

for epoch in range(EPOCHS):
    t = time.perf_counter()
    loss = train_one_epoch(model, train_loader, criterion, optimiser, DEVICE, epoch)
    m    = evaluate(model, val_loader, DEVICE)
    scheduler.step()
    print(
        f'  Epoch {epoch+1:02d}/{EPOCHS:02d}  '
        f'train_loss={loss:.4f}  '
        f'val_acc={m["top1_acc"]*100:.2f}%  '
        f'({time.perf_counter()-t:.1f}s)'
    )
print()

# ── 3. Evaluate with each reducer ─────────────────────────────────────────────
print('[3/3] Sweeping reducers across retention rates ...')
print()

# Each entry: (name, constructor_fn)
reducer_factories = [
    ('Uniform Random', lambda r: UniformRandomReducer(retention_rate=r, seed=42)),
    ('Uniform Grid',   lambda r: UniformGridReducer(retention_rate=r)),
    ('Random Drop',    lambda r: RandomDropReducer(retention_rate=r, seed=42)),
]

# Results table: {reducer_name: {retention: acc}}
results: dict[str, dict[float, float]] = {}

for reducer_name, make_reducer in reducer_factories:
    results[reducer_name] = {}
    print(f'  Reducer: {reducer_name}')

    for r in RETENTION_RATES:
        reducer = make_reducer(r)

        # Evaluate: apply reducer to each batch on-the-fly
        model.eval()
        correct = total = 0
        loss_sum = 0.0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs_reduced = reducer(imgs)          # apply reduction
                imgs_reduced = imgs_reduced.to(DEVICE)
                labels       = labels.to(DEVICE)
                logits       = model(imgs_reduced)
                loss_sum    += nn.CrossEntropyLoss()(logits, labels).item() * imgs.size(0)
                correct     += (logits.argmax(1) == labels).sum().item()
                total       += imgs.size(0)

        acc = correct / total * 100
        results[reducer_name][r] = acc

        # Actual retention may differ from target (uniform grid rounding)
        if hasattr(reducer, 'actual_retention'):
            actual = reducer.actual_retention(IMAGE_SIZE, IMAGE_SIZE)
            print(f'    r={r:.0%}  (actual {actual:.0%})  ->  val_acc = {acc:.2f}%')
        else:
            print(f'    r={r:.0%}                  ->  val_acc = {acc:.2f}%')

    print()

# ── Summary table ─────────────────────────────────────────────────────────────
print()
print('-' * 65)
print('  DUMB BASELINE RESULTS  (train on full, test on reduced)')
print('-' * 65)

header = f"  {'Retention':>10}"
for name in results:
    header += f'  {name:>14}'
print(header)
print('  ' + '-' * 42)

for r in RETENTION_RATES:
    row = f'  {r:>9.0%} '
    for name in results:
        acc = results[name][r]
        row += f'  {acc:>13.2f}%'
    print(row)

print('-' * 65)
print()
print('  Interpretation:')
print('  * r=100% row = your trained baseline (goal: match this after reduction)')
print('  * Other rows = accuracy when pixels are naively dropped')
print('  * PreserveNet must beat these numbers at each retention level')
print()
print('Step 4 (Dumb Baselines) PASSED.')
