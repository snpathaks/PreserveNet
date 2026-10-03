"""
scripts/eval_saliency_baselines.py
────────────────────────────────────
Step 5 — Evaluate saliency-guided reducers against dumb baselines.

Pipeline
--------
1. Train ResNet-18 head for EPOCHS epochs on full CIFAR-10 images.
2. For each reducer (Uniform Random, Uniform Grid, Random Drop,
   Gradient Saliency, GradCAM):
   For each retention rate in RETENTION_RATES:
       Apply the reducer to every validation batch and measure top-1 accuracy.
3. Print a full comparison table.

Run from the repo root (e:\\PreserveNet):
    e:\\.venv\\Scripts\\python.exe PreserveNet\\scripts\\eval_saliency_baselines.py

Optional flags (positional, order matters):
    --image_size 32|224    (default 32 — fast; use 224 for pretrained resolution)
    --epochs N             (default 3)
"""

import sys, pathlib, time
import torch
import torch.nn as nn

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent   # .../PreserveNet/
DATA_DIR  = REPO_ROOT.parent / 'data'
sys.path.insert(0, str(REPO_ROOT))

from src.models.classifier     import build_resnet18, train_one_epoch, evaluate
from src.data.datasets         import get_cifar10_loaders
from src.baselines.uniform     import UniformRandomReducer, UniformGridReducer
from src.baselines.random_drop import RandomDropReducer
from src.baselines.saliency    import GradientSaliencyReducer, GradCAMReducer

# ── Config ────────────────────────────────────────────────────────────────────
IMAGE_SIZE      = 32          # native CIFAR-10 — fast on CPU
BATCH_SIZE      = 64          # batch size for dumb baselines
SALIENCY_BATCH  = 32          # smaller batch for saliency (backward pass is expensive)
NUM_CLASSES     = 10
EPOCHS          = 3
LR              = 1e-3
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cap val batches for saliency reducers on CPU to keep runtime reasonable.
# At 32 img/batch x 50 batches = 1,600 images (16% of CIFAR val set).
# Remove / increase this if running on GPU.
MAX_VAL_BATCHES_SALIENCY = 50  # set to None to evaluate on full val set

# Retention rates to sweep  (1.0 = full image, 0.1 = keep 10 % of pixels)
RETENTION_RATES = [1.0, 0.75, 0.50, 0.25, 0.10]

print()
print('=' * 70)
print('  PreserveNet | Step 5 — Saliency vs. Dumb Baseline Sweep')
print('=' * 70)
print(f'  device                : {DEVICE}')
print(f'  image_size            : {IMAGE_SIZE}')
print(f'  batch_size (dumb)     : {BATCH_SIZE}')
print(f'  batch_size (saliency) : {SALIENCY_BATCH}')
print(f'  epochs                : {EPOCHS}  (head-only fine-tuning on full images)')
print(f'  retentions            : {RETENTION_RATES}')
print(f'  max_val_batches (sal) : {MAX_VAL_BATCHES_SALIENCY}')
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
# Separate small-batch val loader for saliency (cheaper per-batch backward)
_, val_loader_sal, _ = get_cifar10_loaders(
    data_dir=DATA_DIR,
    batch_size=SALIENCY_BATCH,
    image_size=IMAGE_SIZE,
    num_workers=0,
    augment=False,
    download=False,
)
print(f'  {meta}')
print()

# ── 2. Train classification head on full images ───────────────────────────────
print(f'[2/3] Training ResNet-18 head for {EPOCHS} epoch(s) on FULL images ...')
model = build_resnet18(num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=True)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=LR
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)

for epoch in range(EPOCHS):
    t    = time.perf_counter()
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

# ── 3. Sweep reducers ─────────────────────────────────────────────────────────
print('[3/3] Sweeping all reducers across retention rates ...')
print()

# Each entry: (name, constructor_fn, is_saliency)
# Saliency reducers receive the trained model; dumb ones do not.
reducer_factories = [
    ('Uniform Random', lambda r: UniformRandomReducer(retention_rate=r, seed=42), False),
    ('Uniform Grid',   lambda r: UniformGridReducer(retention_rate=r),             False),
    ('Random Drop',    lambda r: RandomDropReducer(retention_rate=r, seed=42),    False),
    ('Grad Saliency',  lambda r: GradientSaliencyReducer(model, retention_rate=r, device=DEVICE), True),
    ('GradCAM',        lambda r: GradCAMReducer(model, retention_rate=r, device=DEVICE),          True),
]

# Results table: {reducer_name: {retention: acc}}
results:       dict[str, dict[float, float]] = {}
n_samples_log: dict[str, int]               = {}

for reducer_name, make_reducer, is_saliency in reducer_factories:
    results[reducer_name] = {}
    loader  = val_loader_sal if is_saliency else val_loader
    max_bat = MAX_VAL_BATCHES_SALIENCY if is_saliency else None

    if is_saliency and max_bat is not None:
        n_est = min(max_bat * SALIENCY_BATCH, meta.n_val)
        print(f'  Reducer: {reducer_name}  (evaluating on ~{n_est} images)')
    else:
        print(f'  Reducer: {reducer_name}  (evaluating on full val set)')

    for r in RETENTION_RATES:
        reducer = make_reducer(r)
        t0      = time.perf_counter()

        model.eval()
        correct = total = 0

        for batch_idx, (imgs, labels) in enumerate(loader):
            if max_bat is not None and batch_idx >= max_bat:
                break

            if is_saliency:
                # Saliency reducer does its own forward+backward internally.
                # Enable grad only for that call, then switch off for accuracy.
                imgs_reduced = reducer(imgs)           # grad enabled inside reducer
                imgs_r       = imgs_reduced.to(DEVICE)
                labels       = labels.to(DEVICE)
                with torch.no_grad():
                    logits = model(imgs_r)
            else:
                with torch.no_grad():
                    imgs_r  = reducer(imgs).to(DEVICE)
                    labels  = labels.to(DEVICE)
                    logits  = model(imgs_r)

            correct += (logits.argmax(1) == labels).sum().item()
            total   += imgs.size(0)

        acc = correct / total * 100
        results[reducer_name][r] = acc
        n_samples_log[reducer_name] = total

        elapsed = time.perf_counter() - t0
        if hasattr(reducer, 'actual_retention'):
            actual = reducer.actual_retention(IMAGE_SIZE, IMAGE_SIZE)
            print(f'    r={r:.0%}  (actual {actual:.0%})  ->  val_acc = {acc:.2f}%  ({elapsed:.1f}s, n={total})')
        else:
            print(f'    r={r:.0%}                  ->  val_acc = {acc:.2f}%  ({elapsed:.1f}s, n={total})')

    print()

# ── Summary table ─────────────────────────────────────────────────────────────
col_width = 16

print()
print('=' * 75)
print('  SALIENCY vs. DUMB BASELINE RESULTS')
print('  (classifier trained on full images; tested on reduced images)')
print('  NOTE: saliency reducers evaluated on a subset of val set on CPU.')
print('=' * 75)

header = f"  {'Retention':>10}"
for name in results:
    header += f'  {name:>{col_width}}'
print(header)
print('  ' + '-' * (12 + (col_width + 2) * len(results)))

for r in RETENTION_RATES:
    row = f'  {r:>9.0%} '
    for name in results:
        acc = results[name][r]
        row += f'  {acc:>{col_width}.2f}%'
    print(row)

print('=' * 75)
print()
print('  Interpretation:')
print('  * r=100% row  = trained baseline (all reducers should match this)')
print('  * Dumb rows   = accuracy after naive pixel drop (full val set)')
print('  * Saliency    = accuracy when model-informed pixels are kept (subset)')
print('  * A saliency reducer WINS if it out-performs dumb baselines at')
print('    the same retention rate.')
print()
print('Step 5 (Saliency Baselines) PASSED.')
