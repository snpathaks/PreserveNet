"""
scripts/eval_cifar_baseline.py
──────────────────────────────
Baseline accuracy: pretrained ResNet-18, no reduction.

This script:
  1. Verifies CIFAR-10 is extracted (extracts if needed).
  2. Loads a timm ResNet-18 with pretrained ImageNet weights.
  3. Evaluates it ZERO-SHOT on the full CIFAR-10 test set (10 000 images)
     at 224×224 — no fine-tuning — to give the raw transfer baseline.
  4. Fine-tunes only the classification head for a small number of epochs,
     then re-evaluates on the full test set.
  5. Prints both numbers clearly. The fine-tuned number is the reference
     point (the "baseline") for all future PreserveNet experiments.

Run from the repo root:
    python scripts/eval_cifar_baseline.py
"""

import sys, tarfile, pathlib, hashlib, time
import torch
import torch.nn as nn

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent   # PreserveNet/
DATA_DIR  = REPO_ROOT / 'data'                               # PreserveNet/data
CIFAR_DIR = DATA_DIR / 'cifar10'
TARBALL   = CIFAR_DIR / 'cifar-10-python.tar.gz'
EXTRACTED = CIFAR_DIR / 'cifar-10-batches-py'

EXPECTED_MD5 = 'c58f30108f718f92721af3b95e74349a'

sys.path.insert(0, str(REPO_ROOT))

# ── 0. Extract CIFAR-10 tarball if needed ────────────────────────────────────
if EXTRACTED.exists():
    print(f'[ok] Dataset already extracted: {EXTRACTED}')
elif TARBALL.exists():
    print('Extracting CIFAR-10 tarball ...')
    with tarfile.open(TARBALL) as tf:
        tf.extractall(CIFAR_DIR)
    print(f'  → {EXTRACTED}')
else:
    # Let torchvision download and extract automatically
    print('CIFAR-10 not found locally — torchvision will download it now.')
    print(f'  Target: {CIFAR_DIR}')

# ── Imports ───────────────────────────────────────────────────────────────────
from src.models.classifier import (
    build_resnet18,
    check_output_shape,
    train_one_epoch,
    evaluate,
)
from src.data.datasets import get_cifar10_loaders

# ── Config ────────────────────────────────────────────────────────────────────
BATCH_SIZE  = 128
IMAGE_SIZE  = 224       # resize CIFAR→ImageNet res so pretrained weights make sense
NUM_CLASSES = 10
EPOCHS      = 5         # head-only fine-tuning; backbone stays frozen
LR          = 1e-3
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print()
print('=' * 60)
print('  PreserveNet | Baseline ResNet-18 on CIFAR-10')
print('=' * 60)
print(f'  device     : {DEVICE}')
print(f'  image_size : {IMAGE_SIZE}')
print(f'  batch_size : {BATCH_SIZE}')
print(f'  epochs     : {EPOCHS}')
print(f'  data_dir   : {DATA_DIR}')
print()

# ── 1. Build model ────────────────────────────────────────────────────────────
print('[1/5] Building ResNet-18 (pretrained ImageNet weights) ...')
model = build_resnet18(
    num_classes=NUM_CLASSES,
    pretrained=True,
    freeze_backbone=True,   # only train the new 10-class head
)
total_p   = sum(p.numel() for p in model.parameters())
train_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'  total params    : {total_p:,}')
print(f'  trainable params: {train_p:,}  (classification head only)')
print()

# ── 2. Shape check ────────────────────────────────────────────────────────────
print('[2/5] Forward-pass shape check on a dummy batch ...')
check_output_shape(model, batch_size=4, num_classes=NUM_CLASSES,
                   image_size=IMAGE_SIZE, device=DEVICE)
print()

# ── 3. Load data ──────────────────────────────────────────────────────────────
print('[3/5] Loading CIFAR-10 at 224×224 ...')
train_loader, val_loader, meta = get_cifar10_loaders(
    data_dir=DATA_DIR,
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    num_workers=0,        # Windows-safe
    augment=True,
    download=True,        # no-op if already present
)
print(f'  {meta}')
print(f'  train batches : {len(train_loader)}')
print(f'  val   batches : {len(val_loader)}')
print()

# ── 4. Zero-shot evaluation (no fine-tuning) ──────────────────────────────────
print('[4/5] Zero-shot evaluation (pretrained head, no fine-tuning) ...')
model.to(DEVICE)
zs_metrics = evaluate(model, val_loader, DEVICE)
zs_acc     = zs_metrics['top1_acc'] * 100
print(f'  Zero-shot val accuracy : {zs_acc:.2f}%')
print()

# ── 5. Fine-tune head and re-evaluate ─────────────────────────────────────────
print(f'[5/5] Fine-tuning classification head for {EPOCHS} epochs ...')
criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=LR
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)

best_acc = 0.0
t_start  = time.perf_counter()

for epoch in range(EPOCHS):
    t_ep     = time.perf_counter()
    avg_loss = train_one_epoch(model, train_loader, criterion, optimiser, DEVICE, epoch)
    metrics  = evaluate(model, val_loader, DEVICE)
    scheduler.step()

    acc = metrics['top1_acc']
    if acc > best_acc:
        best_acc = acc

    elapsed = time.perf_counter() - t_ep
    print(
        f'  Epoch {epoch+1:02d}/{EPOCHS:02d}  '
        f'train_loss={avg_loss:.4f}  '
        f'val_loss={metrics["loss"]:.4f}  '
        f'val_acc={acc*100:.2f}%  '
        f'({elapsed:.1f}s)'
    )

total_time = time.perf_counter() - t_start

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print('━' * 60)
print('  BASELINE RESULTS  (full-image, no reduction)')
print('━' * 60)
print(f'  Zero-shot accuracy (pretrained head)  : {zs_acc:.2f}%')
print(f'  Fine-tuned accuracy (head, {EPOCHS} epochs) : {best_acc*100:.2f}%')
print(f'  Total fine-tuning time               : {total_time/60:.1f} min')
print('━' * 60)
print()
print('  ★  WRITE THIS DOWN  ★')
print(f'  Baseline top-1 accuracy = {best_acc*100:.2f}%')
print('  Every PreserveNet experiment must be compared against this number.')
print()
print('Step 3 PASSED.')
