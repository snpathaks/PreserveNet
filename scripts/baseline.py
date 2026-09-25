"""
scripts/baseline.py
────────────────────
Extract CIFAR-10 (if needed) then run the baseline classifier.

Run from the repo root:
    python scripts/baseline.py
"""
import sys, tarfile, pathlib, hashlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent   # e:\PreserveNet\PreserveNet
DATA_DIR  = REPO_ROOT.parent / 'data'                        # e:\PreserveNet\data
CIFAR_DIR = DATA_DIR / 'cifar10'
TARBALL   = CIFAR_DIR / 'cifar-10-python.tar.gz'
EXTRACTED = CIFAR_DIR / 'cifar-10-batches-py'

EXPECTED_MD5 = 'c58f30108f718f92721af3b95e74349a'

# ── Step 1: verify / extract ──────────────────────────────────────────────────
if EXTRACTED.exists():
    print(f'[ok] Already extracted: {EXTRACTED}')
elif TARBALL.exists():
    # Verify MD5 first
    print(f'Verifying {TARBALL.name} ...')
    md5 = hashlib.md5(TARBALL.read_bytes()).hexdigest()
    if md5 != EXPECTED_MD5:
        print(f'  WARNING: MD5 mismatch (got {md5})')
        print('  Delete the tarball and re-download from:')
        print('    https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')
        sys.exit(1)
    print(f'  MD5 OK. Extracting ...')
    with tarfile.open(TARBALL) as tf:
        tf.extractall(CIFAR_DIR)
    print(f'  Extracted to {EXTRACTED}')
else:
    print('ERROR: CIFAR-10 tarball not found.')
    print(f'  Expected: {TARBALL}')
    print('  Either let torchvision download it (download=True) or manually place')
    print('  cifar-10-python.tar.gz in:', CIFAR_DIR)
    sys.exit(1)

# ── Step 2: run the classifier ────────────────────────────────────────────────
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import time
from pathlib import Path
from src.models.classifier import build_resnet18, check_output_shape, train_one_epoch, evaluate
from src.data.datasets import get_cifar10_loaders

BATCH_SIZE  = 128
IMAGE_SIZE  = 224
NUM_CLASSES = 10
EPOCHS      = 5
LR          = 1e-3
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print()
print('=' * 60)
print('  PreserveNet -- Step 2  |  Baseline ResNet-18 Classifier')
print('=' * 60)
print(f'  device     : {DEVICE}')
print(f'  image_size : {IMAGE_SIZE}')
print(f'  batch_size : {BATCH_SIZE}')
print(f'  epochs     : {EPOCHS}')
print(f'  data_dir   : {DATA_DIR}')
print()

print('[1/4] Building ResNet-18 (pretrained, head frozen) ...')
model = build_resnet18(num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=True)

print('[2/4] Checking output shape with a dummy batch ...')
check_output_shape(model, batch_size=8, num_classes=NUM_CLASSES, image_size=IMAGE_SIZE, device=DEVICE)
total     = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'  total params    : {total:,}')
print(f'  trainable params: {trainable:,}  (head only)')
print()

print('[3/4] Loading CIFAR-10 at 224 x 224 ...')
train_loader, val_loader, meta = get_cifar10_loaders(
    data_dir=DATA_DIR,
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    num_workers=0,
    augment=True,
    download=False,   # already extracted
)
print(f'  {meta}')
print(f'  train batches : {len(train_loader)}')
print(f'  val   batches : {len(val_loader)}')
print()

print(f'[4/4] Fine-tuning classification head for {EPOCHS} epochs ...')
model.to(DEVICE)
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
print()
print('-' * 60)
print(f'  BEST val accuracy (no reduction): {best_acc * 100:.2f}%')
print(f'  Total training time : {total_time / 60:.1f} min')
print('-' * 60)
print()
print('  This is your BASELINE number.')
print('  Any reducer you add must beat (or at least not degrade) this.')
print()
print('Step 2 PASSED.')
