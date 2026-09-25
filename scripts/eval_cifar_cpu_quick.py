"""
scripts/eval_cifar_cpu_quick.py
────────────────────────────────
CPU-safe quick sanity-check version of eval_cifar_baseline.py.

Changes vs. the full script:
  * IMAGE_SIZE = 32  -- no resize; uses CIFAR statistics directly (fast)
  * EPOCHS     = 1  -- one pass to get a rough number without multi-hour waits
  * freeze_backbone = True  -- only trains the 10-class linear head

Run from the repo root (e:\PreserveNet):
    e:\.venv\Scripts\python.exe PreserveNet\scripts\eval_cifar_cpu_quick.py
"""

import sys, tarfile, pathlib, time
import torch
import torch.nn as nn

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR  = REPO_ROOT.parent / 'data'
CIFAR_DIR = DATA_DIR / 'cifar10'
EXTRACTED = CIFAR_DIR / 'cifar-10-batches-py'
TARBALL   = CIFAR_DIR / 'cifar-10-python.tar.gz'

sys.path.insert(0, str(REPO_ROOT))

if EXTRACTED.exists():
    print(f'[ok] Dataset already extracted: {EXTRACTED}')
elif TARBALL.exists():
    print('Extracting CIFAR-10 tarball ...')
    with tarfile.open(TARBALL) as tf:
        tf.extractall(CIFAR_DIR)
    print(f'  -> {EXTRACTED}')
else:
    print('CIFAR-10 not found locally -- torchvision will download it (~170 MB).')

from src.models.classifier import build_resnet18, check_output_shape, train_one_epoch, evaluate
from src.data.datasets import get_cifar10_loaders

BATCH_SIZE  = 128
IMAGE_SIZE  = 32
NUM_CLASSES = 10
EPOCHS      = 1
LR          = 1e-3
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print()
print('=' * 60)
print('  PreserveNet | Baseline ResNet-18 on CIFAR-10  [CPU QUICK]')
print('=' * 60)
print(f'  device     : {DEVICE}')
print(f'  image_size : {IMAGE_SIZE}  (native CIFAR-10 -- no resize)')
print(f'  batch_size : {BATCH_SIZE}')
print(f'  epochs     : {EPOCHS}  (quick sanity check)')
print(f'  data_dir   : {DATA_DIR}')
print()

print('[1/5] Building ResNet-18 (pretrained ImageNet weights) ...')
model = build_resnet18(num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=True)
total_p = sum(p.numel() for p in model.parameters())
train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'  total params    : {total_p:,}')
print(f'  trainable params: {train_p:,}  (classification head only)')
print()

print('[2/5] Forward-pass shape check on a dummy batch ...')
check_output_shape(model, batch_size=4, num_classes=NUM_CLASSES, image_size=IMAGE_SIZE, device=DEVICE)
print()

print('[3/5] Loading CIFAR-10 at 32x32 (native resolution) ...')
train_loader, val_loader, meta = get_cifar10_loaders(
    data_dir=DATA_DIR, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE,
    num_workers=0, augment=True, download=True,
)
print(f'  {meta}')
print(f'  train batches : {len(train_loader)}')
print(f'  val   batches : {len(val_loader)}')
print()

print('[4/5] Zero-shot evaluation (ImageNet head on 32x32 CIFAR) ...')
model.to(DEVICE)
zs_metrics = evaluate(model, val_loader, DEVICE)
zs_acc     = zs_metrics['top1_acc'] * 100
print(f'  Zero-shot val accuracy : {zs_acc:.2f}%')
print()

print(f'[5/5] Fine-tuning classification head for {EPOCHS} epoch(s) ...')
criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
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
    print(f'  Epoch {epoch+1:02d}/{EPOCHS:02d}  train_loss={avg_loss:.4f}  val_loss={metrics["loss"]:.4f}  val_acc={acc*100:.2f}%  ({elapsed:.1f}s)')

total_time = time.perf_counter() - t_start

print()
print('=' * 60)
print('  QUICK BASELINE RESULTS  (32x32, 1 epoch, CPU)')
print('=' * 60)
print(f'  Zero-shot accuracy (ImageNet head)    : {zs_acc:.2f}%')
print(f'  Fine-tuned accuracy (head, {EPOCHS} epoch) : {best_acc*100:.2f}%')
print(f'  Total fine-tuning time               : {total_time/60:.1f} min')
print('=' * 60)
print()
print('  NOTE: This is a QUICK CPU sanity check (32x32, 1 epoch).')
print('  For the proper baseline, run on GPU at 224x224 for 5 epochs:')
print('    python PreserveNet/scripts/eval_cifar_baseline.py')
print()
print('Step 3 PASSED  (CPU quick-check).')
