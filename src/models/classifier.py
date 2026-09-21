"""
src/models/classifier.py
────────────────────────
Step 2 — Baseline classifier (no reduction at all).

Loads a pretrained ResNet-18 through ``timm``, runs it on a CIFAR-10 batch
(images resized to 224 × 224 so the ImageNet weights make sense), and reports
top-1 accuracy on the *full* validation set.

This number is your North Star: every future experiment that adds a reducer
in front of this classifier must be measured against it.

Usage (standalone)
------------------
    python -m src.models.classifier

Or import the pieces you need:
    from src.models.classifier import build_resnet18, evaluate

Architecture note
-----------------
We replace the final FC layer (1 000 ImageNet classes) with a new linear head
of size ``num_classes`` (10 for CIFAR-10), then fine-tune **only that head**
for a handful of epochs so the backbone features stay frozen and we converge
fast.  This is standard linear probing / transfer learning.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── timm import ───────────────────────────────────────────────────────────────
try:
    import timm
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "timm is required for Step 2.  Install it with:\n"
        "    pip install timm"
    ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def build_resnet18(
    num_classes: int = 10,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    """
    Return a ResNet-18 with a fresh classification head for *num_classes*.

    Args:
        num_classes:      Number of output classes (10 for CIFAR-10).
        pretrained:       Load ImageNet-1k pretrained weights via timm.
        freeze_backbone:  If True, freeze all layers except the final FC so
                          only the head is trained.  Set False for full
                          fine-tuning.

    Returns:
        A ``torch.nn.Module`` ready for training / inference.

    Example:
        >>> model = build_resnet18(num_classes=10, pretrained=True)
        >>> x = torch.randn(4, 3, 224, 224)
        >>> model(x).shape
        torch.Size([4, 10])
    """
    model = timm.create_model(
        'resnet18',
        pretrained=pretrained,
        num_classes=num_classes,   # timm replaces the head automatically
    )

    if freeze_backbone:
        # Freeze everything ...
        for param in model.parameters():
            param.requires_grad = False
        # ... then unfreeze only the classification head.
        for param in model.get_classifier().parameters():
            param.requires_grad = True

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Training / evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    device:    torch.device,
    epoch_idx: int = 0,
) -> float:
    """Run one training epoch and return the average loss."""
    model.train()
    running_loss = 0.0
    n_samples    = 0

    bar = tqdm(loader, desc=f'  Epoch {epoch_idx + 1:02d} [train]', leave=False)
    for imgs, labels in bar:
        imgs, labels = imgs.to(device), labels.to(device)

        optimiser.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimiser.step()

        bs            = imgs.size(0)
        running_loss += loss.item() * bs
        n_samples    += bs
        bar.set_postfix(loss=f'{running_loss / n_samples:.4f}')

    return running_loss / n_samples


@torch.no_grad()
def evaluate(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """
    Evaluate *model* on *loader* and return accuracy / loss metrics.

    Returns:
        dict with keys ``'top1_acc'``, ``'loss'``, ``'n_samples'``.
    """
    model.eval()
    criterion   = nn.CrossEntropyLoss()
    correct     = 0
    total_loss  = 0.0
    n_samples   = 0

    for imgs, labels in tqdm(loader, desc='  [eval ]', leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        logits       = model(imgs)
        loss         = criterion(logits, labels)

        bs            = imgs.size(0)
        total_loss   += loss.item() * bs
        correct      += (logits.argmax(1) == labels).sum().item()
        n_samples    += bs

    return {
        'top1_acc':  correct / n_samples,
        'loss':      total_loss / n_samples,
        'n_samples': n_samples,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Output-shape sanity check
# ─────────────────────────────────────────────────────────────────────────────

def check_output_shape(
    model:       nn.Module,
    batch_size:  int = 8,
    num_classes: int = 10,
    image_size:  int = 224,
    device:      Optional[torch.device] = None,
) -> torch.Size:
    """
    Run a dummy forward pass and assert the output shape is correct.

    Args:
        model:       The classifier to probe.
        batch_size:  Fake batch size.
        num_classes: Expected number of output logits.
        image_size:  Spatial resolution of the dummy input.
        device:      Where to run the check (defaults to CPU).

    Returns:
        The actual output ``torch.Size``.

    Raises:
        AssertionError: If the shape does not match ``(batch_size, num_classes)``.
    """
    if device is None:
        device = torch.device('cpu')

    model.eval().to(device)
    dummy = torch.randn(batch_size, 3, image_size, image_size, device=device)

    with torch.no_grad():
        out = model(dummy)

    expected = torch.Size([batch_size, num_classes])
    assert out.shape == expected, (
        f"Shape mismatch — got {out.shape}, expected {expected}"
    )
    print(f'  * Output shape check passed: {tuple(out.shape)}')
    return out.shape


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entry-point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.data.datasets import get_cifar10_loaders

    # ── Config ────────────────────────────────────────────────────────────────
    BATCH_SIZE      = 128
    IMAGE_SIZE      = 224        # resize CIFAR to ImageNet resolution
    NUM_CLASSES     = 10
    EPOCHS          = 5          # head-only fine-tuning; backbone is frozen
    LR              = 1e-3
    DATA_DIR        = Path(__file__).resolve().parents[3] / 'data'
    DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=' * 60)
    print('  PreserveNet -- Step 2  |  Baseline ResNet-18 Classifier')
    print('=' * 60)
    print(f'  device     : {DEVICE}')
    print(f'  image_size : {IMAGE_SIZE}')
    print(f'  batch_size : {BATCH_SIZE}')
    print(f'  epochs     : {EPOCHS}')
    print(f'  data_dir   : {DATA_DIR}')
    print()

    # ── 1. Build model ────────────────────────────────────────────────────────
    print('[1/4] Building ResNet-18 (pretrained, head frozen) ...')
    model = build_resnet18(
        num_classes=NUM_CLASSES,
        pretrained=True,
        freeze_backbone=True,
    )

    # ── 2. Sanity-check output shape ──────────────────────────────────────────
    print('[2/4] Checking output shape with a dummy batch ...')
    check_output_shape(
        model,
        batch_size=8,
        num_classes=NUM_CLASSES,
        image_size=IMAGE_SIZE,
        device=DEVICE,
    )

    # Count trainable parameters
    total_params   = sum(p.numel() for p in model.parameters())
    trainable      = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  total params    : {total_params:,}')
    print(f'  trainable params: {trainable:,}  (head only)')
    print()

    # ── 3. Load data ──────────────────────────────────────────────────────────
    print('[3/4] Loading CIFAR-10 at 224 x 224 ...')
    train_loader, val_loader, meta = get_cifar10_loaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_workers=0,        # safe on Windows
        augment=True,
    )
    print(f'  {meta}')
    print(f'  train batches : {len(train_loader)}')
    print(f'  val   batches : {len(val_loader)}')
    print()

    # ── 4. Fine-tune head & evaluate ──────────────────────────────────────────
    print(f'[4/4] Fine-tuning classification head for {EPOCHS} epochs ...')
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)

    best_acc = 0.0
    t_start  = time.perf_counter()

    for epoch in range(EPOCHS):
        t_ep = time.perf_counter()

        avg_loss = train_one_epoch(
            model, train_loader, criterion, optimiser, DEVICE, epoch
        )
        metrics = evaluate(model, val_loader, DEVICE)
        scheduler.step()

        acc = metrics['top1_acc']
        if acc > best_acc:
            best_acc = acc

        elapsed = time.perf_counter() - t_ep
        print(
            f'  Epoch {epoch + 1:02d}/{EPOCHS:02d}  '
            f'train_loss={avg_loss:.4f}  '
            f'val_loss={metrics["loss"]:.4f}  '
            f'val_acc={acc * 100:.2f}%  '
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
