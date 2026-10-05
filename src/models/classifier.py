"""
src/models/classifier.py
------------------------
Baseline classifiers for PreserveNet.

Provides two families of models:

1. CIFAR-10 transfer (fine-tune head)
   build_resnet18(num_classes=10)
   Replaces the 1000-class ImageNet head with a fresh 10-class head, then
   fine-tunes only that head on CIFAR-10 with the backbone frozen.

2. Imagenette zero-shot / identity remap  (no fine-tuning needed)
   build_imagenette_classifier(arch='resnet18')
   build_vit_imagenette_classifier()
   build_timm_imagenette_classifier(arch)

   Imagenette's 10 classes are a strict subset of ImageNet-1000, so the
   pretrained 1000-class head already knows all of them.  We wrap the model
   in ImagenetteZeroShotClassifier, which slices out the 10 relevant
   logit columns and re-indexes them to [0, 9].  This gives 90%+ top-1
   accuracy with zero fine-tuning.

Usage
-----
    from src.models.classifier import (
        build_resnet18,
        build_imagenette_classifier,
        build_vit_imagenette_classifier,
        evaluate,
    )

    # Zero-shot on Imagenette
    model = build_imagenette_classifier('resnet18')  # or 'vit_small_patch16_224'
    # model(x).shape == (B, 10)  -- already 10 classes, no training
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# -- timm import --------------------------------------------------------------
try:
    import timm
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "timm is required.  Install it with:\n"
        "    pip install timm"
    ) from exc


# =============================================================================
# CIFAR-10: head-replacement + fine-tune
# =============================================================================

def build_resnet18(
    num_classes: int = 10,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    """
    Return a ResNet-18 with a fresh classification head for num_classes.

    Args:
        num_classes:      Number of output classes (10 for CIFAR-10).
        pretrained:       Load ImageNet-1k pretrained weights via timm.
        freeze_backbone:  If True, freeze all layers except the final FC so
                          only the head is trained.

    Returns:
        A torch.nn.Module ready for training / inference.
    """
    model = timm.create_model(
        'resnet18',
        pretrained=pretrained,
        num_classes=num_classes,   # timm replaces the head automatically
    )

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.get_classifier().parameters():
            param.requires_grad = True

    return model


# =============================================================================
# Imagenette zero-shot classifier  (no fine-tuning required)
# =============================================================================

# Imagenette label index (0-9) -> ImageNet-1k class index (0-999)
# Source: https://github.com/fastai/imagenette
# n01440764=tench(0), n02102040=Eng.springer(217), n02979186=cassette(482),
# n03000684=chain saw(491), n03028079=church(497), n03394916=French horn(566),
# n03417042=garbage truck(569), n03425413=gas pump(571),
# n03445777=golf ball(574), n03888257=parachute(701)
IMAGENETTE_IMAGENET_INDICES: list[int] = [
    0,    # tench
    217,  # English springer
    482,  # cassette player
    491,  # chain saw
    497,  # church
    566,  # French horn
    569,  # garbage truck
    571,  # gas pump
    574,  # golf ball
    701,  # parachute
]


class ImagenetteZeroShotClassifier(nn.Module):
    """
    Wraps any timm ImageNet-1k model to produce 10-class Imagenette logits.

    The backbone is loaded with its full 1000-class pretrained head intact.
    During forward(), we extract only the 10 columns that correspond to
    Imagenette classes (using IMAGENETTE_IMAGENET_INDICES).  This is a
    zero-shot setup -- no fine-tuning is required.

    Args:
        backbone:   A timm model with a 1000-class output head.
        freeze:     If True, all backbone parameters are frozen (no grad).

    Shape:
        - Input:  (B, 3, 224, 224) normalised with ImageNet stats.
        - Output: (B, 10) logits for the 10 Imagenette classes.
    """

    _IDX = torch.tensor(IMAGENETTE_IMAGENET_INDICES, dtype=torch.long)

    def __init__(self, backbone: nn.Module, freeze: bool = True) -> None:
        super().__init__()
        self.backbone = backbone
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
        # Register as buffer so it moves to the right device with .to(device)
        self.register_buffer('_class_idx', self._IDX.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits_1000 = self.backbone(x)         # (B, 1000)
        return logits_1000[:, self._class_idx]  # (B, 10)

    def get_classifier(self) -> nn.Module:
        """Expose the backbone classifier head (for parameter counting)."""
        return self.backbone.get_classifier()


def build_timm_imagenette_classifier(
    arch:   str  = 'resnet18',
    freeze: bool = True,
) -> ImagenetteZeroShotClassifier:
    """
    Generic factory: load any timm ImageNet-1k model and wrap it for
    zero-shot Imagenette inference.

    Args:
        arch:   Any timm model name, e.g. 'resnet18', 'vit_small_patch16_224'.
        freeze: Freeze all backbone parameters (recommended for zero-shot).

    Returns:
        ImagenetteZeroShotClassifier ready for eval.
    """
    backbone = timm.create_model(arch, pretrained=True, num_classes=1000)
    return ImagenetteZeroShotClassifier(backbone, freeze=freeze)


def build_imagenette_classifier(
    arch:   str  = 'resnet18',
    freeze: bool = True,
) -> ImagenetteZeroShotClassifier:
    """
    Load a pretrained ResNet-18 (default) for zero-shot Imagenette.

    Imagenette's 10 classes are a strict subset of ImageNet-1k, so the
    pretrained 1000-class head already knows all of them.  No fine-tuning
    required; typically achieves >= 90% top-1 accuracy.

    Args:
        arch:   timm architecture name (default: 'resnet18').
        freeze: Freeze backbone (True by default for zero-shot mode).

    Returns:
        ImagenetteZeroShotClassifier wrapping the backbone.
    """
    return build_timm_imagenette_classifier(arch=arch, freeze=freeze)


def build_vit_imagenette_classifier(
    freeze: bool = True,
) -> ImagenetteZeroShotClassifier:
    """
    Load a pretrained ViT-Small/16 for zero-shot Imagenette inference.

    Uses vit_small_patch16_224 from timm (trained on ImageNet-1k).
    Achieves >= 90% top-1 on Imagenette without any fine-tuning.

    Args:
        freeze: Freeze backbone parameters.

    Returns:
        ImagenetteZeroShotClassifier wrapping the ViT backbone.
    """
    return build_timm_imagenette_classifier(
        arch='vit_small_patch16_224', freeze=freeze
    )


# =============================================================================
# Training / evaluation helpers
# =============================================================================

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
    Evaluate model on loader and return accuracy / loss metrics.

    Returns:
        dict with keys 'top1_acc', 'loss', 'n_samples'.
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


# =============================================================================
# Output-shape sanity check
# =============================================================================

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
        The actual output torch.Size.

    Raises:
        AssertionError: If the shape does not match (batch_size, num_classes).
    """
    if device is None:
        device = torch.device('cpu')

    model.eval().to(device)
    dummy = torch.randn(batch_size, 3, image_size, image_size, device=device)

    with torch.no_grad():
        out = model(dummy)

    expected = torch.Size([batch_size, num_classes])
    assert out.shape == expected, (
        f"Shape mismatch -- got {out.shape}, expected {expected}"
    )
    print(f'  * Output shape check passed: {tuple(out.shape)}')
    return out.shape


# =============================================================================
# Standalone entry-point (CIFAR-10 head fine-tune demo)
# =============================================================================

if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.data.datasets import get_cifar10_loaders

    # -- Config ----------------------------------------------------------------
    BATCH_SIZE      = 128
    IMAGE_SIZE      = 224
    NUM_CLASSES     = 10
    EPOCHS          = 5
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

    print('[1/4] Building ResNet-18 (pretrained, head frozen) ...')
    model = build_resnet18(
        num_classes=NUM_CLASSES,
        pretrained=True,
        freeze_backbone=True,
    )

    print('[2/4] Checking output shape with a dummy batch ...')
    check_output_shape(
        model,
        batch_size=8,
        num_classes=NUM_CLASSES,
        image_size=IMAGE_SIZE,
        device=DEVICE,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  total params    : {total_params:,}')
    print(f'  trainable params: {trainable:,}  (head only)')
    print()

    print('[3/4] Loading CIFAR-10 at 224 x 224 ...')
    train_loader, val_loader, meta = get_cifar10_loaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_workers=0,
        augment=True,
    )
    print(f'  {meta}')
    print(f'  train batches : {len(train_loader)}')
    print(f'  val   batches : {len(val_loader)}')
    print()

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
    print('Step 2 PASSED.')
