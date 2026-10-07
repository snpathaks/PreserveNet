"""
src/training/trainer.py
-----------------------
PreserveNet end-to-end trainer -- Step 7.

Trains PatchScorer to select patches that:
  1. Preserve the classifier prediction (L_agreement -- the main contribution)
  2. Keep the classifier correct (L_task -- secondary, often saturated)
  3. Stay within a patch budget (L_budget -- sparsity constraint)

Gumbel-Softmax temperature is annealed from tau_start -> tau_end over
training; loss weights (lambda_agree, lambda_budget) are warmed up
linearly so the scorer has time to learn basic structure first.

Quick debug run:  CIFAR-10 @ 32x32  (tiny dataset, fast iteration)
Full run:         Imagenette @ 224x224

Usage
-----
    from src.training.trainer import PreserveNetTrainer, TrainerConfig

    cfg = TrainerConfig(
        dataset='imagenette',
        data_dir='../../data',
        epochs=30,
        batch_size=32,
        retention_rate=0.5,
    )
    trainer = PreserveNetTrainer(cfg)
    trainer.train()
"""

from __future__ import annotations

import os
import sys
import time
import pathlib
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.reducer    import PatchScorer
from src.models.operators  import MaskOperator
from src.models.pipeline   import PreserveNetPipeline
from src.losses.budget     import L1BudgetLoss
from src.losses.agreement  import AgreementLoss, TaskLoss
from src.training.schedulers import GumbelTauScheduler, LambdaWarmup


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainerConfig:
    """
    All hyper-parameters for one PreserveNet training run.

    Attributes
    ----------
    dataset:         'cifar10' or 'imagenette'
    data_dir:        Path to the data root directory.
    arch:            Classifier backbone (timm model name).
    retention_rate:  Target fraction of 16x16 patches to keep (0 < r <= 1).
    epochs:          Training epochs.
    batch_size:      Mini-batch size.
    lr:              PatchScorer learning rate.
    weight_decay:    AdamW weight decay.
    lambda_task:     Weight for L_task (cross-entropy on masked image).
    lambda_budget:   Final weight for L_budget (after warmup).
    lambda_agree:    Final weight for L_agreement (after warmup).
    warmup_steps:    Steps to linearly ramp lambda_budget and lambda_agree.
    tau_start:       Initial Gumbel temperature.
    tau_end:         Final Gumbel temperature.
    tau_steps:       Steps over which tau is annealed.
    agree_temp:      Softmax temperature inside AgreementLoss.
    agree_direction: 'forward', 'reverse', or 'symmetric'.
    save_dir:        Directory to save checkpoints and logs.
    log_every:       Print a log line every N batches.
    eval_every:      Evaluate on val set every N epochs.
    device:          'cuda', 'cpu', or 'auto'.
    base_channels:   PatchScorer width (32 = lightweight).
    max_train_batches: Cap training batches per epoch (None = full). Useful for quick debug.
    max_val_batches:   Cap val batches per epoch (None = full).
    """
    dataset:          str   = 'imagenette'
    data_dir:         str   = '../../data'
    arch:             str   = 'resnet18'
    retention_rate:   float = 0.50
    epochs:           int   = 30
    batch_size:       int   = 32
    lr:               float = 1e-3
    weight_decay:     float = 1e-4
    lambda_task:      float = 1.0
    lambda_budget:    float = 2.0
    lambda_agree:     float = 4.0
    warmup_steps:     int   = 500
    tau_start:        float = 5.0
    tau_end:          float = 0.5
    tau_steps:        int   = 0          # 0 = auto: set to epochs * steps_per_epoch
    agree_temp:       float = 2.0
    agree_direction:  str   = 'forward'
    save_dir:         str   = 'checkpoints'
    log_every:        int   = 20
    eval_every:       int   = 1
    device:           str   = 'auto'
    base_channels:    int   = 32
    max_train_batches: Optional[int] = None
    max_val_batches:   Optional[int] = None


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class PreserveNetTrainer:
    """
    Full training harness for PreserveNet PatchScorer.

    Args
    ----
    cfg:  TrainerConfig instance.
    """

    def __init__(self, cfg: TrainerConfig) -> None:
        self.cfg = cfg

        # ── Device ──────────────────────────────────────────────────────────
        if cfg.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(cfg.device)

        # ── Data ────────────────────────────────────────────────────────────
        self.train_loader, self.val_loader, self.meta = self._build_loaders()

        # ── Models ──────────────────────────────────────────────────────────
        self.scorer    = PatchScorer(base_channels=cfg.base_channels).to(self.device)
        self.classifier = self._build_classifier()
        self.operator  = MaskOperator(
            retention_rate=cfg.retention_rate,
            patch_size=16,
        )
        self.pipeline = PreserveNetPipeline(
            scorer=self.scorer,
            classifier=self.classifier,
            operator=self.operator,
        ).to(self.device)

        # ── Losses ──────────────────────────────────────────────────────────
        self.loss_task   = TaskLoss(label_smoothing=0.1)
        self.loss_budget = L1BudgetLoss(target_rate=cfg.retention_rate)
        self.loss_agree  = AgreementLoss(
            temperature=cfg.agree_temp,
            direction=cfg.agree_direction,
        )

        # ── Optimiser ───────────────────────────────────────────────────────
        # Only PatchScorer has requires_grad=True
        self.optimiser = torch.optim.AdamW(
            self.scorer.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        # ── LR scheduler: cosine over epochs ────────────────────────────────
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimiser, T_max=cfg.epochs, eta_min=cfg.lr * 0.01
        )

        # ── Gumbel + lambda schedules ────────────────────────────────────────
        steps_per_epoch = len(self.train_loader)
        tau_steps = cfg.tau_steps if cfg.tau_steps > 0 else cfg.epochs * steps_per_epoch
        self.tau_sched      = GumbelTauScheduler(
            tau_start=cfg.tau_start, tau_end=cfg.tau_end, total_steps=tau_steps
        )
        self.lam_budget_sched = LambdaWarmup(
            warmup_steps=cfg.warmup_steps, target=cfg.lambda_budget
        )
        self.lam_agree_sched  = LambdaWarmup(
            warmup_steps=cfg.warmup_steps, target=cfg.lambda_agree
        )

        # ── Save dir ────────────────────────────────────────────────────────
        self.save_dir = pathlib.Path(cfg.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # ── History ─────────────────────────────────────────────────────────
        self.history: list[dict] = []

    # ── Data factories ──────────────────────────────────────────────────────

    def _build_loaders(self):
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
        from src.data.datasets import get_cifar10_loaders, get_imagenette_loaders

        data_dir = pathlib.Path(self.cfg.data_dir).resolve()
        bs       = self.cfg.batch_size

        if self.cfg.dataset == 'cifar10':
            return get_cifar10_loaders(
                data_dir=data_dir,
                batch_size=bs,
                image_size=224,   # upscale CIFAR so ImageNet backbone works
                augment=True,
                download=True,
            )
        else:
            return get_imagenette_loaders(
                data_dir=data_dir,
                batch_size=bs,
                augment=True,
                source='fastai',
            )

    def _build_classifier(self) -> nn.Module:
        from src.models.classifier import build_imagenette_classifier, build_resnet18
        if self.cfg.dataset == 'cifar10':
            # Use the pretrained ImageNet head; treat CIFAR-10 labels as
            # a known 10-class subset by re-using ImagenetteZeroShotClassifier
            # logic but mapping CIFAR label indices through the full 1k head.
            # Simpler: just use build_resnet18 with num_classes=10 so we can
            # compute CE.  Freeze backbone, train head only to get a decent clf.
            clf = build_resnet18(num_classes=10, pretrained=True, freeze_backbone=True)
        else:
            clf = build_imagenette_classifier(arch=self.cfg.arch, freeze=True)
        return clf.to(self.device).eval()

    # ── Training loop ───────────────────────────────────────────────────────

    def train(self) -> list[dict]:
        """Run the full training loop and return the epoch history list."""
        cfg = self.cfg
        print('=' * 65)
        print(f'  PreserveNet Trainer -- Step 7')
        print(f'  dataset         : {cfg.dataset}')
        print(f'  backbone        : {cfg.arch}')
        print(f'  retention_rate  : {cfg.retention_rate:.0%}')
        print(f'  epochs          : {cfg.epochs}')
        print(f'  batch_size      : {cfg.batch_size}')
        print(f'  device          : {self.device}')
        print(f'  scorer params   : {sum(p.numel() for p in self.scorer.parameters()):,}')
        print(f'  agree_temp      : {cfg.agree_temp}')
        print(f'  lambda_agree    : {cfg.lambda_agree}  (warmed up over {cfg.warmup_steps} steps)')
        print(f'  lambda_budget   : {cfg.lambda_budget}  (warmed up over {cfg.warmup_steps} steps)')
        print(f'  tau schedule    : {cfg.tau_start} -> {cfg.tau_end}')
        print('=' * 65)

        best_agree = float('inf')

        for epoch in range(cfg.epochs):
            t_ep   = time.perf_counter()
            train_metrics = self._train_epoch(epoch)
            elapsed = time.perf_counter() - t_ep

            val_metrics = {}
            if (epoch + 1) % cfg.eval_every == 0:
                val_metrics = self._val_epoch(epoch)

            row = {
                'epoch': epoch + 1,
                **train_metrics,
                **{f'val_{k}': v for k, v in val_metrics.items()},
                'elapsed_s': elapsed,
                'tau': self.tau_sched.current_tau,
                'lam_agree': self.lam_agree_sched.current_value,
                'lam_budget': self.lam_budget_sched.current_value,
            }
            self.history.append(row)
            self._print_row(row)

            self.lr_scheduler.step()

            # Save best checkpoint by lowest agreement loss on val
            if 'val_agree' in val_metrics:
                if val_metrics['val_agree'] < best_agree:
                    best_agree = val_metrics['val_agree']
                    self._save_checkpoint('best.pt', epoch)

            # Always save latest
            self._save_checkpoint('latest.pt', epoch)

        print()
        print(f'  Training complete.  Best val agree loss: {best_agree:.4f}')
        print(f'  Checkpoint saved to: {self.save_dir}')
        return self.history

    # ── Single epoch ────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> dict:
        self.pipeline.train()
        self.scorer.train()
        # Classifier stays frozen
        self.classifier.eval()

        total_task = total_budget = total_agree = total_loss = 0.0
        n_correct  = n_total = 0
        n_batches  = 0

        max_batches = self.cfg.max_train_batches

        for batch_idx, (imgs, labels) in enumerate(self.train_loader):
            if max_batches and batch_idx >= max_batches:
                break

            imgs   = imgs.to(self.device)
            labels = labels.to(self.device)

            # ── Schedule values ─────────────────────────────────────────────
            tau        = self.tau_sched.step()
            lam_budget = self.lam_budget_sched.step()
            lam_agree  = self.lam_agree_sched.step()

            # ── Forward ─────────────────────────────────────────────────────
            out = self.pipeline(imgs, tau=tau, hard=True)

            # ── Losses ──────────────────────────────────────────────────────
            l_task   = self.loss_task(out.masked_logits, labels)
            l_budget = self.loss_budget(out.scores)
            l_agree  = self.loss_agree(out.full_logits, out.masked_logits)

            loss = (
                self.cfg.lambda_task   * l_task
                + lam_budget           * l_budget
                + lam_agree            * l_agree
            )

            # ── Backward ────────────────────────────────────────────────────
            self.optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.scorer.parameters(), max_norm=1.0)
            self.optimiser.step()

            # ── Metrics ─────────────────────────────────────────────────────
            bs              = imgs.size(0)
            total_task     += l_task.item()   * bs
            total_budget   += l_budget.item() * bs
            total_agree    += l_agree.item()  * bs
            total_loss     += loss.item()     * bs
            preds           = out.masked_logits.argmax(dim=1)
            n_correct      += (preds == labels).sum().item()
            n_total        += bs
            n_batches      += 1

            if (batch_idx + 1) % self.cfg.log_every == 0:
                print(
                    f'    [{epoch+1}/{self.cfg.epochs}] '
                    f'batch {batch_idx+1}  '
                    f'loss={loss.item():.4f}  '
                    f'task={l_task.item():.4f}  '
                    f'budget={l_budget.item():.4f}  '
                    f'agree={l_agree.item():.4f}  '
                    f'tau={tau:.3f}'
                )

        N = n_total or 1
        return {
            'task':   total_task   / N,
            'budget': total_budget / N,
            'agree':  total_agree  / N,
            'loss':   total_loss   / N,
            'acc':    n_correct    / N,
        }

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> dict:
        self.pipeline.eval()
        self.scorer.eval()
        self.classifier.eval()

        total_task = total_budget = total_agree = 0.0
        n_correct = n_total = 0
        max_batches = self.cfg.max_val_batches

        for batch_idx, (imgs, labels) in enumerate(self.val_loader):
            if max_batches and batch_idx >= max_batches:
                break

            imgs   = imgs.to(self.device)
            labels = labels.to(self.device)

            # Hard mask at inference: no Gumbel noise
            scores       = self.scorer(imgs)
            mask         = self.operator.hard_mask(scores, imgs.shape[-2:])
            x_masked     = imgs * mask

            full_logits   = self.classifier(imgs)
            masked_logits = self.classifier(x_masked)

            l_task   = self.loss_task(masked_logits, labels)
            l_budget = self.loss_budget(scores)
            l_agree  = self.loss_agree(full_logits, masked_logits)

            bs             = imgs.size(0)
            total_task    += l_task.item()   * bs
            total_budget  += l_budget.item() * bs
            total_agree   += l_agree.item()  * bs
            preds          = masked_logits.argmax(dim=1)
            n_correct     += (preds == labels).sum().item()
            n_total       += bs

        N = n_total or 1
        return {
            'val_task':   total_task   / N,
            'val_budget': total_budget / N,
            'val_agree':  total_agree  / N,
            'val_acc':    n_correct    / N,
        }

    # ── Utilities ───────────────────────────────────────────────────────────

    def _print_row(self, row: dict) -> None:
        ep   = row['epoch']
        acc  = row.get('acc',       0.0)
        vacc = row.get('val_val_acc', row.get('val_acc', float('nan')))
        agr  = row.get('agree',     0.0)
        vagr = row.get('val_val_agree', row.get('val_agree', float('nan')))
        bud  = row.get('budget',    0.0)
        tau  = row.get('tau',       0.0)
        el   = row.get('elapsed_s', 0.0)

        print(
            f'  Epoch {ep:03d}  '
            f'acc={acc*100:5.2f}%  val_acc={vacc*100:5.2f}%  '
            f'agree={agr:.4f}  val_agree={vagr:.4f}  '
            f'budget={bud:.4f}  tau={tau:.3f}  '
            f'({el:.1f}s)'
        )

    def _save_checkpoint(self, name: str, epoch: int) -> None:
        path = self.save_dir / name
        torch.save(
            {
                'epoch':       epoch,
                'scorer':      self.scorer.state_dict(),
                'optimiser':   self.optimiser.state_dict(),
                'tau_sched':   self.tau_sched.state_dict(),
                'lam_agree':   self.lam_agree_sched.state_dict(),
                'lam_budget':  self.lam_budget_sched.state_dict(),
                'cfg':         self.cfg,
                'history':     self.history,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        """Resume from a saved checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.scorer.load_state_dict(ckpt['scorer'])
        self.optimiser.load_state_dict(ckpt['optimiser'])
        self.tau_sched.load_state_dict(ckpt['tau_sched'])
        self.lam_agree_sched.load_state_dict(ckpt['lam_agree'])
        self.lam_budget_sched.load_state_dict(ckpt['lam_budget'])
        self.history = ckpt.get('history', [])
        print(f'  Resumed from epoch {ckpt["epoch"]+1}')


__all__ = ["PreserveNetTrainer", "TrainerConfig"]
