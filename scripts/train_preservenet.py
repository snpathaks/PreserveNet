"""
scripts/train_preservenet.py
-----------------------------
Step 7 launch script: end-to-end PreserveNet training.

Two modes (controlled by --dataset flag):

  cifar10     -- fast debug on CIFAR-10 upscaled to 224x224
                 3 epochs, 40 batches/epoch -> ~2 min on CPU
  imagenette  -- full run on Imagenette 224x224 (default)
                 30 epochs, full dataset

Run from repo root:
    # Quick CIFAR debug (proof of concept, ~2 min CPU):
    python PreserveNet/scripts/train_preservenet.py --dataset cifar10 --epochs 3 --max_train_batches 40 --max_val_batches 10

    # Full Imagenette run:
    python PreserveNet/scripts/train_preservenet.py --dataset imagenette --epochs 30 --retention_rate 0.5
"""

from __future__ import annotations

import argparse
import sys
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.training.trainer import PreserveNetTrainer, TrainerConfig


def parse_args():
    p = argparse.ArgumentParser(description='PreserveNet Step 7 Training')
    p.add_argument('--dataset',            default='imagenette', choices=['cifar10', 'imagenette'])
    p.add_argument('--arch',               default='resnet18')
    p.add_argument('--epochs',             type=int,   default=30)
    p.add_argument('--batch_size',         type=int,   default=32)
    p.add_argument('--lr',                 type=float, default=1e-3)
    p.add_argument('--retention_rate',     type=float, default=0.5)
    p.add_argument('--lambda_task',        type=float, default=1.0)
    p.add_argument('--lambda_budget',      type=float, default=2.0)
    p.add_argument('--lambda_agree',       type=float, default=4.0)
    p.add_argument('--warmup_steps',       type=int,   default=500)
    p.add_argument('--tau_start',          type=float, default=5.0)
    p.add_argument('--tau_end',            type=float, default=0.5)
    p.add_argument('--agree_temp',         type=float, default=2.0)
    p.add_argument('--agree_direction',    default='forward', choices=['forward', 'reverse', 'symmetric'])
    p.add_argument('--base_channels',      type=int,   default=32)
    p.add_argument('--save_dir',           default=str(REPO_ROOT / 'checkpoints'))
    p.add_argument('--data_dir',           default=str(REPO_ROOT.parent / 'data'))
    p.add_argument('--max_train_batches',  type=int,   default=None,
                   help='Cap training batches/epoch (None=full). Use for fast debug.')
    p.add_argument('--max_val_batches',    type=int,   default=None)
    p.add_argument('--log_every',          type=int,   default=20)
    p.add_argument('--device',             default='auto')
    p.add_argument('--resume',             default=None, help='Path to checkpoint to resume from')
    return p.parse_args()


def main():
    args = parse_args()

    cfg = TrainerConfig(
        dataset           = args.dataset,
        data_dir          = args.data_dir,
        arch              = args.arch,
        retention_rate    = args.retention_rate,
        epochs            = args.epochs,
        batch_size        = args.batch_size,
        lr                = args.lr,
        lambda_task       = args.lambda_task,
        lambda_budget     = args.lambda_budget,
        lambda_agree      = args.lambda_agree,
        warmup_steps      = args.warmup_steps,
        tau_start         = args.tau_start,
        tau_end           = args.tau_end,
        agree_temp        = args.agree_temp,
        agree_direction   = args.agree_direction,
        base_channels     = args.base_channels,
        save_dir          = args.save_dir,
        log_every         = args.log_every,
        device            = args.device,
        max_train_batches = args.max_train_batches,
        max_val_batches   = args.max_val_batches,
    )

    trainer = PreserveNetTrainer(cfg)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    history = trainer.train()

    print()
    print('Training complete.')
    print(f'  Final val accuracy : {history[-1].get("val_acc", float("nan"))*100:.2f}%')
    print(f'  Final agree loss   : {history[-1].get("val_agree", float("nan")):.4f}')
    print()
    print('DONE')


if __name__ == '__main__':
    main()
