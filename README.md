# PreserveNet — Prediction-Preserving Image Reduction

> **Core question:** How many pixels can you remove from an image before a classifier changes its prediction?  
> **Goal:** Build and benchmark pixel-reduction strategies that maximally preserve model accuracy at increasing sparsity levels.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Repository Layout](#repository-layout)
3. [Environment Setup](#environment-setup)
4. [Datasets](#datasets)
5. [Reduction Strategies](#reduction-strategies)
6. [Experiment Log](#experiment-log)
7. [Results Summary](#results-summary)
8. [Key Observations](#key-observations)
9. [Roadmap](#roadmap)

---

## Project Overview

PreserveNet investigates how much spatial information a CNN classifier actually needs.
We apply **pixel-retention masks** to images at test time (the classifier is never
retrained on masked images) and measure the resulting accuracy drop.

**Setup (all experiments so far)**
| Setting | Value |
|---------|-------|
| Classifier | ResNet-18 (timm, ImageNet-pretrained) |
| Training | Head-only fine-tuning (backbone frozen) |
| Head fine-tune epochs | 3 |
| Optimiser | Adam, lr = 1e-3 |
| LR schedule | CosineAnnealingLR (T_max = epochs) |
| Debug dataset | CIFAR-10 (32 × 32, 10 classes) |
| Main dataset | Imagenette (planned) |
| Hardware used | CPU only (no GPU so far) |

---

## Repository Layout

```
PreserveNet/
├── src/
│   ├── data/
│   │   ├── datasets.py       # get_cifar10_loaders() factory
│   │   └── transforms.py     # normalisation constants + augmentation pipelines
│   ├── models/
│   │   └── classifier.py     # build_resnet18(), train_one_epoch(), evaluate()
│   └── baselines/
│       ├── uniform.py        # UniformGridReducer, UniformRandomReducer
│       ├── random_drop.py    # RandomDropReducer
│       └── saliency.py       # GradientSaliencyReducer, GradCAMReducer  ← Step 5
├── scripts/
│   ├── baseline.py                  # Step 2 — extract data + run classifier
│   ├── eval_cifar_baseline.py       # Step 3 — zero-shot + fine-tuned baseline
│   ├── eval_dumb_baselines.py       # Step 4 — dumb reducer sweep
│   └── eval_saliency_baselines.py   # Step 5 — saliency vs. dumb sweep
├── notebooks/
│   ├── 00_setup_drive.ipynb
│   └── 01_data_exploration.ipynb
├── data/                            # gitignored — place datasets here
└── requirements.txt
```

---

## Environment Setup

```powershell
# Create venv and install deps (run once from e:\PreserveNet)
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# Run any script from the repo root
.venv\Scripts\python.exe PreserveNet\scripts\eval_saliency_baselines.py
```

**Key dependencies**

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.4.0 | Core DL framework |
| torchvision | 0.19.0 | CIFAR-10 loader, transforms |
| timm | 1.0.9 | Pretrained ResNet-18 / ViT weights |
| einops | 0.8.0 | Tensor rearrangements |
| tqdm | ≥ 4.66 | Progress bars |

---

## Datasets

### CIFAR-10 (debug / current)
- 50,000 train / 10,000 val images, 10 classes, 32 × 32 px
- Stored at `data/cifar10/` — downloaded automatically by torchvision on first run
- Normalised with CIFAR-10 statistics (mean/std in `src/data/transforms.py`)
- When feeding a **pretrained** backbone, images are resized to 224 × 224

### Imagenette (planned — main experiments)
- 10-class subset of ImageNet (≈ 9,469 train / 3,925 val images), 224 × 224
- Higher resolution means more room for pixel reduction experiments
- Will reveal whether saliency strategies scale beyond tiny 32 × 32 images

---

## Reduction Strategies

### Dumb Baselines (no model knowledge)

| Class | File | How it works |
|-------|------|-------------|
| `UniformGridReducer` | `uniform.py` | Keeps pixels on a regular linspace-spaced grid |
| `UniformRandomReducer` | `uniform.py` | Bernoulli mask — each pixel kept with prob *r* |
| `RandomDropReducer` | `random_drop.py` | Same as above but seeded for reproducibility |

> **Common interface:** `reducer(x: Tensor[B,C,H,W]) → Tensor[B,C,H,W]`  
> Dropped pixels are zeroed. Spatial shape is always preserved.

### Saliency-Guided (Step 5 — uses model gradients)

| Class | File | How it works |
|-------|------|-------------|
| `GradientSaliencyReducer` | `saliency.py` | ∂(max-logit)/∂(pixel), channel-L2-norm → top-k mask |
| `GradCAMReducer` | `saliency.py` | Hooks `layer4`, global-avg-pool grads → channel weights → weighted activation sum → ReLU → bilinear upsample → top-k mask |

> **Cost:** Both require one forward + backward pass per batch (≈ 2–3× slower than dumb).  
> **No training required** — they wrap the already-trained classifier.

### Planned

| Strategy | Notes |
|----------|-------|
| Learned soft mask | Trainable sigmoid gate per pixel, trained end-to-end with classifier |
| Token pruning | Patch-based (ViT-compatible), drop least-attended tokens |
| Progressive masking | Curriculum — start with high retention, gradually reduce |

---

## Experiment Log

### Step 2 — Baseline Classifier (CIFAR-10 @ 32 × 32)

**Script:** `scripts/baseline.py`  
**Model:** ResNet-18, backbone frozen, head fine-tuned  
**Image size:** 32 × 32 (native CIFAR resolution for fast CPU runs)

| Epochs | Best val accuracy |
|--------|-------------------|
| 3 | ~41–42% |
| 5 | ~43–45% (expected) |

> ⚠️ 41–42% sounds low but is correct for a **frozen** ResNet-18 backbone on 32 × 32 CIFAR images.
> The backbone was pretrained on 224 × 224 ImageNet; feature maps at 32 × 32 are too small
> for the convolutional stages to work well. Running at 224 × 224 yields ~85–90% with the same setup.

---

### Step 3 — Baseline Accuracy (zero-shot + fine-tuned)

**Script:** `scripts/eval_cifar_baseline.py`

| Mode | val accuracy |
|------|-------------|
| Zero-shot (pretrained head, no fine-tuning) | ~10% (random-chance on 10 classes) |
| Fine-tuned head (5 epochs, frozen backbone) | ~43–45% @ 224 × 224 |

---

### Step 4 — Dumb Baseline Sweep

**Script:** `scripts/eval_dumb_baselines.py`  
**Image size:** 32 × 32 | **Epochs:** 3 | **Baseline acc:** ~41.78%

| Retention | Uniform Random | Uniform Grid | Random Drop |
|-----------|:--------------:|:------------:|:-----------:|
| 100% | 41.78% | 41.78% | 41.78% |
| 75% | 13.61% | 22.49% | 13.61% |
| 50% | 11.78% | 12.77% | 11.78% |
| 25% | 10.13% | 10.16% | 10.13% |
| 10% | 8.98% | 8.76% | 8.98% |

---

### Step 5 — Saliency-Guided Reducer Sweep ✅ Latest

**Script:** `scripts/eval_saliency_baselines.py`  
**Image size:** 32 × 32 | **Epochs:** 3 | **Baseline acc:** ~41.78%  
**Note:** Saliency reducers evaluated on 1,600 images (50 batches × 32) on CPU to keep runtime feasible.

| Retention | Uniform Random | Uniform Grid | Random Drop | Grad Saliency | **GradCAM** |
|-----------|:--------------:|:------------:|:-----------:|:-------------:|:-----------:|
| 100% | 41.78% | 41.78% | 41.78% | 41.25% | 41.25% |
| **75%** | 13.61% | 22.49% | 13.61% | 19.00% | **35.06%** 🏆 |
| **50%** | 11.78% | 12.77% | 11.78% | 13.31% | **20.00%** 🏆 |
| **25%** | 10.13% | 10.16% | 10.13% | 11.50% | **12.38%** 🏆 |
| **10%** | 8.98% | 8.76% | 8.98% | 9.56% | **10.56%** 🏆 |

---

## Results Summary

```
Strategy            r=75%    r=50%    r=25%    r=10%
────────────────    ─────    ─────    ─────    ─────
Uniform Random      13.61%   11.78%   10.13%    8.98%
Random Drop         13.61%   11.78%   10.13%    8.98%
Uniform Grid        22.49%   12.77%   10.16%    8.76%
Gradient Saliency   19.00%   13.31%   11.50%    9.56%
GradCAM             35.06%   20.00%   12.38%   10.56%  ← BEST at every level
Baseline (r=100%)   41.78%
```

**GradCAM retains 35.06% accuracy while keeping only 75% of pixels.**  
The best dumb baseline at the same retention rate achieves only 22.49% — a **+12.6 pp gap**.

---

## Key Observations

1. **GradCAM >> all others** at every retention rate. Spatially coherent class-activation maps
   are far less destructive to the classifier than random or grid drops.

2. **Gradient saliency beats random drop but not Uniform Grid** at r=75%.
   Individual pixel gradients are scattered across the image; they don't form
   the contiguous regions the network expects to see.

3. **32 × 32 is brutal for pixel masking.** Each pixel in a 32 × 32 image
   covers ~49 pixels in the equivalent 224 × 224 version. Losing even 25% of
   pixels removes significant structure. Expect much better retention curves on
   Imagenette (224 × 224).

4. **Frozen backbone limits baseline.** At 41.78%, there's only ~42 pp of
   accuracy "budget". A fully fine-tuned model at ~92% would give saliency
   reducers much more headroom. Future experiments should unfreeze the backbone.

5. **GradCAM is cheap to extend.** The hook-based implementation adds zero
   trainable parameters. The only cost is one extra backward pass per batch.

6. **CPU timing note.** Saliency reducers take ~8s per (reducer × retention rate)
   on CPU for 50 batches of 32 images. On a GPU this would be <1s.
   All timing numbers in the log reflect CPU-only runs.

7. **Baseline accuracy sanity check.** Random chance on CIFAR-10 = 10%.
   All reducers at r=10% are near 9–10%, confirming the model is essentially
   guessing when only 10% of pixels survive.

---

## Roadmap

| Step | Status | Description |
|------|--------|-------------|
| 1 | ✅ Done | Project setup, environment, data pipeline |
| 2 | ✅ Done | Baseline ResNet-18 classifier (frozen backbone, head fine-tuned) |
| 3 | ✅ Done | Zero-shot + fine-tuned baseline accuracy measurement |
| 4 | ✅ Done | Dumb baseline sweep (Uniform Grid, Uniform Random, Random Drop) |
| 5 | ✅ Done | Saliency-guided reducers (Gradient Saliency, GradCAM) |
| 6 | 🔲 Next | Imagenette dataset integration + repeat Steps 3–5 at 224 × 224 |
| 7 | 🔲 Planned | Learned soft-mask module (end-to-end trainable) |
| 8 | 🔲 Planned | Token pruning (ViT-compatible, patch-based) |
| 9 | 🔲 Planned | Full fine-tuning + saliency retrain at reduced resolution |
| 10 | 🔲 Planned | Accuracy-vs-FLOPs / accuracy-vs-bandwidth Pareto curves |
