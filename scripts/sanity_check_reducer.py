"""
scripts/sanity_check_reducer.py
--------------------------------
Step 6 sanity check: feed one image through the *untrained* PatchScorer,
print shape / range assertions, and save a raw-score heatmap PNG.

Run from repo root:
    python PreserveNet/scripts/sanity_check_reducer.py

Expected output (no crash):
    [1/5] Building PatchScorer (untrained) ...
    [2/5] Forward pass shapes ...
          scores  : torch.Size([1, 1, 14, 14])   PASS
          probs   : torch.Size([1, 1, 14, 14])   PASS
          mask    : torch.Size([1, 1, 224, 224])  PASS
          x_out   : torch.Size([1, 3, 224, 224])  PASS
    [3/5] Score range ...
          raw   min/max : <some floats>
          sigmoid range : [<val>, <val>]  (must be in [0,1])  PASS
    [4/5] MaskOperator format-compatibility check ...
          hard mask unique values : {0.0, 1.0}  PASS
          mask fraction kept      : ~0.50  PASS
    [5/5] Saving heatmap to ...  DONE
    ALL CHECKS PASSED. Step 6 sanity check complete.
"""

from __future__ import annotations

import sys
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent   # e:/PreserveNet/PreserveNet
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

# -- try matplotlib; fall back to a PIL-only save if not available ----------
try:
    import matplotlib
    matplotlib.use('Agg')          # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print('  [warn] matplotlib not found -- will save heatmap via PIL only')

try:
    from PIL import Image as PILImage
    import numpy as np
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from src.models.reducer  import PatchScorer, PatchScoreReducer, build_patch_score_reducer
from src.models.operators import MaskOperator

# --------------------------------------------------------------------------
RETENTION_RATE = 0.50
IMAGE_SIZE     = 224
OUT_DIR        = REPO_ROOT / 'notebooks'
OUT_DIR.mkdir(exist_ok=True)
OUT_PATH       = OUT_DIR / 'reducer_sanity_heatmap.png'
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# --------------------------------------------------------------------------

PASS = '\033[92mPASS\033[0m'
FAIL = '\033[91mFAIL\033[0m'

errors = []

def check(condition: bool, msg: str) -> None:
    tag = PASS if condition else FAIL
    print(f'          {msg}  {tag}')
    if not condition:
        errors.append(msg)

# =============================================================================
print('=' * 65)
print('  PreserveNet -- Step 6  |  PatchScorer Sanity Check')
print('=' * 65)
print(f'  device     : {DEVICE}')
print(f'  retain     : {RETENTION_RATE:.0%}')
print(f'  image_size : {IMAGE_SIZE}x{IMAGE_SIZE}')
print()

# --------------- [1/5] build ------------------------------------------------
print('[1/5] Building PatchScorer (untrained) ...')
scorer  = PatchScorer(base_channels=32).to(DEVICE).eval()
reducer = PatchScoreReducer(scorer, retention_rate=RETENTION_RATE, device=DEVICE)
op      = MaskOperator(retention_rate=RETENTION_RATE)

total_params = sum(p.numel() for p in scorer.parameters())
print(f'      PatchScorer  -- {total_params:,} parameters')
print(f'      reducer repr -- {reducer}')
print(f'      operator repr-- {op}')
print()

# --------------- [2/5] shapes -----------------------------------------------
print('[2/5] Forward pass shapes ...')
torch.manual_seed(0)
x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)   # fake single image

with torch.no_grad():
    scores = scorer(x)                                          # (1, 1, 14, 14)
    probs  = torch.sigmoid(scores)                             # (1, 1, 14, 14)
    mask   = op.hard_mask(scores, (IMAGE_SIZE, IMAGE_SIZE))    # (1, 1, 224, 224)
    x_out  = x * mask                                          # (1, 3, 224, 224)

expected_score = torch.Size([1, 1, 14, 14])
expected_img   = torch.Size([1, 3, IMAGE_SIZE, IMAGE_SIZE])
expected_mask  = torch.Size([1, 1, IMAGE_SIZE, IMAGE_SIZE])

check(scores.shape == expected_score,
      f'scores  shape {tuple(scores.shape)} == {tuple(expected_score)}')
check(probs.shape  == expected_score,
      f'probs   shape {tuple(probs.shape)}  == {tuple(expected_score)}')
check(mask.shape   == expected_mask,
      f'mask    shape {tuple(mask.shape)}   == {tuple(expected_mask)}')
check(x_out.shape  == expected_img,
      f'x_out   shape {tuple(x_out.shape)}  == {tuple(expected_img)}')
print()

# --------------- [3/5] range ------------------------------------------------
print('[3/5] Score range ...')
raw_min  = scores.min().item()
raw_max  = scores.max().item()
prob_min = probs.min().item()
prob_max = probs.max().item()

print(f'      raw scores  min/max : {raw_min:.4f} / {raw_max:.4f}')
print(f'      sigmoid     min/max : {prob_min:.6f} / {prob_max:.6f}')

check(prob_min >= 0.0 and prob_min <= 1.0,
      f'sigmoid_min {prob_min:.6f} in [0,1]')
check(prob_max >= 0.0 and prob_max <= 1.0,
      f'sigmoid_max {prob_max:.6f} in [0,1]')
print()

# --------------- [4/5] MaskOperator format-compat check --------------------
print('[4/5] MaskOperator format-compatibility check ...')
unique_vals = mask.unique().cpu().tolist()
frac_kept   = mask.mean().item()

print(f'      mask unique values  : {[round(v,1) for v in unique_vals]}')
print(f'      mask fraction kept  : {frac_kept:.4f}  (target {RETENTION_RATE:.2f})')

check(
    set(round(v, 1) for v in unique_vals).issubset({0.0, 1.0}),
    f'mask values in {{0,1}}'
)
check(
    abs(frac_kept - RETENTION_RATE) < 0.05,
    f'fraction kept {frac_kept:.4f} within 5pp of target {RETENTION_RATE:.2f}'
)

# Also check PatchScoreReducer.__call__ matches
x_red = reducer(x)
check(
    x_red.shape == expected_img,
    f'PatchScoreReducer(x) shape {tuple(x_red.shape)} == {tuple(expected_img)}'
)
print()

# --------------- [5/5] heatmap ----------------------------------------------
print(f'[5/5] Saving heatmap to {OUT_PATH} ...')

# Score heatmap data: squeeze to (14,14)
score_map_np = scores[0, 0].cpu().float().numpy()
prob_map_np  = probs[0, 0].cpu().float().numpy()
mask_map_np  = mask[0, 0].cpu().float().numpy()

# Original image (random noise -- but we also try to load sample_image.png)
sample_img_path = REPO_ROOT / 'sample_image.png'
if sample_img_path.exists() and HAS_PIL:
    import numpy as np
    from torchvision import transforms
    _tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    pil_img  = PILImage.open(sample_img_path).convert('RGB')
    x        = _tf(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        scores       = scorer(x)
        probs        = torch.sigmoid(scores)
        mask_out     = op.hard_mask(scores, (IMAGE_SIZE, IMAGE_SIZE))
    score_map_np = scores[0, 0].cpu().float().numpy()
    prob_map_np  = probs[0, 0].cpu().float().numpy()
    mask_map_np  = mask_out[0, 0].cpu().float().numpy()
    x_vis_np     = x[0].cpu().permute(1, 2, 0).numpy().clip(0, 1)
    used_real    = True
    print('      (used sample_image.png for visualisation)')
else:
    import numpy as np
    x_vis_np  = (torch.randn(IMAGE_SIZE, IMAGE_SIZE, 3) * 0.2 + 0.5).clamp(0,1).numpy()
    used_real = False
    print('      (no sample_image.png found -- using random noise image)')

if HAS_MPL:
    fig = plt.figure(figsize=(14, 4), facecolor='#0f0f1a')
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.05)

    def _ax(idx, title):
        ax = fig.add_subplot(gs[idx])
        ax.set_title(title, color='white', fontsize=9, pad=4)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#555')
        return ax

    # Panel 0 — original
    ax0 = _ax(0, 'Original image')
    ax0.imshow(x_vis_np)

    # Panel 1 — raw logit scores
    ax1 = _ax(1, f'Raw scores (14×14)\nmin={score_map_np.min():.3f}  max={score_map_np.max():.3f}')
    im1 = ax1.imshow(score_map_np, cmap='plasma', interpolation='nearest')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04,
                 label='logit').ax.yaxis.label.set_color('white')
    im1.colorbar.ax.tick_params(colors='white', labelsize=7)

    # Panel 2 — sigmoid probabilities
    ax2 = _ax(2, f'Sigmoid probs (14×14)\nmin={prob_map_np.min():.3f}  max={prob_map_np.max():.3f}')
    im2 = ax2.imshow(prob_map_np, cmap='viridis', vmin=0, vmax=1, interpolation='nearest')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04,
                 label='prob').ax.yaxis.label.set_color('white')
    im2.colorbar.ax.tick_params(colors='white', labelsize=7)

    # Panel 3 — binary mask overlaid on image
    ax3 = _ax(3, f'Binary mask @ {RETENTION_RATE:.0%} kept\n(16×16 patch granularity)')
    ax3.imshow(x_vis_np)
    # overlay dropped patches as semi-transparent red
    dropped = 1.0 - mask_map_np
    red_overlay = np.zeros((*dropped.shape, 4), dtype=np.float32)
    red_overlay[..., 0] = 1.0   # R
    red_overlay[..., 3] = dropped * 0.65  # alpha
    ax3.imshow(red_overlay, interpolation='nearest')

    # Title
    fig.suptitle(
        'PatchScorer Sanity Check  |  Untrained Network  |  Step 6',
        color='white', fontsize=11, y=1.01
    )
    fig.patch.set_facecolor('#0f0f1a')
    plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f'      Saved: {OUT_PATH}  DONE')

elif HAS_PIL:
    import numpy as np
    # Simple PIL fallback: save the raw score heatmap as a greyscale PNG
    norm = (prob_map_np - prob_map_np.min()) / (prob_map_np.ptp() + 1e-8)
    grey = (norm * 255).astype(np.uint8)
    PILImage.fromarray(grey).save(str(OUT_PATH))
    print(f'      Saved greyscale heatmap: {OUT_PATH}  DONE')
else:
    print('      [warn] Neither matplotlib nor PIL available -- heatmap skipped.')

print()

# --------------- summary ----------------------------------------------------
print('=' * 65)
if errors:
    print(f'  {len(errors)} CHECK(S) FAILED:')
    for e in errors:
        print(f'    - {e}')
    sys.exit(1)
else:
    print('  ALL CHECKS PASSED.')
    print('  Step 6 sanity check complete.')
print('=' * 65)
print()
print('DONE')
