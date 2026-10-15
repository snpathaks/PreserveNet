"""
scripts/export_for_demo.py
--------------------------
Exports trained PreserveNet PatchScorer reducer weights from training checkpoints
into standalone, lightweight artifacts (PyTorch state_dict .pt and safetensors)
and optionally uploads them to the Hugging Face Hub.

Usage:
    # 1. Local export only (saves to app/weights/):
    python scripts/export_for_demo.py

    # 2. Export and upload to Hugging Face Hub:
    python scripts/export_for_demo.py --repo_id <username>/preservenet-reducer --push_to_hub --token <hf_token>

    # 3. Using HF_TOKEN environment variable:
    set HF_TOKEN=hf_...
    python scripts/export_for_demo.py --repo_id <username>/preservenet-reducer --push_to_hub
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Add possible repo paths to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
if (REPO_ROOT / 'PreserveNet').exists():
    sys.path.insert(0, str(REPO_ROOT / 'PreserveNet'))

import torch
import torch.nn as nn

try:
    import safetensors.torch
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

try:
    from huggingface_hub import HfApi, create_repo
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False


# =============================================================================
# Standalone PatchScorer Definition (Guarantees zero-dependency loading)
# =============================================================================

def _conv_bn_relu(
    in_ch: int,
    out_ch: int,
    kernel: int = 3,
    stride: int = 1,
    padding: int = 1,
) -> nn.Sequential:
    """Conv -> BatchNorm -> ReLU block."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class PatchScorer(nn.Module):
    """
    Lightweight CNN that maps a (B, 3, 224, 224) image to a (B, 1, 14, 14) score map.
    Each of the 196 spatial positions receives a scalar logit importance score.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        b = base_channels
        self.stem = _conv_bn_relu(in_channels, b, stride=2)       # 224 -> 112
        self.block1 = _conv_bn_relu(b, b * 2, stride=2)           # 112 -> 56
        self.block2 = _conv_bn_relu(b * 2, b * 2, stride=2)       # 56  -> 28
        self.block3 = _conv_bn_relu(b * 2, b, stride=2)           # 28  -> 14
        self.head = nn.Conv2d(b, 1, kernel_size=1)                # 14  -> 14

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        return self.head(h)


# =============================================================================
# Model Card Generator
# =============================================================================

def generate_model_card(repo_id: str) -> str:
    return f"""---
license: apache-2.0
tags:
- vision
- image-reduction
- explainability
- saliency
- pytorch
- safetensors
pipeline_tag: feature-extraction
---

# PreserveNet — Dynamic Image Reduction Scorer

PreserveNet probes spatial redundancy in computer vision models by identifying the minimal subset of 16×16 patches required to preserve a classifier's prediction.

This repository hosts the trained **PatchScorer** CNN weights evaluated on ImageNet / Imagenette ($224 \\times 224$ px resolution, $14 \\times 14 = 196$ patches).

## Model Architecture
- **Input**: `(B, 3, 224, 224)` normalized RGB image
- **Output**: `(B, 1, 14, 14)` raw logit patch importance scores
- **Number of Parameters**: ~110K parameters (ultra-lightweight, <0.5MB)
- **Patch Granularity**: $16 \\times 16$ px (196 total patches)

## Quick Usage

```python
import torch
from huggingface_hub import hf_hub_download

# Download weights
weights_path = hf_hub_download(repo_id="{repo_id}", filename="preservenet_reducer.pt")
state_dict = torch.load(weights_path, map_location="cpu")

# Load model
from app.inference import PatchScorer
scorer = PatchScorer()
scorer.load_state_dict(state_dict)
scorer.eval()

# Forward pass
x = torch.randn(1, 3, 224, 224)
scores = scorer(x)  # shape: (1, 1, 14, 14)
```

## Citation & Authors
- **Repository**: [PreserveNet on GitHub](https://github.com/snpathaks/PreserveNet)
"""


# =============================================================================
# Export Logic
# =============================================================================

def find_default_checkpoint() -> Path:
    candidates = [
        REPO_ROOT / 'checkpoints' / 'best.pt',
        REPO_ROOT / 'PreserveNet' / 'checkpoints' / 'best.pt',
        REPO_ROOT / 'checkpoints' / 'latest.pt',
        REPO_ROOT / 'PreserveNet' / 'checkpoints' / 'latest.pt',
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError("Could not find a valid checkpoint in checkpoints/best.pt")


def export_weights(
    checkpoint_path: Path,
    output_dir: Path,
    repo_id: str | None = None,
    push_to_hub: bool = False,
    token: str | None = None,
    private: bool = False,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading checkpoint from: {checkpoint_path}")
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except Exception as exc:
        print(f"[-] Standard torch.load failed ({exc}), attempting with fallback...")
        ckpt = torch.load(checkpoint_path, map_location='cpu')

    if isinstance(ckpt, dict) and 'scorer' in ckpt:
        state_dict = ckpt['scorer']
        epoch = ckpt.get('epoch', None)
        print(f"[+] Found 'scorer' state_dict with {len(state_dict)} tensors (epoch {epoch})")
    elif isinstance(ckpt, dict) and 'stem.0.weight' in ckpt:
        state_dict = ckpt
        print(f"[+] Loaded raw state_dict with {len(state_dict)} tensors")
    else:
        raise KeyError("Could not find scorer state_dict in checkpoint")

    # Sanity check with PatchScorer
    scorer = PatchScorer()
    scorer.load_state_dict(state_dict)
    scorer.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = scorer(dummy_input)
    assert out.shape == (1, 1, 14, 14), f"Unexpected output shape: {out.shape}"
    print(f"[+] Verification passed: input (1, 3, 224, 224) -> score map {tuple(out.shape)}")

    # 1. Save PyTorch state_dict (.pt)
    pt_path = output_dir / 'preservenet_reducer.pt'
    torch.save(state_dict, pt_path)
    print(f"[+] Saved PyTorch weights: {pt_path} ({pt_path.stat().st_size / 1024:.1f} KB)")

    # 2. Save Safetensors
    st_path = output_dir / 'preservenet_reducer.safetensors'
    if HAS_SAFETENSORS:
        safetensors.torch.save_file(state_dict, str(st_path))
        print(f"[+] Saved Safetensors weights: {st_path} ({st_path.stat().st_size / 1024:.1f} KB)")

    # 3. Save config.json
    config_data = {
        "model_type": "PatchScorer",
        "patch_size": 16,
        "grid_size": 14,
        "num_patches": 196,
        "base_channels": 32,
        "in_channels": 3,
        "target_resolution": 224,
        "classifier_backbone": "resnet18",
        "num_classes": 10,
        "classes": [
            "tench", "English springer", "cassette player", "chain saw", "church",
            "French horn", "garbage truck", "gas pump", "golf ball", "parachute"
        ],
    }
    cfg_path = output_dir / 'config.json'
    with open(cfg_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2)
    print(f"[+] Saved config: {cfg_path}")

    # 4. Save Model Card README.md
    target_repo = repo_id or "username/preservenet-reducer"
    readme_content = generate_model_card(target_repo)
    readme_path = output_dir / 'README.md'
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"[+] Saved model card README: {readme_path}")

    # Also mirror into app/weights if exported to a different location
    app_weights_dir = REPO_ROOT / 'app' / 'weights'
    app_weights_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, app_weights_dir / 'preservenet_reducer.pt')

    pres_app_weights_dir = REPO_ROOT / 'PreserveNet' / 'app' / 'weights'
    pres_app_weights_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, pres_app_weights_dir / 'preservenet_reducer.pt')

    # 5. Hugging Face Hub upload
    if push_to_hub:
        if not HAS_HF_HUB:
            print("[-] huggingface_hub is not installed. Run: pip install huggingface_hub")
            return {"pt": pt_path, "safetensors": st_path, "config": cfg_path}

        hf_token = token or os.environ.get('HF_TOKEN')
        if not repo_id:
            print("[-] Cannot push to Hub: --repo_id must be provided (e.g. --repo_id your-username/preservenet-reducer)")
            return {"pt": pt_path, "safetensors": st_path, "config": cfg_path}

        if not hf_token:
            print("[-] No Hugging Face token found.")
            print("    Please provide --token <hf_token> or set the HF_TOKEN environment variable.")
            print("    You can create a write token at: https://huggingface.co/settings/tokens")
            return {"pt": pt_path, "safetensors": st_path, "config": cfg_path}

        print(f"[*] Uploading to Hugging Face Hub: {repo_id}...")
        api = HfApi(token=hf_token)
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=private)
        api.upload_folder(
            folder_path=str(output_dir),
            repo_id=repo_id,
            repo_type="model",
            commit_message="Export PreserveNet PatchScorer reducer weights and config",
        )
        print(f"[✓] Successfully deployed weights to Hugging Face Hub!")
        print(f"    View model at: https://huggingface.co/{repo_id}")

    return {
        "pt": pt_path,
        "safetensors": st_path,
        "config": cfg_path,
        "readme": readme_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Export PreserveNet Reducer Weights")
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint (best.pt)')
    parser.add_argument('--output_dir', type=str, default=str(REPO_ROOT / 'app' / 'weights'),
                        help='Output directory for exported artifacts')
    parser.add_argument('--repo_id', type=str, default=None,
                        help='Hugging Face repository ID (e.g. username/preservenet-reducer)')
    parser.add_argument('--push_to_hub', action='store_true',
                        help='Upload exported files to Hugging Face Hub')
    parser.add_argument('--token', type=str, default=None,
                        help='Hugging Face write token (or set HF_TOKEN env variable)')
    parser.add_argument('--private', action='store_true',
                        help='Make the Hugging Face repository private')
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint) if args.checkpoint else find_default_checkpoint()
    out_dir = Path(args.output_dir)

    print("=" * 60)
    print("PreserveNet — Reducer Weights Exporter")
    print("=" * 60)
    export_weights(
        checkpoint_path=ckpt_path,
        output_dir=out_dir,
        repo_id=args.repo_id,
        push_to_hub=args.push_to_hub,
        token=args.token,
        private=args.private,
    )
    print("=" * 60)
    print("Export process finished successfully!")


if __name__ == '__main__':
    main()
