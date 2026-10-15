"""
app/inference.py
----------------
Inference wrapper for PreserveNet.

Provides the Predictor class:
    Predictor.run(img, retention) -> dict of:
        - orig        : PIL Image (Original 224x224 input)
        - overlay     : PIL Image (Dropped patches dimmed to ~15%, kept patches tinted amber at 25%)
        - reduced     : PIL Image (Model input with dropped patches zeroed out)
        - full_probs  : dict of {class_name: probability} for full image
        - red_probs   : dict of {class_name: probability} for reduced image
        - full        : alias for full_probs
        - red         : alias for red_probs
        - kept        : int (number of patches retained out of 196)
        - total       : int (196 total 16x16 patches)
        - agree       : bool (True if top-1 class prediction is preserved)
        - kl          : float (KL divergence between full and reduced predictions)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

try:
    import timm
except ImportError as exc:
    raise ImportError("timm is required for PreserveNet inference: pip install timm") from exc

try:
    from huggingface_hub import hf_hub_download
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False


# =============================================================================
# Constants & Metadata
# =============================================================================

IMAGENETTE_CLASSES: list[str] = [
    'tench', 'English springer', 'cassette player',
    'chain saw', 'church', 'French horn',
    'garbage truck', 'gas pump', 'golf ball', 'parachute',
]

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

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
AMBER_COLOR   = np.array([255.0, 180.0, 0.0], dtype=np.float32)  # #FFB400


# =============================================================================
# Model Architectures
# =============================================================================

def _conv_bn_relu(
    in_ch: int,
    out_ch: int,
    kernel: int = 3,
    stride: int = 1,
    padding: int = 1,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class PatchScorer(nn.Module):
    """
    Lightweight CNN mapping (B, 3, 224, 224) to (B, 1, 14, 14) patch logit scores.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        b = base_channels
        self.stem   = _conv_bn_relu(in_channels, b,     stride=2)  # 224 -> 112
        self.block1 = _conv_bn_relu(b,           b * 2, stride=2)  # 112 -> 56
        self.block2 = _conv_bn_relu(b * 2,       b * 2, stride=2)  # 56  -> 28
        self.block3 = _conv_bn_relu(b * 2,       b,     stride=2)  # 28  -> 14
        self.head   = nn.Conv2d(b, 1, kernel_size=1)               # 14  -> 14

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        return self.head(h)


class ImagenetteZeroShotClassifier(nn.Module):
    """
    Zero-shot classifier slicing 10 Imagenette class logits from a pretrained ResNet-18.
    """

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.register_buffer(
            '_class_idx',
            torch.tensor(IMAGENETTE_IMAGENET_INDICES, dtype=torch.long),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits_1000 = self.backbone(x)
        return logits_1000[:, self._class_idx]


# =============================================================================
# Predictor Wrapper
# =============================================================================

class Predictor:
    """
    Main inference controller for PreserveNet demo.
    """

    _instance: Predictor | None = None

    def __init__(
        self,
        weights_path: str | Path | None = None,
        hf_repo_id: str = "snpathaks/preservenet-reducer",
        device: str | None = None,
    ) -> None:
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"[*] Initializing PreserveNet Predictor on device: {self.device}")

        # 1. Initialize PatchScorer
        self.scorer = PatchScorer(base_channels=32).to(self.device)
        self._load_reducer_weights(weights_path, hf_repo_id)
        self.scorer.eval()

        # 2. Initialize ResNet-18 Imagenette Classifier
        print("[*] Loading ResNet-18 pretrained ImageNet backbone...")
        backbone = timm.create_model('resnet18', pretrained=True, num_classes=1000)
        self.classifier = ImagenetteZeroShotClassifier(backbone).to(self.device)
        self.classifier.eval()

        # Normalization tensors
        self.mean_tensor = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std_tensor  = torch.tensor(IMAGENET_STD,  device=self.device).view(1, 3, 1, 1)

    def _load_reducer_weights(
        self,
        weights_path: str | Path | None,
        hf_repo_id: str,
    ) -> None:
        """Finds and loads the PatchScorer weights from local files or HF Hub."""
        resolved_path: Path | None = None

        if weights_path and Path(weights_path).is_file():
            resolved_path = Path(weights_path)
        else:
            # Search possible local locations
            cwd = Path.cwd()
            script_dir = Path(__file__).resolve().parent
            candidates = [
                script_dir / 'weights' / 'preservenet_reducer.pt',
                cwd / 'app' / 'weights' / 'preservenet_reducer.pt',
                cwd / 'PreserveNet' / 'app' / 'weights' / 'preservenet_reducer.pt',
                cwd / 'checkpoints' / 'best.pt',
                cwd / 'PreserveNet' / 'checkpoints' / 'best.pt',
                script_dir.parent / 'checkpoints' / 'best.pt',
                script_dir.parent / 'PreserveNet' / 'checkpoints' / 'best.pt',
            ]
            for c in candidates:
                if c.is_file():
                    resolved_path = c
                    break

        # If not found locally, try Hugging Face Hub
        if resolved_path is None and HAS_HF_HUB:
            try:
                print(f"[*] Downloading reducer weights from Hugging Face Hub: {hf_repo_id}...")
                downloaded = hf_hub_download(
                    repo_id=hf_repo_id,
                    filename="preservenet_reducer.pt",
                    token=os.environ.get("HF_TOKEN"),
                )
                resolved_path = Path(downloaded)
            except Exception as exc:
                print(f"[-] Could not download from HF Hub: {exc}")

        if resolved_path and resolved_path.is_file():
            print(f"[+] Loading reducer weights from: {resolved_path}")
            try:
                ckpt = torch.load(resolved_path, map_location=self.device, weights_only=False)
            except Exception:
                ckpt = torch.load(resolved_path, map_location=self.device)

            if isinstance(ckpt, dict) and 'scorer' in ckpt:
                state_dict = ckpt['scorer']
            elif isinstance(ckpt, dict):
                state_dict = ckpt
            else:
                raise ValueError(f"Invalid checkpoint format in {resolved_path}")

            self.scorer.load_state_dict(state_dict)
            print("[+] PatchScorer weights loaded successfully.")
        else:
            print("[!] Warning: No checkpoint found. Using randomly initialized PatchScorer.")

    @classmethod
    def get_instance(cls) -> Predictor:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def run(cls, img: Any, retention: float | int) -> dict[str, Any]:
        """
        Public entry point matching the user contract:
        Predictor.run(img, retention) -> dict of:
            orig, overlay, reduced, full_probs, red_probs, full, red, kept, total, agree, kl
        """
        instance = cls.get_instance()
        return instance.predict(img, retention)

    def predict(self, img: Any, retention: float | int) -> dict[str, Any]:
        if img is None:
            raise ValueError("No image provided. Please upload an image or choose an example.")

        # 1. Normalize retention rate to [0.05, 1.0]
        if retention > 1.0:
            retention_rate = float(retention) / 100.0
        else:
            retention_rate = float(retention)
        retention_rate = max(0.05, min(1.0, retention_rate))

        # 2. Preprocess input image to PIL 224x224 RGB
        if isinstance(img, str):
            pil_img = Image.open(img).convert("RGB")
        elif isinstance(img, np.ndarray):
            pil_img = Image.fromarray(img).convert("RGB")
        elif isinstance(img, Image.Image):
            pil_img = img.convert("RGB")
        else:
            raise TypeError(f"Unsupported image format: {type(img)}")

        orig_img = pil_img.resize((224, 224), Image.Resampling.BILINEAR)
        img_np = np.array(orig_img, dtype=np.float32)  # (224, 224, 3) in [0, 255]

        # 3. Create normalized tensor (1, 3, 224, 224)
        x_raw = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).to(self.device) / 255.0
        x_norm = (x_raw - self.mean_tensor) / self.std_tensor

        # 4. PatchScorer inference (14x14 = 196 patches)
        with torch.no_grad():
            scores = self.scorer(x_norm)  # (1, 1, 14, 14)

        n_total = 196
        n_keep = max(1, min(n_total, int(round(n_total * retention_rate))))

        flat_scores = scores.view(1, -1)
        _, topk_idx = torch.topk(flat_scores, n_keep, dim=1, largest=True, sorted=False)

        mask_flat = torch.zeros_like(flat_scores)
        mask_flat.scatter_(1, topk_idx, 1.0)
        patch_mask = mask_flat.view(1, 1, 14, 14)

        # Upsample 14x14 patch mask to 224x224 pixel mask
        pixel_mask = F.interpolate(patch_mask, size=(224, 224), mode='nearest')
        mask_2d = pixel_mask[0, 0].cpu().numpy()  # (224, 224) in {0.0, 1.0}

        # 5. Classifier forward passes (Full image vs Masked image)
        x_masked = x_norm * pixel_mask
        with torch.no_grad():
            full_logits = self.classifier(x_norm)    # (1, 10)
            red_logits  = self.classifier(x_masked)  # (1, 10)

        full_probs_t = F.softmax(full_logits, dim=-1)[0]
        red_probs_t  = F.softmax(red_logits,  dim=-1)[0]

        full_probs = full_probs_t.cpu().numpy()
        red_probs  = red_probs_t.cpu().numpy()

        full_top1_idx = int(np.argmax(full_probs))
        red_top1_idx  = int(np.argmax(red_probs))
        agree = bool(full_top1_idx == red_top1_idx)

        # KL divergence: KL(p_full || p_reduced)
        # F.kl_div(log_pred, target) expects log_pred = log(p_reduced), target = p_full
        kl_div = F.kl_div(
            torch.log_softmax(red_logits, dim=-1),
            F.softmax(full_logits, dim=-1),
            reduction='batchmean',
        ).item()
        kl = max(0.0, float(kl_div))

        full_dict = {IMAGENETTE_CLASSES[i]: float(full_probs[i]) for i in range(10)}
        red_dict  = {IMAGENETTE_CLASSES[i]: float(red_probs[i])  for i in range(10)}

        # 6. Generate Selected Patches Overlay
        # Specification:
        # "the overlay dims dropped patches to ~15% brightness and tints kept patches amber at 25%."
        mask_3d = mask_2d[:, :, None]  # (224, 224, 1)
        kept_pixels = (mask_3d > 0.5)
        dropped_pixels = ~kept_pixels

        overlay_np = np.empty_like(img_np)
        overlay_np[dropped_pixels[:, :, 0]] = img_np[dropped_pixels[:, :, 0]] * 0.15
        overlay_np[kept_pixels[:, :, 0]] = (
            img_np[kept_pixels[:, :, 0]] * 0.75 + AMBER_COLOR * 0.25
        )

        # Subtle 16x16 patch boundary grid lines for razor-sharp visual HUD
        overlay_np[::16, :, :] *= 0.8
        overlay_np[:, ::16, :] *= 0.8
        overlay_img = Image.fromarray(np.clip(overlay_np, 0, 255).astype(np.uint8))

        # 7. Generate Reduced Image (Model Input)
        reduced_np = np.zeros_like(img_np)
        reduced_np[kept_pixels[:, :, 0]] = img_np[kept_pixels[:, :, 0]]
        reduced_img = Image.fromarray(np.clip(reduced_np, 0, 255).astype(np.uint8))

        return {
            'orig': orig_img,
            'overlay': overlay_img,
            'reduced': reduced_img,
            'full_probs': full_dict,
            'red_probs': red_dict,
            'full': full_dict,
            'red': red_dict,
            'kept': n_keep,
            'total': n_total,
            'agree': agree,
            'kl': round(kl, 4),
        }
