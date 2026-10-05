"""
src/baselines/__init__.py
─────────────────────────
Patch-reduction and pixel-reduction baselines for PreserveNet.

Each reducer is a callable:
    reducer(x: Tensor[B, C, H, W]) -> Tensor[B, C, H, W]

Dropped patches/pixels are set to zero (black).
The spatial shape is preserved so any classifier can be evaluated directly.

Baselines:
  * UniformRandomReducer    — random bernoulli patch drop (default 16×16)
  * UniformGridReducer      — regular spatial grid of patches (default 16×16)
  * RandomDropReducer       — random bernoulli patch drop with seed (default 16×16)
  * GradCAMReducer          — class activation map patch drop (default 16×16)
  * GradientSaliencyReducer — input-gradient patch drop (default 16×16)
"""

from src.baselines.uniform import UniformGridReducer, UniformRandomReducer
from src.baselines.random_drop import RandomDropReducer
from src.baselines.gradcam import GradCAMReducer
from src.baselines.saliency import GradientSaliencyReducer

__all__ = [
    'UniformGridReducer',
    'UniformRandomReducer',
    'RandomDropReducer',
    'GradCAMReducer',
    'GradientSaliencyReducer',
]
