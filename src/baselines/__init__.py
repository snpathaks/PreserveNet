"""
src/baselines/__init__.py
─────────────────────────
Pixel-reduction baselines for PreserveNet.

Each reducer is a callable:
    reducer(x: Tensor[B, C, H, W]) -> Tensor[B, C, H, W]

Pixels that are "dropped" are set to zero (black).
The spatial shape is preserved so the same classifier can be reused.

Dumb baselines (no model access required):
  * UniformGridReducer   — regular spatial grid
  * UniformRandomReducer — random bernoulli mask
  * RandomDropReducer    — random bernoulli mask (seeded)

Saliency-guided baselines (require a pretrained classifier):
  * GradientSaliencyReducer — input-gradient importance
  * GradCAMReducer          — class activation map (GradCAM)
"""

from src.baselines.uniform import UniformGridReducer, UniformRandomReducer
from src.baselines.random_drop import RandomDropReducer
from src.baselines.saliency import GradientSaliencyReducer, GradCAMReducer

__all__ = [
    'UniformGridReducer',
    'UniformRandomReducer',
    'RandomDropReducer',
    'GradientSaliencyReducer',
    'GradCAMReducer',
]
