"""
src/baselines/__init__.py
─────────────────────────
Dumb reduction baselines for PreserveNet.

Each reducer is a callable:
    reducer(x: Tensor[B, C, H, W]) -> Tensor[B, C, H, W]

Pixels that are "dropped" are set to zero (black).
The spatial shape is preserved so the same classifier can be reused.
"""

from src.baselines.uniform import UniformGridReducer
from src.baselines.random_drop import RandomDropReducer

__all__ = ['UniformGridReducer', 'RandomDropReducer']
