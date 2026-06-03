"""warp_perception - fused GPU-native depth + occupancy-grid ray-casting sensor.

A drop-in replacement for an RTX depth camera plus PyTorch occupancy-grid
post-processing in box-room RL environments (e.g. IsaacLab). A single NVIDIA
Warp kernel ray-casts an analytic box scene and emits a depth image, an
ego-local critic occupancy grid, and an env-local visibility grid in one launch,
keeping every tensor on the GPU (zero-copy ``torch`` interop).

See :class:`WarpRoomPerception` for the public API.
"""

from __future__ import annotations

from .sensor import GridSpec, WarpRoomPerception

__all__ = [
    "WarpRoomPerception",
    "GridSpec",
]

__version__ = "0.1.0"
