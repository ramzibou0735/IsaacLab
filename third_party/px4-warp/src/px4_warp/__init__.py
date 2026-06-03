"""px4_warp - GPU-native batched PX4-style quadrotor controllers.

This package reimplements the batched controllers from the C++/Eigen
``rlPx4Controller.pyParallelControl`` module as NVIDIA Warp kernels so that the
whole control step stays on the GPU. It is designed to be driven directly by
IsaacLab tensors: inputs and outputs are ``torch`` CUDA ``float32`` tensors and
no host (CPU/NumPy) copy happens on the control path.

The public classes mirror the original API:

- :class:`ParallelPosControl`
- :class:`ParallelVelControl`
- :class:`ParallelAttiControl`
- :class:`ParallelRateControl`

See ``MATH_MODEL.md`` for the exact mathematical model these kernels implement.
"""

from __future__ import annotations

from .controllers import (
    ParallelAttiControl,
    ParallelPosControl,
    ParallelRateControl,
    ParallelVelControl,
)
from .selector import (
    ACTION_DIMS,
    CTL_MODES,
    ParallelController,
    PropController,
    make_controller,
)

__all__ = [
    "ParallelAttiControl",
    "ParallelPosControl",
    "ParallelRateControl",
    "ParallelVelControl",
    # ctl_mode-style selection
    "ParallelController",
    "PropController",
    "make_controller",
    "CTL_MODES",
    "ACTION_DIMS",
]

__version__ = "0.1.0"
