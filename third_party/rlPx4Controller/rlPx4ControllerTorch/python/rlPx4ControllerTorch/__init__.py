"""Torch-native parallel controller package."""

from .pyParallelControl import (
    ParallelAttiControl,
    ParallelPosControl,
    ParallelRateControl,
    ParallelVelControl,
)

__all__ = [
    "ParallelAttiControl",
    "ParallelPosControl",
    "ParallelRateControl",
    "ParallelVelControl",
]

__version__ = "0.1.0"
