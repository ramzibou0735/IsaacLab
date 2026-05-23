"""Python wrapper for the torch-native parallel controller extension."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from . import _parallel_control


def _resolve_device(device: Optional[object]) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _require_tensor(name: str, value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    return value


@dataclass(frozen=True)
class _ControllerInfo:
    name: str
    action_cols: int


class _BaseController:
    _info: _ControllerInfo

    def __init__(self, envs_num: int, device: Optional[object] = None) -> None:
        if envs_num <= 0:
            raise ValueError("envs_num must be positive")
        self.envs_num = int(envs_num)
        self.device = _resolve_device(device)

    def _coerce(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        return _require_tensor(name, tensor)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(envs_num={self.envs_num}, "
            f"device='{self.device}')"
        )


class ParallelPosControl(_BaseController):
    """Batched position + yaw controller.

    Parameters
    ----------
    envs_num:
        Number of parallel environments.
    device:
        Optional device specifier. Defaults to ``cuda`` when available and
        otherwise ``cpu``.
    """

    _info = _ControllerInfo("ParallelPosControl", action_cols=4)

    def __init__(self, envs_num: int, device: Optional[object] = None) -> None:
        super().__init__(envs_num, device=device)
        self._impl = _parallel_control.ParallelPosControl(
            self.envs_num, str(self.device)
        )

    def set_status(
        self,
        pos: torch.Tensor,
        q_matrix: torch.Tensor,
        vel: torch.Tensor,
        ang_vel: torch.Tensor,
        dt: float,
    ) -> None:
        self._impl.set_status(
            self._coerce("pos", pos),
            self._coerce("q_matrix", q_matrix),
            self._coerce("vel", vel),
            self._coerce("ang_vel", ang_vel),
            float(dt),
        )

    def update(self, actions: torch.Tensor) -> torch.Tensor:
        return self._impl.update(self._coerce("actions", actions))


class ParallelVelControl(_BaseController):
    """Batched velocity + yaw controller."""

    _info = _ControllerInfo("ParallelVelControl", action_cols=4)

    def __init__(self, envs_num: int, device: Optional[object] = None) -> None:
        super().__init__(envs_num, device=device)
        self._impl = _parallel_control.ParallelVelControl(
            self.envs_num, str(self.device)
        )

    def set_status(
        self,
        pos: torch.Tensor,
        q_matrix: torch.Tensor,
        vel: torch.Tensor,
        ang_vel: torch.Tensor,
        dt: float,
    ) -> None:
        self._impl.set_status(
            self._coerce("pos", pos),
            self._coerce("q_matrix", q_matrix),
            self._coerce("vel", vel),
            self._coerce("ang_vel", ang_vel),
            float(dt),
        )

    def update(self, actions: torch.Tensor) -> torch.Tensor:
        return self._impl.update(self._coerce("actions", actions))


class ParallelAttiControl(_BaseController):
    """Batched attitude + thrust controller."""

    _info = _ControllerInfo("ParallelAttiControl", action_cols=5)

    def __init__(self, envs_num: int, device: Optional[object] = None) -> None:
        super().__init__(envs_num, device=device)
        self._impl = _parallel_control.ParallelAttiControl(
            self.envs_num, str(self.device)
        )

    def set_status(
        self,
        pos: torch.Tensor,
        q_matrix: torch.Tensor,
        vel: torch.Tensor,
        ang_vel: torch.Tensor,
        dt: float,
    ) -> None:
        self._impl.set_status(
            self._coerce("pos", pos),
            self._coerce("q_matrix", q_matrix),
            self._coerce("vel", vel),
            self._coerce("ang_vel", ang_vel),
            float(dt),
        )

    def update(self, actions: torch.Tensor) -> torch.Tensor:
        return self._impl.update(self._coerce("actions", actions))


class ParallelRateControl(_BaseController):
    """Batched body-rate + thrust controller."""

    _info = _ControllerInfo("ParallelRateControl", action_cols=4)

    def __init__(self, envs_num: int, device: Optional[object] = None) -> None:
        super().__init__(envs_num, device=device)
        self._impl = _parallel_control.ParallelRateControl(
            self.envs_num, str(self.device)
        )

    def set_q_world(self, q_world: torch.Tensor) -> None:
        self._impl.set_q_world(self._coerce("q_world", q_world))

    def update(self, actions: torch.Tensor, rate: torch.Tensor, dt: float) -> torch.Tensor:
        return self._impl.update(
            self._coerce("actions", actions),
            self._coerce("rate", rate),
            float(dt),
        )


__all__ = [
    "ParallelAttiControl",
    "ParallelPosControl",
    "ParallelRateControl",
    "ParallelVelControl",
]
