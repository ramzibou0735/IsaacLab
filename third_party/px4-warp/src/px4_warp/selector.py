"""``ctl_mode``-style controller selection.

Mirrors the mode mapping used by AirGym (see ``CONTROLLERS.md``) so an RL env can
pick a controller from a config string and drive it through one ``step()`` call:

- ``"pos"``  -> :class:`ParallelPosControl`,  actions ``[x_sp, y_sp, z_sp, yaw_sp]``
- ``"vel"``  -> :class:`ParallelVelControl`,  actions ``[vx_sp, vy_sp, vz_sp, yaw_sp]``
- ``"atti"`` -> :class:`ParallelAttiControl`, actions ``[qw, qx, qy, qz, thrust]``
- ``"rate"`` -> :class:`ParallelRateControl`, actions ``[p, q, r, thrust]``
- ``"prop"`` -> passthrough, actions ``[m0, m1, m2, m3]`` clamped to ``[0, 1]``

All modes output ``(N, 4)`` normalized motor commands in ``[0, 1]`` and keep the
data on the GPU.
"""

from __future__ import annotations

import torch

from .controllers import (
    ParallelAttiControl,
    ParallelPosControl,
    ParallelRateControl,
    ParallelVelControl,
    _ControllerBase,
)

#: Number of action columns expected by each mode.
ACTION_DIMS = {"pos": 4, "vel": 4, "atti": 5, "rate": 4, "prop": 4}

#: Valid ``ctl_mode`` strings.
CTL_MODES = tuple(ACTION_DIMS.keys())


class PropController(_ControllerBase):
    """Pass-through 'controller' for ``ctl_mode='prop'`` (direct rotor commands).

    ``update(actions)`` clamps the four per-rotor commands to ``[0, 1]`` and writes
    them into the same persistent buffer the real controllers use, so the
    zero-copy / reused-buffer contract is identical across modes.
    """

    def update(self, actions) -> torch.Tensor:
        a = self._check("actions", actions, 4)
        torch.clamp(a, 0.0, 1.0, out=self._commands_torch)
        return self._commands_torch


def make_controller(ctl_mode: str, envs_num: int, device=None):
    """Return the controller implementation for ``ctl_mode``.

    The returned object is the native controller class (or :class:`PropController`
    for ``"prop"``), exposing the standard ``set_status``/``set_q_world`` +
    ``update`` API. Use :class:`ParallelController` for a single unified
    ``step()`` entry point.
    """
    mode = str(ctl_mode).lower()
    if mode == "pos":
        return ParallelPosControl(envs_num, device=device)
    if mode == "vel":
        return ParallelVelControl(envs_num, device=device)
    if mode == "atti":
        return ParallelAttiControl(envs_num, device=device)
    if mode == "rate":
        return ParallelRateControl(envs_num, device=device)
    if mode == "prop":
        return PropController(envs_num, device=device)
    raise ValueError(f"unknown ctl_mode {ctl_mode!r}, expected one of {list(CTL_MODES)}")


class ParallelController:
    """Unified, ``ctl_mode``-selected controller with one :meth:`step` call.

    Parameters
    ----------
    ctl_mode:
        One of :data:`CTL_MODES`.
    envs_num:
        Number of parallel environments.
    device:
        Torch device (defaults to ``"cuda:0"`` when available).
    """

    def __init__(self, ctl_mode: str, envs_num: int, device=None) -> None:
        mode = str(ctl_mode).lower()
        if mode not in ACTION_DIMS:
            raise ValueError(f"unknown ctl_mode {ctl_mode!r}, expected one of {list(CTL_MODES)}")
        self.ctl_mode = mode
        self.action_dim = ACTION_DIMS[mode]
        self._impl = make_controller(mode, envs_num, device=device)
        self.envs_num = self._impl.envs_num
        self.device = self._impl.device

    def step(
        self,
        actions,
        pos=None,
        quat=None,
        vel=None,
        ang_vel=None,
        dt: float = 0.01,
    ) -> torch.Tensor:
        """Run one control step and return ``(N, 4)`` motor commands in ``[0, 1]``.

        ``pos``/``quat``/``vel``/``ang_vel`` are the current vehicle state (from the
        simulator). They are required for every mode except ``"prop"``; ``"rate"``
        uses ``quat`` (as the world frame) and ``ang_vel`` (as the measured rate).
        The returned tensor aliases a persistent buffer; clone it to retain.
        """
        mode = self.ctl_mode
        if mode == "prop":
            return self._impl.update(actions)
        if mode == "rate":
            if quat is None or ang_vel is None:
                raise ValueError("ctl_mode='rate' requires quat and ang_vel")
            self._impl.set_q_world(quat)
            return self._impl.update(actions, ang_vel, dt)
        # pos / vel / atti
        if pos is None or quat is None or vel is None or ang_vel is None:
            raise ValueError(f"ctl_mode='{mode}' requires pos, quat, vel, and ang_vel")
        self._impl.set_status(pos, quat, vel, ang_vel, dt)
        return self._impl.update(actions)

    def reset(self) -> None:
        """Zero the body-rate integrator state (no-op effect for ``'prop'``)."""
        self._impl.reset()

    @property
    def impl(self):
        """The underlying native controller instance."""
        return self._impl

    def __repr__(self) -> str:
        return (
            f"ParallelController(ctl_mode='{self.ctl_mode}', "
            f"envs_num={self.envs_num}, device='{self.device}')"
        )
