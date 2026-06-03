"""Public, GPU-native controller classes.

These mirror the original ``rlPx4Controller.pyParallelControl`` API but run as
NVIDIA Warp kernels and operate directly on ``torch`` CUDA tensors. Inputs are
consumed zero-copy via :func:`warp.from_torch`; the ``(N, 4)`` motor-command
output aliases a persistent Warp buffer exposed through :func:`warp.to_torch`.

Strict I/O contract (kept deliberately simple so nothing silently copies to the
host): every tensor argument must be a contiguous ``float32`` tensor on the
controller's device. Quaternions are ``[w, x, y, z]``.

The returned command tensor is reused across calls; clone it if you need to keep
a result past the next :meth:`update`.
"""

from __future__ import annotations

import torch
import warp as wp

from ._kernels import atti_update, pos_vel_update, rate_update

# Must match the MODE_* wp.constant values in _kernels.py.
_MODE_ALL = 0
_MODE_VEL = 2

# Initialize Warp once at import (idempotent).
wp.init()


def _resolve_device(device) -> torch.device:
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    return dev


class _ControllerBase:
    """Shared device setup, validation, persistent buffers, and launch helper."""

    def __init__(self, envs_num: int, device=None) -> None:
        if int(envs_num) <= 0:
            raise ValueError(f"envs_num must be positive, got {envs_num}")
        wp.init()

        self.envs_num = int(envs_num)
        self._torch_device = _resolve_device(device)
        self._is_cuda = self._torch_device.type == "cuda"
        self._wp_device = wp.device_from_torch(self._torch_device)
        self.device = str(self._torch_device)

        # Persistent per-env state and output buffer (allocated once).
        self._rate_int = wp.zeros((self.envs_num, 3), dtype=wp.float32, device=self._wp_device)
        self._commands = wp.zeros((self.envs_num, 4), dtype=wp.float32, device=self._wp_device)
        # Torch view that aliases the Warp command buffer; returned every update().
        self._commands_torch = wp.to_torch(self._commands)

        self._dt = 0.01

        # Make sure the zero-init above is visible before any torch-stream launch.
        if self._is_cuda:
            wp.synchronize_device(self._wp_device)

    # -- validation -------------------------------------------------------- #
    def _check(self, name: str, tensor, cols: int):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be float32, got {tensor.dtype}")
        if tensor.device != self._torch_device:
            raise ValueError(
                f"{name} must be on device {self._torch_device}, got {tensor.device}"
            )
        if tensor.dim() != 2 or tensor.shape[0] != self.envs_num or tensor.shape[1] != cols:
            raise ValueError(
                f"{name} must have shape ({self.envs_num}, {cols}), got {tuple(tensor.shape)}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        return tensor

    # -- launch ------------------------------------------------------------ #
    def _launch(self, kernel, inputs, outputs) -> None:
        if self._is_cuda:
            # Run on PyTorch's current stream so ordering is correct without
            # any explicit synchronization on the control path.
            stream = wp.stream_from_torch(torch.cuda.current_stream(self._torch_device))
            wp.launch(kernel, dim=self.envs_num, inputs=inputs, outputs=outputs, stream=stream)
        else:
            wp.launch(kernel, dim=self.envs_num, inputs=inputs, outputs=outputs, device=self._wp_device)

    def reset(self) -> None:
        """Zero the body-rate integrator state for all environments.

        Useful at episode boundaries to avoid integrator wind-up carrying across
        resets. Not part of the original API.
        """
        self._rate_int.zero_()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(envs_num={self.envs_num}, device='{self.device}')"


class _CascadeController(_ControllerBase):
    """Position/velocity controllers: full pos/vel -> attitude -> rate -> mixer."""

    _mode = _MODE_ALL

    def __init__(self, envs_num: int, device=None) -> None:
        super().__init__(envs_num, device=device)
        self._pos_wp = None
        self._vel_wp = None
        self._quat_wp = None
        self._ang_wp = None
        self._status_refs = None

    def set_status(self, pos, q_matrix, vel, ang_vel, dt) -> None:
        p = self._check("pos", pos, 3)
        q = self._check("q_matrix", q_matrix, 4)
        v = self._check("vel", vel, 3)
        a = self._check("ang_vel", ang_vel, 3)
        self._dt = float(dt)
        # Keep references alive so the underlying storage is not freed while the
        # Warp views below point at it.
        self._status_refs = (p, q, v, a)
        self._pos_wp = wp.from_torch(p)
        self._quat_wp = wp.from_torch(q)
        self._vel_wp = wp.from_torch(v)
        self._ang_wp = wp.from_torch(a)

    def update(self, actions) -> torch.Tensor:
        if self._pos_wp is None:
            raise RuntimeError("call set_status() before update()")
        a = self._check("actions", actions, 4)
        acts = wp.from_torch(a)
        self._launch(
            pos_vel_update,
            inputs=[
                self._pos_wp,
                self._vel_wp,
                self._quat_wp,
                self._ang_wp,
                acts,
                self._rate_int,
                self._dt,
                self._mode,
            ],
            outputs=[self._commands],
        )
        return self._commands_torch


class ParallelPosControl(_CascadeController):
    """Batched position + yaw controller. ``actions[i] = [x_sp, y_sp, z_sp, yaw_sp]``."""

    _mode = _MODE_ALL


class ParallelVelControl(_CascadeController):
    """Batched velocity + yaw controller. ``actions[i] = [vx_sp, vy_sp, vz_sp, yaw_sp]``."""

    _mode = _MODE_VEL


class ParallelAttiControl(_ControllerBase):
    """Batched attitude + thrust controller.

    ``actions[i] = [qw_sp, qx_sp, qy_sp, qz_sp, thrust_sp]``.
    """

    def __init__(self, envs_num: int, device=None) -> None:
        super().__init__(envs_num, device=device)
        self._quat_wp = None
        self._ang_wp = None
        self._status_refs = None

    def set_status(self, pos, q_matrix, vel, ang_vel, dt) -> None:
        # pos and vel are validated for API parity but unused by this controller.
        self._check("pos", pos, 3)
        q = self._check("q_matrix", q_matrix, 4)
        self._check("vel", vel, 3)
        a = self._check("ang_vel", ang_vel, 3)
        self._dt = float(dt)
        self._status_refs = (q, a)
        self._quat_wp = wp.from_torch(q)
        self._ang_wp = wp.from_torch(a)

    def update(self, actions) -> torch.Tensor:
        if self._quat_wp is None:
            raise RuntimeError("call set_status() before update()")
        a = self._check("actions", actions, 5)
        acts = wp.from_torch(a)
        self._launch(
            atti_update,
            inputs=[self._quat_wp, self._ang_wp, acts, self._rate_int, self._dt],
            outputs=[self._commands],
        )
        return self._commands_torch


class ParallelRateControl(_ControllerBase):
    """Batched body-rate + thrust controller. ``actions[i] = [p_sp, q_sp, r_sp, thrust_sp]``."""

    def __init__(self, envs_num: int, device=None) -> None:
        super().__init__(envs_num, device=device)
        self._qworld_wp = None
        self._status_refs = None

    def set_q_world(self, q_world) -> None:
        q = self._check("q_world", q_world, 4)
        self._status_refs = (q,)
        self._qworld_wp = wp.from_torch(q)

    def update(self, actions, rate, dt) -> torch.Tensor:
        if self._qworld_wp is None:
            raise RuntimeError("call set_q_world() before update()")
        a = self._check("actions", actions, 4)
        r = self._check("rate", rate, 3)
        acts = wp.from_torch(a)
        rr = wp.from_torch(r)
        self._launch(
            rate_update,
            inputs=[self._qworld_wp, acts, rr, self._rate_int, float(dt)],
            outputs=[self._commands],
        )
        return self._commands_torch
