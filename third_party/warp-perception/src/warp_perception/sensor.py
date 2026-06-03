"""GPU-native fused room-perception sensor built on NVIDIA Warp.

``WarpRoomPerception`` replaces an RTX depth camera plus the PyTorch
occupancy-grid post-processing with a single Warp kernel that ray-casts an
analytic box scene (walls + pillars) and emits, per step:

- ``depth``       : ``(N, H, W)`` perspective depth (distance to image plane).
- ``critic_grid`` : ``(N, 2, Hc, Wc)`` ego-local free/occupied BEV grid.
- ``vis_grid``    : ``(N, 2, Hv, Wv)`` env-local free/occupied visibility grid.

All buffers are persistent ``torch`` tensors aliased to Warp arrays (zero-copy).
Inputs are written in place each step, so no host (CPU) transfer happens on the
perception path. Quaternions are accepted in IsaacLab order ``[w, x, y, z]`` and
re-ordered once to Warp ``[x, y, z, w]``.
"""

from __future__ import annotations

import math

import torch
import warp as wp

from ._kernels import clear_grid, perception_kernel

wp.init()


def _resolve_device(device) -> torch.device:
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    return dev


class GridSpec:
    """Axis-aligned BEV grid limits and cell size (x is rows, y is columns)."""

    def __init__(self, x_limits, y_limits, z_limits, cell_size):
        self.x0, self.x1 = float(x_limits[0]), float(x_limits[1])
        self.y0, self.y1 = float(y_limits[0]), float(y_limits[1])
        self.z0, self.z1 = float(z_limits[0]), float(z_limits[1])
        self.cell = float(cell_size)
        self.h = int(math.ceil((self.x1 - self.x0) / self.cell))
        self.w = int(math.ceil((self.y1 - self.y0) / self.cell))


class WarpRoomPerception:
    """Fused depth + occupancy-grid ray-casting sensor for box rooms."""

    def __init__(
        self,
        num_envs: int,
        num_boxes: int,
        *,
        height: int,
        width: int,
        horizontal_fov_deg: float,
        max_range: float,
        critic_grid: GridSpec,
        vis_grid: GridSpec,
        grid_stride: int = 4,
        free_samples: int = 48,
        hit_margin: float = 1.0e-3,
        device=None,
        use_cuda_graph: bool = False,
    ) -> None:
        if int(num_envs) <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        if int(num_boxes) <= 0:
            raise ValueError(f"num_boxes must be positive, got {num_boxes}")
        wp.init()

        self.num_envs = int(num_envs)
        self.num_boxes = int(num_boxes)
        self.height = int(height)
        self.width = int(width)
        self.max_range = float(max_range)
        self.grid_stride = int(grid_stride)
        self.free_samples = int(free_samples)
        self.hit_margin = float(hit_margin)
        self.critic = critic_grid
        self.vis = vis_grid

        self._torch_device = _resolve_device(device)
        self._is_cuda = self._torch_device.type == "cuda"
        self._wp_device = wp.device_from_torch(self._torch_device)

        # Pinhole intrinsics: square pixels, principal point at the image center.
        self.fx = (self.width / 2.0) / math.tan(math.radians(horizontal_fov_deg) / 2.0)
        self.fy = self.fx
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0

        dev = self._torch_device
        f32 = torch.float32

        # -- persistent input buffers (written in place each step) ------------ #
        self._cam_pos = torch.zeros((self.num_envs, 3), device=dev, dtype=f32)
        self._cam_quat = torch.zeros((self.num_envs, 4), device=dev, dtype=f32)
        self._cam_quat[:, 3] = 1.0  # identity in [x, y, z, w]
        self._w2l = torch.eye(3, device=dev, dtype=f32).unsqueeze(0).repeat(self.num_envs, 1, 1).contiguous()
        self._root_pos = torch.zeros((self.num_envs, 3), device=dev, dtype=f32)
        self._env_origin = torch.zeros((self.num_envs, 3), device=dev, dtype=f32)
        self._box_center = torch.zeros((self.num_envs, self.num_boxes, 3), device=dev, dtype=f32)
        self._box_half = torch.zeros((self.num_envs, self.num_boxes, 3), device=dev, dtype=f32)
        self._box_quat = torch.zeros((self.num_envs, self.num_boxes, 4), device=dev, dtype=f32)
        self._box_quat[:, :, 3] = 1.0

        # -- persistent output buffers --------------------------------------- #
        self._depth = torch.zeros((self.num_envs, self.height, self.width), device=dev, dtype=f32)
        self._critic_grid = torch.zeros((self.num_envs, 2, self.critic.h, self.critic.w), device=dev, dtype=f32)
        self._vis_grid = torch.zeros((self.num_envs, 2, self.vis.h, self.vis.w), device=dev, dtype=f32)

        # -- zero-copy Warp aliases (allocated once) ------------------------- #
        self._wp_cam_pos = wp.from_torch(self._cam_pos, dtype=wp.vec3)
        self._wp_cam_quat = wp.from_torch(self._cam_quat, dtype=wp.quat)
        self._wp_w2l = wp.from_torch(self._w2l, dtype=wp.mat33)
        self._wp_root_pos = wp.from_torch(self._root_pos, dtype=wp.vec3)
        self._wp_env_origin = wp.from_torch(self._env_origin, dtype=wp.vec3)
        self._wp_box_center = wp.from_torch(self._box_center, dtype=wp.vec3)
        self._wp_box_half = wp.from_torch(self._box_half, dtype=wp.vec3)
        self._wp_box_quat = wp.from_torch(self._box_quat, dtype=wp.quat)
        self._wp_depth = wp.from_torch(self._depth, dtype=wp.float32)
        self._wp_critic = wp.from_torch(self._critic_grid, dtype=wp.float32)
        self._wp_vis = wp.from_torch(self._vis_grid, dtype=wp.float32)

        self._use_cuda_graph = bool(use_cuda_graph) and self._is_cuda
        self._graph = None
        self._graph_stream = None

        if self._is_cuda:
            wp.synchronize_device(self._wp_device)

    # ----------------------------------------------------------------------- #
    # static / reset-time configuration
    # ----------------------------------------------------------------------- #
    def set_env_origins(self, env_origins: torch.Tensor) -> None:
        """Set per-env world origins (used by the env-local visibility grid)."""
        self._env_origin.copy_(env_origins.to(self._torch_device, torch.float32))

    def set_box_half_extents(self, box_half: torch.Tensor) -> None:
        """Set per-box half-extents. Accepts ``(B, 3)`` or ``(N, B, 3)``."""
        box_half = box_half.to(self._torch_device, torch.float32)
        if box_half.dim() == 2:
            box_half = box_half.unsqueeze(0).expand(self.num_envs, -1, -1)
        self._box_half.copy_(box_half)

    def set_box_transforms(
        self,
        box_center_w: torch.Tensor,
        box_quat_w: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Write world-frame box centers (and optional quats) for reset envs.

        Box poses change only on episode reset, so they are stored persistently
        and reused on every per-step :meth:`compute`. Quaternions use IsaacLab
        order ``[w, x, y, z]`` and are re-ordered to Warp ``[x, y, z, w]``.
        """
        box_center_w = box_center_w.to(self._torch_device, torch.float32)
        if env_ids is None:
            self._box_center.copy_(box_center_w)
            if box_quat_w is not None:
                self._box_quat.copy_(box_quat_w.to(self._torch_device, torch.float32)[:, :, [1, 2, 3, 0]])
        else:
            self._box_center[env_ids] = box_center_w
            if box_quat_w is not None:
                self._box_quat[env_ids] = box_quat_w.to(self._torch_device, torch.float32)[:, :, [1, 2, 3, 0]]

    # ----------------------------------------------------------------------- #
    # per-step compute
    # ----------------------------------------------------------------------- #
    def _inputs(self):
        return [
            self._wp_cam_pos,
            self._wp_cam_quat,
            self._wp_w2l,
            self._wp_root_pos,
            self._wp_env_origin,
            self._wp_box_center,
            self._wp_box_half,
            self._wp_box_quat,
            self.num_boxes,
            float(self.fx),
            float(self.fy),
            float(self.cx),
            float(self.cy),
            float(self.max_range),
            float(self.hit_margin),
            int(self.grid_stride),
            int(self.free_samples),
            self.critic.x0, self.critic.x1, self.critic.y0, self.critic.y1,
            self.critic.z0, self.critic.z1, self.critic.cell, self.critic.h, self.critic.w,
            self.vis.x0, self.vis.x1, self.vis.y0, self.vis.y1,
            self.vis.z0, self.vis.z1, self.vis.cell, self.vis.h, self.vis.w,
        ]

    def _launch(self) -> None:
        dim = (self.num_envs, self.height, self.width)
        outputs = [self._wp_depth, self._wp_critic, self._wp_vis]
        if self._is_cuda:
            stream = wp.stream_from_torch(torch.cuda.current_stream(self._torch_device))
            wp.launch(perception_kernel, dim=dim, inputs=self._inputs(), outputs=outputs, stream=stream)
        else:
            wp.launch(perception_kernel, dim=dim, inputs=self._inputs(), outputs=outputs, device=self._wp_device)

    def _launch_graph(self) -> None:
        # CUDA graphs cannot be captured on the default/legacy stream, so use a
        # dedicated Warp stream and bracket it with cross-stream waits to keep
        # ordering correct against PyTorch's current stream (which ran the
        # in-place input copies just before this call).
        cdim = (self.num_envs, 2, self.critic.h, self.critic.w)
        vdim = (self.num_envs, 2, self.vis.h, self.vis.w)
        pdim = (self.num_envs, self.height, self.width)
        outputs = [self._wp_depth, self._wp_critic, self._wp_vis]

        torch_stream = wp.stream_from_torch(torch.cuda.current_stream(self._torch_device))
        if self._graph_stream is None:
            self._graph_stream = wp.Stream(self._wp_device)
        self._graph_stream.wait_stream(torch_stream)

        if self._graph is None:
            wp.capture_begin(self._wp_device, stream=self._graph_stream)
            try:
                wp.launch(clear_grid, dim=cdim, inputs=[self._wp_critic], stream=self._graph_stream)
                wp.launch(clear_grid, dim=vdim, inputs=[self._wp_vis], stream=self._graph_stream)
                wp.launch(
                    perception_kernel, dim=pdim, inputs=self._inputs(), outputs=outputs, stream=self._graph_stream
                )
            finally:
                self._graph = wp.capture_end(self._wp_device, stream=self._graph_stream)

        wp.capture_launch(self._graph, stream=self._graph_stream)
        torch_stream.wait_stream(self._graph_stream)

    def compute(
        self,
        cam_pos_w: torch.Tensor,
        cam_quat_w: torch.Tensor,
        world_to_local: torch.Tensor,
        root_pos_w: torch.Tensor,
        box_center_w: torch.Tensor | None = None,
        box_quat_w: torch.Tensor | None = None,
    ):
        """Run the fused kernel and return ``(depth, critic_grid, vis_grid)`` views.

        All tensors are on the sensor device. ``cam_quat_w`` / ``box_quat_w`` use
        IsaacLab order ``[w, x, y, z]``. Box transforms are optional; when omitted
        the persistent transforms (set via :meth:`set_box_transforms`) are reused,
        avoiding a per-step copy for static-per-episode scenes. The returned
        tensors alias persistent buffers and are overwritten on the next call.
        """
        self._cam_pos.copy_(cam_pos_w)
        self._cam_quat.copy_(cam_quat_w[:, [1, 2, 3, 0]])  # wxyz -> xyzw
        self._w2l.copy_(world_to_local)
        self._root_pos.copy_(root_pos_w)
        if box_center_w is not None:
            self._box_center.copy_(box_center_w)
        if box_quat_w is not None:
            self._box_quat.copy_(box_quat_w[:, :, [1, 2, 3, 0]])

        if self._use_cuda_graph:
            self._launch_graph()
        else:
            # Accumulating channels must start cleared (kernel uses atomic_max).
            self._critic_grid.zero_()
            self._vis_grid.zero_()
            self._launch()

        return self._depth, self._critic_grid, self._vis_grid

    @property
    def depth(self) -> torch.Tensor:
        return self._depth

    @property
    def critic_grid(self) -> torch.Tensor:
        return self._critic_grid

    @property
    def visibility_grid(self) -> torch.Tensor:
        return self._vis_grid
