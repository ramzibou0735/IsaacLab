# warp-perception

GPU-native, batched **depth + occupancy-grid** perception for box-room RL
environments, implemented as a single fused [NVIDIA Warp](https://github.com/NVIDIA/warp)
kernel. It is a drop-in replacement for an RTX depth camera plus PyTorch
occupancy post-processing: one ray-cast launch produces a depth image, an
ego-local critic occupancy grid, and an env-local visibility grid, with every
tensor kept on the GPU (zero-copy `torch` interop).

- **Version:** 0.1.0
- **Import name:** `warp_perception`
- **Quaternion convention (inputs):** `[w, x, y, z]` (re-ordered internally)
- **Tensor contract:** contiguous `float32` `torch` tensors on the sensor device

## Why

RTX tiled rendering of a trivial box world is dominated by Omniverse render-graph
overhead and runs every physics step. For an analytic box scene (walls + pillars),
Warp ray-casting is dramatically cheaper and, fused with free-space marching,
also removes the separate PyTorch point-cloud scatter passes and their large
intermediate tensors.

## Install

```bash
# from this project root, into the IsaacLab venv (do not let pip replace torch/warp)
pip install -e . --no-deps
```

## Quickstart

```python
import torch
from warp_perception import WarpRoomPerception, GridSpec

sensor = WarpRoomPerception(
    num_envs=64, num_boxes=12,
    height=135, width=240, horizontal_fov_deg=87.0, max_range=10.0,
    critic_grid=GridSpec((0.0, 4.5), (-2.25, 2.25), (-0.5, 1.6), 0.25),
    vis_grid=GridSpec((-5.0, 5.0), (-5.0, 5.0), (0.2, 2.2), 0.5),
    grid_stride=4, free_samples=48, device="cuda:0",
)
sensor.set_env_origins(env_origins)             # (N, 3)
sensor.set_box_half_extents(box_half)           # (B, 3) or (N, B, 3)

depth, critic_grid, vis_grid = sensor.compute(
    cam_pos_w, cam_quat_w, world_to_local, root_pos_w, box_center_w, box_quat_w,
)
```

The returned tensors alias persistent buffers and are overwritten on the next
call; clone if you need to keep them.

## Notes

- `compute()` writes its inputs into persistent buffers in place, so the kernel
  always reads current data and no host (CPU) copy happens on the perception path.
- `use_cuda_graph=True` captures the clear + ray-cast launches once and replays
  them; the default (`False`) launches directly on PyTorch's current stream,
  which is robust and already removes the dominant rendering cost.
- Box obstacles are treated as oriented boxes; with identity orientation they are
  axis-aligned. Quaternions are accepted in `[w, x, y, z]` order.
