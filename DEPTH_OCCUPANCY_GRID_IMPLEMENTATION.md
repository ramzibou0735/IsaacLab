# Depth Occupancy Grid For IsaacLab AirGym Tasks

## Goal

Build a small depth-based occupancy representation for the AirGym x152b tasks and use it as a critic-only observation in `rsl_rl`.

This is not a planner-grade global map. It is a compact, ego-centric spatial observation derived from the onboard depth camera.

## Source Logic Extracted From EGO-Planner

Relevant source files:

- `/home/ramzi/ego-planner-swarm/src/planner/plan_env/src/grid_map.cpp`
- `/home/ramzi/ego-planner-swarm/src/planner/plan_env/src/raycast.cpp`
- `/home/ramzi/ego-planner-swarm/src/planner/plan_env/include/plan_env/grid_map.h`
- `/home/ramzi/ego-planner-swarm/src/planner/plan_env/include/plan_env/raycast.h`

The original pipeline is:

1. Keep a voxel map with fixed bounds and resolution.
2. Read synchronized depth and pose.
3. Backproject depth pixels using camera intrinsics:
   - `x = (u - cx) * z / fx`
   - `y = (v - cy) * z / fy`
   - `z = depth`
4. Transform the 3D point from camera frame to world frame.
5. For each projected point:
   - mark the endpoint as occupied if it is a valid hit,
   - otherwise clip it to map bounds or max ray length and treat it as free-space only.
6. Raycast from the camera origin to the endpoint through the voxel grid.
7. Mark traversed voxels as free and endpoint voxels as occupied.
8. Fuse updates into a log-odds occupancy buffer.
9. Clear stale regions outside a local window and inflate occupied voxels.

The ray traversal uses Amanatides-Woo style voxel stepping.

## Simplification For Our RL Scope

For IsaacLab RL observations, most of the planner machinery is unnecessary.

Keep:

- depth backprojection
- camera-to-body transform
- local voxelization
- optional free-space ray marking
- obstacle inflation

Drop:

- ROS synchronization
- global persistent map
- previous-frame consistency filter
- full log-odds occupancy fusion
- map clearing outside a rolling world window
- planner publishing and visualization

## Recommended Representation

Use a small ego-centric BEV grid for the critic only.

Suggested format:

- shape: `2 x H x W`
- channel 0: free-space
- channel 1: occupied-space

Suggested first size:

- `H = 18`, `W = 18`
- cell size: `0.25 m`
- forward range: `0.0 m` to `4.5 m`
- lateral range: `-2.25 m` to `2.25 m`
- vertical filter in body frame: `[-0.5 m, 2.0 m]`

This is much cheaper than a full 3D grid and should be sufficient for `avoid` and `planning`.

## IsaacLab Implementation Plan

### 1. Use the raw depth output

Do not build occupancy from the normalized tensor returned by `_camera_depth_image()`. That method rescales depth to `[0, 1]`.

Use raw metric depth instead:

```python
depth_m = self._onboard_camera.data.output["depth"][..., 0]
```

Relevant current files:

- `/home/ramzi/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/airgym_x152b/base_env.py`
- `/home/ramzi/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/airgym_x152b/task_common.py`

### 2. Get camera intrinsics from IsaacLab

IsaacLab already exposes camera intrinsics:

```python
intrinsics = self._onboard_camera.data.intrinsic_matrices
```

This avoids hardcoding intrinsics and can be passed directly into IsaacLab's built-in unprojection utility.

### 3. Use IsaacLab's built-in unprojection

Primary API to use:

```python
isaaclab.utils.math.unproject_depth(depth: torch.Tensor, intrinsics: torch.Tensor, is_ortho: bool = True) -> torch.Tensor
```

This function un-projects a depth image into a pointcloud.

Recommended use in our implementation:

```python
from isaaclab.utils.math import unproject_depth

depth = self._onboard_camera.data.output["depth"][..., 0]  # (N, H, W), meters
intrinsics = self._onboard_camera.data.intrinsic_matrices  # (N, 3, 3)

pts_cam = unproject_depth(depth, intrinsics, is_ortho=True)  # (N, P, 3)
```

Why `is_ortho=True`:

- the current `TiledCamera` setup requests `"depth"`
- in IsaacLab `TiledCamera`, `"depth"` is an alias for `"distance_to_image_plane"`
- that is orthogonal depth, not optical-center distance

If the sensor is ever switched to a perspective distance type such as `distance_to_camera`, then `is_ortho=False` should be used instead.

### 4. Subsample the image

Use a fixed stride, for example `stride = 4` or `stride = 8`.

This mirrors the EGO-Planner `skip_pixel_` idea and cuts the cost immediately.

There are two valid ways to use this with `unproject_depth`:

- unproject the full image, then subsample the resulting points
- subsample the depth image first, then unproject the reduced image

The second option is cheaper and is the preferred first implementation.

### 5. Backproject to 3D in torch

With `unproject_depth`, there is no need to manually implement:

```python
pts_cam = unproject_depth(depth, intrinsics, is_ortho=True)
```

This should replace the manual pinhole backprojection block.

### 6. Transform from camera frame to drone-local frame

Use the known camera offset from the onboard camera config and keep the representation in body frame or yaw-local frame.

Body/yaw-local coordinates are the correct frame for the policy and critic.

Since `unproject_depth` returns camera-frame points with shape `(N, P, 3)`, the next step is:

```python
pts_local = camera_to_local(pts_cam)
```

### 7. Filter invalid or irrelevant points

Discard:

- `depth <= near_clip`
- `depth >= camera_max_distance`
- points outside the chosen vertical band
- points outside the chosen local XY bounds

### 8. Build occupancy

Compute BEV cell indices:

```python
ix = torch.floor((pts_local[..., 0] - x_min) / cell_size).long()
iy = torch.floor((pts_local[..., 1] - y_min) / cell_size).long()
```

Mark valid endpoint cells as occupied.

### 9. Optionally mark free-space

If free-space is desired, do not port the exact CPU DDA raycaster from EGO-Planner.

For IsaacLab, the simpler GPU-friendly version is:

- sample a small fixed number of points along each depth ray
- voxelize those intermediate points
- write them into a `free` channel
- then overwrite the final hit cell in the `occupied` channel

This preserves the useful semantics without dragging a planner-style voxel traversal into the RL loop.

### 10. Inflate occupied cells

Use pooling instead of explicit neighborhood loops:

```python
occupied = torch.nn.functional.max_pool2d(
    occupied.unsqueeze(1), kernel_size=3, stride=1, padding=1
).squeeze(1)
```

This approximates obstacle inflation cheaply on GPU.

### 11. Feed only to the critic

Return a separate observation group:

```python
{
    "observation": state_obs,
    "image": depth_obs,
    "critic_grid": critic_grid,
}
```

Then map groups in the `rsl_rl` runner config:

```python
obs_groups = {
    "actor": ["observation", "image"],
    "critic": ["observation", "critic_grid"],
}
```

This keeps the actor lightweight while giving the critic richer spatial structure.

## Minimal Torch Pseudocode

```python
from isaaclab.utils.math import unproject_depth

depth = self._onboard_camera.data.output["depth"][..., 0]  # (N, H, W), meters
depth = depth[:, ::stride, ::stride]

intrinsics = self._onboard_camera.data.intrinsic_matrices  # (N, 3, 3)

pts_cam = unproject_depth(depth, intrinsics, is_ortho=True)  # (N, P, 3)
pts_local = camera_to_local(pts_cam)

valid = torch.isfinite(pts_local).all(dim=-1)
valid = valid & (pts_local[..., 0] > x_min) & (pts_local[..., 0] < x_max)
valid = valid & (pts_local[..., 1] > y_min) & (pts_local[..., 1] < y_max)
valid = valid & (pts_local[..., 2] > z_min) & (pts_local[..., 2] < z_max)

ix = torch.floor((pts_local[..., 0] - x_min) / cell_size).long()
iy = torch.floor((pts_local[..., 1] - y_min) / cell_size).long()

occupied = scatter_points_to_grid(ix, iy, valid)
occupied = inflate(occupied)
```

## Recommendation

Implement the first version as:

- current-frame only
- body-frame or yaw-local BEV
- critic-only
- 2 channels: free + occupied
- stride-based depth subsampling
- pooling-based inflation

Do not start with a persistent 3D planner map. That would add cost and complexity without clear RL benefit.

## Optimization Opportunities Using IsaacLab APIs

Based on IsaacLab's camera-scaling guidance and the currently available sensor APIs, the occupancy-grid pipeline can be optimized substantially.

### 1. Keep `TiledCamera` for the current dynamic-scene baseline

IsaacLab's camera benchmark guidance states that tiled cameras are currently the most performant camera type that can handle multiple dynamic objects.

That matches the current AirGym setup:

- the env already uses `TiledCameraCfg`
- it only requests `"depth"`
- the camera update period is already slower than physics (`0.04 s` vs `0.01 s`)

For the `avoid` task, where the obstacle can move, `TiledCamera` is the safest baseline.

### 2. Do not recompute the occupancy grid every physics step

The camera updates at a slower rate than the simulator.

Therefore:

- compute the occupancy grid only when the camera sensor produces a new frame
- cache the result
- reuse the cached grid on intermediate physics steps

This avoids repeated depth unprojection on identical frames.

### 3. Use IsaacLab's built-in depth math utilities

IsaacLab already provides vectorized GPU helpers:

- `isaaclab.utils.math.orthogonalize_perspective_depth`
- `isaaclab.utils.math.unproject_depth`

These are a better fit than custom pixel-to-point code because they are already batched and torch-based.

Recommended path:

- use `unproject_depth(depth, intrinsics, is_ortho=True)` when reading `TiledCamera` `"depth"`
- if using a perspective-distance sensor variant later, use `orthogonalize_perspective_depth` first

This reduces custom math and keeps the whole path on GPU.

### 4. Lower the effective ray count aggressively

Because the critic only needs a compact occupancy cue:

- reduce image resolution further, or
- keep camera resolution but subsample pixels before unprojection

The current camera is `212 x 120`. For occupancy, a much smaller effective grid is usually enough.

Reasonable options:

- keep `212 x 120` and use `stride=4` or `stride=6`
- or reduce the camera to `160 x 90` or `128 x 72`

The second option reduces rendering cost. The first reduces only post-processing cost.

### 5. Keep the sensor depth-only

The benchmark guide explicitly recommends decreasing the number of data types per camera.

For this occupancy use-case, keep:

- `"depth"` only

Do not request:

- RGB
- normals
- segmentation
- motion vectors

unless another observation path actually needs them.

### 6. Use pooling and scatter ops, not Python loops

The occupancy-grid pipeline should remain:

- batched
- torch-based
- GPU-resident

Use:

- `unproject_depth`
- tensor masking
- tensorized index computation
- `scatter` or indexed writes
- `max_pool2d` for inflation

Avoid:

- per-ray Python loops
- per-pixel Python loops
- planner-style CPU voxel traversal in the training loop

### 7. A better geometry-only option exists for `planning`

If the observation only needs geometry rather than a rendered depth image, a mesh ray-caster is likely the stronger long-term option.

Relevant IsaacLab APIs:

- `MultiMeshRayCaster`
- `MultiMeshRayCasterCamera`
- `PinholeCameraPatternCfg`
- `GridPatternCfg`

Why this is attractive:

- the occupancy grid only needs geometric visibility, not raster rendering
- `MultiMeshRayCaster` supports multiple targets
- it can track dynamic meshes when needed
- static meshes can disable transform tracking for better performance
- duplicated meshes can use shared references to reduce memory and startup cost

Useful config features:

- `track_mesh_transforms=False` for static obstacles and ground
- `track_mesh_transforms=True` only for moving obstacles
- `is_shared=True` when the same mesh asset is duplicated across environments
- `reference_meshes=True` to reuse duplicate meshes in memory

### 8. Prefer `MultiMeshRayCaster` over `MultiMeshRayCasterCamera` if you only want occupancy

This is an inference from the available APIs and the occupancy objective, not a direct benchmark claim.

If the final observation is just a small occupancy grid, then a full camera-style image is unnecessary.

In that case:

- use `MultiMeshRayCaster` with a sparse pinhole-like or grid ray pattern
- collect ray hit points directly
- voxelize those hits into the critic BEV grid

This can avoid:

- tiled raster rendering
- full image storage
- image-to-pointcloud conversion

For `planning`, where obstacles are mostly static and geometric, this is likely the cleanest optimization path.

### 9. Task-specific recommendation

For `avoid`:

- keep `TiledCamera` first
- optimize with sensor-rate caching, subsampling, and GPU unprojection

For `planning`:

- seriously consider replacing the depth camera occupancy path with `MultiMeshRayCaster` or `MultiMeshRayCasterCamera`
- the task is more geometry-driven and likely benefits more from direct ray-based sensing

### 10. Benchmark the decision with IsaacLab's benchmark utility

Use:

- `scripts/benchmarks/benchmark_cameras.py`

This should be used before committing to a camera-type switch, especially if you want to compare:

- current `TiledCamera` depth baseline
- lower-resolution tiled depth
- mesh ray-caster based alternatives
