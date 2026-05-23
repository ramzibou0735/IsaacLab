# rlPx4ControllerTorch Python API Guide

This document explains how to use the Python API exposed by
`rlPx4ControllerTorch`. The package is a torch-native sibling of
`rlPx4Controller` and is intended to be an almost drop-in replacement for the
parallel controller module while keeping the controller behavior close to the
legacy implementation.

## 1. What this package provides

The public Python entrypoint is:

```python
from rlPx4ControllerTorch.pyParallelControl import (
    ParallelPosControl,
    ParallelVelControl,
    ParallelAttiControl,
    ParallelRateControl,
)
```

All four classes operate on batched `torch.Tensor` inputs and return batched
motor commands with shape `(N, 4)`.

The wrapper keeps the same controller naming and call pattern as the legacy
parallel API:

- `ParallelPosControl`
- `ParallelVelControl`
- `ParallelAttiControl`
- `ParallelRateControl`

The main intended caller change is the import path:

```python
# Old
from rlPx4Controller.pyParallelControl import ParallelPosControl

# New
from rlPx4ControllerTorch.pyParallelControl import ParallelPosControl
```

## 2. Build and import

Build with the IsaacLab interpreter that already contains the matching Torch
toolchain:

```bash
/home/ramzi/IsaacLab/env_isaaclab/bin/python setup.py build_ext --inplace
```

If your target environment has `pip`, editable install is also supported:

```bash
python -m pip install --no-build-isolation -e .
```

After a local build, import from the package root:

```python
import torch
from rlPx4ControllerTorch.pyParallelControl import ParallelPosControl
```

## 3. Device behavior

Every controller instance owns a single runtime device.

Constructor behavior:

```python
controller = ParallelPosControl(envs_num=4096, device=None)
```

Rules:

- If `device` is omitted, the wrapper chooses `cuda` when
  `torch.cuda.is_available()` is `True`, otherwise `cpu`.
- You can force a device with strings like `"cpu"`, `"cuda"`, or `"cuda:0"`.
- Inputs are converted internally to the controller device and to `float32`.
- Outputs are returned as `torch.float32` tensors on the controller device.

Inspect the active device:

```python
print(controller.device)
```

## 4. Tensor conventions

All controller inputs are `torch.Tensor`.

Quaternion order is always:

```text
[w, x, y, z]
```

Returned motor commands are normalized and shaped:

```text
(N, 4)
```

## 5. Controller APIs

### 5.1 ParallelPosControl

Position plus yaw controller. The action layout is:

```text
actions[i] = [x_sp, y_sp, z_sp, yaw_sp]
```

Usage:

```python
import torch
from rlPx4ControllerTorch.pyParallelControl import ParallelPosControl

num_envs = 2
controller = ParallelPosControl(num_envs, device="cpu")

pos = torch.zeros((num_envs, 3), dtype=torch.float32)
quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32).repeat(num_envs, 1)
vel = torch.zeros((num_envs, 3), dtype=torch.float32)
ang_vel = torch.zeros((num_envs, 3), dtype=torch.float32)

controller.set_status(pos, quat, vel, ang_vel, 0.01)

actions = torch.tensor(
    [
        [0.0, 0.0, 2.0, 0.0],
        [1.0, 0.0, 1.5, 0.2],
    ],
    dtype=torch.float32,
)

motor_commands = controller.update(actions)
print(motor_commands.shape)  # (2, 4)
```

Methods:

- `set_status(pos, q_matrix, vel, ang_vel, dt)`
- `update(actions) -> Tensor[N, 4]`

Required tensor shapes:

- `pos`: `(N, 3)`
- `q_matrix`: `(N, 4)`
- `vel`: `(N, 3)`
- `ang_vel`: `(N, 3)`
- `actions`: `(N, 4)`

### 5.2 ParallelVelControl

Velocity plus yaw controller. The action layout is:

```text
actions[i] = [vx_sp, vy_sp, vz_sp, yaw_sp]
```

Usage:

```python
import torch
from rlPx4ControllerTorch.pyParallelControl import ParallelVelControl

num_envs = 2
controller = ParallelVelControl(num_envs)

pos = torch.zeros((num_envs, 3), dtype=torch.float32)
quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32).repeat(num_envs, 1)
vel = torch.zeros((num_envs, 3), dtype=torch.float32)
ang_vel = torch.zeros((num_envs, 3), dtype=torch.float32)

controller.set_status(pos, quat, vel, ang_vel, 0.01)

actions = torch.tensor(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -0.5, 0.2, 0.1],
    ],
    dtype=torch.float32,
)

motor_commands = controller.update(actions)
```

Methods:

- `set_status(pos, q_matrix, vel, ang_vel, dt)`
- `update(actions) -> Tensor[N, 4]`

Required tensor shapes:

- `actions`: `(N, 4)`

### 5.3 ParallelAttiControl

Attitude plus thrust controller. The action layout is:

```text
actions[i] = [qw_sp, qx_sp, qy_sp, qz_sp, thrust_sp]
```

Usage:

```python
import torch
from rlPx4ControllerTorch.pyParallelControl import ParallelAttiControl

num_envs = 2
controller = ParallelAttiControl(num_envs)

pos = torch.zeros((num_envs, 3), dtype=torch.float32)
quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32).repeat(num_envs, 1)
vel = torch.zeros((num_envs, 3), dtype=torch.float32)
ang_vel = torch.zeros((num_envs, 3), dtype=torch.float32)

controller.set_status(pos, quat, vel, ang_vel, 0.01)

actions = torch.tensor(
    [
        [1.0, 0.0, 0.0, 0.0, 0.3],
        [0.9238795, 0.0, 0.3826834, 0.0, 0.4],
    ],
    dtype=torch.float32,
)

motor_commands = controller.update(actions)
```

Methods:

- `set_status(pos, q_matrix, vel, ang_vel, dt)`
- `update(actions) -> Tensor[N, 4]`

Required tensor shapes:

- `actions`: `(N, 5)`

### 5.4 ParallelRateControl

Body-rate plus thrust controller. The action layout is:

```text
actions[i] = [p_sp, q_sp, r_sp, thrust_sp]
```

Usage:

```python
import torch
from rlPx4ControllerTorch.pyParallelControl import ParallelRateControl

num_envs = 2
controller = ParallelRateControl(num_envs)

quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32).repeat(num_envs, 1)
ang_vel = torch.zeros((num_envs, 3), dtype=torch.float32)

controller.set_q_world(quat)

actions = torch.tensor(
    [
        [0.1, 0.2, 0.3, 0.4],
        [0.0, 0.1, -0.2, 0.3],
    ],
    dtype=torch.float32,
)

motor_commands = controller.update(actions, ang_vel, 0.01)
```

Methods:

- `set_q_world(q_world)`
- `update(actions, rate, dt) -> Tensor[N, 4]`

Required tensor shapes:

- `q_world`: `(N, 4)`
- `actions`: `(N, 4)`
- `rate`: `(N, 3)`

## 6. Error behavior

The wrapper validates Python-side input types and the C++ backend validates
shapes.

Examples:

- non-tensor input raises `TypeError`
- wrong matrix shape raises `RuntimeError`
- invalid or non-positive `dt` raises `RuntimeError`

## 7. Behavioral compatibility notes

The implementation intentionally preserves the current controller behavior from
`rlPx4Controller` as closely as possible.

Preserved behavior:

- hover thrust reset to `0.1533`
- controller gains and clamp ranges
- mixer desaturation ordering
- stateful rate integrator behavior

Intentionally corrected:

- obvious row-indexing mistakes in the legacy parallel wrapper logic

## 8. Running the tests and reading metrics

The parity tests compare the new package directly against the legacy package and
print metrics to the terminal.

Run them with output capture disabled:

```bash
PYTHONPATH=/home/ramzi/rlPx4Controller/rlPx4ControllerTorch/python:/home/ramzi/rlPx4Controller \
/home/ramzi/IsaacLab/env_isaaclab/bin/python -m pytest rlPx4ControllerTorch/tests -s
```

The printed metrics include:

- max absolute difference
- mean absolute difference
- root-mean-square error
- output tensor shape
- preview of the first two rows from the new output tensor
- preview of the first two rows from the legacy output tensor

The latency benchmark also prints:

- average torch update latency in microseconds
- average legacy update latency in microseconds, including CUDA tensor to CPU NumPy input conversion and legacy output conversion back to a CUDA torch tensor
- update-path speedup ratio
- average torch full-step latency in microseconds
- average legacy full-step latency in microseconds, including the Torch/NumPy bridge on each step
- full-step speedup ratio

The 100-step loop benchmark also prints:

- total torch loop time in milliseconds for 100 steps
- total legacy loop time in milliseconds for 100 steps, including the Torch/NumPy bridge on every step
- total-loop speedup ratio
- average torch step time in microseconds across the 100-step loop
- average legacy step time in microseconds across the 100-step loop, including the Torch/NumPy bridge

These metrics are printed for:

- single-step controller parity
- multi-step stateful parity for the rate controller
- latency comparisons for `rate`, `pos`, `vel`, and `atti` with the new package on CUDA and the legacy package on CPU, including CUDA Torch to NumPy to CUDA Torch bridging for the legacy path
- 100-step loop comparisons for `rate`, `pos`, `vel`, and `atti` with batch size `4000`, including CUDA Torch to NumPy to CUDA Torch bridging for the legacy path

The test suite is configured around:

- `parallel_envs = 4000`
- Torch controller execution on CUDA
- legacy controller execution on CPU with per-step CUDA Torch to NumPy input conversion and output conversion back to a CUDA torch tensor

## 9. Legacy comparison example

```python
import torch
from rlPx4ControllerTorch.pyParallelControl import ParallelRateControl as NewRate
from rlPx4Controller.pyParallelControl import ParallelRateControl as OldRate

q_world = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
rate = torch.zeros((1, 3), dtype=torch.float32)
actions = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)

new_ctl = NewRate(1, device="cpu")
old_ctl = OldRate(1)

new_ctl.set_q_world(q_world)
old_ctl.set_q_world(q_world.numpy())

new_output = new_ctl.update(actions, rate, 0.01)
old_output = torch.tensor(old_ctl.update(actions.numpy(), rate.numpy(), 0.01))

print("new:", new_output)
print("old:", old_output)
print("abs diff:", (new_output - old_output).abs())
```

## 10. Summary

Use `rlPx4ControllerTorch` when you want the same parallel controller surface as
`rlPx4Controller.pyParallelControl`, but with a torch-native backend that is
ready for GPU-oriented IsaacLab workflows.
