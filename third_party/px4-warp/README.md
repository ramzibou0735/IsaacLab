# px4-warp

GPU-native, batched PX4-style quadrotor controllers implemented as
[NVIDIA Warp](https://github.com/NVIDIA/warp) kernels.

This library is a drop-in replacement for the CPU-bound C++/Eigen
`rlPx4Controller.pyParallelControl` module. The original controllers run a
Python loop over a `std::vector` of single-drone controllers, which becomes a
hard bottleneck when training thousands of environments on a GPU. `px4-warp`
runs the whole cascaded controller (position/velocity -> attitude -> body-rate ->
mixer) as a single Warp kernel launch per step, one thread per environment, and
keeps every tensor on the GPU.

- One `wp.launch` per `update()`, one thread per environment.
- Zero-copy `torch` <-> Warp interop (`wp.from_torch` / `wp.to_torch`).
- No NumPy and no host/device copies on the control path.
- Deterministic (no atomics, fixed gains).
- Same public API and the same hardcoded gains as the original module.

The exact math is documented in [MATH_MODEL.md](MATH_MODEL.md).

## Install

Install into the environment that already provides a CUDA-enabled `torch` and
`warp` (for example an IsaacLab venv):

```bash
pip install -e . --no-deps
```

`--no-deps` avoids pip replacing the existing CUDA `torch` / local Warp build.

## API

All controllers operate on contiguous `float32` CUDA `torch` tensors on the
controller's device. Quaternions are `[w, x, y, z]` (Isaac Sim / IsaacLab
convention). `update()` returns an `(N, 4)` tensor of normalized motor commands
in `[0, 1]`.

```python
import torch
from px4_warp import (
    ParallelPosControl, ParallelVelControl, ParallelAttiControl, ParallelRateControl,
)

N = 4096
device = "cuda:0"

pos = torch.zeros((N, 3), dtype=torch.float32, device=device)
quat = torch.zeros((N, 4), dtype=torch.float32, device=device); quat[:, 0] = 1.0  # wxyz
vel = torch.zeros((N, 3), dtype=torch.float32, device=device)
ang = torch.zeros((N, 3), dtype=torch.float32, device=device)

# Position control: actions = [x_sp, y_sp, z_sp, yaw_sp]
ctl = ParallelPosControl(N, device=device)
ctl.set_status(pos, quat, vel, ang, 0.01)
actions = torch.zeros((N, 4), dtype=torch.float32, device=device)
actions[:, 2] = 2.0
motor_cmd = ctl.update(actions)            # (N, 4) on cuda:0, in [0, 1]
```

| Controller | `set_*` | `update(...)` | action columns |
| --- | --- | --- | --- |
| `ParallelPosControl` | `set_status(pos, q, vel, ang_vel, dt)` | `update(actions)` | `[x_sp, y_sp, z_sp, yaw_sp]` |
| `ParallelVelControl` | `set_status(pos, q, vel, ang_vel, dt)` | `update(actions)` | `[vx_sp, vy_sp, vz_sp, yaw_sp]` |
| `ParallelAttiControl` | `set_status(pos, q, vel, ang_vel, dt)` | `update(actions)` | `[qw, qx, qy, qz, thrust]` |
| `ParallelRateControl` | `set_q_world(q)` | `update(actions, rate, dt)` | `[p, q, r, thrust]` |

### Important: the returned buffer is reused

For minimal overhead `update()` returns a `torch` tensor that aliases a
persistent internal Warp buffer. It is overwritten on the next `update()` call.
If you need to keep a result across steps, clone it:

```python
motor_cmd = ctl.update(actions).clone()
```

### `ctl_mode` selector

To pick a controller from a config string (as AirGym does with
`cfg.env.ctl_mode`), use `ParallelController`. It maps the mode to the right
controller and exposes a single `step()` that hides each mode's call pattern,
including a `"prop"` passthrough (direct rotor commands clamped to `[0, 1]`).

```python
from px4_warp import ParallelController, ACTION_DIMS

ctl = ParallelController(cfg.env.ctl_mode, num_envs, device=sim_device)
action_dim = ctl.action_dim          # 4 for pos/vel/rate/prop, 5 for atti

# one call per step; pass the current state every time (ignored for "prop")
cmd = ctl.step(actions, pos=root_pos_w, quat=root_quat_w,
               vel=root_lin_vel_w, ang_vel=root_ang_vel_w, dt=dt)  # (N, 4) in [0, 1]
```

| `ctl_mode` | controller | action columns |
| --- | --- | --- |
| `"pos"` | `ParallelPosControl` | `[x_sp, y_sp, z_sp, yaw_sp]` |
| `"vel"` | `ParallelVelControl` | `[vx_sp, vy_sp, vz_sp, yaw_sp]` |
| `"atti"` | `ParallelAttiControl` | `[qw, qx, qy, qz, thrust]` |
| `"rate"` | `ParallelRateControl` | `[p, q, r, thrust]` |
| `"prop"` | passthrough | `[m0, m1, m2, m3]` |

`make_controller(ctl_mode, N, device)` returns the native controller instance
directly if you prefer the two-call `set_*` + `update` API.

## IsaacLab integration sketch

```python
# root_quat_w is IsaacLab's body orientation, already [w, x, y, z], float32, cuda.
ctl = ParallelPosControl(num_envs, device=sim_device)

def apply_action(actions):  # actions: (num_envs, 4) float32 cuda
    ctl.set_status(root_pos_w, root_quat_w, root_lin_vel_w, root_ang_vel_w, dt)
    cmd = ctl.update(actions)              # (num_envs, 4), stays on GPU
    forces = cmd * max_rotor_force         # feed straight back into the sim
    ...
```

Everything stays on the GPU: IsaacLab tensors flow in, motor commands flow out,
with no synchronization or host transfer.

## Scope

This library ports the four batched controllers. The single-drone `pyControl`
classes, the trajectory tools, and the `HoverThrustEkf` (which is dead code for
the controller output in the original) are intentionally not ported.
