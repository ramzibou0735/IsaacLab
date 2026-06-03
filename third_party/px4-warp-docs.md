# px4-warp — Complete Documentation

GPU-native, batched PX4-style quadrotor controllers implemented as
[NVIDIA Warp](https://github.com/NVIDIA/warp) kernels, designed as a drop-in,
zero-copy replacement for the CPU-bound C++/Eigen `rlPx4Controller.pyParallelControl`
module used to drive vectorized RL environments (e.g. IsaacLab / Isaac Sim).

- **Version:** 0.1.0
- **Import name:** `px4_warp`
- **Quaternion convention:** `[w, x, y, z]`
- **Tensor contract:** contiguous `float32` `torch` tensors on the controller device
- **Math spec:** see [`MATH_MODEL.md`](MATH_MODEL.md) (exact equations + C++ provenance)

---

## Table of contents

1. [Why this library](#1-why-this-library)
2. [Installation](#2-installation)
3. [Quickstart](#3-quickstart)
4. [Architecture](#4-architecture)
5. [I/O contract](#5-io-contract)
6. [Zero-copy buffer semantics](#6-zero-copy-buffer-semantics)
7. [API reference](#7-api-reference)
8. [`ctl_mode` selector](#8-ctl_mode-selector)
9. [IsaacLab integration](#9-isaaclab-integration)
10. [Mathematical model & gains](#10-mathematical-model--gains)
11. [Performance](#11-performance)
12. [Determinism & numerical parity](#12-determinism--numerical-parity)
13. [Testing](#13-testing)
14. [Scope & non-goals](#14-scope--non-goals)
15. [Project layout](#15-project-layout)
16. [Troubleshooting / FAQ](#16-troubleshooting--faq)

---

## 1. Why this library

The original controllers run a Python loop over a `std::vector` of single-drone
C++ controllers. With thousands of parallel environments this becomes a hard
CPU bottleneck and forces a host↔device copy of every state/action tensor each
step. `px4-warp` runs the **entire cascaded controller** (position/velocity →
attitude → body-rate → mixer) as a **single Warp kernel launch per step, one
thread per environment**, and keeps every tensor on the GPU.

Key properties:

- One `wp.launch` per `update()`, one thread per environment.
- Zero-copy `torch` ↔ Warp interop (`wp.from_torch` / `wp.to_torch`).
- No NumPy and no host/device copies on the control path.
- Deterministic (no atomics, no reductions, fixed gains).
- Same public API and the same hardcoded gains as the original module.

---

## 2. Installation

`px4-warp` depends on a CUDA-enabled `torch` and `warp-lang`, which an IsaacLab
environment already provides. Install **without** dependencies so pip/uv does not
replace the existing CUDA `torch` or local Warp build:

```bash
# from the px4-warp/ project root, into the IsaacLab venv
uv pip install -e . --no-deps --python /home/ramzi/IsaacLab/env_isaaclab/bin/python
```

or, with plain pip from inside the activated venv:

```bash
pip install -e . --no-deps
```

Editable (`-e`) installs keep the package pointing at the source tree, so edits
take effect immediately. To verify:

```bash
python -c "import px4_warp; print(px4_warp.__version__)"
```

> If you relocate the project (e.g. to `IsaacLab/third_party/px4-warp`), re-run
> the editable install from the new path to make that copy the active one.

---

## 3. Quickstart

```python
import torch
from px4_warp import ParallelPosControl

N = 4096
device = "cuda:0"

# State (from the simulator). Quaternions are [w, x, y, z].
pos  = torch.zeros((N, 3), dtype=torch.float32, device=device)
quat = torch.zeros((N, 4), dtype=torch.float32, device=device); quat[:, 0] = 1.0
vel  = torch.zeros((N, 3), dtype=torch.float32, device=device)
ang  = torch.zeros((N, 3), dtype=torch.float32, device=device)

ctl = ParallelPosControl(N, device=device)

# actions = [x_sp, y_sp, z_sp, yaw_sp]
actions = torch.zeros((N, 4), dtype=torch.float32, device=device)
actions[:, 2] = 2.0

ctl.set_status(pos, quat, vel, ang, dt=0.01)
motor_cmd = ctl.update(actions)        # (N, 4) float32 on cuda:0, each in [0, 1]
```

Or pick the controller from a config string with the selector:

```python
from px4_warp import ParallelController

ctl = ParallelController("pos", N, device=device)   # "pos"|"vel"|"atti"|"rate"|"prop"
motor_cmd = ctl.step(actions, pos=pos, quat=quat, vel=vel, ang_vel=ang, dt=0.01)
```

---

## 4. Architecture

All controllers share one cascaded pipeline; each layer feeds setpoints to the
next. Different controller classes simply enter the cascade at a different layer.

```
            actions + state (pos, quat, vel, ang_vel)
                              │
        ┌─────────────────────┴───────────────────────┐
 pos/vel│  Layer 1  Position / Velocity                │
        │     → desired accel → (q_sp, thrust_sp)       │
        └─────────────────────┬───────────────────────┘
   atti │  Layer 2  Attitude                            │  enters here
        │     (q_sp, q) → rate_sp (body)                │  (actions = q_sp, thrust)
        └─────────────────────┬───────────────────────┘
   rate │  Layer 3  Body-rate PID (+ integrator state)  │  enters here
        │     (rate_sp, measured rate) → torque         │  (actions = rates, thrust)
        └─────────────────────┬───────────────────────┘
   prop │  Layer 4  Quad-X Mixer (+ desaturation)       │  bypassed
        │     [torque, thrust] → 4 motor cmds in [0,1]  │  (actions = motor cmds)
        └─────────────────────┬───────────────────────┘
                              ▼
                  motor commands (N, 4) ∈ [0, 1]
```

Implementation: the Python classes in `controllers.py` do device resolution,
input validation, and persistent-buffer management, then launch one of three
kernels in `_kernels.py` (`pos_vel_update`, `atti_update`, `rate_update`). Each
kernel runs the full remaining cascade per thread in registers.

The only state carried between calls is the **body-rate integrator**
`rate_int` of shape `(N, 3)`; everything else is recomputed each step.

---

## 5. I/O contract

The contract is intentionally strict so nothing silently copies to the host.
Every tensor argument must be:

| Requirement | Rule | Violation raises |
| --- | --- | --- |
| Type | a `torch.Tensor` | `TypeError` |
| dtype | `torch.float32` | `ValueError` |
| device | the controller's device | `ValueError` |
| shape | exactly `(envs_num, cols)` (2-D) | `ValueError` |
| memory | contiguous | `ValueError` |

Additional rules:

- `envs_num` must be a positive integer (else `ValueError` at construction).
- Calling `update()` before `set_status()` / `set_q_world()` raises `RuntimeError`.
- Quaternions are `[w, x, y, z]`. They are normalized inside the kernels, so they
  need not be pre-normalized.
- Outputs are always `(N, 4)` `float32` motor commands clamped to `[0, 1]`, on the
  controller device.

Column layouts:

| Tensor | shape | columns |
| --- | --- | --- |
| `pos`, `vel`, `ang_vel`, `rate` | `(N, 3)` | `[x, y, z]` |
| `quat` / `q_world` | `(N, 4)` | `[w, x, y, z]` |
| pos/vel actions | `(N, 4)` | `[sp_x, sp_y, sp_z, yaw_sp]` |
| atti actions | `(N, 5)` | `[qw, qx, qy, qz, thrust]` |
| rate / prop actions | `(N, 4)` | `[p, q, r, thrust]` / `[m0, m1, m2, m3]` |
| output | `(N, 4)` | `[m0, m1, m2, m3]` ∈ `[0, 1]` |

---

## 6. Zero-copy buffer semantics

For minimal overhead, `update()` (and `step()`) returns a `torch` tensor that
**aliases a persistent internal Warp buffer**. The same memory is overwritten on
the next call. If you need to keep a result across steps, clone it:

```python
cmd = ctl.update(actions).clone()      # safe to keep
```

Inputs are also consumed zero-copy via `wp.from_torch`, so the controller reads
directly from your simulator's state tensors — do not free/resize them between
`set_status()` and `update()`.

On CUDA, kernels are launched on PyTorch's current stream
(`wp.stream_from_torch(torch.cuda.current_stream())`), so ordering with
surrounding torch ops is correct **without any explicit synchronization** on the
control path.

---

## 7. API reference

All controllers live in `px4_warp` and accept `(envs_num: int, device=None)`.
`device` defaults to `"cuda:0"` when CUDA is available, else `"cpu"`.

Common attributes / methods on every controller:

- `.envs_num: int` — number of environments.
- `.device: str` — resolved torch device string (e.g. `"cuda:0"`).
- `.reset() -> None` — zero the body-rate integrator state for all envs (use at
  episode boundaries to avoid wind-up carrying across resets).

### 7.1 `ParallelPosControl(envs_num, device=None)`

Position + yaw controller (enters at Layer 1, `mode = ALL`).

- `set_status(pos, q_matrix, vel, ang_vel, dt) -> None`
  - `pos (N,3)`, `q_matrix (N,4)`, `vel (N,3)`, `ang_vel (N,3)`, `dt: float`.
- `update(actions) -> torch.Tensor`
  - `actions (N,4) = [x_sp, y_sp, z_sp, yaw_sp]` → `(N,4)` motor commands.

### 7.2 `ParallelVelControl(envs_num, device=None)`

Velocity + yaw controller (Layer 1, `mode = VEL_ONLY`). Same methods as
`ParallelPosControl`; `actions (N,4) = [vx_sp, vy_sp, vz_sp, yaw_sp]`.

### 7.3 `ParallelAttiControl(envs_num, device=None)`

Attitude + thrust controller (enters at Layer 2).

- `set_status(pos, q_matrix, vel, ang_vel, dt) -> None`
  - `pos` and `vel` are validated for API parity but unused by this controller.
- `update(actions) -> torch.Tensor`
  - `actions (N,5) = [qw_sp, qx_sp, qy_sp, qz_sp, thrust_sp]` → `(N,4)`.

### 7.4 `ParallelRateControl(envs_num, device=None)`

Body-rate + thrust controller (enters at Layer 3).

- `set_q_world(q_world) -> None`
  - `q_world (N,4)` — world orientation used to map the measured rate into body frame.
- `update(actions, rate, dt) -> torch.Tensor`
  - `actions (N,4) = [p_sp, q_sp, r_sp, thrust_sp]`, `rate (N,3)` measured
    angular velocity (world frame), `dt: float` → `(N,4)`.

### 7.5 Selector helpers

- `ParallelController(ctl_mode, envs_num, device=None)` — unified wrapper, see [§8](#8-ctl_mode-selector).
- `make_controller(ctl_mode, envs_num, device=None)` — factory returning the
  native controller instance (or `PropController` for `"prop"`).
- `PropController(envs_num, device=None)` — passthrough for direct rotor commands;
  `update(actions (N,4)) -> (N,4)` clamps to `[0, 1]` into the persistent buffer.
- `CTL_MODES: tuple` — `("pos", "vel", "atti", "rate", "prop")`.
- `ACTION_DIMS: dict` — `{"pos":4, "vel":4, "atti":5, "rate":4, "prop":4}`.

### 7.6 Exceptions summary

| Exception | Cause |
| --- | --- |
| `TypeError` | a tensor argument is not a `torch.Tensor` |
| `ValueError` | wrong dtype / device / shape / non-contiguous / `envs_num <= 0` / unknown `ctl_mode` |
| `RuntimeError` | `update()` called before `set_status()` / `set_q_world()` |

---

## 8. `ctl_mode` selector

`ParallelController` mirrors the AirGym `cfg.env.ctl_mode` mapping and exposes a
single `step()` that hides each mode's call pattern (including the `prop`
passthrough). Ideal for an RL env that selects a controller from config.

```python
from px4_warp import ParallelController, ACTION_DIMS

ctl = ParallelController(cfg.env.ctl_mode, num_envs, device=sim_device)
action_dim = ctl.action_dim          # size your policy's action head

cmd = ctl.step(
    actions,
    pos=root_pos_w, quat=root_quat_w,
    vel=root_lin_vel_w, ang_vel=root_ang_vel_w,
    dt=dt,
)                                     # (N, 4) ∈ [0, 1]
```

`step(actions, pos=None, quat=None, vel=None, ang_vel=None, dt=0.01)` dispatches:

| `ctl_mode` | controller | `step` uses | action columns |
| --- | --- | --- | --- |
| `"pos"` | `ParallelPosControl` | `set_status` + `update` | `[x, y, z, yaw]` |
| `"vel"` | `ParallelVelControl` | `set_status` + `update` | `[vx, vy, vz, yaw]` |
| `"atti"` | `ParallelAttiControl` | `set_status` + `update` | `[qw, qx, qy, qz, thrust]` |
| `"rate"` | `ParallelRateControl` | `set_q_world(quat)` + `update(actions, ang_vel, dt)` | `[p, q, r, thrust]` |
| `"prop"` | `PropController` | `update` (clamp to `[0,1]`) | `[m0, m1, m2, m3]` |

Notes:

- `ctl_mode` is case-insensitive; an unknown value raises `ValueError`.
- For every mode except `"prop"`, `step()` requires `pos`, `quat`, `vel`,
  `ang_vel` (else `ValueError`); `"rate"` requires at least `quat` and `ang_vel`.
- Extra attributes: `.ctl_mode`, `.action_dim`, `.envs_num`, `.device`,
  `.impl` (the underlying native controller), and `.reset()`.
- `"prop"` clamps direct rotor commands to `[0, 1]` to stay consistent with the
  mixer's normalized output range and reuses the same persistent buffer.

If you prefer the explicit two-call API, use `make_controller(...)` to get the
native instance and call `set_*` / `update` yourself.

---

## 9. IsaacLab integration

IsaacLab root-state tensors are already `float32` CUDA tensors and its quaternions
are `[w, x, y, z]`, so they feed in directly with no conversion.

```python
from px4_warp import ParallelController

class MyDroneEnv:
    def __init__(self, cfg):
        self.ctl = ParallelController(cfg.ctl_mode, self.num_envs, device=self.device)
        # size the action space from the selected mode:
        self.action_dim = self.ctl.action_dim

    def _apply_action(self, actions):              # (num_envs, action_dim), f32, cuda
        cmd = self.ctl.step(
            actions,
            pos=self.robot.data.root_pos_w,
            quat=self.robot.data.root_quat_w,       # [w, x, y, z]
            vel=self.robot.data.root_lin_vel_w,
            ang_vel=self.robot.data.root_ang_vel_w,
            dt=self.physics_dt,
        )
        forces = cmd * self.max_rotor_force         # (num_envs, 4), still on GPU
        self.robot.set_external_force_and_torque(...)  # feed straight back into sim

    def _reset_idx(self, env_ids):
        self.ctl.reset()                            # clear rate integrator wind-up
```

Everything stays on the GPU: IsaacLab tensors flow in, motor commands flow out,
with no host transfer and no explicit synchronization.

> If your policy outputs actions in `[-1, 1]` (common for thrust), remap to the
> controller's expected ranges in the env before calling `step()`; `px4-warp`
> does not assume a normalization scheme beyond the documented column layouts.

---

## 10. Mathematical model & gains

The full equations, the C++ source they were extracted from, and the quaternion
helpers are documented in [`MATH_MODEL.md`](MATH_MODEL.md). Summary of the
hardcoded constants (identical to the original C++/Eigen controllers):

| Symbol | Value | Meaning |
| --- | --- | --- |
| `g` | `9.80665` | gravity [m/s²] |
| `hover_thrust` | `0.1533` | fixed normalized hover throttle |
| `Kp` | `(1.5, 1.5, 1.5)` | position P gain |
| `Kv` | `(1.5, 1.5, 1.5)` | velocity P gain |
| `acc_lim` | `4.0` | per-axis desired-acceleration clamp |
| `att_P` | `(8.0, 8.0, 2.5)` | attitude P gain (yaw = 1.0 / yaw_w) |
| `yaw_w` | `0.4` | attitude yaw weight |
| `rate_lim` | `(1600/57.3, 1600/57.3, 1000/57.3)` | rate-setpoint clamp [rad/s] |
| `rate_P` | `(0.5, 0.5, 0.2)` | body-rate P gain |
| `rate_I` | `(0.08, 0.08, 0.05)` | body-rate I gain |
| `rate_D` | `(0.001, 0.001, 0.0)` | body-rate D gain |
| `int_lim` | `(0.3, 0.3, 0.3)` | integrator clamp |
| `i_factor_ref` | `radians(400)` | integral anti-windup reference |

Quad-X mixer per-rotor scales (rotor order matches `_config_quad_x`):

| rotor | roll | pitch | yaw | thrust |
| --- | --- | --- | --- | --- |
| 0 | -0.707107 | -0.707107 | -1.0 | 1.0 |
| 1 | +0.707107 | +0.707107 | -1.0 | 1.0 |
| 2 | +0.707107 | -0.707107 | +1.0 | 1.0 |
| 3 | -0.707107 | +0.707107 | +1.0 | 1.0 |

Fidelity notes:

- The original `SimplePositionController` runs a `HoverThrustEkf`, then overwrites
  its result with the constant `0.1533`. The EKF is therefore dead code for the
  output and is **omitted** here (deterministic, identical output).
- `Px4AttitudeController` initializes P gain `(8.0, 8.0, 1.0)` then divides the yaw
  term by `yaw_w = 0.4`, giving the effective `(8.0, 8.0, 2.5)` used directly.
- The `yawspeed_setpoint` feed-forward is `0`, so it is dropped.

---

## 11. Performance

`examples/benchmark.py` compares a full `set_status` + `update` step against the
legacy CPU C++ controller across batch sizes:

```bash
python examples/benchmark.py --envs 4096 --steps 200
```

Representative results on an RTX 3060 Laptop GPU (your numbers will vary with
hardware, batch size, and surrounding workload):

| envs (N) | speedup vs legacy CPU |
| --- | --- |
| 4096 | ~13.7× |
| 16384 | ~54.7× |

The speedup grows with `N` because the Warp path is GPU-parallel and copy-free
while the legacy path scales with a per-env CPU loop plus host↔device transfers.

---

## 12. Determinism & numerical parity

- The pipeline is branch-per-thread but contains **no atomics or reductions**, so
  output is deterministic across runs for a given input.
- The original C++ uses `double` for most of the cascade; `px4-warp` runs the
  whole pipeline in `float32` for zero-copy interop. Parity with the C++ ground
  truth holds to about **`1e-4` absolute** on representative inputs (the tolerance
  used by the parity tests).

---

## 13. Testing

Run the suite with the IsaacLab interpreter from the project root:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Coverage:

- `tests/test_parity_legacy.py` — numerical parity against the legacy C++
  `rlPx4Controller.pyParallelControl` (skipped automatically if it is not
  importable). Single-env cases exercise diverse inputs for rigorous math checks;
  batched cases use uniform rows to also validate the multi-step integrator.
- `tests/test_api.py` — output shape/dtype/device/range, zero-copy buffer
  aliasing, integrator evolution + `reset()`, determinism, and the full input
  validation matrix.
- `tests/test_selector.py` — factory typing, case-insensitivity, unknown-mode
  errors, per-mode `step()` shape/range, `prop` clamp + buffer aliasing,
  `step()` parity vs the native classes, and missing-state errors.

All tests pass on CUDA; they fall back to CPU when no GPU is present.

---

## 14. Scope & non-goals

This library ports the **four batched controllers** and adds the `ctl_mode`
selector. Intentionally **not** ported:

- The single-drone `pyControl` classes.
- The trajectory tools (`traj_tools`).
- The `HoverThrustEkf` (dead code for the controller output; see §10).

---

## 15. Project layout

```
px4-warp/
├── pyproject.toml              # build/metadata; deps: warp-lang>=1.0, torch>=2.0
├── README.md                   # short overview / quickstart
├── MATH_MODEL.md               # exact equations + C++ provenance
├── px4-warp-docs.md            # this document
├── src/px4_warp/
│   ├── __init__.py             # public exports + __version__
│   ├── _kernels.py             # wp.constant gains, @wp.func helpers, @wp.kernel cascade
│   ├── controllers.py          # public controller classes + interop/validation
│   ├── selector.py             # ParallelController, make_controller, PropController
│   └── py.typed                # PEP 561 typing marker
├── tests/
│   ├── conftest.py             # adds src/ to sys.path
│   ├── test_parity_legacy.py
│   ├── test_api.py
│   └── test_selector.py
└── examples/
    └── benchmark.py            # GPU vs legacy CPU timing
```

---

## 16. Troubleshooting / FAQ

**`RuntimeError: call set_status() before update()`** — you must set state each
step before `update()`. With the selector, use `step()` which does both.

**`ValueError: <name> must be float32 / on device ... / contiguous`** — the input
contract is strict. Cast with `.float()`, move with `.to(device)`, and make
contiguous with `.contiguous()` *before* passing tensors in (ideally keep your
sim tensors in the right format to avoid per-step allocations).

**My returned commands changed unexpectedly** — the output buffer is reused; call
`.clone()` if you need to retain a result across steps (see §6).

**`pip` replaced my CUDA torch / Warp** — always install with `--no-deps`.

**Quaternion looks wrong** — ensure `[w, x, y, z]` order (not `[x, y, z, w]`).
Inputs do not need to be normalized; the kernels normalize internally.

**Does `prop` need state?** — no. `step()` ignores `pos/quat/vel/ang_vel` for
`"prop"` and just clamps the four rotor commands to `[0, 1]`.

**How do I clear integrator wind-up at episode reset?** — call `ctl.reset()`.
