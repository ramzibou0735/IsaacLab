# Controllers Documentation

This document explains the controller stack exposed by `rlPx4Controller`, the pybind11 Python APIs, and how these APIs are used in AirGym (`~/AirGym/airgym/envs`).

## 1) What is provided

`rlPx4Controller` exposes two pybind modules:

- `rlPx4Controller.pyControl` (single-drone building blocks)
- `rlPx4Controller.pyParallelControl` (batched controllers for vectorized RL envs)

Bindings are declared in:

- `bind/pyControl.cpp`
- `bind/pyParallelControl.cpp`

Core implementation is in `include/*.hpp`.

## 2) Controller pipeline (high level)

The default cascaded flow is:

1. **Position/Velocity controller** (`SimplePositionController`)
2. **Attitude controller** (`Px4AttitudeController`)
3. **Rate controller** (`Px4RateController`)
4. **Mixer** (`Px4Mixer`) -> 4 normalized motor commands

Python exposure nuance:

- Single-drone API exposes this first layer as `PosControl` (no standalone `VelControl` class).
- Dedicated velocity-only mode (`CTRL_VEL_ONLY`) is exposed through `ParallelVelControl`.

For modes closer to the motors, the upper layers are bypassed:

- `atti`: starts from attitude setpoint + thrust
- `rate`: starts from rate setpoint + thrust
- `prop`: bypasses all controllers and uses 4 prop commands directly (AirGym side)

## 3) API reference: `pyControl` (single vehicle)

## `PosControl`

Methods:

- `set_pid_params(pos_gains: (3,), vel_gains: (3,))`
- `set_status(pos: (3,), vel: (3,), angular_velocity: (3,), q: (4,), dt: float)`
- `update(pos_sp: (3,), vel_sp: (3,), acc_sp: (3,), yaw_sp: float) -> (5,)`
- `get_hover_thrust() -> float`

Notes:

- Quaternion input order is **`[w, x, y, z]`**.
- `update()` returns `[qw, qx, qy, qz, thrust_sp]`.
- In current implementation, hover thrust is hardcoded each status update:
  - `_hover_thrust = 0.1533`

## Velocity control in `pyControl` (single drone)

There is **no separate `VelControl` class** in `pyControl`.  
Single-drone velocity commands go through `PosControl.update(...)` by:

- setting `vel_sp` to the desired velocity command, and
- keeping `pos_sp` consistent with current position if you want to reduce position feedback.

Important: `pyControl.PosControl` does **not** expose `set_control_mode`, so true single-drone `CTRL_VEL_ONLY` is not selectable from Python.  
`ParallelVelControl` is the API path that explicitly enables `CTRL_VEL_ONLY` in this repository.

## `AttiControl`

Methods:

- `set_pid_params(p_gains: (3,))`
- `update(q_sp: (4,), q: (4,)) -> (3,)`

Returns desired body rates. Quaternion order is `[w, x, y, z]`.

## `RateControl`

Methods:

- `set_q_world(q_world: (4,))`
- `set_pid_params(p_gains: (3,), i_gains: (3,), d_gains: (3,))`
- `update(rate_sp: (3,), rate: (3,), angular_accel: (3,), dt: float) -> (3,)`

Returns desired torque vector.

## `Mixer`

Methods:

- `update(torque: (4,)) -> (4,)`

Input order: `[roll_torque, pitch_torque, yaw_torque, thrust]`  
Output: four normalized motor commands, constrained to `[0, 1]`.

## 4) API reference: `pyParallelControl` (batched)

All batched controllers output `(N, 4)` motor commands in `[0, 1]`.

## `ParallelPosControl(N)`

- `set_status(pos: (N,3), q: (N,4), vel: (N,3), ang_vel: (N,3), dt)`
- `update(actions: (N,4))`

`actions[i] = [x_sp, y_sp, z_sp, yaw_sp]`.

## `ParallelVelControl(N)`

- same `set_status(...)`
- `update(actions: (N,4))`

`actions[i] = [vx_sp, vy_sp, vz_sp, yaw_sp]`.

Velocity-mode flow in implementation:

1. `SimplePositionController` runs in `CTRL_VEL_ONLY`.
2. Its output attitude/thrust setpoint is passed to `Px4AttitudeController`.
3. `Px4RateController` computes torque setpoint.
4. `Px4Mixer` converts to 4 motor commands.

## `ParallelAttiControl(N)`

- same `set_status(...)`
- `update(actions: (N,5))`

`actions[i] = [qw_sp, qx_sp, qy_sp, qz_sp, thrust_sp]`.

## `ParallelRateControl(N)`

- `set_q_world(q: (N,4))`
- `update(actions: (N,4), observations: (N,3), dt)`

`actions[i] = [p_sp, q_sp, r_sp, thrust_sp]`,  
`observations[i]` is angular velocity vector used by rate control.

## 5) Inner control logic (important behavior)

## Position/velocity layer

`SimplePositionController` computes:

- `des_acc = acc_sp + Kv*(vel_sp-vel) + Kp*(pos_sp-pos)` (depending on mode)
- `des_acc` is clamped axis-wise to `[-4, 4]`
- thrust setpoint:
  - `thrust_sp = des_acc.z * (hover_thrust / g) + hover_thrust`

Then it computes roll/pitch from lateral acceleration and current yaw, builds quaternion setpoint, and returns `[q_sp, thrust_sp]`.

## Attitude layer

`Px4AttitudeController` uses quaternion attitude error and proportional gains to generate body-rate setpoints, with rate limits.

Default gains/rates:

- P gains initialized to `(8.0, 8.0, 1.0)` (yaw weighted)
- Rate limits about `(1600/57.3, 1600/57.3, 1000/57.3)` rad/s

## Rate layer

`Px4RateController` computes:

- body-rate error (after frame transform via `q_world`)
- torque = `P*error + I - D*angular_accel`

Default gains:

- `P = (0.5, 0.5, 0.2)`
- `I = (0.08, 0.08, 0.05)`
- `D = (0.001, 0.001, 0.0)`

## Mixer layer

`Px4Mixer` applies a Quad-X mixing matrix and desaturation steps (thrust/roll/pitch/yaw), then clamps outputs to `[0, 1]`.

## 6) How AirGym uses these controllers

Controllers are imported and selected by `cfg.env.ctl_mode` in:

- `~/AirGym/airgym/envs/base/hovering.py`
- `~/AirGym/airgym/envs/base/customized.py`
- `~/AirGym/airgym/envs/base/depthgen.py`
- `~/AirGym/airgym/envs/task/tracking.py`
- `~/AirGym/airgym/envs/task/maplanning.py`

Mode-to-controller mapping used in AirGym:

- `ctl_mode == "pos"` -> `ParallelPosControl`, actions `[x_sp, y_sp, z_sp, yaw_sp]`
- `ctl_mode == "vel"` -> `ParallelVelControl`, actions `[vx_sp, vy_sp, vz_sp, yaw_sp]`
- `ctl_mode == "atti"` -> `ParallelAttiControl`, actions `[qw_sp, qx_sp, qy_sp, qz_sp, thrust]`
- `ctl_mode == "rate"` -> `ParallelRateControl`, actions `[p_sp, q_sp, r_sp, thrust]`
- `ctl_mode == "prop"` -> no controller, actions are direct rotor commands

Common per-step pattern:

1. Read IsaacGym root states.
2. Reorder quaternion from IsaacGym `[x,y,z,w]` to controller `[w,x,y,z]`.
3. For `rate` and `atti`, remap last action from `[-1,1]` to `[0,1]`:
   - `a_last = 0.5 + 0.5 * a_last`
4. Clamp actions by mode-specific limits.
5. Call controller (`set_status` and `update`, or `set_q_world` + `update` for `rate`).
6. Receive normalized motor commands `cmd_thrusts` in `[0,1]`.
7. Convert to simulator force:
   - `thrust_force = cmd_thrusts * 9.59` (per-rotor Z force in AirGym code path)
8. Apply rotor forces and yaw reaction torques in IsaacGym.

## Multi-agent case (`maplanning.py`)

- One parallel controller instance is created **per robot index**.
- Each instance is still batched over `num_envs`.
- Per-step loop iterates robots, runs controller, and writes commands into `self.cmd_thrusts[:, i, :]`.

## 7) Minimal usage examples

## Single drone

```python
import numpy as np
from rlPx4Controller.pyControl import PosControl, AttiControl, RateControl, Mixer

pos = PosControl()
atti = AttiControl()
rate = RateControl()
mix = Mixer()

q_wxyz = np.array([1., 0., 0., 0.])
pos.set_status(np.zeros(3), np.zeros(3), np.zeros(3), q_wxyz, 0.01)
atti_thrust = pos.update(np.array([0., 0., 1.5]), np.zeros(3), np.zeros(3), 0.0)
rate_sp = atti.update(atti_thrust[:4], q_wxyz)
rate.set_q_world(q_wxyz)
torque = rate.update(rate_sp, np.zeros(3), np.zeros(3), 0.01)
motor = mix.update(np.array([torque[0], torque[1], torque[2], atti_thrust[4]]))
```

## Single drone velocity-mode path (no separate `VelControl` class)

```python
import numpy as np
from rlPx4Controller.pyControl import PosControl

pos = PosControl()
q_wxyz = np.array([1., 0., 0., 0.])
current_pos = np.array([0., 0., 1.0])
pos.set_status(
    pos=current_pos,
    vel=np.zeros(3),
    angular_velocity=np.zeros(3),
    q=q_wxyz,
    dt=0.01,
)
# Velocity command: [vx, vy, vz], yaw command in radians
atti_thrust = pos.update(
    pos_sp=current_pos,           # keeps position error small
    vel_sp=np.array([1.0, 0.0, 0.0]),
    acc_sp=np.zeros(3),
    yaw_sp=0.0,
)
```

## Batched position control

```python
import numpy as np
from rlPx4Controller.pyParallelControl import ParallelPosControl

N = 256
ctl = ParallelPosControl(N)
ctl.set_status(
    pos=np.zeros((N, 3)),
    q_matrix=np.tile(np.array([[1., 0., 0., 0.]]), (N, 1)),  # wxyz
    vel=np.zeros((N, 3)),
    ang_vel=np.zeros((N, 3)),
    dt=0.01,
)
actions = np.zeros((N, 4))   # [x_sp, y_sp, z_sp, yaw_sp]
motor_cmd = ctl.update(actions.astype(np.float64))
```

## Batched velocity control

```python
import numpy as np
from rlPx4Controller.pyParallelControl import ParallelVelControl

N = 256
ctl = ParallelVelControl(N)
ctl.set_status(
    pos=np.zeros((N, 3)),
    q_matrix=np.tile(np.array([[1., 0., 0., 0.]]), (N, 1)),  # wxyz
    vel=np.zeros((N, 3)),
    ang_vel=np.zeros((N, 3)),
    dt=0.01,
)
actions = np.zeros((N, 4))   # [vx_sp, vy_sp, vz_sp, yaw_sp]
motor_cmd = ctl.update(actions.astype(np.float64))
```

## 8) Integration checklist

- Use quaternion order **`w,x,y,z`** when calling controller APIs.
- Keep matrix shapes exact (`N x dim`) to satisfy internal assertions.
- Pass consistent `dt` (AirGym currently uses `0.01` in controller calls).
- Expect motor command output in `[0,1]`; convert to simulator thrust externally.
- If hover behavior differs across vehicles, calibrate hover thrust handling in `SimplePositionController`.
