# PX4 Parallel Controller - Mathematical Model

This document extracts the exact mathematical model implemented by the CPU-bound
C++/Eigen controllers in `rlPx4Controller` (`include/*.hpp`, exposed through
`bind/pyParallelControl.cpp`). It is the specification implemented by the Warp
kernels in `px4_warp`.

The model was cross-checked against two sources:

- The original C++ headers: `SimplePositionController.hpp`, `Px4AttitudeController.hpp`,
  `Px4RateController.hpp`, `Px4Mixer.hpp`.
- The vectorized, parity-tested reference in
  `rlPx4ControllerTorch/csrc/parallel_controllers.cpp`.

## Conventions

- Quaternions are stored `[w, x, y, z]` (Eigen / Isaac Sim / IsaacLab convention).
- All vectors are 3D world-frame unless noted; rates returned by the rate layer
  are body-frame.
- `N` is the number of parallel environments (one drone per env).
- Gravity `g = 9.80665 m/s^2`.
- A rotation matrix `R(q)` is the standard quaternion-to-matrix map; `R(q).col(2)`
  is the body z-axis expressed in world, and `R(q)^T` maps world vectors to body.

### Quaternion helpers

Given `q = (w, x, y, z)`:

- Hamilton product `q_a * q_b` (used everywhere "multiply" appears).
- Inverse of a unit quaternion: `q^-1 = (w, -x, -y, -z)`.
- Body z-axis (third column of `R(q)`):
  - `e_z = ( 2(xz + yw), 2(yz - xw), 1 - 2(x^2 + y^2) )`
- World-to-body rotation of a vector `r` (i.e. `R(q)^T r`):
  - `b_x = (1 - 2y^2 - 2z^2) r_x + (2xy + 2zw) r_y + (2xz - 2yw) r_z`
  - `b_y = (2xy - 2zw) r_x + (1 - 2x^2 - 2z^2) r_y + (2yz + 2xw) r_z`
  - `b_z = (2xz + 2yw) r_x + (2yz - 2xw) r_y + (1 - 2x^2 - 2y^2) r_z`
- Yaw from quaternion (`fromQuaternion2yaw`):
  - `yaw = atan2( 2(xy + wz), w^2 + x^2 - y^2 - z^2 )`
- Quaternion from roll/pitch/yaw, `R = Rz(yaw) Ry(pitch) Rx(roll)`
  (with half-angles `cr = cos(roll/2)` etc.):
  - `w = cr cp cy + sr sp sy`
  - `x = sr cp cy - cr sp sy`
  - `y = cr sp cy + sr cp sy`
  - `z = cr cp sy - sr sp cy`
- Canonical form: multiply all components by `sign(w)` (sign(0) = 0), so `w >= 0`.

## Constants and gains

| Symbol | Value | Meaning |
| --- | --- | --- |
| `g` | `9.80665` | gravity |
| `hover_thrust` | `0.1533` | fixed normalized hover throttle (see note) |
| `Kp` | `(1.5, 1.5, 1.5)` | position P gain |
| `Kv` | `(1.5, 1.5, 1.5)` | velocity P gain |
| `acc_lim` | `4.0` | per-axis desired-acceleration clamp |
| `att_P` | `(8.0, 8.0, 2.5)` | attitude P gain (effective, see note) |
| `yaw_w` | `0.4` | attitude yaw weight |
| `rate_lim` | `(1600/57.3, 1600/57.3, 1000/57.3)` | attitude rate-setpoint clamp [rad/s] |
| `rate_P` | `(0.5, 0.5, 0.2)` | body-rate P gain |
| `rate_I` | `(0.08, 0.08, 0.05)` | body-rate I gain |
| `rate_D` | `(0.001, 0.001, 0.0)` | body-rate D gain |
| `int_lim` | `(0.3, 0.3, 0.3)` | integrator clamp |
| `i_factor_ref` | `radians(400)` | integral anti-windup reference |

Notes:

- The original `SimplePositionController::set_status` runs a `HoverThrustEkf` and a
  velocity `Derivate`, but then overwrites the result with `_hover_thrust = 0.1533`.
  The EKF output is therefore dead code for the controller output. `px4_warp` omits
  the EKF entirely (deterministic, identical output).
- `Px4AttitudeController` initializes P gain `(8.0, 8.0, 1.0)`, then divides the yaw
  term by `yaw_w = 0.4`, giving the effective gain `(8.0, 8.0, 2.5)`.
- The `yawspeed_setpoint` feed-forward is `0`, so it is dropped.

## Persistent state

The only state carried between `update()` calls is the rate-controller integrator
`rate_int` (shape `(N, 3)`), one per env. Everything else is recomputed from the
status set each step.

## Layer 1 - Position / Velocity (`SimplePositionController`)

Inputs: current `pos`, `vel`, `q`; setpoints `pos_sp`, `vel_sp`, `acc_sp`,
scalar `yaw_sp`; a `mode` in `{ALL, POS_ONLY, VEL_ONLY}`.

Desired acceleration:

- `ALL`:      `des_acc = acc_sp + Kv (vel_sp - vel) + Kp (pos_sp - pos)`
- `POS_ONLY`: `des_acc = acc_sp + Kp (pos_sp - pos)`
- `VEL_ONLY`: `des_acc = acc_sp + Kv (vel_sp - vel)`

Then, per axis: `des_acc = clamp(des_acc, -acc_lim, +acc_lim)`.

Thrust setpoint (normalized):

- `thrust_sp = des_acc_z * (hover_thrust / g) + hover_thrust`

Tilt from horizontal acceleration, using current yaw `psi = yaw(q)`:

- `roll  = ( des_acc_x sin(psi) - des_acc_y cos(psi) ) / g`
- `pitch = ( des_acc_x cos(psi) + des_acc_y sin(psi) ) / g`

Attitude setpoint quaternion: `q_sp = quat_from_rpy(roll, pitch, yaw_sp)`.

Output: `(q_sp, thrust_sp)`.

In the batched pipelines: `acc_sp = 0`; pos control uses `mode = ALL` with
`pos_sp = action[:3]`, `vel_sp = 0`; vel control uses `mode = VEL_ONLY` with
`vel_sp = action[:3]`, `pos_sp = 0`. `yaw_sp = action[3]`.

## Layer 2 - Attitude (`Px4AttitudeController`)

Inputs: desired `q_sp`, current `q`. Both normalized first.

1. Thrust directions: `e_z = R(q).col(2)`, `e_z_d = R(q_sp).col(2)`.
2. Reduced attitude error `qd_red = atti_err(e_z, e_z_d)` (shortest-arc quaternion
   rotating `e_z` onto `e_z_d`, defined below).
3. If `|qd_red_x| > 1 - 1e-5` or `|qd_red_y| > 1 - 1e-5` (thrust nearly opposite):
   `qd_red = q_sp`; else `qd_red = qd_red * q` (Hamilton product).
4. Mixing of full and reduced attitude:
   - `q_mix = canonical( qd_red^-1 * q_sp )`
   - clamp `q_mix_w` and `q_mix_z` to `[-1, 1]`
   - `q_yaw = ( cos(yaw_w * acos(q_mix_w)), 0, 0, sin(yaw_w * asin(q_mix_z)) )`
   - `q_d = qd_red * q_yaw`
5. Attitude error: `q_e = canonical( q^-1 * q_d )`.
6. `e_q = 2 (q_e_x, q_e_y, q_e_z)`.
7. `rate_sp = e_q * att_P` (element-wise), then clamp to `+/- rate_lim`.

Output: `rate_sp` (body frame).

### `atti_err(src, dst)` (shortest rotation from `src` to `dst`)

Let `c = src x dst`, `d = src . dst`.

- If `|c| < 1e-5` and `d < 0` (anti-parallel corner case):
  - Choose a basis axis `a` along the smallest-magnitude component of `src`:
    - if `|src_x| < |src_y|`: if `|src_x| < |src_z|` then `a = x_hat` else `a = z_hat`
    - else: if `|src_y| < |src_z|` then `a = y_hat` else `a = z_hat`
  - `c = src x a`, `w0 = 0`.
- Else: `w0 = d + sqrt(|src|^2 |dst|^2)` (for unit vectors, `= d + 1`).
- Return `normalize( (w0, c_x, c_y, c_z) )`.

## Layer 3 - Body-rate (`Px4RateController`)

Inputs: `rate_sp` (body), measured angular velocity `rate` (world frame),
`q_world`, `angular_accel` (always `0` here), `dt`, and persistent `rate_int`.

- `body_rate = R(q_world)^T * rate`
- `rate_error = rate_sp - body_rate`
- `torque = rate_P * rate_error + rate_int - rate_D * angular_accel`
- Anti-windup factor (element-wise): `i_factor = max(0, 1 - (rate_error / radians(400))^2)`
- `rate_int <- clamp( rate_int + i_factor * rate_I * rate_error * dt, -int_lim, +int_lim )`

Output: `torque` (3-vector). The updated `rate_int` is written back to state.

## Layer 4 - Mixer (`Px4Mixer`, Quad-X)

Per-rotor mixing scales (rotor order matches `_config_quad_x`):

| rotor | roll | pitch | yaw | thrust |
| --- | --- | --- | --- | --- |
| 0 | -0.707107 | -0.707107 | -1.0 | 1.0 |
| 1 | +0.707107 | +0.707107 | -1.0 | 1.0 |
| 2 | +0.707107 | -0.707107 | +1.0 | 1.0 |
| 3 | -0.707107 | +0.707107 | +1.0 | 1.0 |

Input `torque_thrust = [roll, pitch, yaw, thrust]`.

1. Clamp `roll, pitch, yaw` to `[-1, 1]`, `thrust` to `[0, 1]`.
2. `out_i = roll * roll_i + pitch * pitch_i + thrust * thrust_i` (no yaw yet).
3. `minimize_saturation(thrust_scale, 0, 1, reduce_only=True)`
4. `minimize_saturation(roll_scale, 0, 1)`
5. `minimize_saturation(pitch_scale, 0, 1)`
6. `out_i += yaw * yaw_i`
7. `minimize_saturation(yaw_scale, 0, 1.15)`
8. `minimize_saturation(thrust_scale, 0, 1, reduce_only=True)`
9. Clamp `out` to `[0, 1]`.

Output: four normalized motor commands in `[0, 1]`.

### `compute_desaturation_gain(desat, out, lo, hi)`

`k_min = 0`, `k_max = 0`. For each rotor `i` with `|desat_i| >= FLT_EPSILON`:

- if `out_i < lo`: `k = (lo - out_i) / desat_i`; `k_min = min(k_min, k)`, `k_max = max(k_max, k)`
- if `out_i > hi`: `k = (hi - out_i) / desat_i`; `k_min = min(k_min, k)`, `k_max = max(k_max, k)`

Return `k_min + k_max`.

### `minimize_saturation(desat, out, lo, hi, reduce_only=False)`

- `k1 = compute_desaturation_gain(desat, out, lo, hi)`
- if `reduce_only` and `k1 > 0`: return `out` unchanged
- `out += k1 * desat`
- `k2 = 0.5 * compute_desaturation_gain(desat, out, lo, hi)`
- `out += k2 * desat`

## Full pipelines

```
ParallelPosControl(actions=[x_sp, y_sp, z_sp, yaw_sp]):
  pos layer (mode=ALL) -> (q_sp, thrust_sp)
  atti layer(q_sp, q)  -> rate_sp
  rate layer(rate_sp, ang_vel, q_world=q, dt) -> torque
  mixer([torque, thrust_sp]) -> motor_cmd (N,4)

ParallelVelControl(actions=[vx_sp, vy_sp, vz_sp, yaw_sp]):
  pos layer (mode=VEL_ONLY) -> (q_sp, thrust_sp)
  atti layer -> rate layer -> mixer  (same as above)

ParallelAttiControl(actions=[qw, qx, qy, qz, thrust]):
  atti layer(action[:4], q) -> rate_sp
  rate layer(rate_sp, ang_vel, q_world=q, dt) -> torque
  mixer([torque, action[4]]) -> motor_cmd (N,4)

ParallelRateControl(actions=[p, q, r, thrust], rate=observation):
  rate layer(action[:3], rate, q_world set via set_q_world, dt) -> torque
  mixer([torque, action[3]]) -> motor_cmd (N,4)
```

All four output `(N, 4)` motor commands in `[0, 1]`.

## Numerical notes

- The original C++ uses `double` for most of the cascade and `float` in a few spots
  (e.g. the mixer). `px4_warp` runs the whole pipeline in `float32` to allow
  zero-copy interop with IsaacLab tensors. Parity with the C++ ground truth holds to
  about `1e-4` absolute on representative inputs.
- The pipeline is branch-per-thread but contains no atomics or reductions, so the
  output is deterministic across runs for a given input.
