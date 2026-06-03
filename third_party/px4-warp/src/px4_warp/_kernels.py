"""Warp kernels implementing the batched PX4-style controller cascade.

Every controller runs as a single ``wp.launch`` with one thread per environment.
Each thread evaluates the full cascade (position/velocity -> attitude -> body-rate
-> mixer) in registers. Quaternions are ``[w, x, y, z]``. See ``MATH_MODEL.md`` for
the exact equations and the C++ source they were extracted from.

All math is ``float32`` so the arrays can alias IsaacLab ``torch`` CUDA tensors
with zero copies.
"""

# NOTE: deliberately no ``from __future__ import annotations`` here. Warp inspects
# real annotation objects (e.g. ``wp.array2d(dtype=wp.float32)``) at decoration
# time; PEP 563 stringized annotations break struct/kernel code generation.

import math

import warp as wp

# --------------------------------------------------------------------------- #
# Constants (match the original C++/Eigen controllers exactly)
# --------------------------------------------------------------------------- #
G = wp.constant(9.80665)            # gravity [m/s^2]
HOVER = wp.constant(0.1533)         # fixed normalized hover throttle
ACC_LIM = wp.constant(4.0)          # per-axis desired-acceleration clamp
YAW_W = wp.constant(0.4)            # attitude yaw weight
RAD400 = wp.constant(float(math.radians(400.0)))  # rate-integrator anti-windup ref
FLT_EPS = wp.constant(1.1920929e-07)  # FLT_EPSILON, mixer desaturation guard

# Position / velocity P gains.
KP = wp.constant(wp.vec3(1.5, 1.5, 1.5))
KV = wp.constant(wp.vec3(1.5, 1.5, 1.5))

# Attitude P gain (effective: yaw 1.0 / yaw_w 0.4 = 2.5) and rate-setpoint limit.
ATT_P = wp.constant(wp.vec3(8.0, 8.0, 2.5))
RATE_LIM = wp.constant(wp.vec3(1600.0 / 57.3, 1600.0 / 57.3, 1000.0 / 57.3))

# Body-rate PID gains and integrator clamp.
RATE_P = wp.constant(wp.vec3(0.5, 0.5, 0.2))
RATE_I = wp.constant(wp.vec3(0.08, 0.08, 0.05))
RATE_D = wp.constant(wp.vec3(0.001, 0.001, 0.0))
INT_LIM = wp.constant(wp.vec3(0.3, 0.3, 0.3))

# Quad-X mixer per-rotor scales (rotor order matches _config_quad_x in Px4Mixer).
ROLL_SCALE = wp.constant(wp.vec4(-0.707107, 0.707107, 0.707107, -0.707107))
PITCH_SCALE = wp.constant(wp.vec4(-0.707107, 0.707107, -0.707107, 0.707107))
YAW_SCALE = wp.constant(wp.vec4(-1.0, -1.0, 1.0, 1.0))
THRUST_SCALE = wp.constant(wp.vec4(1.0, 1.0, 1.0, 1.0))

# Control modes for the shared position/velocity kernel.
MODE_ALL = wp.constant(0)
MODE_POS = wp.constant(1)
MODE_VEL = wp.constant(2)


# --------------------------------------------------------------------------- #
# Small structs to return more than one value from a @wp.func
# --------------------------------------------------------------------------- #
@wp.struct
class PosOut:
    q_sp: wp.vec4
    thrust: wp.float32


@wp.struct
class RateOut:
    torque: wp.vec3
    integ: wp.vec3


# --------------------------------------------------------------------------- #
# Scalar / vector helpers
# --------------------------------------------------------------------------- #
@wp.func
def sign0(x: wp.float32) -> wp.float32:
    # Matches MyMath::sign: -1, 0, or +1 (zero-preserving).
    if x > 0.0:
        return 1.0
    if x < 0.0:
        return -1.0
    return 0.0


@wp.func
def clamp_vec3(v: wp.vec3, lo: wp.float32, hi: wp.float32) -> wp.vec3:
    return wp.vec3(wp.clamp(v[0], lo, hi), wp.clamp(v[1], lo, hi), wp.clamp(v[2], lo, hi))


@wp.func
def clamp_sym(v: wp.vec3, lim: wp.vec3) -> wp.vec3:
    return wp.vec3(
        wp.clamp(v[0], -lim[0], lim[0]),
        wp.clamp(v[1], -lim[1], lim[1]),
        wp.clamp(v[2], -lim[2], lim[2]),
    )


# --------------------------------------------------------------------------- #
# Quaternion helpers (all in [w, x, y, z])
# --------------------------------------------------------------------------- #
@wp.func
def qmul(a: wp.vec4, b: wp.vec4) -> wp.vec4:
    aw = a[0]
    ax = a[1]
    ay = a[2]
    az = a[3]
    bw = b[0]
    bx = b[1]
    by = b[2]
    bz = b[3]
    return wp.vec4(
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


@wp.func
def qinv(q: wp.vec4) -> wp.vec4:
    return wp.vec4(q[0], -q[1], -q[2], -q[3])


@wp.func
def qnorm(q: wp.vec4) -> wp.vec4:
    n = wp.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    n = wp.max(n, 1.0e-12)
    inv = 1.0 / n
    return wp.vec4(q[0] * inv, q[1] * inv, q[2] * inv, q[3] * inv)


@wp.func
def qcanon(q: wp.vec4) -> wp.vec4:
    s = sign0(q[0])
    return wp.vec4(q[0] * s, q[1] * s, q[2] * s, q[3] * s)


@wp.func
def ez_from_q(q: wp.vec4) -> wp.vec3:
    # Third column of R(q): the body z-axis in world frame.
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]
    return wp.vec3(2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y))


@wp.func
def world_to_body(q: wp.vec4, r: wp.vec3) -> wp.vec3:
    # R(q)^T * r : rotate a world-frame vector into the body frame.
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]
    rx = r[0]
    ry = r[1]
    rz = r[2]
    ox = (1.0 - 2.0 * y * y - 2.0 * z * z) * rx + (2.0 * x * y + 2.0 * z * w) * ry + (2.0 * x * z - 2.0 * y * w) * rz
    oy = (2.0 * x * y - 2.0 * z * w) * rx + (1.0 - 2.0 * x * x - 2.0 * z * z) * ry + (2.0 * y * z + 2.0 * x * w) * rz
    oz = (2.0 * x * z + 2.0 * y * w) * rx + (2.0 * y * z - 2.0 * x * w) * ry + (1.0 - 2.0 * x * x - 2.0 * y * y) * rz
    return wp.vec3(ox, oy, oz)


@wp.func
def yaw_from_q(q: wp.vec4) -> wp.float32:
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]
    return wp.atan2(2.0 * (x * y + w * z), w * w + x * x - y * y - z * z)


@wp.func
def q_from_rpy(roll: wp.float32, pitch: wp.float32, yaw: wp.float32) -> wp.vec4:
    # R = Rz(yaw) Ry(pitch) Rx(roll)
    cr = wp.cos(roll * 0.5)
    sr = wp.sin(roll * 0.5)
    cp = wp.cos(pitch * 0.5)
    sp = wp.sin(pitch * 0.5)
    cy = wp.cos(yaw * 0.5)
    sy = wp.sin(yaw * 0.5)
    return wp.vec4(
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


@wp.func
def atti_err(src: wp.vec3, dst: wp.vec3) -> wp.vec4:
    # Shortest-arc quaternion rotating src onto dst (getAttiErr in C++).
    c = wp.cross(src, dst)
    d = wp.dot(src, dst)
    cn = wp.length(c)

    corner = False
    if cn < 1.0e-5:
        if d < 0.0:
            corner = True

    w0 = wp.float32(0.0)
    if corner:
        ax = wp.abs(src[0])
        ay = wp.abs(src[1])
        az = wp.abs(src[2])
        axis = wp.vec3(0.0, 0.0, 1.0)
        if ax < ay:
            if ax < az:
                axis = wp.vec3(1.0, 0.0, 0.0)
            else:
                axis = wp.vec3(0.0, 0.0, 1.0)
        else:
            if ay < az:
                axis = wp.vec3(0.0, 1.0, 0.0)
            else:
                axis = wp.vec3(0.0, 0.0, 1.0)
        c = wp.cross(src, axis)
        w0 = 0.0
    else:
        w0 = d + wp.sqrt(wp.dot(src, src) * wp.dot(dst, dst))

    return qnorm(wp.vec4(w0, c[0], c[1], c[2]))


# --------------------------------------------------------------------------- #
# Mixer (Quad-X with desaturation)
# --------------------------------------------------------------------------- #
@wp.func
def desat_gain(desat: wp.vec4, out: wp.vec4, lo: wp.float32, hi: wp.float32) -> wp.float32:
    k_min = wp.float32(0.0)
    k_max = wp.float32(0.0)
    for i in range(4):
        dv = desat[i]
        if wp.abs(dv) >= FLT_EPS:
            ov = out[i]
            k = wp.float32(0.0)
            if ov < lo:
                k = (lo - ov) / dv
                k_min = wp.min(k_min, k)
                k_max = wp.max(k_max, k)
            if ov > hi:
                k = (hi - ov) / dv
                k_min = wp.min(k_min, k)
                k_max = wp.max(k_max, k)
    return k_min + k_max


@wp.func
def minimize_sat(
    desat: wp.vec4, out: wp.vec4, lo: wp.float32, hi: wp.float32, reduce_only: int
) -> wp.vec4:
    k1 = desat_gain(desat, out, lo, hi)
    if reduce_only == 1:
        if k1 > 0.0:
            return out
    res = out + k1 * desat
    k2 = 0.5 * desat_gain(desat, res, lo, hi)
    res = res + k2 * desat
    return res


@wp.func
def mixer(roll: wp.float32, pitch: wp.float32, yaw: wp.float32, thrust: wp.float32) -> wp.vec4:
    r = wp.clamp(roll, -1.0, 1.0)
    p = wp.clamp(pitch, -1.0, 1.0)
    yw = wp.clamp(yaw, -1.0, 1.0)
    th = wp.clamp(thrust, 0.0, 1.0)

    out = r * ROLL_SCALE + p * PITCH_SCALE + th * THRUST_SCALE
    out = minimize_sat(THRUST_SCALE, out, 0.0, 1.0, 1)  # only reduce thrust
    out = minimize_sat(ROLL_SCALE, out, 0.0, 1.0, 0)
    out = minimize_sat(PITCH_SCALE, out, 0.0, 1.0, 0)
    out = out + yw * YAW_SCALE
    out = minimize_sat(YAW_SCALE, out, 0.0, 1.15, 0)
    out = minimize_sat(THRUST_SCALE, out, 0.0, 1.0, 1)
    return wp.vec4(
        wp.clamp(out[0], 0.0, 1.0),
        wp.clamp(out[1], 0.0, 1.0),
        wp.clamp(out[2], 0.0, 1.0),
        wp.clamp(out[3], 0.0, 1.0),
    )


# --------------------------------------------------------------------------- #
# Controller layers
# --------------------------------------------------------------------------- #
@wp.func
def position_layer(
    mode: int,
    pos: wp.vec3,
    vel: wp.vec3,
    q: wp.vec4,
    pos_sp: wp.vec3,
    vel_sp: wp.vec3,
    yaw_sp: wp.float32,
) -> PosOut:
    # acc_sp is always zero in the batched pipelines.
    des_acc = wp.vec3(0.0, 0.0, 0.0)
    if mode == MODE_POS:
        des_acc = wp.cw_mul(KP, pos_sp - pos)
    elif mode == MODE_VEL:
        des_acc = wp.cw_mul(KV, vel_sp - vel)
    else:
        des_acc = wp.cw_mul(KV, vel_sp - vel) + wp.cw_mul(KP, pos_sp - pos)

    des_acc = clamp_vec3(des_acc, -ACC_LIM, ACC_LIM)

    thrust = des_acc[2] * (HOVER / G) + HOVER

    psi = yaw_from_q(q)
    s = wp.sin(psi)
    c = wp.cos(psi)
    roll = (des_acc[0] * s - des_acc[1] * c) / G
    pitch = (des_acc[0] * c + des_acc[1] * s) / G

    out = PosOut()
    out.q_sp = q_from_rpy(roll, pitch, yaw_sp)
    out.thrust = thrust
    return out


@wp.func
def attitude_layer(q_sp_in: wp.vec4, q_in: wp.vec4) -> wp.vec3:
    q = qnorm(q_in)
    qd_full = qnorm(q_sp_in)

    e_z = ez_from_q(q)
    e_z_d = ez_from_q(qd_full)

    qd_red = atti_err(e_z, e_z_d)

    opp = False
    if wp.abs(qd_red[1]) > (1.0 - 1.0e-5):
        opp = True
    if wp.abs(qd_red[2]) > (1.0 - 1.0e-5):
        opp = True

    qd_reduced = wp.vec4(0.0, 0.0, 0.0, 0.0)
    if opp:
        qd_reduced = qd_full
    else:
        qd_reduced = qmul(qd_red, q)

    q_mix = qcanon(qmul(qinv(qd_reduced), qd_full))
    qmw = wp.clamp(q_mix[0], -1.0, 1.0)
    qmz = wp.clamp(q_mix[3], -1.0, 1.0)
    q_yaw = wp.vec4(wp.cos(YAW_W * wp.acos(qmw)), 0.0, 0.0, wp.sin(YAW_W * wp.asin(qmz)))

    q_d = qmul(qd_reduced, q_yaw)
    q_e = qcanon(qmul(qinv(q), q_d))

    eq = wp.vec3(q_e[1] * 2.0, q_e[2] * 2.0, q_e[3] * 2.0)
    rate_sp = wp.cw_mul(ATT_P, eq)
    return clamp_sym(rate_sp, RATE_LIM)


@wp.func
def rate_layer(
    q_world: wp.vec4, integ: wp.vec3, rate_sp: wp.vec3, rate: wp.vec3, dt: wp.float32
) -> RateOut:
    body_rate = world_to_body(q_world, rate)
    rate_error = rate_sp - body_rate

    # torque = P*error + I_state - D*angular_accel; angular_accel is 0 here.
    torque = wp.cw_mul(RATE_P, rate_error) + integ

    ex = rate_error[0] / RAD400
    ey = rate_error[1] / RAD400
    ez = rate_error[2] / RAD400
    i_factor = wp.vec3(
        wp.max(0.0, 1.0 - ex * ex),
        wp.max(0.0, 1.0 - ey * ey),
        wp.max(0.0, 1.0 - ez * ez),
    )

    inc = wp.cw_mul(wp.cw_mul(i_factor, RATE_I), rate_error) * dt
    new_int = clamp_sym(integ + inc, INT_LIM)

    out = RateOut()
    out.torque = torque
    out.integ = new_int
    return out


# --------------------------------------------------------------------------- #
# Kernels: one thread per environment, one launch per update()
# --------------------------------------------------------------------------- #
@wp.kernel
def pos_vel_update(
    pos: wp.array2d(dtype=wp.float32),
    vel: wp.array2d(dtype=wp.float32),
    quat: wp.array2d(dtype=wp.float32),
    ang_vel: wp.array2d(dtype=wp.float32),
    actions: wp.array2d(dtype=wp.float32),
    rate_int: wp.array2d(dtype=wp.float32),
    dt: wp.float32,
    mode: int,
    out: wp.array2d(dtype=wp.float32),
):
    i = wp.tid()
    p = wp.vec3(pos[i, 0], pos[i, 1], pos[i, 2])
    v = wp.vec3(vel[i, 0], vel[i, 1], vel[i, 2])
    q = wp.vec4(quat[i, 0], quat[i, 1], quat[i, 2], quat[i, 3])
    av = wp.vec3(ang_vel[i, 0], ang_vel[i, 1], ang_vel[i, 2])

    sp = wp.vec3(actions[i, 0], actions[i, 1], actions[i, 2])
    yaw_sp = actions[i, 3]

    pos_sp = wp.vec3(0.0, 0.0, 0.0)
    vel_sp = wp.vec3(0.0, 0.0, 0.0)
    if mode == MODE_VEL:
        vel_sp = sp
    else:
        pos_sp = sp

    po = position_layer(mode, p, v, q, pos_sp, vel_sp, yaw_sp)
    rate_sp = attitude_layer(po.q_sp, q)

    integ = wp.vec3(rate_int[i, 0], rate_int[i, 1], rate_int[i, 2])
    ro = rate_layer(q, integ, rate_sp, av, dt)
    rate_int[i, 0] = ro.integ[0]
    rate_int[i, 1] = ro.integ[1]
    rate_int[i, 2] = ro.integ[2]

    cmd = mixer(ro.torque[0], ro.torque[1], ro.torque[2], po.thrust)
    out[i, 0] = cmd[0]
    out[i, 1] = cmd[1]
    out[i, 2] = cmd[2]
    out[i, 3] = cmd[3]


@wp.kernel
def atti_update(
    quat: wp.array2d(dtype=wp.float32),
    ang_vel: wp.array2d(dtype=wp.float32),
    actions: wp.array2d(dtype=wp.float32),
    rate_int: wp.array2d(dtype=wp.float32),
    dt: wp.float32,
    out: wp.array2d(dtype=wp.float32),
):
    i = wp.tid()
    q = wp.vec4(quat[i, 0], quat[i, 1], quat[i, 2], quat[i, 3])
    av = wp.vec3(ang_vel[i, 0], ang_vel[i, 1], ang_vel[i, 2])
    q_sp = wp.vec4(actions[i, 0], actions[i, 1], actions[i, 2], actions[i, 3])
    thrust = actions[i, 4]

    rate_sp = attitude_layer(q_sp, q)

    integ = wp.vec3(rate_int[i, 0], rate_int[i, 1], rate_int[i, 2])
    ro = rate_layer(q, integ, rate_sp, av, dt)
    rate_int[i, 0] = ro.integ[0]
    rate_int[i, 1] = ro.integ[1]
    rate_int[i, 2] = ro.integ[2]

    cmd = mixer(ro.torque[0], ro.torque[1], ro.torque[2], thrust)
    out[i, 0] = cmd[0]
    out[i, 1] = cmd[1]
    out[i, 2] = cmd[2]
    out[i, 3] = cmd[3]


@wp.kernel
def rate_update(
    q_world: wp.array2d(dtype=wp.float32),
    actions: wp.array2d(dtype=wp.float32),
    rate: wp.array2d(dtype=wp.float32),
    rate_int: wp.array2d(dtype=wp.float32),
    dt: wp.float32,
    out: wp.array2d(dtype=wp.float32),
):
    i = wp.tid()
    qw = wp.vec4(q_world[i, 0], q_world[i, 1], q_world[i, 2], q_world[i, 3])
    rate_sp = wp.vec3(actions[i, 0], actions[i, 1], actions[i, 2])
    thrust = actions[i, 3]
    meas = wp.vec3(rate[i, 0], rate[i, 1], rate[i, 2])

    integ = wp.vec3(rate_int[i, 0], rate_int[i, 1], rate_int[i, 2])
    ro = rate_layer(qw, integ, rate_sp, meas, dt)
    rate_int[i, 0] = ro.integ[0]
    rate_int[i, 1] = ro.integ[1]
    rate_int[i, 2] = ro.integ[2]

    cmd = mixer(ro.torque[0], ro.torque[1], ro.torque[2], thrust)
    out[i, 0] = cmd[0]
    out[i, 1] = cmd[1]
    out[i, 2] = cmd[2]
    out[i, 3] = cmd[3]
