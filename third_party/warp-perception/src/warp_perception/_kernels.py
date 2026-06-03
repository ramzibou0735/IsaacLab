"""Warp kernels for the fused room-perception sensor.

A single kernel ray-casts an analytic box scene (axis-aligned or oriented boxes)
per camera pixel to produce, in one launch:

- a perspective depth image (distance to the camera image plane), and
- a two-channel ego-local critic occupancy grid (free, occupied), and
- a two-channel env-local room visibility grid (free, occupied).

The free/occupied channels are filled by marching samples along each ray and
writing with :func:`warp.atomic_max`, mirroring the semantics of the previous
PyTorch ``build_*_from_depth`` helpers. Quaternions are expected in Warp order
``[x, y, z, w]`` (the sensor wrapper re-orders IsaacLab ``[w, x, y, z]`` once).
"""

import warp as wp

# Distance returned when a ray misses every box. Large but finite so it is also
# a usable "far plane" sentinel inside arithmetic.
NO_HIT = wp.constant(1.0e6)


@wp.func
def ray_box_t(ro: wp.vec3, rd: wp.vec3, center: wp.vec3, half: wp.vec3, q: wp.quat) -> float:
    """Return the forward hit distance of a ray against an oriented box.

    The ray ``(ro, rd)`` is given in world space; ``rd`` must be normalized so the
    returned ``t`` is a metric distance. The box is centered at ``center`` with
    per-axis half-extents ``half`` and orientation ``q``. Returns :data:`NO_HIT`
    when there is no forward intersection.
    """
    # Transform the ray into the box-local frame (slab test in that frame).
    rol = wp.quat_rotate_inv(q, ro - center)
    rdl = wp.quat_rotate_inv(q, rd)

    tmin = float(0.0)
    tmax = float(NO_HIT)
    hit = int(1)

    for k in range(3):
        o = rol[k]
        d = rdl[k]
        h = half[k]
        if wp.abs(d) < 1.0e-8:
            # Ray parallel to this slab: miss if the origin is outside it.
            if (o < -h) or (o > h):
                hit = 0
        else:
            inv = 1.0 / d
            t1 = (-h - o) * inv
            t2 = (h - o) * inv
            tlo = wp.min(t1, t2)
            thi = wp.max(t1, t2)
            tmin = wp.max(tmin, tlo)
            tmax = wp.min(tmax, thi)

    if hit == 0:
        return NO_HIT
    if tmin > tmax:
        return NO_HIT
    if tmax < 0.0:
        return NO_HIT
    return tmin


@wp.func
def mark_point(
    grid: wp.array4d(dtype=wp.float32),
    env: int,
    ch: int,
    px: float,
    py: float,
    pz: float,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    z0: float,
    z1: float,
    cell: float,
    gh: int,
    gw: int,
    val: float,
):
    """Write ``val`` into a BEV grid cell if the point is inside the limits."""
    if (px >= x0) and (px < x1) and (py >= y0) and (py < y1) and (pz >= z0) and (pz < z1):
        ix = int(wp.floor((px - x0) / cell))
        iy = int(wp.floor((py - y0) / cell))
        if (ix >= 0) and (ix < gh) and (iy >= 0) and (iy < gw):
            wp.atomic_max(grid, env, ch, ix, iy, val)


@wp.kernel(enable_backward=False)
def clear_grid(grid: wp.array4d(dtype=wp.float32)):
    """Zero a ``(N, C, H, W)`` grid (used by the CUDA-graph capture path)."""
    env, ch, i, j = wp.tid()
    grid[env, ch, i, j] = 0.0


@wp.kernel(enable_backward=False)
def perception_kernel(
    cam_pos: wp.array(dtype=wp.vec3),
    cam_quat: wp.array(dtype=wp.quat),
    w2l: wp.array(dtype=wp.mat33),
    root_pos: wp.array(dtype=wp.vec3),
    env_origin: wp.array(dtype=wp.vec3),
    box_center: wp.array2d(dtype=wp.vec3),
    box_half: wp.array2d(dtype=wp.vec3),
    box_quat: wp.array2d(dtype=wp.quat),
    num_boxes: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    far: float,
    hit_margin: float,
    stride: int,
    free_samples: int,
    # critic grid (ego-local) limits / shape
    cgx0: float,
    cgx1: float,
    cgy0: float,
    cgy1: float,
    cgz0: float,
    cgz1: float,
    ccell: float,
    cgh: int,
    cgw: int,
    # visibility grid (env-local) limits / shape
    vgx0: float,
    vgx1: float,
    vgy0: float,
    vgy1: float,
    vgz0: float,
    vgz1: float,
    vcell: float,
    vgh: int,
    vgw: int,
    # outputs
    depth: wp.array3d(dtype=wp.float32),
    critic_grid: wp.array4d(dtype=wp.float32),
    vis_grid: wp.array4d(dtype=wp.float32),
):
    env, v, u = wp.tid()

    # Body-frame ray for pixel (u, v): camera looks along body +x, image x maps to
    # body -y (right), image y maps to body -z (down).
    xc = (float(u) - cx) / fx
    yc = (float(v) - cy) / fy
    rb = wp.normalize(wp.vec3(1.0, -xc, -yc))
    mult = rb[0]  # cos(angle to principal axis) -> range to image-plane depth

    ro = cam_pos[env]
    rd = wp.quat_rotate(cam_quat[env], rb)

    best = far
    for b in range(num_boxes):
        t = ray_box_t(ro, rd, box_center[env, b], box_half[env, b], box_quat[env, b])
        if t < best:
            best = t

    dval = far
    if best < far:
        dval = best * mult
    depth[env, v, u] = wp.min(dval, far)

    # Only a strided subset of rays contributes to the occupancy grids.
    if (u % stride != 0) or (v % stride != 0):
        return

    hit = int(0)
    march_len = far
    if best < (far - hit_margin):
        hit = 1
        march_len = best

    rp = root_pos[env]
    eo = env_origin[env]
    m = w2l[env]

    for s in range(free_samples):
        f = (float(s) + 0.5) / float(free_samples)
        p = ro + (f * march_len) * rd
        # critic grid: ego-local (yaw-aligned) frame
        pl = m * (p - rp)
        mark_point(
            critic_grid, env, 0, pl[0], pl[1], pl[2],
            cgx0, cgx1, cgy0, cgy1, cgz0, cgz1, ccell, cgh, cgw, 1.0,
        )
        # visibility grid: env-local frame
        pr = p - eo
        mark_point(
            vis_grid, env, 0, pr[0], pr[1], pr[2],
            vgx0, vgx1, vgy0, vgy1, vgz0, vgz1, vcell, vgh, vgw, 1.0,
        )

    if hit == 1:
        ph = ro + best * rd
        pl = m * (ph - rp)
        mark_point(
            critic_grid, env, 1, pl[0], pl[1], pl[2],
            cgx0, cgx1, cgy0, cgy1, cgz0, cgz1, ccell, cgh, cgw, 1.0,
        )
        pr = ph - eo
        mark_point(
            vis_grid, env, 1, pr[0], pr[1], pr[2],
            vgx0, vgx1, vgy0, vgy1, vgz0, vgz1, vcell, vgh, vgw, 1.0,
        )
