"""Benchmark px4_warp (GPU, zero-copy) against the legacy C++ controller (CPU).

Run with the IsaacLab interpreter, for example::

    python px4-warp/examples/benchmark.py --envs 4096 --steps 200

The Warp path keeps everything on the GPU; the legacy path is CPU/NumPy and is
what creates the training-loop bottleneck this library removes.
"""

import argparse
import time

import numpy as np
import torch

import px4_warp

try:
    from rlPx4Controller import pyParallelControl as legacy

    HAS_LEGACY = True
except Exception:
    HAS_LEGACY = False


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench_warp(n, steps, device):
    pos = torch.zeros((n, 3), dtype=torch.float32, device=device)
    quat = torch.zeros((n, 4), dtype=torch.float32, device=device)
    quat[:, 0] = 1.0
    vel = torch.zeros((n, 3), dtype=torch.float32, device=device)
    ang = torch.zeros((n, 3), dtype=torch.float32, device=device)
    actions = torch.zeros((n, 4), dtype=torch.float32, device=device)
    actions[:, 2] = 2.0

    ctl = px4_warp.ParallelPosControl(n, device=device)
    # warmup (compiles / caches kernel)
    for _ in range(10):
        ctl.set_status(pos, quat, vel, ang, 0.01)
        ctl.update(actions)
    _sync()

    start = time.perf_counter()
    for _ in range(steps):
        ctl.set_status(pos, quat, vel, ang, 0.01)
        out = ctl.update(actions)
    _sync()
    elapsed = time.perf_counter() - start
    return elapsed / steps, out.detach().cpu().numpy()[0]


def bench_legacy(n, steps):
    pos = np.zeros((n, 3))
    quat = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]]), (n, 1))
    vel = np.zeros((n, 3))
    ang = np.zeros((n, 3))
    actions = np.zeros((n, 4))
    actions[:, 2] = 2.0

    ctl = legacy.ParallelPosControl(n)
    for _ in range(3):
        ctl.set_status(pos, quat, vel, ang, 0.01)
        ctl.update(actions)

    start = time.perf_counter()
    for _ in range(steps):
        ctl.set_status(pos, quat, vel, ang, 0.01)
        out = np.asarray(ctl.update(actions))
    elapsed = time.perf_counter() - start
    return elapsed / steps, out[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"device={device} envs={args.envs} steps={args.steps}")

    warp_per_step, warp_row0 = bench_warp(args.envs, args.steps, device)
    print(f"[px4_warp ] full step: {warp_per_step * 1e6:9.2f} us/step  row0={warp_row0}")

    if HAS_LEGACY:
        legacy_per_step, legacy_row0 = bench_legacy(args.envs, args.steps)
        print(f"[legacy   ] full step: {legacy_per_step * 1e6:9.2f} us/step  row0={legacy_row0}")
        print(f"[speedup  ] {legacy_per_step / warp_per_step:6.1f}x  (legacy / px4_warp)")
        if not np.allclose(warp_row0, legacy_row0, atol=1e-4):
            print("WARNING: row0 mismatch between warp and legacy")
    else:
        print("[legacy   ] not importable; skipping comparison")


if __name__ == "__main__":
    main()
