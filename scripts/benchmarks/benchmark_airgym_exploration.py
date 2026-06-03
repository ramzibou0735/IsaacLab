# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Headless smoke test + throughput benchmark for the AirGym x152b exploration env.

The exploration task replaces the RTX depth camera with the fused Warp perception
sensor and a frozen depth-VAE encoder, so this script measures steps/sec and peak
VRAM with no rendering in the loop. It also validates the observation dict
(``observation`` / ``latent`` / ``critic_grid``) and reward finiteness.

Example::

    ./isaaclab.sh -p scripts/benchmarks/benchmark_airgym_exploration.py \
        --headless --num_envs 256 --num_steps 300
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Benchmark the AirGym x152b exploration environment.")
parser.add_argument("--task", type=str, default="Isaac-AirGym-X152b-Exploration-Direct-v0", help="Task name.")
parser.add_argument("--num_envs", type=int, default=256, help="Number of environments to simulate.")
parser.add_argument("--num_steps", type=int, default=300, help="Number of timed environment steps.")
parser.add_argument("--warmup_steps", type=int, default=30, help="Untimed warmup steps before measuring.")
parser.add_argument("--seed", type=int, default=0, help="Environment seed.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import time

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed

    env = gym.make(args_cli.task, cfg=env_cfg)
    device = torch.device(env.unwrapped.device)
    num_envs = env.unwrapped.num_envs
    action_dim = gym.spaces.flatdim(env.unwrapped.single_action_space)
    print(f"[benchmark] task={args_cli.task} num_envs={num_envs} device={device} action_dim={action_dim}")

    obs, _ = env.reset()
    _validate_obs(obs, num_envs)

    def random_actions() -> torch.Tensor:
        return (2.0 * torch.rand((num_envs, action_dim), device=device) - 1.0)

    for _ in range(args_cli.warmup_steps):
        obs, rew, terminated, truncated, _ = env.step(random_actions())
    _validate_obs(obs, num_envs)
    assert torch.isfinite(rew).all(), "reward contains non-finite values"

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    for _ in range(args_cli.num_steps):
        obs, rew, terminated, truncated, _ = env.step(random_actions())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    env_steps = args_cli.num_steps * num_envs
    sps = env_steps / elapsed
    print("[benchmark] ---------------------------------------------")
    print(f"[benchmark] timed steps         : {args_cli.num_steps} ({env_steps} env-steps)")
    print(f"[benchmark] wall time           : {elapsed:.3f} s")
    print(f"[benchmark] throughput          : {sps:,.0f} env-steps/s")
    print(f"[benchmark] per-iteration       : {1e3 * elapsed / args_cli.num_steps:.2f} ms/step")
    if device.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
        print(f"[benchmark] peak VRAM (torch)   : {peak_gb:.2f} GiB")
    print("[benchmark] ---------------------------------------------")

    env.close()


def _validate_obs(obs, num_envs: int) -> None:
    for key in ("observation", "latent", "critic_grid"):
        assert key in obs, f"missing observation group '{key}'. got: {list(obs.keys())}"
    assert obs["latent"].shape[0] == num_envs and obs["latent"].dim() == 2, obs["latent"].shape
    assert obs["critic_grid"].shape[1:] == (2, 18, 18), obs["critic_grid"].shape
    assert "image" not in obs, "depth image observation should be removed in favor of the VAE latent"
    for key, value in obs.items():
        assert torch.isfinite(value).all(), f"non-finite values in observation '{key}'"


if __name__ == "__main__":
    main()
    simulation_app.close()
