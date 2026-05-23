# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Diagnostic script for the AirGym X152b velocity controller.

This script applies a deterministic velocity-mode action to the registered
AirGym X152b task and reports whether the measured root velocity tracks the
requested velocity after a settling window.

.. code-block:: bash

    ./isaaclab.sh -p scripts/demos/airgym_x152b_velocity_check.py --headless
    ./isaaclab.sh -p scripts/demos/airgym_x152b_velocity_check.py --action 1.0 0.0 0.0 0.0 --duration 4.0

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Check AirGym X152b velocity tracking for a deterministic action.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-AirGym-X152b-Hovering-Direct-v0",
    help="Registered AirGym X152b task to instantiate.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of cloned environments to run.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument(
    "--action",
    type=float,
    nargs=4,
    default=(1.0, 0.0, 0.0, 0.0),
    metavar=("VX", "VY", "VZ", "YAW"),
    help="Deterministic velocity-mode action: [vx, vy, vz, yaw].",
)
parser.add_argument("--duration", type=float, default=4.0, help="Total diagnostic duration in seconds.")
parser.add_argument(
    "--settle_time",
    "--settle-time",
    type=float,
    default=1.5,
    help="Seconds to ignore before computing tracking metrics.",
)
parser.add_argument(
    "--start_height", "--start-height", type=float, default=1.0, help="Initial drone height in meters."
)
parser.add_argument("--seed", type=int, default=0, help="Environment seed.")
parser.add_argument(
    "--tolerance",
    type=float,
    default=0.25,
    help="Pass/fail tolerance on the norm of mean measured velocity error in m/s.",
)
parser.add_argument(
    "--print_every",
    "--print-every",
    type=float,
    default=0.5,
    help="Print live velocity every N seconds. Use 0 to disable.",
)
parser.add_argument(
    "--no_fail",
    "--no-fail",
    action="store_true",
    default=False,
    help="Always exit with status 0 after printing metrics.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import sys

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi
from isaaclab_tasks.utils import parse_env_cfg


def _format_vector(values: torch.Tensor | list[float] | tuple[float, ...]) -> str:
    """Return a compact fixed-width vector string."""
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().tolist()
    return "[" + ", ".join(f"{value: .4f}" for value in values) + "]"


def _reset_robot_to_known_state(env, start_height: float):
    """Write the drone into a deterministic pose and velocity."""
    env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    pos = torch.zeros(env.num_envs, 3, device=env.device)
    pos[:, 2] = start_height
    quat_w = torch.zeros(env.num_envs, 4, device=env.device)
    quat_w[:, 0] = 1.0
    lin_vel_w = torch.zeros(env.num_envs, 3, device=env.device)
    ang_vel_w = torch.zeros(env.num_envs, 3, device=env.device)

    env._write_robot_state_local(env_ids, pos, quat_w, lin_vel_w, ang_vel_w)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(dt=env.physics_dt)


def _step_without_task_resets(env, action: torch.Tensor):
    """Step the controller and physics path without invoking RL done/reset logic."""
    env._pre_physics_step(action)
    is_rendering = env.sim.has_gui() or env.sim.has_rtx_sensors()
    for _ in range(env.cfg.decimation):
        env._sim_step_counter += 1
        env._apply_action()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        if env._sim_step_counter % env.cfg.sim.render_interval == 0 and is_rendering:
            env.sim.render()
        env.scene.update(dt=env.physics_dt)


def main() -> int:
    """Run the deterministic velocity diagnostic."""
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    env_cfg.ctl_mode = "vel"
    env_cfg.action_space = 4

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.sim.set_camera_view(eye=[3.0, -4.0, 2.0], target=[0.0, 0.0, args_cli.start_height])

    try:
        env.reset(seed=args_cli.seed)
        _reset_robot_to_known_state(env, args_cli.start_height)

        action = torch.tensor(args_cli.action, dtype=torch.float32, device=env.device).repeat(env.num_envs, 1)
        total_steps = max(1, round(args_cli.duration / env.step_dt))
        settle_steps = max(0, min(total_steps - 1, round(args_cli.settle_time / env.step_dt)))
        print_interval = round(args_cli.print_every / env.step_dt) if args_cli.print_every > 0.0 else 0

        velocities = []
        yaw_angles = []
        positions = []
        cmd_thrusts = []
        processed_action = None

        print("[INFO]: Setup complete.")
        print(f"[INFO]: Task               : {args_cli.task}")
        print(f"[INFO]: Control mode       : {env.cfg.ctl_mode}")
        print(f"[INFO]: Physics dt         : {env.physics_dt:.4f} s")
        print(f"[INFO]: Environment dt     : {env.step_dt:.4f} s")
        print(f"[INFO]: Raw action         : {_format_vector(args_cli.action)}")
        print(f"[INFO]: Duration           : {args_cli.duration:.2f} s")
        print(f"[INFO]: Settling window    : first {settle_steps} steps ({settle_steps * env.step_dt:.2f} s)")

        with torch.inference_mode():
            for step in range(total_steps):
                if not simulation_app.is_running():
                    break

                _step_without_task_resets(env, action)

                if processed_action is None:
                    processed_action = env._actions[0].detach().clone()

                root_vel_w = env._robot.data.root_lin_vel_w.detach().clone()
                _, _, root_yaw_w = euler_xyz_from_quat(env._robot.data.root_quat_w)
                root_yaw_w = root_yaw_w.detach().clone()
                root_pos_local = env._root_pos_local().detach().clone()

                if step >= settle_steps:
                    velocities.append(root_vel_w)
                    yaw_angles.append(root_yaw_w)
                    positions.append(root_pos_local)
                    cmd_thrusts.append(env._cmd_thrusts.detach().clone())

                if print_interval > 0 and (step + 1) % print_interval == 0:
                    sim_time = (step + 1) * env.step_dt
                    print(
                        f"[INFO]: t={sim_time:5.2f}s vel_w[0]={_format_vector(root_vel_w[0])} "
                        f"yaw_w[0]={root_yaw_w[0].item(): .4f} rad"
                    )

        if processed_action is None:
            raise RuntimeError("Simulation did not step; no processed action was produced.")

        if not velocities:
            raise RuntimeError("No samples were collected after the settling window.")

        velocity_history = torch.stack(velocities)
        yaw_history = torch.stack(yaw_angles)
        position_history = torch.stack(positions)
        thrust_history = torch.stack(cmd_thrusts)

        target_velocity = processed_action[:3]
        velocity_error = velocity_history - target_velocity.view(1, 1, 3)
        velocity_error_norm = torch.linalg.vector_norm(velocity_error, dim=-1)
        mean_velocity = velocity_history.mean(dim=(0, 1))
        final_velocity = velocity_history[-1, 0]
        mean_error = mean_velocity - target_velocity
        mean_error_norm = torch.linalg.vector_norm(mean_error).item()
        max_error_norm = velocity_error_norm.max().item()
        rms_error_norm = torch.sqrt(torch.mean(torch.square(velocity_error_norm))).item()
        target_yaw = processed_action[3]
        yaw_error = wrap_to_pi(yaw_history - target_yaw)
        mean_yaw = yaw_history.mean().item()
        final_yaw = yaw_history[-1, 0].item()
        mean_yaw_error = yaw_error.mean().item()
        final_yaw_error = yaw_error[-1, 0].item()
        rms_yaw_error = torch.sqrt(torch.mean(torch.square(yaw_error))).item()
        max_abs_yaw_error = yaw_error.abs().max().item()
        final_position = position_history[-1, 0]
        mean_motor_command = thrust_history.mean(dim=(0, 1))

        if not torch.allclose(processed_action, action[0]):
            print(f"[WARN]: Action was clamped to : {_format_vector(processed_action)}")
        else:
            print(f"[INFO]: Processed action    : {_format_vector(processed_action)}")

        print("\n[RESULT]: AirGym X152b deterministic velocity check")
        print(f"  target velocity          : {_format_vector(target_velocity)} m/s")
        print(f"  mean measured velocity   : {_format_vector(mean_velocity)} m/s")
        print(f"  final measured velocity  : {_format_vector(final_velocity)} m/s")
        print(f"  mean velocity error      : {_format_vector(mean_error)} m/s")
        print(f"  mean error norm          : {mean_error_norm:.4f} m/s")
        print(f"  RMS error norm           : {rms_error_norm:.4f} m/s")
        print(f"  max sample error norm    : {max_error_norm:.4f} m/s")
        print(f"  target yaw               : {target_yaw.item():.4f} rad")
        print(f"  mean yaw                 : {mean_yaw:.4f} rad")
        print(f"  final yaw                : {final_yaw:.4f} rad")
        print(f"  mean yaw error           : {mean_yaw_error:.4f} rad")
        print(f"  final yaw error          : {final_yaw_error:.4f} rad")
        print(f"  RMS yaw error            : {rms_yaw_error:.4f} rad")
        print(f"  max abs yaw error        : {max_abs_yaw_error:.4f} rad")
        print(f"  final local position     : {_format_vector(final_position)} m")
        print(f"  mean motor command       : {_format_vector(mean_motor_command)}")

        passed = mean_error_norm <= args_cli.tolerance
        status = "PASS" if passed else "FAIL"
        print(f"\n[RESULT]: {status} mean error norm <= {args_cli.tolerance:.4f} m/s")
        return 0 if passed or args_cli.no_fail else 1
    finally:
        env.close()


if __name__ == "__main__":
    try:
        exit_code = main()
    finally:
        simulation_app.close()
    sys.exit(exit_code)
