# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import normalize, quat_apply, quat_from_euler_xyz, sample_uniform

from .base_env import AirGymX152bBaseEnv, compute_yaw_diff
from .base_env_cfg import AirGymX152bBaseEnvCfg


@configclass
class AirGymX152bHoveringEnvCfg(AirGymX152bBaseEnvCfg):
    episode_length_s = 24.0
    observation_space = 18
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=256, env_spacing=1.0, replicate_physics=True)
    terrain = None


class AirGymX152bHoveringEnv(AirGymX152bBaseEnv):
    cfg: AirGymX152bHoveringEnvCfg

    def _reset_task(self, env_ids: torch.Tensor):
        num_resets = len(env_ids)
        pos = torch.zeros(num_resets, 3, device=self.device)
        pos[:, :2] = sample_uniform(-1.0, 1.0, (num_resets, 2), self.device)
        pos[:, 2] = sample_uniform(-1.0, 1.0, (num_resets, 1), self.device).squeeze(-1)
        roll = 0.01 * sample_uniform(-torch.pi, torch.pi, (num_resets,), self.device)
        pitch = 0.01 * sample_uniform(-torch.pi, torch.pi, (num_resets,), self.device)
        yaw = 0.05 * sample_uniform(-torch.pi, torch.pi, (num_resets,), self.device)
        quat = quat_from_euler_xyz(roll, pitch, yaw)
        lin_vel = 0.5 * sample_uniform(-1.0, 1.0, (num_resets, 3), self.device)
        ang_vel = 0.2 * sample_uniform(-1.0, 1.0, (num_resets, 3), self.device)
        self._write_robot_state_local(env_ids, pos, quat, lin_vel, ang_vel)

    def _get_task_observations(self) -> dict[str, torch.Tensor]:
        obs = torch.cat(
            (
                self._root_rotation_matrix().reshape(self.num_envs, -1),
                self._root_pos_local(),
                self._robot.data.root_lin_vel_w,
                self._robot.data.root_ang_vel_w,
            ),
            dim=-1,
        )
        obs = self._add_state_noise(obs)
        obs = obs - self._target_states
        return {"policy": obs}

    def _compute_reward_and_metrics(self) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        thrust_cmds = self._cmd_thrusts.clamp(0.0, 1.0)
        effort_reward = 0.1 * (1.0 - thrust_cmds).sum(dim=-1) / 4.0

        action_diff = self._actions - self._previous_actions
        thrust_reward = torch.zeros(self.num_envs, device=self.device)
        if self.cfg.ctl_mode in {"pos", "vel", "prop"}:
            continuous_action_reward = 0.2 * torch.exp(-torch.linalg.vector_norm(action_diff, dim=-1))
        else:
            continuous_action_reward = 0.2 * torch.exp(-torch.linalg.vector_norm(action_diff[:, :-1], dim=-1))
            continuous_action_reward += 0.5 / (1.0 + torch.square(3.0 * action_diff[:, -1]))
            thrust_reward = 0.1 * (1.0 - torch.abs(self.cfg.hover_thrust - self._actions[:, -1]))

        target_pos = self._target_states[:, 9:12]
        relative_positions = target_pos - self._root_pos_local()
        pos_diff = torch.linalg.vector_norm(relative_positions, dim=-1)
        pos_reward = 0.7 / (1.0 + torch.square(1.6 * pos_diff))

        tar_direction = normalize(relative_positions)
        vel_direction = normalize(self._robot.data.root_lin_vel_w)
        dot_product = (tar_direction * vel_direction).sum(dim=-1).clamp(-1.0, 1.0)
        angle_diff = torch.acos(dot_product).abs()
        vel_direction_reward = 0.1 * torch.exp(-angle_diff / torch.pi)

        root_euler = self._root_euler_xyz()
        yaw_diff = compute_yaw_diff(torch.zeros_like(root_euler[:, 2]), root_euler[:, 2]) / torch.pi
        yaw_reward = 1.0 / (1.0 + torch.square(3.0 * yaw_diff))

        spinnage = torch.square(self._robot.data.root_ang_vel_w[:, 2])
        spin_reward = 1.0 / (1.0 + torch.square(3.0 * spinnage))

        ups = quat_apply(self._robot.data.root_quat_w, torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1))
        ups_reward = torch.square((ups[:, 2] + 1.0) / 2.0)

        reward = continuous_action_reward + effort_reward + pos_reward
        reward += pos_reward * (vel_direction_reward + ups_reward + spin_reward + yaw_reward)
        if self.cfg.ctl_mode not in {"pos", "vel", "prop"}:
            reward += thrust_reward

        reward_info = {
            "continous_action_reward": continuous_action_reward,
            "effort_reward": effort_reward,
            "thrust_reward": thrust_reward,
            "pos_reward": pos_reward,
            "vel_direction_reward": vel_direction_reward,
            "ups_reward": ups_reward,
            "spin_reward": spin_reward,
            "yaw_reward": yaw_reward,
            "reward": reward,
        }
        return reward, reward_info

    def _compute_terminated(self) -> torch.Tensor:
        target_pos = self._target_states[:, 9:12]
        relative_positions = target_pos - self._root_pos_local()
        ups = quat_apply(self._robot.data.root_quat_w, torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1))
        terminated = torch.linalg.vector_norm(relative_positions, dim=-1) > 4.0
        terminated |= relative_positions[:, 2] < -2.0
        terminated |= relative_positions[:, 2] > 2.0
        terminated |= ups[:, 2] < 0.0
        if self.cfg.ctl_mode == "atti":
            terminated |= self._actions[:, 0] < 0.0
        return terminated
