# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, sample_uniform

from .base_env import AirGymX152bBaseEnv, compute_yaw_diff
from .base_env_cfg import AirGymX152bBaseEnvCfg
from .task_common import make_plane_cfg


@configclass
class AirGymX152bTrackingEnvCfg(AirGymX152bBaseEnvCfg):
    episode_length_s = 36.0
    observation_space = 48
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=256, env_spacing=10.0, replicate_physics=True)
    terrain = make_plane_cfg()
    pos_action_limits = ((-6.0, -6.0, -6.0, -6.0), (6.0, 6.0, 6.0, 6.0))


class AirGymX152bTrackingEnv(AirGymX152bBaseEnv):
    cfg: AirGymX152bTrackingEnvCfg

    def __init__(self, cfg: AirGymX152bTrackingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._previous_root_pos = torch.zeros(self.num_envs, 3, device=self.device)

    def _reset_task(self, env_ids: torch.Tensor):
        num_resets = len(env_ids)
        pos = torch.zeros(num_resets, 3, device=self.device)
        pos[:, :2] = 0.1 * sample_uniform(-1.0, 1.0, (num_resets, 2), self.device)
        pos[:, 2] = 0.1 * sample_uniform(-1.0, 1.0, (num_resets, 1), self.device).squeeze(-1) + 1.0
        roll = 0.1 * sample_uniform(-torch.pi, torch.pi, (num_resets,), self.device)
        pitch = 0.1 * sample_uniform(-torch.pi, torch.pi, (num_resets,), self.device)
        yaw = 0.2 * sample_uniform(-torch.pi, torch.pi, (num_resets,), self.device)
        quat = quat_from_euler_xyz(roll, pitch, yaw)
        lin_vel = 0.5 * sample_uniform(-1.0, 1.0, (num_resets, 3), self.device)
        ang_vel = 0.2 * sample_uniform(-1.0, 1.0, (num_resets, 3), self.device)
        self._write_robot_state_local(env_ids, pos, quat, lin_vel, ang_vel)
        self._previous_root_pos[env_ids] = 0.0

    def _compute_reference_positions(self, n_steps: int = 10, step_size: int = 5, scale: float = 0.25) -> torch.Tensor:
        step = self.episode_length_buf.unsqueeze(1) + torch.arange(n_steps, device=self.device).view(1, -1) * step_size
        t = step * self.step_dt * scale
        ref_x = 3.0 * torch.sin(t) / (1.0 + torch.cos(t) ** 2)
        ref_y = 3.0 * torch.sin(t) * torch.cos(t) / (1.0 + torch.cos(t) ** 2)
        ref_z = torch.ones_like(ref_x)
        return torch.stack((ref_x, ref_y, ref_z), dim=-1)

    def _get_task_observations(self) -> dict[str, torch.Tensor]:
        ref_positions = self._compute_reference_positions()
        obs_state = torch.cat(
            (
                self._root_rotation_matrix().reshape(self.num_envs, -1),
                self._root_pos_local(),
                self._robot.data.root_lin_vel_w,
                self._robot.data.root_ang_vel_w,
            ),
            dim=-1,
        )
        obs_state = self._add_state_noise(obs_state)
        future_pos = (ref_positions - self._root_pos_local().unsqueeze(1)).reshape(self.num_envs, -1)
        obs = torch.cat((obs_state, future_pos), dim=-1)
        return {"policy": obs}

    def _compute_reward_and_metrics(self) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        ref_positions = self._compute_reference_positions()
        thrust_cmds = self._cmd_thrusts.clamp(0.0, 1.0)
        effort_reward = 0.1 * (1.0 - thrust_cmds).sum(dim=-1) / 4.0

        action_diff = self._actions - self._previous_actions
        thrust_reward = torch.zeros(self.num_envs, device=self.device)
        if self.cfg.ctl_mode in {"pos", "vel", "prop"}:
            continuous_action_reward = 0.2 * torch.exp(-torch.linalg.vector_norm(action_diff, dim=-1))
        else:
            continuous_action_reward = 0.1 * torch.exp(-torch.linalg.vector_norm(action_diff[:, :-1], dim=-1))
            continuous_action_reward += 0.5 / (1.0 + torch.square(2.0 * action_diff[:, -1]))
            thrust_reward = 0.1 * (1.0 - torch.abs(self.cfg.hover_thrust - self._actions[:, -1]))

        dist_diff = ref_positions[:, 0] - self._root_pos_local()
        dist_norm = torch.linalg.vector_norm(dist_diff, dim=-1)
        dist_reward = 1.0 / (1.0 + torch.square(1.8 * dist_norm))

        root_euler = self._root_euler_xyz()
        yaw_diff = compute_yaw_diff(torch.zeros_like(root_euler[:, 2]), root_euler[:, 2]) / torch.pi
        yaw_reward = 1.0 / (1.0 + torch.square(4.0 * yaw_diff))

        spinnage = torch.square(self._robot.data.root_ang_vel_w[:, 2])
        spin_reward = 1.0 / (1.0 + torch.square(2.0 * spinnage))

        ups = quat_apply(self._robot.data.root_quat_w, torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1))
        ups_reward = torch.square((ups[:, 2] + 1.0) / 2.0)

        reward = continuous_action_reward + effort_reward + dist_reward
        reward += dist_reward * (spin_reward + yaw_reward + ups_reward)
        if self.cfg.ctl_mode not in {"pos", "vel", "prop"}:
            reward += thrust_reward

        reward_info = {
            "dist_norm": dist_norm,
            "dist_reward": dist_reward,
            "yaw_reward": yaw_reward,
            "spin_reward": spin_reward,
            "continous_action_reward": continuous_action_reward,
            "thrust_reward": thrust_reward,
            "effort_reward": effort_reward,
            "ups_reward": ups_reward,
            "reward": reward,
        }
        return reward, reward_info

    def _compute_terminated(self) -> torch.Tensor:
        ref_positions = self._compute_reference_positions()
        dist_norm = torch.linalg.vector_norm(ref_positions[:, 0] - self._root_pos_local(), dim=-1)
        terminated = dist_norm > 1.0
        if self.cfg.ctl_mode == "atti":
            terminated |= self._actions[:, 0] < 0.0
        return terminated

    def _after_rewards(self):
        self._previous_root_pos.copy_(self._root_pos_local())
