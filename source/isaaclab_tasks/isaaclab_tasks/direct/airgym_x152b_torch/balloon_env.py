# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat, normalize, quat_apply, quat_from_euler_xyz, sample_uniform

from .base_env import AirGymX152bBaseEnv, compute_yaw_diff
from .base_env_cfg import AirGymX152bBaseEnvCfg
from .task_common import make_balloon_cfg, make_contact_sensor_cfg, make_plane_cfg


@configclass
class AirGymX152bBalloonEnvCfg(AirGymX152bBaseEnvCfg):
    episode_length_s = 8.0
    observation_space = 18
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=256, env_spacing=10.0, replicate_physics=True)
    terrain = make_plane_cfg()
    balloon = make_balloon_cfg()
    contact_sensor = make_contact_sensor_cfg()


class AirGymX152bBalloonEnv(AirGymX152bBaseEnv):
    cfg: AirGymX152bBalloonEnvCfg

    def __init__(self, cfg: AirGymX152bBalloonEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._previous_root_pos = torch.zeros(self.num_envs, 3, device=self.device)

    def _reset_task(self, env_ids: torch.Tensor):
        num_resets = len(env_ids)

        balloon_pos = torch.zeros(num_resets, 3, device=self.device)
        balloon_pos[:, 0] = 0.5 * sample_uniform(-1.0, 1.0, (num_resets, 1), self.device).squeeze(-1) + 2.5
        balloon_pos[:, 1] = 2.0 * sample_uniform(-1.0, 1.0, (num_resets, 1), self.device).squeeze(-1)
        balloon_pos[:, 2] = 0.3 * sample_uniform(-1.0, 1.0, (num_resets, 1), self.device).squeeze(-1) + 1.0
        balloon_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(num_resets, 1)
        self._write_rigid_object_state_local(self._balloon, env_ids, balloon_pos, balloon_quat)

        pos = torch.zeros(num_resets, 3, device=self.device)
        pos[:, :2] = 0.1 * sample_uniform(-1.0, 1.0, (num_resets, 2), self.device)
        pos[:, 2] = 0.2 * sample_uniform(-1.0, 1.0, (num_resets, 1), self.device).squeeze(-1) + 1.0

        roll = 0.1 * sample_uniform(-torch.pi, torch.pi, (num_resets,), self.device)
        pitch = 0.1 * sample_uniform(0.0, torch.pi, (num_resets,), self.device)
        yaw = 0.2 * sample_uniform(-torch.pi, torch.pi, (num_resets,), self.device)
        quat = quat_from_euler_xyz(roll, pitch, yaw)

        lin_vel = 0.5 * sample_uniform(-1.0, 1.0, (num_resets, 3), self.device)
        ang_vel = 0.2 * sample_uniform(-1.0, 1.0, (num_resets, 3), self.device)
        self._write_robot_state_local(env_ids, pos, quat, lin_vel, ang_vel)

        self._previous_root_pos[env_ids] = 0.0

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
        balloon_rot_matrix = matrix_from_quat(self._balloon.data.root_quat_w).reshape(self.num_envs, -1)
        obs[:, 0:9] -= balloon_rot_matrix
        obs[:, 9:12] -= self._balloon_pos_local()
        return {"policy": obs}

    def _compute_reward_and_metrics(self) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        relative_positions = self._balloon_pos_local() - self._root_pos_local()

        direction_vector = normalize(relative_positions)
        direction_yaw = torch.atan2(direction_vector[:, 1], direction_vector[:, 0])
        root_euler = self._root_euler_xyz()
        relative_heading = compute_yaw_diff(root_euler[:, 2], direction_yaw)
        yaw_distance = torch.abs(relative_heading)
        yaw_reward = 1.0 / (1.0 + torch.square(1.6 * yaw_distance))

        guidance_reward = 30.0 * (
            torch.linalg.vector_norm(self._balloon_pos_local() - self._previous_root_pos, dim=-1)
            - torch.linalg.vector_norm(self._balloon_pos_local() - self._root_pos_local(), dim=-1)
        )

        up_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)
        ups = quat_apply(self._robot.data.root_quat_w, up_axis)
        ups_reward = 0.5 * torch.square((ups[:, 2] + 1.0) / 2.0)

        check = torch.linalg.vector_norm(relative_positions, dim=-1)
        hit_reward = 800.0 * (check < 0.1).float()

        effort_reward = 0.1 * torch.exp(-self._actions.pow(2).sum(dim=-1))
        action_diff = torch.linalg.vector_norm(self._actions - self._previous_actions, dim=-1)
        action_smoothness_reward = 0.1 * torch.exp(-action_diff)

        reward = guidance_reward + yaw_reward + hit_reward + action_smoothness_reward + ups_reward + effort_reward

        reward_info = {
            "guidance_reward": guidance_reward,
            "yaw_reward": yaw_reward,
            "hit_reward": hit_reward,
            "action_smoothness_reward": action_smoothness_reward,
            "ups_reward": ups_reward,
            "effort_reward": effort_reward,
            "reward": reward,
        }
        return reward, reward_info

    def _compute_terminated(self) -> torch.Tensor:
        relative_positions = self._balloon_pos_local() - self._root_pos_local()
        check = torch.linalg.vector_norm(relative_positions, dim=-1)
        terminated = relative_positions[:, 0] < -0.2
        terminated |= self._robot.data.root_lin_vel_w[:, 0] < 0.0
        terminated |= check > 4.0
        terminated |= self._root_pos_local()[:, 2] < 0.5
        terminated |= self._root_pos_local()[:, 2] > 1.5
        terminated |= check < 0.1
        terminated |= self._collision_state()
        return terminated

    def _after_rewards(self):
        self._previous_root_pos.copy_(self._root_pos_local())
