# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import abstractmethod

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, TiledCamera
from isaaclab.utils.math import (
    euler_xyz_from_quat,
    matrix_from_quat,
    normalize,
    quat_from_matrix,
    quat_unique,
    wrap_to_pi,
)
from rlPx4Controller.pyParallelControl import (
    ParallelAttiControl,
    ParallelPosControl,
    ParallelRateControl,
    ParallelVelControl,
)

from .base_env_cfg import AirGymX152bBaseEnvCfg


def compute_yaw_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return wrap_to_pi(b - a)


class AirGymX152bBaseEnv(DirectRLEnv):
    cfg: AirGymX152bBaseEnvCfg

    def __init__(self, cfg: AirGymX152bBaseEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._env_origins = self.scene.env_origins
        self._target_states = torch.tensor(self.cfg.target_state, device=self.device, dtype=torch.float32).repeat(
            self.num_envs, 1
        )

        self._balloon = getattr(self, "_balloon", None)
        self._moving_obstacle = getattr(self, "_moving_obstacle", None)
        self._goal = getattr(self, "_goal", None)
        self._obstacles = getattr(self, "_obstacles", None)
        self._contact_sensor = getattr(self, "_contact_sensor", None)
        self._onboard_camera = getattr(self, "_onboard_camera", None)

        self._prop_body_ids = torch.tensor(
            self._robot.find_bodies(["prop_1", "prop_2", "prop_3", "prop_4"], preserve_order=True)[0],
            dtype=torch.long,
            device=self.device,
        )
        self._base_body_id = self._robot.find_bodies(["base_link"], preserve_order=True)[0][0]
        self._reaction_torque_sign = torch.tensor([-1.0, -1.0, 1.0, 1.0], device=self.device).view(1, 4)

        action_dim = gym.spaces.flatdim(self.single_action_space)
        self._policy_actions = torch.zeros(self.num_envs, action_dim, device=self.device)
        self._actions = torch.zeros_like(self._policy_actions)
        self._previous_actions = torch.zeros_like(self._policy_actions)
        self._cmd_thrusts = torch.zeros(self.num_envs, 4, device=self.device)
        self._rotor_forces = torch.zeros(self.num_envs, 4, 3, device=self.device)
        self._rotor_torques = torch.zeros(self.num_envs, 4, 3, device=self.device)

        self._action_lower, self._action_upper = self._resolve_action_bounds()
        self._controller = self._build_controller()

        self._camera_image = None
        self._camera_metric_image = None

    def _setup_scene(self):
        self._terrain = None
        self._contact_sensor = None
        self._onboard_camera = None
        self._balloon = None
        self._moving_obstacle = None
        self._goal = None
        self._obstacles = None

        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        if self.cfg.balloon is not None:
            self._balloon = RigidObject(self.cfg.balloon)
            self.scene.rigid_objects["balloon"] = self._balloon
        if self.cfg.moving_obstacle is not None:
            self._moving_obstacle = RigidObject(self.cfg.moving_obstacle)
            self.scene.rigid_objects["moving_obstacle"] = self._moving_obstacle
        if self.cfg.goal is not None:
            self._goal = RigidObject(self.cfg.goal)
            self.scene.rigid_objects["goal"] = self._goal
        if self.cfg.obstacles is not None:
            self._obstacles = RigidObjectCollection(self.cfg.obstacles)
            self.scene.rigid_object_collections["obstacles"] = self._obstacles
        if self.cfg.contact_sensor is not None:
            self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
            self.scene.sensors["contact_sensor"] = self._contact_sensor
        if self.cfg.onboard_camera is not None:
            self._onboard_camera = TiledCamera(self.cfg.onboard_camera)
            self.scene.sensors["onboard_camera"] = self._onboard_camera

        if self.cfg.terrain is not None:
            self.cfg.terrain.num_envs = self.scene.cfg.num_envs
            self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
            self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            global_paths = [self.cfg.terrain.prim_path] if self.cfg.terrain is not None else []
            self.scene.filter_collisions(global_prim_paths=global_paths)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _resolve_action_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        limits_by_mode = {
            "pos": self.cfg.pos_action_limits,
            "vel": self.cfg.vel_action_limits,
            "atti": self.cfg.atti_action_limits,
            "rate": self.cfg.rate_action_limits,
            "prop": self.cfg.prop_action_limits,
        }
        lower, upper = limits_by_mode[self.cfg.ctl_mode]
        return (
            torch.tensor(lower, device=self.device, dtype=torch.float32),
            torch.tensor(upper, device=self.device, dtype=torch.float32),
        )

    def _build_controller(self):
        if self.cfg.ctl_mode == "pos":
            return ParallelPosControl(self.num_envs)
        if self.cfg.ctl_mode == "vel":
            return ParallelVelControl(self.num_envs)
        if self.cfg.ctl_mode == "atti":
            return ParallelAttiControl(self.num_envs)
        if self.cfg.ctl_mode == "rate":
            return ParallelRateControl(self.num_envs)
        if self.cfg.ctl_mode == "prop":
            return None
        raise ValueError(f"Unsupported control mode: {self.cfg.ctl_mode}")

    def _pre_physics_step(self, actions: torch.Tensor):
        self._camera_image = None
        self._camera_metric_image = None
        self._policy_actions = actions.clone()
        processed_actions = actions.clone()
        if self.cfg.ctl_mode in {"rate", "atti"}:
            processed_actions[:, -1] = 0.5 + 0.5 * processed_actions[:, -1]
        processed_actions = torch.clamp(processed_actions, self._action_lower, self._action_upper)
        self._actions = processed_actions

        if self._controller is None:
            self._cmd_thrusts = processed_actions
        else:
            root_pos_local = self._root_pos_local().detach().cpu().numpy().astype(np.float64)
            root_quat_w = quat_unique(self._robot.data.root_quat_w).detach().cpu().numpy().astype(np.float64)
            root_lin_vel = self._robot.data.root_lin_vel_w.detach().cpu().numpy().astype(np.float64)
            root_ang_vel = self._robot.data.root_ang_vel_w.detach().cpu().numpy().astype(np.float64)
            actions_cpu = processed_actions.detach().cpu().numpy().astype(np.float64)
            dt = float(self.step_dt)

            if self.cfg.ctl_mode == "pos":
                self._controller.set_status(root_pos_local, root_quat_w, root_lin_vel, root_ang_vel, dt)
                cmd_thrusts = self._controller.update(actions_cpu)
            elif self.cfg.ctl_mode == "vel":
                self._controller.set_status(root_pos_local, root_quat_w, root_lin_vel, root_ang_vel, dt)
                cmd_thrusts = self._controller.update(actions_cpu)
            elif self.cfg.ctl_mode == "atti":
                self._controller.set_status(root_pos_local, root_quat_w, root_lin_vel, root_ang_vel, dt)
                cmd_thrusts = self._controller.update(actions_cpu)
            elif self.cfg.ctl_mode == "rate":
                self._controller.set_q_world(root_quat_w)
                cmd_thrusts = self._controller.update(actions_cpu, root_ang_vel, dt)
            else:
                raise ValueError(f"Unsupported control mode: {self.cfg.ctl_mode}")

            self._cmd_thrusts = torch.tensor(cmd_thrusts, device=self.device, dtype=torch.float32)

        self._rotor_forces.zero_()
        self._rotor_forces[:, :, 2] = self._cmd_thrusts * self.cfg.thrust_to_force
        self._rotor_torques.zero_()
        self._rotor_torques[:, :, 2] = (
            self._cmd_thrusts * self.cfg.reaction_torque_scale * self._reaction_torque_sign
        )

    def _apply_action(self):
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._prop_body_ids,
            forces=self._rotor_forces,
            torques=self._rotor_torques,
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        return self._get_task_observations()

    def _get_rewards(self) -> torch.Tensor:
        rewards, reward_info = self._compute_reward_and_metrics()
        self.extras["item_reward_info"] = reward_info
        self._previous_actions.copy_(self._actions)
        self._after_rewards()
        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = self._compute_terminated()
        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        if self._balloon is not None:
            self._balloon.reset(env_ids)
        if self._moving_obstacle is not None:
            self._moving_obstacle.reset(env_ids)
        if self._goal is not None:
            self._goal.reset(env_ids)
        if self._obstacles is not None:
            self._obstacles.reset(env_ids)
        if self._contact_sensor is not None:
            self._contact_sensor.reset(env_ids)
        if self._onboard_camera is not None:
            self._onboard_camera.reset(env_ids)

        super()._reset_idx(env_ids)

        self._policy_actions[env_ids] = 0.0
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._cmd_thrusts[env_ids] = 0.0
        self._rotor_forces[env_ids] = 0.0
        self._rotor_torques[env_ids] = 0.0
        self._robot.permanent_wrench_composer.reset(env_ids)

        self._reset_task(env_ids)
        self._camera_image = None
        self._camera_metric_image = None

    def _root_pos_local(self) -> torch.Tensor:
        return self._robot.data.root_pos_w - self._env_origins

    def _root_rotation_matrix(self) -> torch.Tensor:
        return matrix_from_quat(self._robot.data.root_quat_w)

    def _root_euler_xyz(self) -> torch.Tensor:
        roll, pitch, yaw = euler_xyz_from_quat(self._robot.data.root_quat_w)
        return torch.stack((roll, pitch, yaw), dim=-1)

    def _compute_yaw_local_frame(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rotation_global = self._root_rotation_matrix()
        yaw = torch.atan2(rotation_global[:, 1, 0], rotation_global[:, 0, 0])
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        zeros = torch.zeros_like(yaw)
        ones = torch.ones_like(yaw)
        world_to_local = torch.stack(
            (
                torch.stack((cos_yaw, -sin_yaw, zeros), dim=-1),
                torch.stack((sin_yaw, cos_yaw, zeros), dim=-1),
                torch.stack((zeros, zeros, ones), dim=-1),
            ),
            dim=1,
        )
        rotation_local = torch.matmul(world_to_local, rotation_global)
        local_quat = quat_from_matrix(rotation_local)
        roll, pitch, yaw_local = euler_xyz_from_quat(local_quat)
        euler_local = torch.stack((roll, pitch, yaw_local), dim=-1)
        lin_vel_local = torch.einsum("bij,bj->bi", world_to_local, self._robot.data.root_lin_vel_w)
        ang_vel_local = torch.einsum("bij,bj->bi", world_to_local, self._robot.data.root_ang_vel_w)
        return world_to_local, euler_local, lin_vel_local, ang_vel_local

    def _collision_state(self) -> torch.Tensor:
        if self._contact_sensor is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        net_forces = self._contact_sensor.data.net_forces_w
        return torch.linalg.vector_norm(net_forces, dim=-1).amax(dim=-1) > self.cfg.contact_force_threshold

    def _camera_depth_metric_image(self) -> torch.Tensor:
        if self._onboard_camera is None:
            raise RuntimeError("This task does not define an onboard camera.")

        if self._camera_metric_image is None:
            depth = self._onboard_camera.data.output["depth"].permute(0, 3, 1, 2).contiguous()
            if self.cfg.camera_additive_noise_std > 0.0:
                depth = depth + self.cfg.camera_additive_noise_std * self.cfg.camera_max_distance * torch.randn_like(depth)
            if self.cfg.camera_multiplicative_noise_std > 0.0:
                depth = depth * (1.0 + self.cfg.camera_multiplicative_noise_std * torch.randn_like(depth))
            self._camera_metric_image = depth.clamp_(0.0, self.cfg.camera_max_distance)
        return self._camera_metric_image

    def _camera_depth_image(self) -> torch.Tensor:
        if self._camera_image is None:
            depth = self._camera_depth_metric_image().clone()
            depth = depth / self.cfg.camera_max_distance
            self._camera_image = depth.clamp_(0.0, 1.0)
        return self._camera_image

    def _add_state_noise(self, observations: torch.Tensor) -> torch.Tensor:
        noisy_obs = observations.clone()
        if noisy_obs.shape[-1] < 18:
            return noisy_obs
        noisy_obs[:, 0:9] += 1.0e-3 * torch.randn(self.num_envs, 9, device=self.device)
        noisy_obs[:, 9:12] += 5.0e-3 * torch.randn(self.num_envs, 3, device=self.device)
        noisy_obs[:, 12:15] += 2.0e-2 * torch.randn(self.num_envs, 3, device=self.device)
        noisy_obs[:, 15:18] += 4.0e-1 * torch.randn(self.num_envs, 3, device=self.device)
        return noisy_obs

    def _write_robot_state_local(
        self,
        env_ids: torch.Tensor,
        pos_local: torch.Tensor,
        quat_w: torch.Tensor,
        lin_vel_w: torch.Tensor,
        ang_vel_w: torch.Tensor,
    ):
        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] = pos_local + self._env_origins[env_ids]
        root_state[:, 3:7] = quat_w
        root_state[:, 7:10] = lin_vel_w
        root_state[:, 10:13] = ang_vel_w
        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _write_rigid_object_state_local(
        self,
        obj: RigidObject,
        env_ids: torch.Tensor,
        pos_local: torch.Tensor,
        quat_w: torch.Tensor,
        lin_vel_w: torch.Tensor | None = None,
        ang_vel_w: torch.Tensor | None = None,
    ):
        root_state = obj.data.default_root_state[env_ids].clone()
        root_state[:, :3] = pos_local + self._env_origins[env_ids]
        root_state[:, 3:7] = quat_w
        if lin_vel_w is None:
            root_state[:, 7:10] = 0.0
        else:
            root_state[:, 7:10] = lin_vel_w
        if ang_vel_w is None:
            root_state[:, 10:13] = 0.0
        else:
            root_state[:, 10:13] = ang_vel_w
        obj.write_root_pose_to_sim(root_state[:, :7], env_ids)
        obj.write_root_velocity_to_sim(root_state[:, 7:], env_ids)

    def _write_object_collection_state_local(
        self,
        obj_collection: RigidObjectCollection,
        env_ids: torch.Tensor,
        pos_local: torch.Tensor,
        quat_w: torch.Tensor,
        lin_vel_w: torch.Tensor | None = None,
        ang_vel_w: torch.Tensor | None = None,
    ):
        object_state = obj_collection.data.default_object_state[env_ids].clone()
        object_state[:, :, :3] = pos_local + self._env_origins[env_ids].unsqueeze(1)
        object_state[:, :, 3:7] = quat_w
        if lin_vel_w is None:
            object_state[:, :, 7:10] = 0.0
        else:
            object_state[:, :, 7:10] = lin_vel_w
        if ang_vel_w is None:
            object_state[:, :, 10:13] = 0.0
        else:
            object_state[:, :, 10:13] = ang_vel_w
        obj_collection.write_object_state_to_sim(object_state, env_ids=env_ids)

    def _balloon_pos_local(self) -> torch.Tensor:
        if self._balloon is None:
            raise RuntimeError("This task does not define a balloon asset.")
        return self._balloon.data.root_pos_w - self._env_origins

    def _goal_pos_local(self) -> torch.Tensor:
        if self._goal is None:
            raise RuntimeError("This task does not define a goal asset.")
        return self._goal.data.root_pos_w - self._env_origins

    def _after_rewards(self):
        pass

    @abstractmethod
    def _reset_task(self, env_ids: torch.Tensor):
        raise NotImplementedError

    @abstractmethod
    def _get_task_observations(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def _compute_reward_and_metrics(self) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        raise NotImplementedError

    @abstractmethod
    def _compute_terminated(self) -> torch.Tensor:
        raise NotImplementedError
