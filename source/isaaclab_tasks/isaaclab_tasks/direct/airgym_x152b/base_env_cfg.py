# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.drone import X152B_CFG


@configclass
class AirGymX152bBaseEnvCfg(DirectRLEnvCfg):
    episode_length_s = 24.0
    decimation = 1
    action_space = 4
    observation_space = 18
    state_space = 0
    debug_vis = False

    ctl_mode = "vel"
    target_state = (
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )

    sim: SimulationCfg = SimulationCfg(
        dt=0.01,
        render_interval=1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=256, env_spacing=1.0, replicate_physics=True)

    robot: ArticulationCfg = X152B_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    terrain: TerrainImporterCfg | None = None
    contact_sensor: ContactSensorCfg | None = None
    onboard_camera: TiledCameraCfg | None = None
    balloon: RigidObjectCfg | None = None
    moving_obstacle: RigidObjectCfg | None = None
    goal: RigidObjectCfg | None = None
    obstacles: RigidObjectCollectionCfg | None = None

    thrust_to_force = 9.59
    reaction_torque_scale = 0.2
    hover_thrust = 0.1533
    contact_force_threshold = 0.1
    camera_max_distance = 4.5
    camera_additive_noise_std = 0.0
    camera_multiplicative_noise_std = 0.0

    pos_action_limits = ((-3.0, -3.0, -3.0, -6.0), (3.0, 3.0, 3.0, 6.0))
    vel_action_limits = ((-6.0, -6.0, -6.0, -6.0), (6.0, 6.0, 6.0, 6.0))
    atti_action_limits = ((-1.0, -1.0, -1.0, -1.0, 0.0), (1.0, 1.0, 1.0, 1.0, 1.0))
    rate_action_limits = ((-6.0, -6.0, -6.0, 0.0), (6.0, 6.0, 6.0, 1.0))
    prop_action_limits = ((0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0))

    def __post_init__(self):
        action_dims = {
            "pos": 4,
            "vel": 4,
            "atti": 5,
            "rate": 4,
            "prop": 4,
        }
        if self.ctl_mode not in action_dims:
            raise ValueError(f"Unsupported control mode: {self.ctl_mode}")
        self.action_space = action_dims[self.ctl_mode]
        if self.onboard_camera is not None:
            self.num_rerenders_on_reset = max(self.num_rerenders_on_reset, 1)
