# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.terrains import TerrainImporterCfg


def make_plane_cfg() -> TerrainImporterCfg:
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0)),
        debug_vis=False,
    )


def make_contact_sensor_cfg() -> ContactSensorCfg:
    return ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        history_length=1,
        update_period=0.01,
        track_pose=False,
        debug_vis=False,
    )


def make_onboard_camera_cfg() -> TiledCameraCfg:
    return TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/base_link/OnboardCamera",
        update_period=0.04,
        update_latest_camera_pose=True,
        offset=TiledCameraCfg.OffsetCfg(pos=(0.15, 0.0, 0.1), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        data_types=["depth"],
        depth_clipping_behavior="max",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=11.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 4.5),
        ),
        width=212,
        height=120,
    )


def make_balloon_cfg() -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/Balloon",
        spawn=sim_utils.SphereCfg(
            radius=0.1,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.4, 0.4)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.5, 0.0, 1.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )


def make_moving_obstacle_cfg() -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/Obstacle",
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.4, 0.4)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(4.2, 0.0, 1.4), rot=(1.0, 0.0, 0.0, 0.0)),
    )


def make_goal_cfg() -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/Goal",
        spawn=sim_utils.SphereCfg(
            radius=0.15,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.4, 0.4)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(8.5, 0.0, 1.5), rot=(1.0, 0.0, 0.0, 0.0)),
    )


def make_planning_obstacles_cfg(num_obstacles: int = 40) -> RigidObjectCollectionCfg:
    obstacle_cfgs: dict[str, RigidObjectCfg] = {}
    for obstacle_id in range(num_obstacles):
        obstacle_cfgs[f"obstacle_{obstacle_id:02d}"] = RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/ThinObstacle_{obstacle_id:02d}",
            spawn=sim_utils.CuboidCfg(
                size=(0.1, 0.8, 2.0),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.27, 0.07)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0), rot=(1.0, 0.0, 0.0, 0.0)),
        )
    return RigidObjectCollectionCfg(rigid_objects=obstacle_cfgs)


def make_exploration_room_cfg(num_pillars: int = 8) -> RigidObjectCollectionCfg:
    obstacle_cfgs: dict[str, RigidObjectCfg] = {}

    wall_specs = {
        "wall_north": ((10.6, 0.2, 2.2), (0.0, 5.1, 1.1)),
        "wall_south": ((10.6, 0.2, 2.2), (0.0, -5.1, 1.1)),
        "wall_east": ((0.2, 10.6, 2.2), (5.1, 0.0, 1.1)),
        "wall_west": ((0.2, 10.6, 2.2), (-5.1, 0.0, 1.1)),
    }
    for wall_name, (size, pos) in wall_specs.items():
        obstacle_cfgs[wall_name] = RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/{wall_name}",
            spawn=sim_utils.CuboidCfg(
                size=size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.82, 0.82, 0.85)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=(1.0, 0.0, 0.0, 0.0)),
        )

    for pillar_id in range(num_pillars):
        obstacle_cfgs[f"pillar_{pillar_id:02d}"] = RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/Pillar_{pillar_id:02d}",
            spawn=sim_utils.CuboidCfg(
                size=(0.5, 0.5, 2.2),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.52, 0.34, 0.16)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.1), rot=(1.0, 0.0, 0.0, 0.0)),
        )

    return RigidObjectCollectionCfg(rigid_objects=obstacle_cfgs)
