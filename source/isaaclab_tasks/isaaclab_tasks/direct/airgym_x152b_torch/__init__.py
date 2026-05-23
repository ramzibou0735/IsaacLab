# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AirGym x152b direct-workflow environments using the torch-native controller."""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-AirGym-X152b-Hovering-Torch-Direct-v0",
    entry_point=f"{__name__}.hovering_env:AirGymX152bHoveringEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hovering_env:AirGymX152bHoveringEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AirGymX152bHoveringPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-AirGym-X152b-Balloon-Torch-Direct-v0",
    entry_point=f"{__name__}.balloon_env:AirGymX152bBalloonEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.balloon_env:AirGymX152bBalloonEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AirGymX152bBalloonPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-AirGym-X152b-Tracking-Torch-Direct-v0",
    entry_point=f"{__name__}.tracking_env:AirGymX152bTrackingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env:AirGymX152bTrackingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AirGymX152bTrackingPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-AirGym-X152b-Avoid-Torch-Direct-v0",
    entry_point=f"{__name__}.avoid_env:AirGymX152bAvoidEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.avoid_env:AirGymX152bAvoidEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AirGymX152bAvoidPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-AirGym-X152b-Planning-Torch-Direct-v0",
    entry_point=f"{__name__}.planning_env:AirGymX152bPlanningEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.planning_env:AirGymX152bPlanningEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AirGymX152bPlanningPPORunnerCfg",
    },
)
