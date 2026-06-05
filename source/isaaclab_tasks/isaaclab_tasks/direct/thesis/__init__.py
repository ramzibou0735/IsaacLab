# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Thesis direct-workflow environments."""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-Thesis-X152b-Exploration-Direct-v0",
    entry_point=f"{__name__}.exploration_env:ThesisX152bExplorationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.exploration_env:ThesisX152bExplorationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ThesisX152bExplorationPPORunnerCfg",
    },
)
