# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlCNNModelCfg,
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


MLP_ACTOR_CFG = RslRlMLPModelCfg(
    hidden_dims=[256, 128, 64],
    activation="elu",
    obs_normalization=False,
    distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5),
)

MLP_CRITIC_CFG = RslRlMLPModelCfg(
    hidden_dims=[256, 128, 64],
    activation="elu",
    obs_normalization=False,
)

VISION_ACTOR_CFG = RslRlCNNModelCfg(
    hidden_dims=[256, 128, 64],
    activation="elu",
    obs_normalization=False,
    distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5),
    cnn_cfg=RslRlCNNModelCfg.CNNCfg(
        output_channels=[16, 32, 64],
        kernel_size=[5, 5, 3],
        stride=[2, 2, 2],
        padding="zeros",
        norm=["batch", "batch", "batch"],
        activation="elu",
        global_pool="avg",
    ),
)

VISION_CRITIC_CFG = RslRlCNNModelCfg(
    hidden_dims=[256, 128, 64],
    activation="elu",
    obs_normalization=False,
    cnn_cfg=RslRlCNNModelCfg.CNNCfg(
        output_channels=[16, 32, 64],
        kernel_size=[5, 5, 3],
        stride=[2, 2, 2],
        padding="zeros",
        norm=["batch", "batch", "batch"],
        activation="elu",
        global_pool="avg",
    ),
)

BASE_ALGORITHM_CFG = RslRlPpoAlgorithmCfg(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=0.001,
    num_learning_epochs=4,
    num_mini_batches=4,
    learning_rate=4.0e-4,
    schedule="adaptive",
    gamma=0.98,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,
)

VISION_ALGORITHM_CFG = RslRlPpoAlgorithmCfg(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=0.001,
    num_learning_epochs=4,
    num_mini_batches=4,
    learning_rate=3.0e-4,
    schedule="adaptive",
    gamma=0.98,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,
    share_cnn_encoders=True,
)


@configclass
class AirGymX152bBasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 1500
    empirical_normalization = False
    clip_actions = 1.0
    save_interval = 50
    experiment_name = "airgym_x152b_torch"
    logger = "wandb"
    wandb_project = "isaaclab-airgym-x152b"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = MLP_ACTOR_CFG.copy()
    critic = MLP_CRITIC_CFG.copy()
    algorithm = BASE_ALGORITHM_CFG.copy()


@configclass
class AirGymX152bHoveringPPORunnerCfg(AirGymX152bBasePPORunnerCfg):
    experiment_name = "airgym_x152b_hovering_torch"


@configclass
class AirGymX152bBalloonPPORunnerCfg(AirGymX152bBasePPORunnerCfg):
    experiment_name = "airgym_x152b_balloon_torch"


@configclass
class AirGymX152bTrackingPPORunnerCfg(AirGymX152bBasePPORunnerCfg):
    experiment_name = "airgym_x152b_tracking_torch"
    max_iterations = 2000


@configclass
class AirGymX152bVisionPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 32
    max_iterations = 2000
    empirical_normalization = False
    clip_actions = 1.0
    save_interval = 50
    experiment_name = "airgym_x152b_vision_torch"
    logger = "wandb"
    wandb_project = "isaaclab-airgym-x152b"
    obs_groups = {"actor": ["observation", "image"], "critic": ["observation", "image"]}
    actor = VISION_ACTOR_CFG.copy()
    critic = VISION_CRITIC_CFG.copy()
    algorithm = VISION_ALGORITHM_CFG.copy()


@configclass
class AirGymX152bAvoidPPORunnerCfg(AirGymX152bVisionPPORunnerCfg):
    experiment_name = "airgym_x152b_avoid_torch"


@configclass
class AirGymX152bPlanningPPORunnerCfg(AirGymX152bVisionPPORunnerCfg):
    experiment_name = "airgym_x152b_planning_torch"
