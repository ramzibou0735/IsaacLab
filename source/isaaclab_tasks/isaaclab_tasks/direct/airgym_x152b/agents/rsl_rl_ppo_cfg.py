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
    RslRlRndCfg,
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

ASYMMETRIC_VISION_ALGORITHM_CFG = VISION_ALGORITHM_CFG.copy()
ASYMMETRIC_VISION_ALGORITHM_CFG.share_cnn_encoders = False


@configclass
class ActiveMapPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "isaaclab_tasks.direct.airgym_x152b.agents.active_map_ppo:ActiveMapPPO"
    model_class_name: str = "isaaclab_tasks.direct.airgym_x152b.agents.active_map_ppo:ActiveMapSharedActorCritic"
    model_cfg: dict = {
        "vector_obs_group": "observations",
        "map_obs_group": "observations_map",
        "map_stages": [[8, 2], [16, 2], [16, 2]],
        "encoder_hidden_dims": [256, 128, 64],
        "activation": "elu",
        "rnn_hidden_dim": 64,
        "rnn_num_layers": 1,
        "obs_normalization": False,
        "distribution_cfg": RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5).to_dict(),
    }


ACTIVE_MAP_ALGORITHM_CFG = ActiveMapPpoAlgorithmCfg(
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
)

EXPLORATION_RND_ALGORITHM_CFG = ASYMMETRIC_VISION_ALGORITHM_CFG.copy()
EXPLORATION_RND_ALGORITHM_CFG.rnd_cfg = RslRlRndCfg(
    weight=0.01,
    reward_normalization=True,
    state_normalization=True,
    learning_rate=1.0e-4,
    num_outputs=16,
    predictor_hidden_dims=[128, 64],
    target_hidden_dims=[128, 64],
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
    experiment_name = "airgym_x152b"
    logger = "wandb"
    wandb_project = "isaaclab-airgym-x152b"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = MLP_ACTOR_CFG.copy()
    critic = MLP_CRITIC_CFG.copy()
    algorithm = BASE_ALGORITHM_CFG.copy()


@configclass
class AirGymX152bHoveringPPORunnerCfg(AirGymX152bBasePPORunnerCfg):
    experiment_name = "airgym_x152b_hovering"


@configclass
class AirGymX152bBalloonPPORunnerCfg(AirGymX152bBasePPORunnerCfg):
    experiment_name = "airgym_x152b_balloon"


@configclass
class AirGymX152bTrackingPPORunnerCfg(AirGymX152bBasePPORunnerCfg):
    experiment_name = "airgym_x152b_tracking"
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
    experiment_name = "airgym_x152b_vision"
    logger = "wandb"
    wandb_project = "isaaclab-airgym-x152b"
    obs_groups = {"actor": ["observation", "image"], "critic": ["observation", "image"]}
    actor = VISION_ACTOR_CFG.copy()
    critic = VISION_CRITIC_CFG.copy()
    algorithm = VISION_ALGORITHM_CFG.copy()


@configclass
class AirGymX152bAvoidPPORunnerCfg(AirGymX152bVisionPPORunnerCfg):
    experiment_name = "airgym_x152b_avoid"


@configclass
class AirGymX152bPlanningPPORunnerCfg(AirGymX152bVisionPPORunnerCfg):
    experiment_name = "airgym_x152b_planning"


@configclass
class AirGymX152bExplorationPPORunnerCfg(AirGymX152bVisionPPORunnerCfg):
    experiment_name = "airgym_x152b_exploration"
    max_iterations = 2500
    num_steps_per_env = 32
    obs_groups = {"actor": ["observations", "observations_map"], "critic": ["observations", "observations_map"]}
    actor = MLP_ACTOR_CFG.copy()
    critic = MLP_CRITIC_CFG.copy()
    algorithm = ACTIVE_MAP_ALGORITHM_CFG.copy()


@configclass
class AirGymX152bExplorationRndPPORunnerCfg(AirGymX152bExplorationPPORunnerCfg):
    experiment_name = "airgym_x152b_exploration_rnd"
    obs_groups = {
        "actor": ["observation", "latent"],
        "critic": ["observation", "critic_grid"],
        "rnd_state": ["observation", "critic_grid", "latent"],
    }
    actor = MLP_ACTOR_CFG.copy()
    critic = VISION_CRITIC_CFG.copy()
    algorithm = EXPLORATION_RND_ALGORITHM_CFG.copy()
