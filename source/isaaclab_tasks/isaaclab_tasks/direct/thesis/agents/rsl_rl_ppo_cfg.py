# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class ThesisActiveMapPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "isaaclab_tasks.direct.thesis.agents.thesis_active_map_ppo:ThesisActiveMapPPO"
    model_class_name: str = (
        "isaaclab_tasks.direct.thesis.agents.thesis_active_map_ppo:ThesisActiveMapActorCritic"
    )
    model_cfg: dict = {
        "vector_obs_group": "observations",
        "image_obs_group": "observations_image",
        "vae_obs_group": "observations_vae",
        "map_obs_group": "observations_map",
        "critic_obs_group": "critic_observations",
        "use_vae_latent": False,
        "activation": "elu",
        "state_hidden_dims": [256, 256],
        "image_latent_dim": 512,
        "map_latent_dim": 512,
        "fusion_hidden_dims": [1024, 512],
        "rnn_hidden_dim": 512,
        "rnn_num_layers": 1,
        "critic_hidden_dims": [512, 512, 256],
        "obs_normalization": True,
        "critic_obs_normalization": True,
        "distribution_cfg": RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5).to_dict(),
    }


THESIS_ALGORITHM_CFG = ThesisActiveMapPpoAlgorithmCfg(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=0.001,
    num_learning_epochs=4,
    num_mini_batches=4,
    learning_rate=3.0e-4,
    schedule="adaptive",
    gamma=0.99,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,
)


@configclass
class ThesisX152bExplorationPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 64
    max_iterations = 5000
    empirical_normalization = False
    clip_actions = 1.0
    save_interval = 50
    experiment_name = "thesis_x152b_exploration"
    logger = "wandb"
    wandb_project = "isaaclab-thesis-x152b"
    obs_groups = {
        "actor": ["observations", "observations_image", "observations_map"],
        "critic": ["critic_observations"],
    }
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 256],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 512, 256],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = THESIS_ALGORITHM_CFG.copy()


__all__ = [
    "ThesisActiveMapPpoAlgorithmCfg",
    "ThesisX152bExplorationPPORunnerCfg",
]
