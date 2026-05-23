# UAV Indoor Exploration DRL Policy Notes

## Summary

The best fit for `exploration_env.py` is a visual PPO exploration policy with asymmetric training: the actor receives UAV state plus forward depth, while the critic receives state plus a privileged local occupancy grid. This matches the existing RSL-RL configuration for `AirGymX152bExplorationPPORunnerCfg`, where the actor uses `["observation", "image"]` and the critic uses `["observation", "critic_grid"]`.

The main implementation gap is the exploration signal. The current environment rewards path coverage from the UAV body position. Indoor exploration policies in the literature more often reward information gain: newly observed free, occupied, or unknown cells from sensor rays, sometimes with frontier or curiosity bonuses. For this environment, the strongest improvement is to make coverage camera-visible instead of body-footprint-only.

## Relevant Policy Families

### Visual Continuous-Control PPO

This environment already matches the common visual UAV RL setup:

- Low-dimensional proprioceptive state for stabilization.
- Forward depth image for obstacle perception.
- Continuous velocity action interface through the existing controller.
- PPO-compatible actor and critic networks.

This is the lowest-friction path because `ctl_mode = "vel"` uses continuous 4D actions, while DQN-style approaches require discretized high-level actions.

### Curiosity-Augmented Exploration

Recent UAV indoor exploration work uses intrinsic motivation, such as Intrinsic Curiosity Module (ICM), and parameter-noise exploration, such as NoisyNet, to improve exploration in unfamiliar indoor scenes.

In this IsaacLab tree, Random Network Distillation (RND) is the closest built-in option. `RslRlPpoAlgorithmCfg` supports `rnd_cfg`, and `RslRlRndCfg` exposes intrinsic reward weight, normalization, learning rate, and predictor/target dimensions. RND should be tried before adding a custom ICM trainer because it requires less infrastructure.

### Frontier or Graph-Based Exploration

Frontier and graph policies typically act over a partial map and select exploration goals or frontiers. They are useful for long-horizon exploration, but they are better implemented as a higher-level planner around this environment rather than as a small change to the current direct 4D velocity-action policy.

## Current Observations

`exploration_env.py` currently returns:

- `policy`: low-dimensional state vector.
- `observation`: same low-dimensional state vector for vision model fusion.
- `image`: normalized forward depth image.
- `critic_grid`: local depth-derived free/occupied BEV grid for asymmetric critic training.

The state vector includes:

- Normalized local x, y, and height error.
- Local roll, pitch, yaw.
- Local linear velocity.
- Local angular velocity.
- Last policy action.
- Global coverage ratio.

This is a reasonable baseline for a reactive visual policy, but it does not expose frontier direction, local unknown-space structure, or persistent map state to the actor.

## Current Reward Function

The reward combines:

- `coverage_reward = 2.0 * new_visit_count`
- `alive_reward`
- upright reward
- target-height reward
- action-smoothness reward
- low-spin reward
- depth proximity penalty
- collision penalty
- bounds penalty
- bad-height penalty

The important limitation is that `new_visit_count` comes from `CoverageGrid2D.update(None, root_pos_local[:, :2])`, so exploration credit is tied to the drone path. A UAV can observe substantial free space without physically visiting it, and that observed information is currently not rewarded.

## Recommended Implementation Direction

1. Keep PPO with the asymmetric critic.

   The existing actor and critic observation split is well aligned with visual UAV exploration. The actor stays deployable because it uses onboard state and depth only, while the critic can use privileged map-like tensors during training.

2. Add camera-visible information gain.

   Maintain a persistent env-local known-space grid built from depth rays. Reward newly observed cells:

   ```python
   info_gain_reward = w_free * new_free_cells + w_occ * new_occupied_cells
   ```

   This can reuse the existing depth-to-pointcloud and free/occupied BEV utilities already used by `build_critic_grid_from_depth`.

3. Add local exploration features to the actor state.

   Useful low-cost additions:

   - nearest frontier distance
   - nearest frontier bearing in yaw-local frame
   - unknown-ahead ratio
   - local known-space ratio

   If these are added to `_state_observation()`, update `observation_space` accordingly.

4. Rebalance survival and collision rewards.

   `alive_reward = 0.1` can accumulate strongly over a 20 second episode. Consider reducing it or only awarding it when the policy is still making exploration progress. Also consider increasing terminal collision cost relative to possible coverage return.

5. Try RND before ICM.

   RND can be enabled through the existing RSL-RL config structure. ICM would require adding learned forward/inverse prediction losses and a custom intrinsic reward path.

## Implementation Hooks

- Environment: `source/isaaclab_tasks/isaaclab_tasks/direct/airgym_x152b/exploration_env.py`
- Depth to critic grid: `source/isaaclab_tasks/isaaclab_tasks/direct/airgym_x152b/exploration_helpers.py`
- Camera configuration: `source/isaaclab_tasks/isaaclab_tasks/direct/airgym_x152b/task_common.py`
- PPO config: `source/isaaclab_tasks/isaaclab_tasks/direct/airgym_x152b/agents/rsl_rl_ppo_cfg.py`
- Coverage helper: `source/isaaclab/isaaclab/utils/coverage.py`
- Occupancy helper: `source/isaaclab/isaaclab/utils/occupancy.py`
- RND config: `source/isaaclab_rl/isaaclab_rl/rsl_rl/rnd_cfg.py`

## Sources

- Air Learning: A deep reinforcement learning gym for autonomous aerial robot visual navigation, Machine Learning, 2021: https://link.springer.com/article/10.1007/s10994-021-06006-6
- UAV exploration for indoor navigation based on deep reinforcement learning and intrinsic curiosity, Results in Engineering, 2026: https://www.sciencedirect.com/science/article/pii/S2667305325001449
- Curiosity-driven Exploration by Self-supervised Prediction, Pathak et al., 2017: https://arxiv.org/abs/1705.05363
- Noisy Networks for Exploration, Fortunato et al., 2017: https://arxiv.org/abs/1706.10295
- Proximal Policy Optimization Algorithms, Schulman et al., 2017: https://arxiv.org/abs/1707.06347
- ARiADNE: A Reinforcement learning approach using Attention-based Deep Networks for Exploration, 2023: https://arxiv.org/abs/2301.11575
- Autonomous Exploration Under Uncertainty via Deep Reinforcement Learning on Graphs, 2020: https://arxiv.org/abs/2007.12640
