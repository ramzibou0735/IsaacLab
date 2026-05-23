# AirGym Exploration Research Base

Status: validated working baseline as of 2026-04-13.

This document defines the current canonical exploration environment for the AirGym x152b IsaacLab port. Future sessions should treat this as the starting point for exploration, obstacle avoidance, mapping, and unknown-environment navigation research.

## Scope

Environment id:

- `Isaac-AirGym-X152b-Exploration-Direct-v0`

Primary intent:

- exploration in an unknown indoor room
- obstacle avoidance with onboard depth sensing
- asymmetric actor-critic training where the actor consumes depth image features and the critic consumes a compact depth-derived occupancy grid

Research rule:

- keep the CPU-controller AirGym task family as the main blueprint
- do not treat the torch-controller copy as the source of truth

## Canonical Files

Task implementation:

- `source/isaaclab_tasks/isaaclab_tasks/direct/airgym_x152b/exploration_env.py`

Task helpers:

- `source/isaaclab_tasks/isaaclab_tasks/direct/airgym_x152b/exploration_helpers.py`
- `source/isaaclab_tasks/isaaclab_tasks/direct/airgym_x152b/task_common.py`

Shared utilities used by this task:

- `source/isaaclab/isaaclab/utils/coverage.py`
- `source/isaaclab/isaaclab/utils/occupancy.py`

Runner config:

- `source/isaaclab_tasks/isaaclab_tasks/direct/airgym_x152b/agents/rsl_rl_ppo_cfg.py`

Registration:

- `source/isaaclab_tasks/isaaclab_tasks/direct/airgym_x152b/__init__.py`

Related design notes:

- `DEPTH_OCCUPANCY_GRID_IMPLEMENTATION.md`
- `EXPLORATION_GRID_HELPER_IMPLEMENTATION.md`

Pure tests:

- `source/isaaclab_tasks/test/test_airgym_exploration_helpers.py`
- `source/isaaclab/test/utils/test_occupancy.py`

## Environment Definition

Room:

- square room in env-local coordinates
- x limits: `[-5.0, 5.0]`
- y limits: `[-5.0, 5.0]`
- four fixed walls
- eight randomized pillars by default
- plane terrain

Flight envelope:

- nominal flight height: `1.2 m`
- terminate below `0.35 m`
- terminate above `2.0 m`
- terminate on leaving room bounds
- terminate on collision
- terminate if uprightness collapses enough that `ups[:, 2] < 0.0`

Spawn logic:

- spawn sampled inside room with `1.0 m` wall margin
- spawn z jitter: `+-0.15 m`
- random yaw in `[-pi, pi]`
- coverage grid resets at spawn and marks spawn cell visited

Obstacle sampling:

- pillar placement uses rejection sampling
- pillar wall margin: `1.0 m`
- pillar minimum spacing: `1.2 m`
- spawn clearance from pillars: `1.4 m`

## Observation Design

### Actor observations

Observation groups used by the actor:

- `observation`
- `image`

`observation` is the default vector state and currently includes:

- normalized local x
- normalized local y
- normalized z error relative to hover height
- local Euler angles
- local linear velocity
- local angular velocity
- current policy actions
- coverage ratio

Vector dimension:

- `17`

`image` is the normalized onboard depth image:

- shape: `1 x 120 x 212`
- depth is normalized by `camera_max_distance`
- this path goes through the CNN exactly like the planning task

### Critic observations

Observation groups used by the critic:

- `observation`
- `critic_grid`

`critic_grid` is built from the same metric depth frame used by the actor image and depth reward terms.

Grid definition:

- 2 channels: free, occupied
- x limits: `(0.0, 4.5)`
- y limits: `(-2.25, 2.25)`
- z limits: `(-0.5, 1.6)`
- cell size: `0.25 m`
- stride: `4`
- inflation radius: `1`
- final shape: `2 x 18 x 18`

Important design rule:

- actor and critic visual inputs are different, so CNN encoders must not be shared
- `share_cnn_encoders=False` is required for this task

## Coverage Grid

Coverage map:

- env-local x-y grid
- limits match room limits
- cell size: `0.5 m`
- grid size: `20 x 20`
- total cells: `400`

Behavior:

- starts as all zeros
- spawn cell is marked visited
- path cells are marked, not only the endpoint
- visited cells stay visited until env reset
- reward is given for newly visited cells only

Current implementation utility:

- `CoverageGrid2D`

## Reward Definition

The current reward is the sum of:

- `coverage_reward = 2.0 * new_visit_count`
- `alive_reward = 0.1`
- `upright_reward = 0.3 * ((ups_z + 1) / 2)^2`
- `height_reward = 0.3 * exp(-4 * (z - 1.2)^2)`
- `action_smoothness_reward = 0.1 * exp(-||a_t - a_{t-1}||)`
- `spin_reward = 0.1 / (1 + ||omega_local||^2)`
- `proximity_penalty = -2.0 * max(0, 0.45 - min_depth)`
- `collision_penalty = -50` on collision
- `bounds_penalty = -25` when out of room bounds
- `height_penalty = -20` when outside height band

Interpretation:

- exploration pressure comes primarily from `coverage_reward`
- stability and controllability come from upright, height, smoothness, and spin shaping
- near-obstacle pressure comes from the depth proximity term
- hard failures are collision, room exit, and large height violations

## Camera And Depth Rules

The onboard camera is configured with:

- `update_latest_camera_pose=True`
- `depth_clipping_behavior="max"`
- clipping range `(0.05, 4.5)`

Important implementation rules that should be preserved:

- metric depth and normalized depth image must come from the same per-step cached frame
- actor image, critic grid, and any depth-based reward terms must not see different noise realizations in the same step
- critic grid uses actual camera world pose plus the yaw-local world-to-local frame transform
- max-range rays should contribute free-space but must not create fake occupied endpoints

## Training Configuration

Runner class:

- `AirGymX152bExplorationPPORunnerCfg`

Key settings:

- logger: `wandb`
- project: `isaaclab-airgym-x152b`
- `max_iterations = 2500`
- actor obs groups: `{"actor": ["observation", "image"]}`
- critic obs groups: `{"critic": ["observation", "critic_grid"]}`
- actor model: CNN + MLP
- critic model: CNN + MLP
- `share_cnn_encoders = False`

## Commands

Training smoke test:

```bash
cd /home/ramzi/IsaacLab && ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-AirGym-X152b-Exploration-Direct-v0 --num_envs 8 --headless --enable_cameras --max_iterations 5
```

Helper tests:

```bash
cd /home/ramzi/IsaacLab && env_isaaclab/bin/python -m pytest source/isaaclab/test/utils/test_occupancy.py source/isaaclab_tasks/test/test_airgym_exploration_helpers.py -q
```

## Extension Guidance

When extending this environment, preserve these invariants unless there is a deliberate research reason to break them:

- keep actor depth image on the planning-style CNN path
- keep critic occupancy grid as a separate observation group
- keep asymmetric actor/critic encoders unshared
- keep coverage tracking env-local and batched
- keep camera pose updates enabled
- keep depth-derived free-space handling for max-range rays
- keep CPU-controller tasks as the blueprint for dynamics and control behavior

Safe research directions from this baseline:

- richer exploration rewards built on top of coverage ratio or frontier discovery
- larger or non-square rooms
- dynamic obstacles or procedurally generated layouts
- occupancy-grid temporal fusion for critic only
- exploration-specific curriculum schedules
- export and publication plots through W&B

## Future Session Checklist

If a future session changes this environment, verify at minimum:

- env still registers under `Isaac-AirGym-X152b-Exploration-Direct-v0`
- actor receives `observation + image`
- critic receives `observation + critic_grid`
- `share_cnn_encoders` is still `False`
- room walls and pillars still spawn correctly
- coverage grid resets correctly across multi-env training
- contact sensor still terminates collisions correctly
- critic grid still uses current camera pose and current depth frame
- smoke-test training still starts successfully with `--enable_cameras`

