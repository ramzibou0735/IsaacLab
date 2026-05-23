# AirGym to IsaacLab Porting Agent Instructions

## Mission
Port the single-agent reinforcement learning environments in `airgym/envs` from Isaac Gym Preview 4 to IsaacLab while preserving compatibility with `rlPx4Controller`.

## Scope
- In scope:
  - `hovering`
  - `customized`
  - `balloon`
  - `tracking`
  - `avoid`
  - `planning`
  - `depthgen` if needed as a utility environment
- Out of scope:
  - Multi-agent reinforcement learning
  - `maplanning`
  - `DirectMARLEnv`
  - Any redesign around multi-robot training

Ignore `airgym/envs/task/maplanning.py` and `airgym/envs/task/maplanning_config.py` completely unless they are needed only as reference for shared assets or controller usage patterns.

## Required Architectural Decisions
- Use IsaacLab direct workflow, not manager-based workflow.
- Implement the port on top of `DirectRLEnv`.
- Keep `rlPx4Controller` as the low-level control backend.
- Do not replace `rlPx4Controller` with a learned motor model, PD approximation, or a simplified base-wrench controller unless explicitly requested.

## Source of Truth
- Environment registration: `airgym/envs/__init__.py`
- Single-agent control loop:
  - `airgym/envs/base/hovering.py`
  - `airgym/envs/base/customized.py`
- Task-specific logic:
  - `airgym/envs/task/balloon.py`
  - `airgym/envs/task/tracking.py`
  - `airgym/envs/task/avoid.py`
  - `airgym/envs/task/planning.py`
- Asset loading and onboard camera behavior:
  - `airgym/assets/asset_manager.py`
  - `airgym/assets/__init__.py`
  - `airgym/assets/robots/X152b/model.urdf`

## Non-Negotiable Constraints
- Preserve control modes: `pos`, `vel`, `atti`, `rate`, `prop`.
- Preserve the use of batched `rlPx4Controller` classes:
  - `ParallelPosControl`
  - `ParallelVelControl`
  - `ParallelAttiControl`
  - `ParallelRateControl`
- Preserve per-rotor force and reaction torque application. The current simulator applies forces to `prop_1` to `prop_4`, not only to the base link.
- Preserve camera-driven observations for `avoid` and `planning`.
- Preserve reset logic, reward structure, and episode termination behavior as closely as possible before making any improvements.

## Implementation Strategy
1. Create a reusable IsaacLab base environment for the X152b quadrotor.
2. Port `hovering` first as the reference implementation.
3. Move shared controller integration into the base environment.
4. Port task-specific reset, observation, reward, and termination logic one task at a time.
5. Only after parity is reached, consider refactors for cleanliness.

## IsaacLab Mapping Rules
- Use `DirectRLEnv` with explicit `_setup_scene()`, `_pre_physics_step()`, `_apply_action()`, `_get_observations()`, `_get_rewards()`, `_get_dones()`, and `_reset_idx()`.
- Keep `sim.dt = 0.01` initially.
- Start with `decimation = 1` so the `rlPx4Controller` update frequency matches the current AirGym control loop.
- If vision sensors are used, configure IsaacLab camera rendering so reset frames are refreshed after reset.
- Use IsaacLab scene cloning instead of the old Isaac Gym `create_env` loop.

## Controller Integration Rules
- Treat the policy action as a high-level command and `rlPx4Controller` output as the motor-space command source.
- Convert IsaacLab tensors to the controller input format only where required.
- Minimize CPU-GPU copies, but do not break controller correctness to avoid them.
- The current controller backend is CPU-based. Preserve that assumption unless the controller itself is replaced.

## Asset and Physics Rules
- Port the X152b URDF carefully.
- Verify that `prop_1`, `prop_2`, `prop_3`, and `prop_4` remain available as distinct rigid bodies after import.
- If IsaacLab URDF conversion collapses fixed joints, adjust the import settings so rotor bodies remain addressable.
- Recreate environment assets through IsaacLab scene configuration instead of `AssetManager`, but preserve the same asset sets, counts, and placement logic.

## Task-Specific Notes
- `hovering`: state-only reference task, lowest-risk starting point.
- `balloon`: one target object, relative-pose reward.
- `tracking`: future trajectory observation must remain intact.
- `avoid`: moving obstacle plus depth image observation.
- `planning`: goal object plus many obstacles plus depth-derived clearance proxy.
- `customized`: use as a shared base behavior reference, not as the first public task to finalize.

## Validation Checklist
- Confirm action ranges match the original environment for each control mode.
- Confirm the reset distribution is numerically similar to the original task.
- Confirm reward components and termination conditions are preserved.
- Confirm onboard depth images have the same shape and normalization semantics.
- Confirm collisions are still detectable and used by `avoid` and `planning`.
- Confirm `hovering` works before porting any camera-based task.

## Explicit Prohibitions
- Do not introduce multi-agent abstractions.
- Do not port `maplanning`.
- Do not build around `DirectMARLEnv`.
- Do not rewrite the task around IsaacLab managers unless explicitly requested.
- Do not silently simplify rotor-level actuation into a single-body wrench model.

## Deliverable Preference
Produce a clean IsaacLab task package with a shared single-agent quadrotor base, task-specific configs, and a thin compatibility layer only if training code requires it.
