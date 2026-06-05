# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
from itertools import chain

import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import RandomNetworkDistillation, resolve_rnd_config, resolve_symmetry_config
from rsl_rl.modules import EmpiricalNormalization, HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import (
    resolve_callable,
    resolve_nn_activation,
    resolve_obs_groups,
    resolve_optimizer,
    split_and_pad_trajectories,
    unpad_trajectories,
)


class ResidualBlock3D(nn.Module):
    """Two-layer 3D residual block used by the active-perception map encoder."""

    def __init__(self, channels: int, activation: str = "elu") -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.activation = resolve_nn_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.activation(self.conv1(x))
        x = self.conv2(x)
        return self.activation(x + residual)


class ActiveMap3DResNet(nn.Module):
    """3D ResNet map encoder for ``(N, 1, 21, 21, 21)`` local occupancy crops."""

    def __init__(
        self,
        input_channels: int = 1,
        stages: tuple[tuple[int, int], ...] | list[list[int]] = ((8, 2), (16, 2), (16, 2)),
        activation: str = "elu",
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = int(input_channels)
        for out_channels, num_blocks in stages:
            out_channels = int(out_channels)
            layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(resolve_nn_activation(activation))
            layers.append(nn.MaxPool3d(kernel_size=3, stride=2, padding=1))
            for _ in range(int(num_blocks)):
                layers.append(ResidualBlock3D(out_channels, activation))
            in_channels = out_channels
        self.net = nn.Sequential(*layers)
        self.output_dim = in_channels * 3 * 3 * 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x.flatten(start_dim=-4)


class ActiveMapSharedActorCritic(nn.Module):
    """Shared 3D map encoder, MLP encoder, GRU trunk, actor head, and critic head."""

    is_recurrent: bool = True

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        action_dim: int,
        *,
        vector_obs_group: str = "observations",
        map_obs_group: str = "observations_map",
        map_stages: tuple[tuple[int, int], ...] | list[list[int]] = ((8, 2), (16, 2), (16, 2)),
        encoder_hidden_dims: tuple[int, ...] | list[int] = (256, 128, 64),
        activation: str = "elu",
        rnn_hidden_dim: int = 64,
        rnn_num_layers: int = 1,
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        active_groups = obs_groups[obs_set]
        if vector_obs_group not in active_groups or map_obs_group not in active_groups:
            raise ValueError(
                f"ActiveMapSharedActorCritic requires '{vector_obs_group}' and '{map_obs_group}' in obs_groups"
            )
        if len(obs[vector_obs_group].shape) != 2:
            raise ValueError(f"Expected 2D vector observations, got {obs[vector_obs_group].shape}")
        if len(obs[map_obs_group].shape) != 5:
            raise ValueError(f"Expected 5D map observations, got {obs[map_obs_group].shape}")

        self.obs_groups = active_groups
        self.vector_obs_group = vector_obs_group
        self.map_obs_group = map_obs_group
        self.vector_obs_dim = int(obs[vector_obs_group].shape[-1])
        self.obs_map_shape = tuple(int(v) for v in obs[map_obs_group].shape[1:])
        self.map_encoder = ActiveMap3DResNet(
            input_channels=int(obs[map_obs_group].shape[1]),
            stages=map_stages,
            activation=activation,
        )
        self.obs_normalization = bool(obs_normalization)
        if self.obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.vector_obs_dim)
        else:
            self.obs_normalizer = nn.Identity()

        activation_mod = resolve_nn_activation(activation)
        dims = [self.vector_obs_dim + self.map_encoder.output_dim, *[int(v) for v in encoder_hidden_dims]]
        encoder_layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            encoder_layers.append(nn.Linear(in_dim, out_dim))
            encoder_layers.append(activation_mod)
        self.encoder = nn.Sequential(*encoder_layers)
        encoder_output_dim = dims[-1]
        self.rnn = nn.GRU(encoder_output_dim, int(rnn_hidden_dim), int(rnn_num_layers))

        if distribution_cfg is not None:
            distribution_cfg = dict(distribution_cfg)
            dist_class: type[Distribution] = resolve_callable(distribution_cfg.pop("class_name"))  # type: ignore
            self.distribution: Distribution | None = dist_class(action_dim, **distribution_cfg)
            actor_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            actor_output_dim = action_dim
        self.actor_head = nn.Linear(int(rnn_hidden_dim), actor_output_dim)
        self.critic_head = nn.Linear(int(rnn_hidden_dim), 1)
        self.hidden_state: torch.Tensor | None = None

    def encode(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        vector_obs = self.obs_normalizer(obs[self.vector_obs_group])
        map_obs = obs[self.map_obs_group].to(dtype=vector_obs.dtype)
        original_shape = vector_obs.shape[:-1]
        flat_vector = vector_obs.reshape(-1, vector_obs.shape[-1])
        flat_map = map_obs.reshape(-1, *map_obs.shape[-4:])
        map_latent = self.map_encoder(flat_map)
        encoded = self.encoder(torch.cat((flat_vector, map_latent), dim=-1)).view(*original_shape, -1)

        if masks is not None:
            if hidden_state is None:
                raise ValueError("Hidden states not passed to active-map GRU during policy update")
            out, _ = self.rnn(encoded, hidden_state)
            out = unpad_trajectories(out, masks)
        elif hidden_state is not None:
            out, _ = self.rnn(encoded.unsqueeze(0), hidden_state)
            out = out.squeeze(0)
        else:
            out, self.hidden_state = self.rnn(encoded.unsqueeze(0), self.hidden_state)
            out = out.squeeze(0)
        return out

    def act_and_evaluate(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(obs, masks, hidden_state)
        actor_output = self.actor_head(latent)
        value = self.critic_head(latent)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(actor_output)
                return self.distribution.sample(), value
            return self.distribution.deterministic_output(actor_output), value
        return actor_output, value

    def act(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        actions, _ = self.act_and_evaluate(obs, masks, hidden_state, stochastic_output)
        return actions

    def evaluate(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        latent = self.encode(obs, masks, hidden_state)
        return self.critic_head(latent)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        return self.act(obs, masks, hidden_state, stochastic_output)

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        if dones is None:
            self.hidden_state = hidden_state
        elif self.hidden_state is not None:
            self.hidden_state[..., dones == 1, :] = 0.0

    def get_hidden_state(self) -> HiddenState:
        if self.hidden_state is None:
            return None
        return self.hidden_state.detach()

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        if self.hidden_state is None:
            return
        if dones is None:
            self.hidden_state = self.hidden_state.detach()
        else:
            self.hidden_state[..., dones == 1, :] = self.hidden_state[..., dones == 1, :].detach()

    @property
    def output_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        return self.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        return self.distribution.kl_divergence(old_params, new_params)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            self.obs_normalizer.update(obs[self.vector_obs_group])  # type: ignore

    def as_jit(self) -> nn.Module:
        return _TorchActiveMapGRUModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        return _OnnxActiveMapGRUModel(self, verbose)


class _ActiveMapExportMixin:
    def _encode_map_vector(self, observations: torch.Tensor, observations_map: torch.Tensor) -> torch.Tensor:
        vector_obs = self.obs_normalizer(observations)
        map_obs = observations_map.to(dtype=vector_obs.dtype)
        map_latent = self.map_encoder(map_obs)
        return self.encoder(torch.cat((vector_obs, map_latent), dim=-1))


class _TorchActiveMapGRUModel(nn.Module, _ActiveMapExportMixin):
    """TorchScript export wrapper for active-map GRU policy inference."""

    def __init__(self, model: ActiveMapSharedActorCritic) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.map_encoder = copy.deepcopy(model.map_encoder)
        self.encoder = copy.deepcopy(model.encoder)
        self.rnn = copy.deepcopy(model.rnn)
        self.actor_head = copy.deepcopy(model.actor_head)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))

    def forward(self, observations: torch.Tensor, observations_map: torch.Tensor) -> torch.Tensor:
        encoded = self._encode_map_vector(observations, observations_map)
        out, hidden_state = self.rnn(encoded.unsqueeze(0), self.hidden_state)
        self.hidden_state[:] = hidden_state
        actor_output = self.actor_head(out.squeeze(0))
        return self.deterministic_output(actor_output)

    @torch.jit.export
    def reset(self) -> None:
        self.hidden_state[:] = 0.0


class _OnnxActiveMapGRUModel(nn.Module, _ActiveMapExportMixin):
    """ONNX export wrapper with explicit GRU hidden-state input/output."""

    def __init__(self, model: ActiveMapSharedActorCritic, verbose: bool = False) -> None:
        super().__init__()
        self.verbose = verbose
        self.vector_obs_dim = model.vector_obs_dim
        self.map_shape = tuple(model.obs_map_shape)
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.map_encoder = copy.deepcopy(model.map_encoder)
        self.encoder = copy.deepcopy(model.encoder)
        self.rnn = copy.deepcopy(model.rnn)
        self.actor_head = copy.deepcopy(model.actor_head)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(
        self, observations: torch.Tensor, observations_map: torch.Tensor, h_in: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self._encode_map_vector(observations, observations_map)
        out, h_out = self.rnn(encoded.unsqueeze(0), h_in)
        actor_output = self.actor_head(out.squeeze(0))
        return self.deterministic_output(actor_output), h_out

    def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        observations = torch.zeros(1, self.vector_obs_dim)
        observations_map = torch.zeros(1, *self.map_shape)
        h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
        return observations, observations_map, h_in

    @property
    def input_names(self) -> list[str]:
        return ["observations", "observations_map", "h_in"]

    @property
    def output_names(self) -> list[str]:
        return ["actions", "h_out"]


class ActiveMapPPO(PPO):
    """PPO variant with one shared recurrent active-map actor-critic trunk."""

    def __init__(
        self,
        actor_critic: ActiveMapSharedActorCritic,
        storage: RolloutStorage,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        self.gpu_global_rank = multi_gpu_cfg["global_rank"] if multi_gpu_cfg is not None else 0
        self.gpu_world_size = multi_gpu_cfg["world_size"] if multi_gpu_cfg is not None else 1

        if rnd_cfg:
            rnd_lr = rnd_cfg.pop("learning_rate", 1e-3)
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            self.rnd_optimizer = optim.Adam(self.rnd.predictor.parameters(), lr=rnd_lr)
        else:
            self.rnd = None
            self.rnd_optimizer = None
        if symmetry_cfg is not None:
            use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            if use_symmetry:
                raise ValueError("Symmetry augmentation is not supported for active-map recurrent policies.")
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        self.actor_critic = actor_critic.to(self.device)
        self.actor = self.actor_critic
        self.critic = self.actor_critic
        self.optimizer = resolve_optimizer(optimizer)(self.actor_critic.parameters(), lr=learning_rate)  # type: ignore
        self.storage = storage
        self.transition = RolloutStorage.Transition()

        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

    def act(self, obs: TensorDict) -> torch.Tensor:
        shared_hidden = self.actor_critic.get_hidden_state()
        self.transition.hidden_states = (shared_hidden, None)
        actions, values = self.actor_critic.act_and_evaluate(obs, stochastic_output=True)
        self.transition.actions = actions.detach()
        self.transition.values = values.detach()
        self.transition.actions_log_prob = self.actor_critic.get_output_log_prob(self.transition.actions).detach()
        self.transition.distribution_params = tuple(p.detach() for p in self.actor_critic.output_distribution_params)
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        self.actor_critic.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if self.rnd:
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            self.transition.rewards += self.intrinsic_rewards
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1
            )
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        st = self.storage
        last_values = self.actor_critic.evaluate(obs, hidden_state=self.actor_critic.get_hidden_state()).detach()
        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            next_is_not_terminal = 1.0 - st.dones[step].float()
            delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            st.returns[step] = advantage + st.values[step]
        st.advantages = st.returns - st.values
        if not self.normalize_advantage_per_mini_batch:
            st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)

    def update(self) -> dict[str, float]:
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        num_mini_batches = min(self.num_mini_batches, self.storage.num_envs)
        num_updates = 0
        generator = self.storage.recurrent_mini_batch_generator(num_mini_batches, self.num_learning_epochs)
        for batch in generator:
            num_updates += 1
            original_batch_size = batch.observations.batch_size[0]
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

            _, values = self.actor_critic.act_and_evaluate(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor_critic.get_output_log_prob(batch.actions)
            distribution_params = tuple(p[:original_batch_size] for p in self.actor_critic.output_distribution_params)
            entropy = self.actor_critic.output_entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor_critic.get_kl_divergence(batch.old_distribution_params, distribution_params)
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()

        self.storage.clear()
        return {
            "value": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
        }

    def train_mode(self) -> None:
        self.actor_critic.train()
        if self.rnd:
            self.rnd.train()

    def eval_mode(self) -> None:
        self.actor_critic.eval()
        if self.rnd:
            self.rnd.eval()

    def save(self) -> dict:
        saved_dict = {
            "actor_state_dict": self.actor_critic.state_dict(),
            "critic_state_dict": self.actor_critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.rnd:
            saved_dict["rnd_state_dict"] = self.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.rnd_optimizer.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        if load_cfg is None:
            load_cfg = {"actor": True, "critic": True, "optimizer": True, "iteration": True, "rnd": True}
        if load_cfg.get("actor"):
            self.actor_critic.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if load_cfg.get("optimizer"):
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        if load_cfg.get("rnd") and self.rnd:
            self.rnd.load_state_dict(loaded_dict["rnd_state_dict"], strict=strict)
            self.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        return load_cfg.get("iteration", False)

    def get_policy(self) -> ActiveMapSharedActorCritic:
        return self.actor_critic

    def broadcast_parameters(self) -> None:
        model_params = [self.actor_critic.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.actor_critic.load_state_dict(model_params[0])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[1])

    def reduce_parameters(self) -> None:
        all_params = self.actor_critic.parameters()
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())
        all_params = list(all_params)
        grads = [param.grad.view(-1) for param in all_params if param.grad is not None]
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "ActiveMapPPO":
        alg_class: type[ActiveMapPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        model_class: type[ActiveMapSharedActorCritic] = resolve_callable(
            cfg["algorithm"].pop("model_class_name", ActiveMapSharedActorCritic)
        )  # type: ignore
        model_cfg = cfg["algorithm"].pop("model_cfg", {})
        cfg["algorithm"].pop("share_cnn_encoders", None)
        default_sets = ["actor", "critic"]
        if "rnd_cfg" in cfg["algorithm"] and cfg["algorithm"]["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        actor_critic = model_class(obs, cfg["obs_groups"], "actor", env.num_actions, **model_cfg).to(device)
        print(f"Active-map Actor-Critic Model: {actor_critic}")
        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)
        return alg_class(actor_critic, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"])


__all__ = [
    "ActiveMap3DResNet",
    "ActiveMapPPO",
    "ActiveMapSharedActorCritic",
    "ResidualBlock3D",
]
