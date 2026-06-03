# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Frozen depth-VAE encoder used as a visual observation backbone for RL.

This wraps the vendored :class:`~isaaclab.utils.vae.vae.VAE` (a DroNet/ResNet8
depth autoencoder) and reproduces the exact preprocessing used when its weights
were trained: metric depth is range-clipped, out-of-range pixels are mapped to
``+max_range`` (far) / ``-max_range`` (too near), normalized by ``max_range`` to
``[-1] ∪ [0, 1]``, then resized to the network's native input resolution before a
frozen, ``no_grad`` encode. Only the latent (mean or sampled) is returned.
"""

from __future__ import annotations

import os
import torch

from .vae import VAE

# Default weights shipped alongside this module (DroNet depth VAE, latent dim 64).
_DEFAULT_WEIGHTS = os.path.join(os.path.dirname(__file__), "weights", "trained_vae.pth")


def clean_state_dict(state_dict: dict) -> dict:
    """Strip ``module.`` (DataParallel) and remap the ``dronet.`` encoder prefix.

    The training code stored the encoder under ``dronet.*`` while the vendored
    :class:`VAE` registers it as ``encoder.*``; remap so the checkpoint loads
    cleanly.
    """
    clean: dict = {}
    for key, value in state_dict.items():
        if "module." in key:
            key = key.replace("module.", "")
        if "dronet." in key:
            key = key.replace("dronet.", "encoder.")
        clean[key] = value
    return clean


class DepthVAEEncoder:
    """Frozen depth VAE encoder producing a compact latent per depth image.

    Args:
        weights_path: Path to the ``state_dict`` checkpoint. Defaults to the
            bundled ``trained_vae.pth``.
        latent_dims: Size of the latent vector produced per image (must match the
            checkpoint). Defaults to 64.
        image_res: ``(height, width)`` the network was trained on; inputs are
            resized to this with nearest-neighbour interpolation. Defaults to
            ``(270, 480)``.
        max_range: Depth normalization range in meters (matches camera far clip).
            Defaults to 10.0.
        min_range: Depths below this are treated as "too near" and mapped to
            ``-max_range`` before normalization. Defaults to 0.2.
        return_sampled_latent: If True, return the reparameterized sample;
            otherwise return the latent mean (deterministic). Defaults to False.
        device: Torch device to place the model on. Defaults to ``cuda`` if
            available else ``cpu``.
    """

    def __init__(
        self,
        weights_path: str | None = None,
        *,
        latent_dims: int = 64,
        image_res: tuple[int, int] = (270, 480),
        max_range: float = 10.0,
        min_range: float = 0.2,
        return_sampled_latent: bool = False,
        device: str | torch.device | None = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.latent_dims = int(latent_dims)
        self.image_res = (int(image_res[0]), int(image_res[1]))
        self.max_range = float(max_range)
        self.min_range = float(min_range)
        self.return_sampled_latent = bool(return_sampled_latent)
        self._far_value = self.max_range
        self._near_value = -self.max_range

        weights_path = weights_path or _DEFAULT_WEIGHTS
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"DepthVAEEncoder weights not found: {weights_path}")

        # Deterministic mean when not sampling; reparameterization noise when sampling.
        self.model = VAE(input_dim=1, latent_dim=self.latent_dims, inference_mode=not self.return_sampled_latent)
        state_dict = clean_state_dict(torch.load(weights_path, map_location="cpu"))
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        # The decoder may be partially unused; only the encoder path matters for RL.
        encoder_missing = [k for k in missing if k.startswith("encoder.")]
        if encoder_missing:
            raise RuntimeError(f"DepthVAEEncoder: missing encoder weights {encoder_missing}")
        self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _to_nchw(self, depth: torch.Tensor) -> torch.Tensor:
        """Coerce a depth batch of various layouts to ``(N, 1, H, W)``."""
        if depth.dim() == 4:
            # (N, C, H, W) or (N, H, W, C) -> single channel.
            if depth.shape[1] == 1:
                return depth
            if depth.shape[-1] == 1:
                return depth.permute(0, 3, 1, 2).contiguous()
            raise ValueError(f"Unsupported 4D depth shape {tuple(depth.shape)}")
        if depth.dim() == 3:
            return depth.unsqueeze(1)
        raise ValueError(f"Unsupported depth shape {tuple(depth.shape)}; expected 3D or 4D")

    def _normalize(self, depth: torch.Tensor) -> torch.Tensor:
        """Range-clip + normalize metric depth to the VAE's training domain."""
        x = depth.to(self.device, dtype=torch.float32).clone()
        x = torch.nan_to_num(x, nan=self._far_value, posinf=self._far_value, neginf=self._near_value)
        x[x > self.max_range] = self._far_value
        x[x < self.min_range] = self._near_value
        x = x / self.max_range
        return x

    @torch.no_grad()
    def encode(self, depth_metric: torch.Tensor) -> torch.Tensor:
        """Encode a batch of metric depth images into latent vectors.

        Args:
            depth_metric: Depth in meters with shape ``(N, H, W)``, ``(N, 1, H, W)``
                or ``(N, H, W, 1)``.

        Returns:
            Latent tensor of shape ``(N, latent_dims)`` on this encoder's device.
        """
        x = self._normalize(self._to_nchw(depth_metric))
        if (x.shape[-2], x.shape[-1]) != self.image_res:
            x = torch.nn.functional.interpolate(x, size=self.image_res, mode="nearest")
        z_sampled, means, _ = self.model.encode(x)
        return z_sampled if self.return_sampled_latent else means

    __call__ = encode
