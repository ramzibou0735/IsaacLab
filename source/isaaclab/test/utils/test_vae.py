# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke tests for the frozen depth-VAE encoder."""

import pytest
import torch

from isaaclab.utils.vae import DepthVAEEncoder

DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_encode_shape_and_layouts(device):
    """Latent is (N, latent_dims) and accepts (N,H,W), (N,1,H,W), (N,H,W,1)."""
    enc = DepthVAEEncoder(device=device, latent_dims=64)
    depth = torch.rand(3, 135, 240, device=device) * enc.max_range
    assert enc.encode(depth).shape == (3, enc.latent_dims)
    assert enc.encode(depth.unsqueeze(1)).shape == (3, enc.latent_dims)
    assert enc.encode(depth.unsqueeze(-1)).shape == (3, enc.latent_dims)


@pytest.mark.parametrize("device", DEVICES)
def test_mean_mode_is_deterministic_and_frozen(device):
    """Mean-mode encodes are deterministic and the model carries no gradients."""
    enc = DepthVAEEncoder(device=device, return_sampled_latent=False)
    depth = torch.rand(4, 135, 240, device=device) * enc.max_range
    z0 = enc.encode(depth)
    z1 = enc.encode(depth)
    torch.testing.assert_close(z0, z1)
    assert not any(p.requires_grad for p in enc.model.parameters())
    assert z0.requires_grad is False


@pytest.mark.parametrize("device", DEVICES)
def test_normalization_is_finite_on_extremes(device):
    """Out-of-range / non-finite depth is mapped to finite latents."""
    enc = DepthVAEEncoder(device=device)
    far = torch.full((2, 135, 240), 1.0e3, device=device)  # beyond max range -> far fill
    near = torch.zeros((2, 135, 240), device=device)  # below min range -> near fill
    nan = torch.full((2, 135, 240), float("nan"), device=device)
    for depth in (far, near, nan):
        z = enc.encode(depth)
        assert torch.isfinite(z).all()


@pytest.mark.parametrize("device", DEVICES)
def test_sampled_mode_varies(device):
    """Sampled-latent mode injects reparameterization noise (run-to-run variation)."""
    enc = DepthVAEEncoder(device=device, return_sampled_latent=True)
    depth = torch.rand(4, 135, 240, device=device) * enc.max_range
    assert not torch.allclose(enc.encode(depth), enc.encode(depth))
