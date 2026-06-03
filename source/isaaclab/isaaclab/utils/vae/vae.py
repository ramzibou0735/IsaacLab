# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# This depth VAE architecture is vendored from the active-perception-RL-navigation
# project (ResNet8/DroNet-style encoder) so the pretrained weights load unchanged.
# The network definition is kept byte-compatible with the original ``state_dict``
# layout; only console prints were removed.

"""Variational autoencoder for depth-image reconstruction (encoder used for RL)."""

from __future__ import annotations

import torch
import torch.nn as nn


class ImgDecoder(nn.Module):
    """Decoder half of the depth VAE (unused for RL, kept for weight parity)."""

    def __init__(self, input_dim=1, latent_dim=64, with_logits=False):
        super().__init__()
        self.with_logits = with_logits
        self.n_channels = input_dim
        self.dense = nn.Linear(latent_dim, 512)
        self.dense1 = nn.Linear(512, 9 * 15 * 128)
        self.deconv1 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(
            128, 64, kernel_size=5, stride=2, padding=(2, 2), output_padding=(0, 1), dilation=1
        )
        self.deconv4 = nn.ConvTranspose2d(
            64, 32, kernel_size=6, stride=4, padding=(2, 2), output_padding=(0, 0), dilation=1
        )
        self.deconv6 = nn.ConvTranspose2d(32, 16, kernel_size=6, stride=2, padding=(0, 0), output_padding=(0, 1))
        self.deconv7 = nn.ConvTranspose2d(16, self.n_channels, kernel_size=4, stride=2, padding=2)

    def forward(self, z):
        return self.decode(z)

    def decode(self, z):
        x = self.dense(z)
        x = torch.relu(x)
        x = self.dense1(x)
        x = x.view(x.size(0), 128, 9, 15)
        x = torch.relu(self.deconv1(x))
        x = torch.relu(self.deconv2(x))
        x = torch.relu(self.deconv4(x))
        x = torch.relu(self.deconv6(x))
        x = self.deconv7(x)
        if self.with_logits:
            return x
        return torch.sigmoid(x)


class ImgEncoder(nn.Module):
    """ResNet8-style encoder producing ``2 * latent_dim`` outputs (mean, logvar)."""

    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.define_encoder()
        self.elu = nn.ELU()

    def define_encoder(self):
        self.conv0 = nn.Conv2d(self.input_dim, 32, kernel_size=5, stride=2, padding=2)
        self.conv0_1 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=2)
        self.conv1_0 = nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=1)
        self.conv1_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_0 = nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3_0 = nn.Conv2d(128, 128, kernel_size=5, stride=2)
        self.conv0_jump_2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv1_jump_3 = nn.Conv2d(64, 128, kernel_size=5, stride=4, padding=(2, 1))
        self.dense0 = nn.Linear(3 * 6 * 128, 512)
        self.dense1 = nn.Linear(512, 2 * self.latent_dim)

    def forward(self, img):
        return self.encode(img)

    def encode(self, img):
        x0_0 = self.conv0(img)
        x0_1 = self.conv0_1(x0_0)
        x0_1 = self.elu(x0_1)

        x1_0 = self.conv1_0(x0_1)
        x1_1 = self.conv1_1(x1_0)
        x0_jump_2 = self.conv0_jump_2(x0_1)
        x1_1 = x1_1 + x0_jump_2
        x1_1 = self.elu(x1_1)

        x2_0 = self.conv2_0(x1_1)
        x2_1 = self.conv2_1(x2_0)
        x1_jump3 = self.conv1_jump_3(x1_1)
        x2_1 = x2_1 + x1_jump3
        x2_1 = self.elu(x2_1)

        x3_0 = self.conv3_0(x2_1)
        x = x3_0.reshape(x3_0.size(0), -1)
        x = self.dense0(x)
        x = self.elu(x)
        x = self.dense1(x)
        return x


class Lambda(nn.Module):
    """Wrap a function as a module (used to slice mean/logvar parameters)."""

    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class VAE(nn.Module):
    """Variational autoencoder for reconstruction of depth images."""

    def __init__(self, input_dim=1, latent_dim=64, with_logits=False, inference_mode=False):
        super().__init__()
        self.with_logits = with_logits
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.inference_mode = inference_mode
        self.encoder = ImgEncoder(input_dim=self.input_dim, latent_dim=self.latent_dim)
        self.img_decoder = ImgDecoder(input_dim=1, latent_dim=self.latent_dim, with_logits=self.with_logits)
        self.mean_params = Lambda(lambda x: x[:, : self.latent_dim])
        self.logvar_params = Lambda(lambda x: x[:, self.latent_dim :])

    def forward(self, img):
        z = self.encoder(img)
        mean = self.mean_params(z)
        logvar = self.logvar_params(z)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        if self.inference_mode:
            eps = torch.zeros_like(eps)
        z_sampled = mean + eps * std
        img_recon = self.img_decoder(z_sampled)
        return img_recon, mean, logvar, z_sampled

    def encode(self, img):
        z = self.encoder(img)
        means = self.mean_params(z)
        logvars = self.logvar_params(z)
        std = torch.exp(0.5 * logvars)
        eps = torch.randn_like(logvars)
        if self.inference_mode:
            eps = torch.zeros_like(eps)
        z_sampled = means + eps * std
        return z_sampled, means, std

    def decode(self, z):
        img_recon = self.img_decoder(z)
        if self.with_logits:
            return torch.sigmoid(img_recon)
        return img_recon

    def set_inference_mode(self, mode):
        self.inference_mode = mode
