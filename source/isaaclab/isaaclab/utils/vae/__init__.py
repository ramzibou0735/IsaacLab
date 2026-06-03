# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Vendored depth VAE and a frozen encoder wrapper for visual RL observations."""

from .encoder import DepthVAEEncoder, clean_state_dict
from .vae import VAE, ImgDecoder, ImgEncoder

__all__ = ["DepthVAEEncoder", "VAE", "ImgEncoder", "ImgDecoder", "clean_state_dict"]
