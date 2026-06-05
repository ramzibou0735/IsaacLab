# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab_tasks.direct.airgym_x152b.active_perception_sensor import ActivePerceptionWarpSensor


class ThesisActivePerceptionWarpSensor(ActivePerceptionWarpSensor):
    """Thesis-local Warp perception sensor extension point."""

    pass


__all__ = ["ThesisActivePerceptionWarpSensor"]
