"""Course 01: three tugs already near their slots, one tug joins from a short route."""

from __future__ import annotations

import math


COURSE = {
    "name": "c01_three_ready",
    "description": "Three ready tugs; one low-noise, short-distance joining tug.",
    "total_steps": 600_000,
    "env_overrides": {
        "tug_init_mixed_ready_counts": (3,),
        "tug_init_force_single_opposite": False,
        "tug_init_mixed_zones": ("outer_slot", "side_lane"),
        "hold_time_s": 1.0,
        "tug_init_mixed_pair_min_dist_m": 100.0,
        "tug_init_mixed_opposite_stern_dist_min_m": 120.0,
        "tug_init_mixed_opposite_stern_dist_max_m": 220.0,
        "tug_init_mixed_opposite_lateral_extra_m": 20.0,
        "tug_init_mixed_approach_speed_min_ms": 0.15,
        "tug_init_mixed_approach_speed_max_ms": 0.45,
        "tug_init_heading_noise_rad": math.radians(2.0),
        "tug_init_yaw_rate_noise_rads": 0.002,
    },
}
