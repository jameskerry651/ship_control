"""Course 03: one tug ready, three tugs start from broader route distributions."""

from __future__ import annotations

import math


COURSE = {
    "name": "c03_one_ready",
    "description": "One ready tug; three joining tugs using near-final full-distance ranges.",
    "total_steps": 1_000_000,
    "env_overrides": {
        "tug_init_mixed_ready_counts": (1,),
        "tug_init_mixed_zones": ("stern_gate", "side_lane", "outer_slot"),
        "hold_time_s": 2.0,
        "tug_init_mixed_pair_min_dist_m": 120.0,
        "tug_init_rear_stern_slot_dist_min_m": 80.0,
        "tug_init_rear_stern_slot_dist_max_m": 220.0,
        "tug_init_rear_bow_slot_dist_min_m": 240.0,
        "tug_init_rear_bow_slot_dist_max_m": 360.0,
        "tug_init_mixed_opposite_stern_dist_min_m": 220.0,
        "tug_init_mixed_opposite_stern_dist_max_m": 380.0,
        "tug_init_mixed_approach_speed_min_ms": 0.20,
        "tug_init_mixed_approach_speed_max_ms": 0.75,
        "tug_init_heading_noise_rad": math.radians(4.0),
        "tug_init_yaw_rate_noise_rads": 0.004,
    },
}
