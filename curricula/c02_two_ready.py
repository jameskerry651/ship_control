"""Course 02: two tugs ready, two tugs join from moderate route starts."""

from __future__ import annotations

import math


COURSE = {
    "name": "c02_two_ready",
    "description": "Two ready tugs; two joining tugs with moderate distances and noise.",
    "total_steps": 800_000,
    "env_overrides": {
        "tug_init_mixed_ready_counts": (2,),
        "tug_init_mixed_zones": ("outer_slot", "side_lane", "stern_gate"),
        "hold_time_s": 1.5,
        "tug_init_mixed_pair_min_dist_m": 110.0,
        "tug_init_rear_stern_slot_dist_min_m": 80.0,
        "tug_init_rear_stern_slot_dist_max_m": 180.0,
        "tug_init_rear_bow_slot_dist_min_m": 220.0,
        "tug_init_rear_bow_slot_dist_max_m": 320.0,
        "tug_init_mixed_opposite_stern_dist_min_m": 180.0,
        "tug_init_mixed_opposite_stern_dist_max_m": 320.0,
        "tug_init_mixed_approach_speed_min_ms": 0.20,
        "tug_init_mixed_approach_speed_max_ms": 0.60,
        "tug_init_heading_noise_rad": math.radians(3.0),
        "tug_init_yaw_rate_noise_rads": 0.003,
    },
}
