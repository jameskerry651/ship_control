"""Strict c04 stage: zero-ready starts, 100 m position tolerance."""

from __future__ import annotations


COURSE = {
    "name": "c04_pos100m",
    "description": "Zero-ready c04 strict-position ladder stage with pos_tol=100 m.",
    "total_steps": 800_000,
    "env_overrides": {
        "tug_init_mixed_ready_counts": (0,),
        "tug_init_mixed_zones": ("stern_gate", "side_lane", "outer_slot"),
        "hold_time_s": 2.0,
        "pos_tol_m": 100.0,
        "speed_tol_ms": 3.0,
        "reward_precision_w": 0.35,
        "reward_precision_scale_m": 40.0,
        "reward_near_hold_w": 0.50,
        "reward_near_hold_scale_m": 80.0,
    },
}
