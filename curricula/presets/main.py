"""Main c01–c04 ready-count curriculum stages."""

from __future__ import annotations

import math

from curricula.presets.helpers import make_course, merge_overrides
from curricula.presets.shared import (
    C04_ZERO_READY_INIT,
    FINAL_SLOT_GUIDANCE_OVERRIDES,
    SAFE_JOIN_REWARD_OVERRIDES,
)

C01_THREE_READY = make_course(
    name="c01_three_ready",
    description="Three ready tugs; one low-noise, short-distance joining tug.",
    total_steps=600_000,
    env_overrides={
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
)

C02_TWO_READY = make_course(
    name="c02_two_ready",
    description="Two ready tugs; two joining tugs from side/outer starts with safety-biased routing.",
    total_steps=800_000,
    env_overrides=merge_overrides(SAFE_JOIN_REWARD_OVERRIDES, FINAL_SLOT_GUIDANCE_OVERRIDES, {
        "tug_init_mixed_ready_counts": (2,),
        "tug_init_mixed_zones": ("outer_slot", "side_lane"),
        "hold_time_s": 1.0,
        "tug_init_mixed_pair_min_dist_m": 135.0,
        "tug_init_rear_stern_slot_dist_min_m": 100.0,
        "tug_init_rear_stern_slot_dist_max_m": 200.0,
        "tug_init_rear_bow_slot_dist_min_m": 240.0,
        "tug_init_rear_bow_slot_dist_max_m": 340.0,
        "tug_init_mixed_opposite_stern_dist_min_m": 180.0,
        "tug_init_mixed_opposite_stern_dist_max_m": 320.0,
        "tug_init_mixed_approach_speed_min_ms": 0.15,
        "tug_init_mixed_approach_speed_max_ms": 0.45,
        "tug_init_heading_noise_rad": math.radians(2.0),
        "tug_init_yaw_rate_noise_rads": 0.002,
        "route_chase_speed_max_ms": 0.45,
        "route_nonfinal_forward_action_cap": 0.28,
    }),
)

C03_ONE_READY = make_course(
    name="c03_one_ready",
    description="Bridge stage with one/two ready tugs and side/outer safe-route exposure.",
    total_steps=1_200_000,
    env_overrides=merge_overrides(SAFE_JOIN_REWARD_OVERRIDES, FINAL_SLOT_GUIDANCE_OVERRIDES, {
        "tug_init_mixed_ready_counts": (2, 2, 2, 1),
        "tug_init_mixed_zones": (
            "outer_slot",
            "outer_slot",
            "outer_slot",
        ),
        "hold_time_s": 1.0,
        "tug_init_mixed_pair_min_dist_m": 180.0,
        "tug_init_rear_stern_slot_dist_min_m": 120.0,
        "tug_init_rear_stern_slot_dist_max_m": 260.0,
        "tug_init_rear_bow_slot_dist_min_m": 280.0,
        "tug_init_rear_bow_slot_dist_max_m": 400.0,
        "tug_init_mixed_opposite_stern_dist_min_m": 220.0,
        "tug_init_mixed_opposite_stern_dist_max_m": 380.0,
        "tug_init_mixed_approach_speed_min_ms": 0.15,
        "tug_init_mixed_approach_speed_max_ms": 0.40,
        "tug_init_heading_noise_rad": math.radians(3.0),
        "tug_init_yaw_rate_noise_rads": 0.003,
        "route_chase_speed_max_ms": 0.40,
        "route_nonfinal_forward_action_cap": 0.25,
    }),
)

C04_ONE_READY_BRIDGE = make_course(
    name="c04_one_ready_bridge",
    description="One-ready bridge with three joining tugs from safe side/outer route starts.",
    total_steps=1_200_000,
    env_overrides=merge_overrides(SAFE_JOIN_REWARD_OVERRIDES, {
        "tug_init_mixed_ready_counts": (1,),
        "tug_init_mixed_zones": (
            "outer_slot",
            "outer_slot",
            "outer_slot",
            "side_lane",
            "side_lane",
            "side_lane",
        ),
        "hold_time_s": 1.0,
        "tug_init_mixed_pair_min_dist_m": 190.0,
        "tug_init_mixed_approach_speed_min_ms": 0.12,
        "tug_init_mixed_approach_speed_max_ms": 0.38,
        "tug_init_heading_noise_rad": math.radians(3.0),
        "tug_init_yaw_rate_noise_rads": 0.003,
        "route_chase_speed_max_ms": 0.42,
        "route_nonfinal_forward_action_cap": 0.26,
        "route_guidance_blend": 0.25,
        "route_guidance_forward_action": 0.26,
        "route_guidance_turn_action": 0.0,
        "reward_route_w": 0.45,
        "reward_shape_w": 0.40,
        "reward_collision_w": 4.5,
        "reward_collision_cap": 2.5,
        "reward_near_hold_w": 0.35,
        "reward_near_hold_scale_m": 120.0,
        "reward_hold_streak_w": 0.35,
    }),
)

C04_ZERO_ROUTE_BRIDGE = make_course(
    name="c04_zero_route_bridge",
    description="Zero-ready bridge with four joining tugs from side/outer starts.",
    total_steps=1_200_000,
    env_overrides=merge_overrides(
        SAFE_JOIN_REWARD_OVERRIDES,
        C04_ZERO_READY_INIT,
        FINAL_SLOT_GUIDANCE_OVERRIDES,
        {
            "tug_init_mixed_zones": (
                "outer_slot",
                "outer_slot",
                "outer_slot",
            ),
            "hold_time_s": 1.0,
            "tug_init_heading_noise_rad": math.radians(3.0),
            "tug_init_yaw_rate_noise_rads": 0.003,
            "route_chase_speed_max_ms": 0.55,
            "route_tug_speed_soft_limit_ms": 2.6,
            "route_nonfinal_forward_action_cap": 0.32,
            "route_guidance_blend": 0.65,
            "route_guidance_forward_action": 0.32,
            "route_guidance_turn_action": 0.0,
            "reward_route_w": 0.65,
            "reward_shape_w": 0.55,
            "reward_collision_w": 4.0,
            "reward_collision_cap": 2.0,
            "reward_near_hold_w": 0.45,
            "reward_near_hold_scale_m": 125.0,
            "reward_hold_streak_w": 0.40,
            "reward_hold_start_m": 180.0,
            "reward_hold_full_m": 60.0,
            "reward_arrival_bonus": 100.0,
        },
    ),
)

C04_ZERO_READY = make_course(
    name="c04_zero_ready",
    description="Zero-ready side/outer safe-route starts after the zero-route bridge.",
    total_steps=1_500_000,
    env_overrides=merge_overrides(
        SAFE_JOIN_REWARD_OVERRIDES,
        C04_ZERO_READY_INIT,
        FINAL_SLOT_GUIDANCE_OVERRIDES,
        {
            "hold_time_s": 1.0,
            "tug_init_heading_noise_rad": math.radians(3.0),
            "tug_init_yaw_rate_noise_rads": 0.003,
            "route_chase_speed_max_ms": 0.55,
            "route_tug_speed_soft_limit_ms": 2.6,
            "route_nonfinal_forward_action_cap": 0.32,
            "route_guidance_blend": 0.65,
            "route_guidance_forward_action": 0.32,
            "route_guidance_turn_action": 0.0,
            "reward_route_w": 0.65,
            "reward_shape_w": 0.55,
            "reward_collision_w": 4.0,
            "reward_collision_cap": 2.0,
            "reward_near_hold_w": 0.45,
            "reward_near_hold_scale_m": 125.0,
            "reward_hold_streak_w": 0.40,
            "reward_hold_start_m": 180.0,
            "reward_hold_full_m": 60.0,
            "reward_arrival_bonus": 100.0,
        },
    ),
)

MAIN_COURSES = {
    "c01_three_ready": C01_THREE_READY,
    "c02_two_ready": C02_TWO_READY,
    "c03_one_ready": C03_ONE_READY,
    "c04_one_ready_bridge": C04_ONE_READY_BRIDGE,
    "c04_zero_route_bridge": C04_ZERO_ROUTE_BRIDGE,
    "c04_zero_ready": C04_ZERO_READY,
}
