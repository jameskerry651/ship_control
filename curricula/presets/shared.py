"""Shared env override fragments reused across curriculum families."""

from __future__ import annotations

# c04 zero-ready task: near/mid route starts (rear/opposite deferred to later stages).
C04_ZERO_READY_INIT = {
    "tug_init_mixed_ready_counts": (0,),
    "tug_init_mixed_zones": (
        "outer_slot",
        "outer_slot",
        "outer_slot",
    ),
    "hold_time_s": 2.0,
    "tug_init_mixed_pair_min_dist_m": 160.0,
    "tug_init_mixed_approach_speed_min_ms": 0.12,
    "tug_init_mixed_approach_speed_max_ms": 0.45,
    "route_guidance_blend": 0.65,
    "route_guidance_forward_action": 0.30,
    "route_guidance_turn_action": 0.0,
    "route_guidance_distance_scale_m": 120.0,
    "route_guidance_allow_reverse": True,
}

# Safety-biased joining controls shared by the ready-count and strict-position
# curricula.  The failing C02 runs were not short on arrival incentive; they
# were accepting high collision rates while two or more tugs converged.  These
# knobs keep route-stage approach speeds moderate and make CPA/clearance
# shaping visible before terminal collisions occur.
SAFE_JOIN_REWARD_OVERRIDES = {
    "route_speed_governor": True,
    "route_chase_speed_max_ms": 0.50,
    "route_tug_speed_soft_limit_ms": 2.2,
    "route_nonfinal_forward_action_cap": 0.30,
    "route_speed_governor_min_forward_action": 0.02,
    "route_speed_governor_cap_slope": 0.45,
    "reward_chase_speed_target_ms": 0.60,
    "reward_route_w": 0.35,
    "reward_team_w": 0.70,
    "reward_collision_w": 4.0,
    "reward_collision_cap": 2.0,
    "reward_collision_ship_safe_m": 75.0,
    "reward_collision_tug_safe_m": 110.0,
    "reward_cpa_horizon_s": 80.0,
    "reward_collision_cpa_w": 3.0,
    "reward_near_hold_w": 0.20,
    "reward_near_hold_scale_m": 120.0,
}

# Final strict task: any number of tugs may already be in formation. Ready
# starts are tightened so "ready" means inside the final 10 m gate.
C04_FINAL_MIXED_READY_INIT = {
    "tug_init_mixed_ready_counts": (0, 1, 2, 3, 4),
    "tug_init_mixed_zones": (
        "outer_slot",
        "outer_slot",
        "outer_slot",
    ),
    "tug_init_force_single_opposite": False,
    "hold_time_s": 2.0,
    "tug_init_mixed_pair_min_dist_m": 70.0,
    "tug_init_ready_outward_offset_m": 4.0,
    "tug_init_ready_pos_jitter_m": 1.5,
    "tug_init_ready_heading_noise_rad": 0.03490658503988659,
    "tug_init_ready_speed_noise_ms": 0.05,
    "tug_init_ready_sway_noise_ms": 0.02,
    "tug_init_ready_yaw_rate_noise_rads": 0.002,
    "route_guidance_blend": 0.65,
    "route_guidance_forward_action": 0.30,
    "route_guidance_turn_action": 0.0,
    "route_guidance_distance_scale_m": 120.0,
    "route_guidance_allow_reverse": True,
}

FINAL_SLOT_GUIDANCE_OVERRIDES = {
    "route_guidance_final_blend": 0.35,
    "route_guidance_final_forward_action": 0.20,
    "route_guidance_final_turn_action": 0.08,
    "route_guidance_final_heading_turn_action": 0.12,
    "route_guidance_final_distance_scale_m": 90.0,
}

# Strict position-tolerance shaping applied on top of c04 init.
STRICT_POS_REWARD_OVERRIDES = {
    "speed_tol_ms": 3.0,
    "reward_route_w": 0.45,
    "reward_shape_w": 0.55,
    "reward_hold_streak_w": 0.45,
    "reward_hold_start_m": 180.0,
    "reward_hold_full_m": 40.0,
    "reward_precision_w": 0.35,
    "reward_precision_scale_m": 40.0,
    "reward_near_hold_w": 0.50,
    "reward_near_hold_scale_m": 80.0,
    "reward_arrival_bonus": 110.0,
}

STRICT_POS_FINAL_REWARD_OVERRIDES = {
    "speed_tol_ms": 3.0,
    "reward_team_w": 0.8,
    "reward_route_w": 0.45,
    "reward_shape_w": 0.60,
    "reward_hold_streak_w": 0.55,
    "reward_hold_start_m": 80.0,
    "reward_hold_full_m": 15.0,
    "reward_precision_w": 0.50,
    "reward_precision_scale_m": 25.0,
    "reward_near_hold_w": 0.65,
    "reward_near_hold_scale_m": 55.0,
    "reward_arrival_bonus": 130.0,
}

# C4 ablation baseline env (zero-ready, hold=2s).
C04_ABLATION_INIT = {
    "tug_init_mixed_ready_counts": (0,),
    "hold_time_s": 2.0,
}

# Original reward stack with incremental components disabled.
C04_ABLATION_REWARD_OFF = {
    "reward_team_w": 0.0,
    "reward_shape_w": 0.0,
    "reward_hold_streak_w": 0.0,
    "reward_hold_streak_decay": 999,
    "reward_collision_w": 5.0,
    "reward_collision_cap": 1.0e9,
}
