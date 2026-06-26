"""C4 消融 #3：P1 + P2 + P3（势函数式接近 shaping）。"""

from __future__ import annotations


COURSE = {
    "name": "c04_abl_p1p2p3",
    "description": "C4 ablation: P1 + P2 + P3 potential-based approach shaping.",
    "total_steps": 1_500_000,
    "env_overrides": {
        "tug_init_mixed_ready_counts": (0,),
        "hold_time_s": 2.0,
        "reward_collision_pen_culprit": 80.0,
        "reward_collision_pen_bystander": 15.0,
        "reward_team_w": 0.5,
        "reward_team_softmin_beta": 4.0,
        # ---- P3 开 ----
        "reward_shape_w": 0.3,
        "reward_shape_gamma": 0.99,
        "reward_shape_d_ref_m": 200.0,
        "reward_shape_clip": 1.0,
        # ---- P4/P5 关 ----
        "reward_hold_streak_w": 0.0,
        "reward_hold_streak_decay": 999,
        "reward_collision_w": 5.0,
        "reward_collision_cap": 1.0e9,
    },
}
