"""C4 消融 #4：P1 + P2 + P3 + P4（hold streak 衰减 + streak 稠密奖励）。"""

from __future__ import annotations


COURSE = {
    "name": "c04_abl_p1p2p3p4",
    "description": "C4 ablation: P1 + P2 + P3 + P4 hold-streak robustness.",
    "total_steps": 1_500_000,
    "env_overrides": {
        "tug_init_mixed_ready_counts": (0,),
        "hold_time_s": 2.0,
        "reward_collision_pen_culprit": 80.0,
        "reward_collision_pen_bystander": 15.0,
        "reward_team_w": 0.5,
        "reward_team_softmin_beta": 4.0,
        "reward_shape_w": 0.3,
        "reward_shape_gamma": 0.99,
        "reward_shape_d_ref_m": 200.0,
        "reward_shape_clip": 1.0,
        # ---- P4 开 ----
        "reward_hold_streak_w": 0.3,
        "reward_hold_streak_decay": 2,
        # ---- P5 关 ----
        "reward_collision_w": 5.0,
        "reward_collision_cap": 1.0e9,
    },
}
