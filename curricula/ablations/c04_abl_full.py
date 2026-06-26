"""C4 消融 #5（全量）：P1 + P2 + P3 + P4 + P5（碰撞/接近再平衡）。

等价于 config.py 当前默认值；显式写出以便复现与归档。
"""

from __future__ import annotations


COURSE = {
    "name": "c04_abl_full",
    "description": "C4 ablation: full P1..P5 redesign.",
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
        "reward_hold_streak_w": 0.3,
        "reward_hold_streak_decay": 2,
        # ---- P5 开 ----
        "reward_collision_w": 3.0,
        "reward_collision_cap": 1.5,
    },
}
