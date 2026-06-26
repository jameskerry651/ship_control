"""C4 消融 #2：P1 + P2（团队同步 softmin 项）。"""

from __future__ import annotations


COURSE = {
    "name": "c04_abl_p1p2",
    "description": "C4 ablation: P1 + P2 team softmin.",
    "total_steps": 1_500_000,
    "env_overrides": {
        "tug_init_mixed_ready_counts": (0,),
        "hold_time_s": 2.0,
        "reward_collision_pen_culprit": 80.0,
        "reward_collision_pen_bystander": 15.0,
        # ---- P2 开 ----
        "reward_team_w": 0.5,
        "reward_team_softmin_beta": 4.0,
        # ---- P3/P4/P5 关 ----
        "reward_shape_w": 0.0,
        "reward_hold_streak_w": 0.0,
        "reward_hold_streak_decay": 999,
        "reward_collision_w": 5.0,
        "reward_collision_cap": 1.0e9,
    },
}
