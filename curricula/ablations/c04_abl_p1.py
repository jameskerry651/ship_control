"""C4 消融 #1：仅启用 P1（终端碰撞惩罚按责分配）。

基线 = 原始奖励（全员共担 −80、无团队项/势函数/streak、w_collision=5、无 cap），
本课程只把 bystander 惩罚降到 15，单独验证 P1 对 critic EV 与碰撞率的修复。
"""

from __future__ import annotations


COURSE = {
    "name": "c04_abl_p1",
    "description": "C4 ablation: P1 per-culprit terminal penalty only.",
    "total_steps": 1_500_000,
    "env_overrides": {
        "tug_init_mixed_ready_counts": (0,),
        "hold_time_s": 2.0,
        # ---- P1 开 ----
        "reward_collision_pen_culprit": 80.0,
        "reward_collision_pen_bystander": 15.0,
        # ---- P2/P3/P4/P5 关（还原原始行为）----
        "reward_team_w": 0.0,
        "reward_shape_w": 0.0,
        "reward_hold_streak_w": 0.0,
        "reward_hold_streak_decay": 999,
        "reward_collision_w": 5.0,
        "reward_collision_cap": 1.0e9,
    },
}
