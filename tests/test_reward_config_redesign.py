"""Defaults for structural reward redesign."""

from config import EnvConfig


def test_reward_redesign_defaults() -> None:
    cfg = EnvConfig()
    assert cfg.reward_dist_progress_frac == 0.7
    assert cfg.reward_stall_w == 0.5
    assert cfg.reward_stall_window_s == 5.0
    assert cfg.reward_stall_min_progress_m == 2.0
    assert cfg.reward_stall_floor == 0.2
    assert cfg.reward_corridor_half_width_m == 40.0
    assert cfg.reward_corridor_axial_slack_m == 30.0
    assert cfg.reward_ship_soft_min_scale == 0.15
    assert cfg.reward_collision_ship_safe_m == 80.0
