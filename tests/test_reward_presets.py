"""Reward preset CLI mapping on EnvConfig."""

from __future__ import annotations

import pytest

from config import EnvConfig, REWARD_PRESETS, apply_reward_preset, list_reward_presets

EXPECTED = {
    "rsc_baseline": {
        "reward_dist_w": 3.0,
        "reward_collision_cap": 2.0,
        "reward_ship_soft_min_scale": 0.15,
    },
    "rsc_dist_soft": {
        "reward_dist_w": 1.5,
        "reward_collision_cap": 2.0,
        "reward_ship_soft_min_scale": 0.15,
    },
    "rsc_coll_mid": {
        "reward_dist_w": 3.0,
        "reward_collision_cap": 4.0,
        "reward_ship_soft_min_scale": 0.15,
    },
    "rsc_coll_hi": {
        "reward_dist_w": 3.0,
        "reward_collision_cap": 6.0,
        "reward_ship_soft_min_scale": 0.15,
    },
    "rsc_balanced": {
        "reward_dist_w": 1.5,
        "reward_collision_cap": 4.0,
        "reward_ship_soft_min_scale": 0.15,
    },
    "rsc_corridor_hard": {
        "reward_dist_w": 3.0,
        "reward_collision_cap": 2.0,
        "reward_ship_soft_min_scale": 0.50,
    },
}


def test_list_reward_presets_matches_rsc_design() -> None:
    assert list_reward_presets() == sorted(EXPECTED)
    assert set(REWARD_PRESETS) == set(EXPECTED)


@pytest.mark.parametrize("preset_id,overrides", sorted(EXPECTED.items()))
def test_apply_reward_preset_overrides_rsc_fields(
    preset_id: str, overrides: dict[str, float]
) -> None:
    cfg = EnvConfig()
    applied = apply_reward_preset(cfg, preset_id)
    assert applied == preset_id
    for key, value in overrides.items():
        assert getattr(cfg, key) == pytest.approx(value)
    # Unlisted fields stay at EnvConfig defaults
    assert cfg.reward_collision_w == pytest.approx(2.0)
    assert cfg.reward_hold_w == pytest.approx(3.0)
    assert cfg.reward_safe_w == pytest.approx(2.0)


def test_apply_reward_preset_none_is_noop() -> None:
    cfg = EnvConfig()
    dist = cfg.reward_dist_w
    assert apply_reward_preset(cfg, None) is None
    assert apply_reward_preset(cfg, "") is None
    assert cfg.reward_dist_w == dist


def test_apply_reward_preset_unknown_raises() -> None:
    cfg = EnvConfig()
    with pytest.raises(ValueError, match="rsc_baseline"):
        apply_reward_preset(cfg, "not_a_preset")
