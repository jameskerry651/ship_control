"""Reward preset overlays on EnvConfig."""

from __future__ import annotations

import pytest

from config import EnvConfig, REWARD_PRESETS, apply_reward_preset, list_reward_presets


EXPECTED = {
    "rw_baseline": {},
    "rw_dist_up": {"reward_dist_w": 6.0},
    "rw_ship_safe_dn": {"reward_collision_ship_safe_m": 60.0},
    "rw_coll_soft": {
        "reward_collision_w": 0.5,
        "reward_collision_cpa_w": 1.0,
    },
    "rw_shape_up": {"reward_shape_w": 0.8},
    "rw_combo": {
        "reward_dist_w": 6.0,
        "reward_collision_ship_safe_m": 60.0,
    },
}


def test_list_reward_presets_matches_design() -> None:
    assert list_reward_presets() == sorted(EXPECTED)
    assert set(REWARD_PRESETS) == set(EXPECTED)


@pytest.mark.parametrize("preset_id,overrides", sorted(EXPECTED.items()))
def test_apply_reward_preset_overrides(preset_id: str, overrides: dict[str, float]) -> None:
    cfg = EnvConfig()
    before = {k: getattr(cfg, k) for k in (
        "reward_dist_w",
        "reward_collision_ship_safe_m",
        "reward_collision_w",
        "reward_collision_cpa_w",
        "reward_shape_w",
        "reward_arrival_bonus",
    )}
    applied = apply_reward_preset(cfg, preset_id)
    assert applied == preset_id
    for key, value in overrides.items():
        assert getattr(cfg, key) == pytest.approx(value)
    # Unmentioned reward knobs stay at defaults (spot-check arrival bonus always untouched).
    assert cfg.reward_arrival_bonus == before["reward_arrival_bonus"]
    for key, value in before.items():
        if key not in overrides and key != "reward_arrival_bonus":
            # only assert keys that are in our watch list and not overridden
            if key in (
                "reward_dist_w",
                "reward_collision_ship_safe_m",
                "reward_collision_w",
                "reward_collision_cpa_w",
                "reward_shape_w",
            ) and key not in overrides:
                assert getattr(cfg, key) == value


def test_apply_reward_preset_none_is_noop() -> None:
    cfg = EnvConfig()
    dist = cfg.reward_dist_w
    assert apply_reward_preset(cfg, None) is None
    assert apply_reward_preset(cfg, "") is None
    assert cfg.reward_dist_w == dist


def test_apply_reward_preset_unknown_raises() -> None:
    cfg = EnvConfig()
    with pytest.raises(ValueError, match="rw_baseline"):
        apply_reward_preset(cfg, "not_a_preset")
