"""Reward preset CLI skeleton on EnvConfig."""

from __future__ import annotations

import pytest

from config import EnvConfig, REWARD_PRESETS, apply_reward_preset, list_reward_presets


def test_reward_presets_table_is_empty_skeleton() -> None:
    assert REWARD_PRESETS == {}
    assert list_reward_presets() == []


def test_apply_reward_preset_none_is_noop() -> None:
    cfg = EnvConfig()
    dist = cfg.reward_dist_w
    assert apply_reward_preset(cfg, None) is None
    assert apply_reward_preset(cfg, "") is None
    assert cfg.reward_dist_w == dist


def test_apply_reward_preset_unknown_raises() -> None:
    cfg = EnvConfig()
    with pytest.raises(ValueError, match="none defined"):
        apply_reward_preset(cfg, "not_a_preset")


def test_apply_reward_preset_applies_registered_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(REWARD_PRESETS, "rw_tmp", {"reward_dist_w": 9.0})
    cfg = EnvConfig()
    assert apply_reward_preset(cfg, "rw_tmp") == "rw_tmp"
    assert cfg.reward_dist_w == pytest.approx(9.0)
