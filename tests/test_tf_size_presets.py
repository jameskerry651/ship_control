"""Transformer actor size presets on PPOConfig."""

from __future__ import annotations

import pytest

from config import PPOConfig, TF_SIZE_PRESETS, apply_tf_size_preset, list_tf_size_presets
from rl.actor import build_actor


EXPECTED = {
    "S": {
        "tf_d_model": 64,
        "tf_nhead": 4,
        "tf_num_layers": 2,
        "tf_ffn_dim": 128,
    },
    "M": {
        "tf_d_model": 128,
        "tf_nhead": 4,
        "tf_num_layers": 3,
        "tf_ffn_dim": 256,
    },
    "L": {
        "tf_d_model": 256,
        "tf_nhead": 8,
        "tf_num_layers": 4,
        "tf_ffn_dim": 512,
    },
}


def test_list_tf_size_presets_matches_design() -> None:
    assert list_tf_size_presets() == ["L", "M", "S"]
    assert set(TF_SIZE_PRESETS) == set(EXPECTED)


@pytest.mark.parametrize("size_id,overrides", sorted(EXPECTED.items()))
def test_apply_tf_size_preset_overrides(size_id: str, overrides: dict[str, int]) -> None:
    cfg = PPOConfig()
    applied = apply_tf_size_preset(cfg, size_id.lower())
    assert applied == size_id
    for key, value in overrides.items():
        assert getattr(cfg, key) == value


def test_apply_tf_size_preset_none_is_noop() -> None:
    cfg = PPOConfig()
    d = cfg.tf_d_model
    assert apply_tf_size_preset(cfg, None) is None
    assert apply_tf_size_preset(cfg, "") is None
    assert cfg.tf_d_model == d


def test_apply_tf_size_preset_unknown_raises() -> None:
    cfg = PPOConfig()
    with pytest.raises(ValueError, match="S"):
        apply_tf_size_preset(cfg, "XL")


@pytest.mark.parametrize(
    "size_id,min_params,max_params",
    [
        ("S", 180_000, 250_000),
        ("M", 450_000, 700_000),
        ("L", 1_800_000, 3_000_000),
    ],
)
def test_build_actor_param_count_in_band(size_id: str, min_params: int, max_params: int) -> None:
    cfg = PPOConfig(actor_arch="transformer")
    apply_tf_size_preset(cfg, size_id)
    actor = build_actor(
        arch="transformer",
        obs_dim=93,
        action_dim=4,
        hist_len=4,
        tf_d_model=cfg.tf_d_model,
        tf_nhead=cfg.tf_nhead,
        tf_num_layers=cfg.tf_num_layers,
        tf_ffn_dim=cfg.tf_ffn_dim,
        tf_dropout=cfg.tf_dropout,
    )
    n = sum(p.numel() for p in actor.parameters())
    assert min_params <= n <= max_params, n
