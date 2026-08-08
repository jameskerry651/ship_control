"""Reward scale ablation naming and promotion rules."""

from __future__ import annotations

import pytest

from rl.reward_scale_ablation import (
    RSC_PRESET_ORDER,
    rsc_run_name,
    rsc_short_name,
    select_rsc_promotions,
)


def test_rsc_preset_order_matches_design() -> None:
    assert RSC_PRESET_ORDER == [
        "rsc_baseline",
        "rsc_dist_soft",
        "rsc_coll_mid",
        "rsc_coll_hi",
        "rsc_balanced",
        "rsc_corridor_hard",
    ]


def test_rsc_short_and_run_name() -> None:
    assert rsc_short_name("rsc_coll_mid") == "coll_mid"
    assert rsc_run_name("rsc_coll_mid", phase="1m") == "rsc_1m_coll_mid"
    assert rsc_run_name("rsc_coll_mid", phase="5m") == "rsc_5m_coll_mid"


def test_rsc_short_name_rejects_bad_id() -> None:
    with pytest.raises(ValueError, match="rsc_"):
        rsc_short_name("baseline")


def test_select_promotions_picks_collision_improvers() -> None:
    metrics = {
        "baseline": {
            "capture_rate": 0.0,
            "final_dist_mean": 140.0,
            "collision_rate": 0.80,
        },
        "dist_soft": {
            "capture_rate": 0.0,
            "final_dist_mean": 150.0,
            "collision_rate": 0.70,  # only -10 pt → fail
        },
        "coll_mid": {
            "capture_rate": 0.0,
            "final_dist_mean": 155.0,
            "collision_rate": 0.60,  # -20 pt → pass
        },
        "coll_hi": {
            "capture_rate": 0.0,
            "final_dist_mean": 210.0,  # >200 and +70 vs baseline → fail dist
            "collision_rate": 0.50,
        },
        "balanced": {
            "capture_rate": 0.0,
            "final_dist_mean": 145.0,
            "collision_rate": 0.55,  # -25 pt → pass, best delta
        },
        "corridor_hard": {
            "capture_rate": 0.0,
            "final_dist_mean": 130.0,
            "collision_rate": 0.64,  # -16 pt → pass
        },
    }
    # top by delta: balanced (-25), coll_mid (-20); corridor_hard (-16) third
    assert select_rsc_promotions(metrics, max_promote=2) == ["balanced", "coll_mid"]


def test_select_promotions_capture_auto_promotes_first() -> None:
    metrics = {
        "baseline": {
            "capture_rate": 0.0,
            "final_dist_mean": 140.0,
            "collision_rate": 0.80,
        },
        "dist_soft": {
            "capture_rate": 0.1,
            "final_dist_mean": 250.0,  # would fail dist gate without capture
            "collision_rate": 0.79,
        },
        "coll_mid": {
            "capture_rate": 0.0,
            "final_dist_mean": 150.0,
            "collision_rate": 0.60,
        },
    }
    assert select_rsc_promotions(metrics, max_promote=2) == ["dist_soft", "coll_mid"]


def test_select_promotions_requires_baseline() -> None:
    with pytest.raises(ValueError, match="baseline"):
        select_rsc_promotions({"coll_mid": {
            "capture_rate": 0.0,
            "final_dist_mean": 100.0,
            "collision_rate": 0.1,
        }})
