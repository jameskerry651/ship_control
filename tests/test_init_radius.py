"""安全圆环初始化的几何、可复现性与失败行为。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from config import EnvConfig
from env.formation_env import FormationEnv
from env.init import InitializationError, assign_tugs_to_slots, sample_tug_init_states


def _mean_ship_radius(env: FormationEnv) -> float:
    rs = [
        math.hypot(tug.eta.x - env.ship.x, tug.eta.y - env.ship.y)
        for tug in env.tugs
    ]
    return float(np.mean(rs))


def test_init_radius_100() -> None:
    cfg = EnvConfig(tug_init_radius_m=100.0)
    env = FormationEnv(cfg=cfg, seed=0)
    with pytest.warns(RuntimeWarning, match="biases accepted positions"):
        env.reset(seed=0)
    mean_r = _mean_ship_radius(env)
    assert abs(mean_r - 100.0) < 1e-3
    for tug in env.tugs:
        r = math.hypot(tug.eta.x - env.ship.x, tug.eta.y - env.ship.y)
        assert abs(r - 100.0) < 1e-3


def test_init_radius_200() -> None:
    cfg = EnvConfig(tug_init_radius_m=200.0)
    env = FormationEnv(cfg=cfg, seed=1)
    env.reset(seed=1)
    for tug in env.tugs:
        r = math.hypot(tug.eta.x - env.ship.x, tug.eta.y - env.ship.y)
        assert abs(r - 200.0) < 1e-3


def test_default_init_radius_is_120() -> None:
    cfg = EnvConfig()
    assert cfg.tug_init_schema == "safe_circle_v2"
    assert cfg.tug_init_radius_m == 120.0
    assert cfg.tug_init_ship_margin_m == 5.0
    assert cfg.tug_init_pair_margin_m == 5.0
    assert cfg.tug_init_max_attempts == 1000
    assert cfg.tug_slot_assignment_mode == "minimax"


def test_default_reset_is_collision_free_across_many_seeds() -> None:
    cfg = EnvConfig()
    required_ship = cfg.ship_collision_dist_m + cfg.tug_init_ship_margin_m
    required_pair = cfg.tug_collision_dist_m + cfg.tug_init_pair_margin_m

    for seed in range(500):
        env = FormationEnv(cfg=cfg, seed=seed)
        env.reset(seed=seed)
        assert min(
            env.ship.distance_from_hull(tug.eta.x, tug.eta.y) for tug in env.tugs
        ) >= required_ship - 1e-9
        assert min(
            math.hypot(
                env.tugs[i].eta.x - env.tugs[j].eta.x,
                env.tugs[i].eta.y - env.tugs[j].eta.y,
            )
            for i in range(env.n_tugs)
            for j in range(i + 1, env.n_tugs)
        ) >= required_pair - 1e-9


def test_safe_reset_is_reproducible() -> None:
    cfg = EnvConfig()
    env_a = FormationEnv(cfg=cfg, seed=7)
    env_b = FormationEnv(cfg=cfg, seed=999)
    obs_a = env_a.reset(seed=123)
    obs_b = env_b.reset(seed=123)

    np.testing.assert_array_equal(obs_a, obs_b)
    np.testing.assert_array_equal(
        [[t.eta.x, t.eta.y, t.eta.z] for t in env_a.tugs],
        [[t.eta.x, t.eta.y, t.eta.z] for t in env_b.tugs],
    )
    assert env_a.last_init_diagnostics == env_b.last_init_diagnostics
    assert env_a.render_snapshot()["init"] == env_a.last_init_diagnostics


def test_safe_sampler_is_rotation_invariant_in_ship_frame() -> None:
    cfg = EnvConfig()
    psi = 1.234
    positions_0, *_ = sample_tug_init_states(
        np.random.default_rng(55), cfg.n_tugs, 0.0, 0.0, 0.0, cfg
    )
    positions_rotated, *_ = sample_tug_init_states(
        np.random.default_rng(55), cfg.n_tugs, 17.0, -9.0, psi, cfg
    )

    c = math.cos(psi)
    s = math.sin(psi)
    expected = np.empty_like(positions_0)
    expected[:, 0] = 17.0 + c * positions_0[:, 0] - s * positions_0[:, 1]
    expected[:, 1] = -9.0 + s * positions_0[:, 0] + c * positions_0[:, 1]
    np.testing.assert_allclose(positions_rotated, expected, atol=1e-12)


def test_biased_legacy_radius_is_safe_and_reported() -> None:
    cfg = EnvConfig(tug_init_radius_m=100.0)
    env = FormationEnv(cfg=cfg, seed=3)
    with pytest.warns(RuntimeWarning, match="full-angle safe radius"):
        env.reset(seed=3)

    assert env.last_init_diagnostics["biased_angles"] is True
    assert env.last_init_diagnostics["ship_clearance_m"] == 11.0
    assert env.last_init_diagnostics["pair_separation_m"] == 25.0


def test_impossible_safe_circle_fails_clearly() -> None:
    cfg = EnvConfig(tug_init_radius_m=20.0)
    env = FormationEnv(cfg=cfg, seed=0)
    with pytest.raises(InitializationError, match="no ship-clearance solution"):
        env.reset(seed=0)


def test_attempt_budget_is_bounded() -> None:
    cfg = EnvConfig(tug_init_max_attempts=1)
    env = FormationEnv(cfg=cfg, seed=0)
    with pytest.raises(InitializationError, match="attempt budget"):
        env.reset(seed=0)


def test_minimax_assignment_prioritizes_the_hardest_tug() -> None:
    tugs = np.asarray([[9.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    slots = np.asarray([[0.0, 0.0], [10.0, 0.0]], dtype=np.float64)

    mapping, diagnostics = assign_tugs_to_slots(tugs, slots, "minimax")

    np.testing.assert_array_equal(mapping, [1, 0])
    assert diagnostics["assignment_mode"] == "minimax"
    assert diagnostics["assignment_max_distance_m"] == 1.0
    assert diagnostics["assignment_total_distance_m"] == 2.0


def test_fixed_assignment_mode_preserves_legacy_roles() -> None:
    cfg = EnvConfig(tug_slot_assignment_mode="fixed")
    env = FormationEnv(cfg=cfg, seed=5)
    env.reset(seed=5)

    np.testing.assert_array_equal(env.tug_to_slot, np.arange(cfg.n_tugs))
    assert env.last_init_diagnostics["assignment_mode"] == "fixed"


def test_default_assignment_canonicalizes_the_minimax_solution() -> None:
    cfg = EnvConfig()
    env = FormationEnv(cfg=cfg, seed=0)
    for seed in range(100):
        env.reset(seed=seed)
        assert sorted(int(v) for v in env.tug_to_slot) == list(range(cfg.n_tugs))

        slots = env.ship.slot_positions_world()
        assigned = [
            math.hypot(
                env.tugs[i].eta.x - slots[env.tug_to_slot[i], 0],
                env.tugs[i].eta.y - slots[env.tug_to_slot[i], 1],
            )
            for i in range(cfg.n_tugs)
        ]
        positions = np.asarray([[t.eta.x, t.eta.y] for t in env.tugs])
        _, optimal = assign_tugs_to_slots(positions, slots, "minimax")
        assert max(assigned) == pytest.approx(optimal["assignment_max_distance_m"])
        assert sum(assigned) == pytest.approx(optimal["assignment_total_distance_m"])
        assert env.last_init_diagnostics["tug_to_slot"] == tuple(env.tug_to_slot)
        assert sorted(env.last_init_diagnostics["sample_to_slot"]) == list(
            range(cfg.n_tugs)
        )
