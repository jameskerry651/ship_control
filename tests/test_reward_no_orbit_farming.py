"""Reward relationships that prevent static or orbital farming."""

from __future__ import annotations

import math

import numpy as np
import pytest

from config import EnvConfig
from env.formation_env import FormationEnv
from physics.tugboat_dynamics_model import Vec3


def _isolated_env(*, team_w: float = 0.0) -> FormationEnv:
    cfg = EnvConfig()
    cfg.reward_collision_w = 0.0
    cfg.reward_velocity_w = 0.0
    cfg.reward_team_w = team_w
    cfg.reward_progress_risk_gate = 1e9  # ρ ≈ 0
    cfg.reward_safe_w = 0.0
    env = FormationEnv(cfg=cfg, seed=17)
    env.reset()
    env.ship.x = 0.0
    env.ship.y = 0.0
    env.ship.psi = 0.0
    env.ship.u = 0.0
    env.ship.v = 0.0
    env.ship.r = 0.0
    return env


def _place_at_slot_distance(env: FormationEnv, tug_idx: int, distance_m: float) -> None:
    slots = env.ship.slot_positions_world()
    slot = slots[env.tug_to_slot[tug_idx]]
    angle = math.atan2(float(slot[1]) - env.ship.y, float(slot[0]) - env.ship.x)
    env.tugs[tug_idx].eta = Vec3(
        float(slot[0]) + distance_m * math.cos(angle),
        float(slot[1]) + distance_m * math.sin(angle),
        float(slot[2]),
    )
    env.tugs[tug_idx].nu = Vec3.zero()


def _components(env: FormationEnv, previous: list[float]) -> dict:
    env._episode.prev_dist[:] = np.asarray(previous, dtype=np.float32)
    actions = np.zeros((env.cfg.n_tugs, 4), dtype=np.float32)
    _, info = env._compute_rewards(actions)
    return info["reward_components"]


@pytest.mark.parametrize(
    ("distance_m", "expected_cost", "expected_total"),
    [(200.0, 1.0, -0.2), (100.0, 0.5, -0.1), (25.0, 0.125, -0.025)],
)
def test_static_outside_target_is_negative(
    distance_m: float, expected_cost: float, expected_total: float
) -> None:
    env = _isolated_env()
    for i, distance in enumerate((distance_m, 400.0, 450.0, 500.0)):
        _place_at_slot_distance(env, i, distance)
    comp = _components(env, [distance_m, 400.0, 450.0, 500.0])
    assert float(comp["r_dist"][0]) == pytest.approx(0.0)
    assert float(comp["p_distance"][0]) == pytest.approx(expected_cost)
    assert float(comp["r_total"][0]) == pytest.approx(expected_total, abs=1e-6)


def test_approach_beats_static_and_retreat() -> None:
    env = _isolated_env()
    for i, distance in enumerate((100.0, 400.0, 450.0, 500.0)):
        _place_at_slot_distance(env, i, distance)
    static = _components(env, [100.0, 400.0, 450.0, 500.0])
    approach = _components(env, [101.0, 400.0, 450.0, 500.0])
    retreat = _components(env, [99.0, 400.0, 450.0, 500.0])
    assert float(approach["r_dist"][0]) == pytest.approx(1.0)
    assert float(retreat["r_dist"][0]) == pytest.approx(-1.0)
    assert float(approach["r_total"][0]) > float(static["r_total"][0])
    assert float(static["r_total"][0]) > float(retreat["r_total"][0])


def test_progress_does_not_drop_at_old_150m_boundary() -> None:
    env = _isolated_env()
    values = []
    for distance_m in (149.0, 151.0):
        _place_at_slot_distance(env, 0, distance_m)
        for i, distance in enumerate((400.0, 450.0, 500.0), start=1):
            _place_at_slot_distance(env, i, distance)
        comp = _components(env, [distance_m + 1.0, 400.0, 450.0, 500.0])
        values.append(float(comp["r_dist"][0]))
    assert values == pytest.approx([1.0, 1.0])


def test_target_center_hold_is_positive() -> None:
    env = _isolated_env()
    for i, distance in enumerate((0.0, 400.0, 450.0, 500.0)):
        _place_at_slot_distance(env, i, distance)
    comp = _components(env, [0.0, 400.0, 450.0, 500.0])
    assert float(comp["p_distance"][0]) == pytest.approx(0.0)
    assert float(comp["r_hold"][0]) == pytest.approx(1.0)
    assert float(comp["r_total"][0]) == pytest.approx(env.cfg.reward_hold_w)


def test_team_cost_tracks_the_lagging_tug() -> None:
    env = _isolated_env(team_w=0.2)
    for i, distance in enumerate((20.0, 20.0, 20.0, 200.0)):
        _place_at_slot_distance(env, i, distance)
    lagging = _components(env, [20.0, 20.0, 20.0, 200.0])
    for i, distance in enumerate((20.0, 20.0, 20.0, 50.0)):
        _place_at_slot_distance(env, i, distance)
    recovered = _components(env, [20.0, 20.0, 20.0, 50.0])
    assert np.all(np.asarray(lagging["r_team"]) < 0.0)
    assert np.ptp(np.asarray(lagging["r_team"])) == pytest.approx(0.0)
    assert abs(float(recovered["r_team"][0])) < abs(float(lagging["r_team"][0]))


def test_approach_and_hold_trajectory_beats_longer_orbit() -> None:
    env = _isolated_env()
    parked = [400.0, 450.0, 500.0]

    def reward_at(distance_m: float, previous_m: float) -> float:
        _place_at_slot_distance(env, 0, distance_m)
        for i, distance in enumerate(parked, start=1):
            _place_at_slot_distance(env, i, distance)
        return float(_components(env, [previous_m, *parked])["r_total"][0])

    orbit_return = sum(reward_at(100.0, 100.0) for _ in range(80))
    distances = list(np.linspace(200.0, 0.0, 51))
    approach_return = sum(
        reward_at(distance, previous)
        for previous, distance in zip(distances, distances[1:])
    )
    approach_return += sum(reward_at(0.0, 0.0) for _ in range(10))
    assert orbit_return < 0.0
    assert approach_return > orbit_return
