"""Safe-slot approach reward: risk-gated progress and R_safe."""

from __future__ import annotations

import math

import numpy as np
import pytest

from config import EnvConfig
from env.formation_env import FormationEnv
from physics.tugboat_dynamics_model import Vec3


def _env(**overrides) -> FormationEnv:
    cfg = EnvConfig()
    cfg.reward_collision_w = 0.0
    cfg.reward_velocity_w = 0.0
    cfg.reward_team_w = 0.0
    cfg.reward_safe_w = 0.0
    for k, v in overrides.items():
        setattr(cfg, k, v)
    env = FormationEnv(cfg=cfg, seed=3)
    env.reset()
    env.ship.x = 0.0
    env.ship.y = 0.0
    env.ship.psi = 0.0
    env.ship.u = 0.0
    env.ship.v = 0.0
    env.ship.r = 0.0
    return env


def _place(env: FormationEnv, i: int, distance_m: float) -> None:
    slots = env.ship.slot_positions_world()
    slot = slots[env.tug_to_slot[i]]
    angle = math.atan2(float(slot[1]) - env.ship.y, float(slot[0]) - env.ship.x)
    env.tugs[i].eta = Vec3(
        float(slot[0]) + distance_m * math.cos(angle),
        float(slot[1]) + distance_m * math.sin(angle),
        float(slot[2]),
    )
    env.tugs[i].nu = Vec3.zero()


def _comp(env: FormationEnv, prev: list[float]) -> dict:
    env._episode.prev_dist[:] = np.asarray(prev, dtype=np.float32)
    actions = np.zeros((env.cfg.n_tugs, 4), dtype=np.float32)
    _, info = env._compute_rewards(actions)
    return info["reward_components"]


def test_positive_progress_gated_by_ship_risk() -> None:
    env = _env(reward_progress_risk_gate=0.5, reward_safe_w=0.0)
    for i, d in enumerate((8.0, 400.0, 450.0, 500.0)):
        _place(env, i, d)
    high = _comp(env, [9.0, 400.0, 450.0, 500.0])
    assert float(high["progress_risk"][0]) > 0.5
    assert float(high["r_dist"][0]) < 0.5

    for i, d in enumerate((120.0, 400.0, 450.0, 500.0)):
        _place(env, i, d)
    low = _comp(env, [121.0, 400.0, 450.0, 500.0])
    assert float(low["progress_risk"][0]) == pytest.approx(0.0, abs=1e-6)
    assert float(low["r_dist"][0]) == pytest.approx(1.0)


def test_negative_progress_not_risk_gated() -> None:
    env = _env(reward_progress_risk_gate=0.5, reward_safe_w=0.0)
    for i, d in enumerate((8.0, 400.0, 450.0, 500.0)):
        _place(env, i, d)
    retreat = _comp(env, [7.0, 400.0, 450.0, 500.0])
    assert float(retreat["progress_risk"][0]) > 0.5
    # Negative progress is not multiplied by (1-ρ); only (1-g) applies.
    d = 8.0
    target_x = float(np.clip(1.0 - d / env.cfg.pos_tol_m, 0.0, 1.0))
    target_gate = target_x * target_x * (3.0 - 2.0 * target_x)
    assert float(retreat["r_dist"][0]) == pytest.approx(-(1.0 - target_gate), abs=1e-5)


def test_r_safe_zero_outside_corridor() -> None:
    env = _env(reward_safe_w=2.0, reward_progress_risk_gate=1e9)
    for i, d in enumerate((200.0, 400.0, 450.0, 500.0)):
        _place(env, i, d)
    comp = _comp(env, [201.0, 400.0, 450.0, 500.0])
    assert float(comp["corridor_gate"][0]) == pytest.approx(0.0)
    assert float(comp["r_safe"][0]) == pytest.approx(0.0)


def test_r_safe_higher_when_centered_and_approaching() -> None:
    env = _env(reward_safe_w=2.0, reward_progress_risk_gate=1e9, reward_dist_w=0.0)
    for i, d in enumerate((40.0, 400.0, 450.0, 500.0)):
        _place(env, i, d)
    on_axis = _comp(env, [41.0, 400.0, 450.0, 500.0])

    slots = env.ship.slot_positions_world()
    slot = slots[env.tug_to_slot[0]]
    ax = float(slot[0]) - env.ship.x
    ay = float(slot[1]) - env.ship.y
    n = math.hypot(ax, ay)
    ex, ey = ax / n, ay / n
    px, py = -ey, ex
    env.tugs[0].eta = Vec3(
        float(slot[0]) + 40.0 * ex + 35.0 * px,
        float(slot[1]) + 40.0 * ey + 35.0 * py,
        float(slot[2]),
    )
    off = _comp(env, [41.0, 400.0, 450.0, 500.0])
    assert float(on_axis["r_safe"][0]) > float(off["r_safe"][0])


def test_r_safe_drops_when_closing_fast_on_ship() -> None:
    env = _env(reward_safe_w=2.0, reward_progress_risk_gate=1e9, reward_dist_w=0.0)
    for i, d in enumerate((40.0, 400.0, 450.0, 500.0)):
        _place(env, i, d)
    tug = env.tugs[0]
    ux = env.ship.x - float(tug.eta.x)
    uy = env.ship.y - float(tug.eta.y)
    tug.eta = Vec3(float(tug.eta.x), float(tug.eta.y), math.atan2(uy, ux))
    tug.nu = Vec3(3.0, 0.0, 0.0)
    fast = _comp(env, [40.0, 400.0, 450.0, 500.0])
    tug.nu = Vec3(0.0, 0.0, 0.0)
    still = _comp(env, [40.0, 400.0, 450.0, 500.0])
    assert float(still["r_safe"][0]) > float(fast["r_safe"][0])
