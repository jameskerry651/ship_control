"""Corridor softening for ship collision risk."""

from __future__ import annotations

import math

import numpy as np
import pytest

from config import EnvConfig
from env.formation_env import FormationEnv
from physics.tugboat_dynamics_model import Vec3


def _env() -> FormationEnv:
    cfg = EnvConfig()
    cfg.reward_team_w = 0.0
    env = FormationEnv(cfg=cfg, seed=3)
    env.reset()
    return env


def _park(env: FormationEnv) -> None:
    for i in (1, 2, 3):
        env.tugs[i].eta = Vec3(800.0 + 50 * i, 800.0, 0.0)
        env.tugs[i].nu = Vec3.zero()


def _comp(env: FormationEnv) -> dict:
    actions = np.zeros((env.cfg.n_tugs, 4), dtype=np.float32)
    _, info = env._compute_rewards(actions)
    return info["reward_components"]


def test_corridor_softens_ship_penalty_on_radial_approach() -> None:
    env = _env()
    env.ship.x = 0.0
    env.ship.y = 0.0
    env.ship.psi = 0.0
    env.ship.u = 0.0
    env.ship.v = 0.0
    env.ship.r = 0.0
    slots = env.ship.slot_positions_world()
    sx, sy = float(slots[0, 0]), float(slots[0, 1])
    ang = math.atan2(sy - env.ship.y, sx - env.ship.x)
    d = 80.0
    env.tugs[0].eta = Vec3(sx + d * math.cos(ang), sy + d * math.sin(ang), env.ship.psi)
    env.tugs[0].nu = Vec3.zero()
    _park(env)

    comp = _comp(env)
    assert float(comp["corridor_gate"][0]) > 0.5
    assert float(comp["ship_soft_scale"][0]) < 0.5
    expected = 1.0 - (1.0 - env.cfg.reward_ship_soft_min_scale) * float(comp["corridor_gate"][0])
    assert float(comp["ship_soft_scale"][0]) == pytest.approx(expected, abs=1e-5)


def test_outside_corridor_keeps_full_ship_soft_scale() -> None:
    env = _env()
    env.ship.x = 0.0
    env.ship.y = 0.0
    env.ship.psi = 0.0
    slots = env.ship.slot_positions_world()
    sx, sy = float(slots[0, 0]), float(slots[0, 1])
    ang = math.atan2(sy, sx)
    perp = ang + math.pi / 2
    env.tugs[0].eta = Vec3(sx + 80.0 * math.cos(perp), sy + 80.0 * math.sin(perp), 0.0)
    env.tugs[0].nu = Vec3.zero()
    _park(env)

    comp = _comp(env)
    assert float(comp["corridor_gate"][0]) < 0.1
    assert float(comp["ship_soft_scale"][0]) > 0.95


def test_tug_collision_not_softened_by_corridor() -> None:
    env = _env()
    env.ship.x = 0.0
    env.ship.y = 0.0
    env.ship.psi = 0.0
    slots = env.ship.slot_positions_world()
    sx, sy = float(slots[0, 0]), float(slots[0, 1])
    # Radial near slot (corridor) with a neighbor within tug soft radius.
    ang = math.atan2(sy, sx)
    env.tugs[0].eta = Vec3(sx + 60.0 * math.cos(ang), sy + 60.0 * math.sin(ang), 0.0)
    env.tugs[0].nu = Vec3.zero()
    env.tugs[1].eta = Vec3(
        env.tugs[0].eta.x + 50.0 * math.cos(ang + math.pi / 2),
        env.tugs[0].eta.y + 50.0 * math.sin(ang + math.pi / 2),
        0.0,
    )
    env.tugs[1].nu = Vec3.zero()
    env.tugs[2].eta = Vec3(900.0, 900.0, 0.0)
    env.tugs[2].nu = Vec3.zero()
    env.tugs[3].eta = Vec3(950.0, 900.0, 0.0)
    env.tugs[3].nu = Vec3.zero()

    comp = _comp(env)
    assert float(comp["p_tug_collision"][0]) > 0.0
    assert float(comp["ship_soft_scale"][0]) <= 1.0
    # Soft scale must not zero out tug risk even when corridor is active.
    if float(comp["corridor_gate"][0]) > 0.5:
        assert float(comp["p_tug_collision"][0]) > 0.0
