"""Stall penalty and stall_scale for reward farming suppression."""

from __future__ import annotations

import numpy as np

from config import EnvConfig
from env.formation_env import FormationEnv
from physics.tugboat_dynamics_model import Vec3


def _make() -> FormationEnv:
    cfg = EnvConfig()
    cfg.reward_stall_w = 0.5
    cfg.reward_stall_window_s = 1.0  # 5 steps at dt=0.2
    cfg.reward_stall_min_progress_m = 2.0
    cfg.reward_stall_floor = 0.2
    cfg.reward_shape_w = 0.0
    cfg.reward_team_w = 0.0
    cfg.reward_collision_w = 0.0  # isolate stall
    env = FormationEnv(cfg=cfg, seed=5)
    env.reset()
    return env


def test_stall_triggers_when_no_net_progress() -> None:
    env = _make()
    env.ship.x = 0.0
    env.ship.y = 0.0
    env.ship.u = 0.0
    slots = env.ship.slot_positions_world()
    sx, sy = float(slots[0, 0]), float(slots[0, 1])
    env.tugs[0].eta = Vec3(sx + 200.0, sy, 0.0)
    env.tugs[0].nu = Vec3.zero()
    for i in (1, 2, 3):
        env.tugs[i].eta = Vec3(800.0 + i * 40.0, 800.0, 0.0)
        env.tugs[i].nu = Vec3.zero()

    zero = np.zeros((env.cfg.n_tugs, 4), dtype=np.float32)
    last = None
    for _ in range(12):
        _, _, _, info = env.step(zero)
        last = info["reward_components"]
        env.tugs[0].eta = Vec3(sx + 200.0, sy, 0.0)
        env.tugs[0].nu = Vec3.zero()

    assert last is not None
    assert float(last["p_stall"][0]) > 0.5
    assert float(last["stall_scale"][0]) < 0.5


def test_stall_disabled_in_hold_region() -> None:
    env = _make()
    env.ship.x = 0.0
    env.ship.y = 0.0
    env.ship.u = 0.0
    slots = env.ship.slot_positions_world()
    sx, sy, spsi = float(slots[0, 0]), float(slots[0, 1]), float(slots[0, 2])
    env.tugs[0].eta = Vec3(sx + 5.0, sy, spsi)
    env.tugs[0].nu = Vec3(env.ship.u, 0.0, 0.0)
    for i in (1, 2, 3):
        env.tugs[i].eta = Vec3(800.0 + i * 40.0, 800.0, 0.0)
        env.tugs[i].nu = Vec3.zero()

    zero = np.zeros((env.cfg.n_tugs, 4), dtype=np.float32)
    last = None
    for _ in range(12):
        _, _, _, info = env.step(zero)
        last = info["reward_components"]
        env.tugs[0].eta = Vec3(sx + 5.0, sy, spsi)
        env.tugs[0].nu = Vec3(env.ship.u, 0.0, 0.0)

    assert last is not None
    assert float(last["hold_gate"][0]) > 0.9
    assert float(last["p_stall"][0]) == 0.0
    assert float(last["stall_scale"][0]) == 1.0
