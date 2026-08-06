"""初始化圆环半径应遵循 EnvConfig.tug_init_radius_m。"""

from __future__ import annotations

import math

import numpy as np

from config import EnvConfig
from env.formation_env import FormationEnv


def _mean_ship_radius(env: FormationEnv) -> float:
    rs = [
        math.hypot(tug.eta.x - env.ship.x, tug.eta.y - env.ship.y)
        for tug in env.tugs
    ]
    return float(np.mean(rs))


def test_init_radius_100() -> None:
    cfg = EnvConfig(tug_init_radius_m=100.0)
    env = FormationEnv(cfg=cfg, seed=0)
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


def test_default_init_radius_is_100() -> None:
    assert EnvConfig().tug_init_radius_m == 100.0
