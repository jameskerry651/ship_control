"""Smoke: redesigned reward components are finite and keyed."""

import numpy as np

from config import EnvConfig
from env.formation_env import FormationEnv


REQUIRED = {
    "r_total", "r_dist", "r_hold", "p_collision", "p_stall",
    "corridor_gate", "ship_soft_scale", "stall_scale", "hold_gate",
}


def test_reward_redesign_smoke_step() -> None:
    env = FormationEnv(cfg=EnvConfig(), seed=0)
    obs = env.reset()
    assert obs is not None
    actions = np.zeros((env.cfg.n_tugs, 4), dtype=np.float32)
    for _ in range(5):
        obs, rew, done, info = env.step(actions)
        comp = info["reward_components"]
        for k in REQUIRED:
            assert k in comp
            assert np.isfinite(np.asarray(comp[k], dtype=np.float64)).all()
        assert np.isfinite(rew).all()
    assert env.cfg.ship_collision_dist_m == 6.0
    assert env.cfg.reward_arrival_bonus == 80.0
