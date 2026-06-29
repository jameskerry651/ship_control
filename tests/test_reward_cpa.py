import numpy as np

from config import EnvConfig
from env.formation_env import FormationEnv
from physics.tugboat_dynamics_model import Vec3


def _make_env() -> FormationEnv:
    cfg = EnvConfig()
    cfg.reward_cpa_horizon_s = 60.0
    cfg.reward_collision_cpa_w = 2.0
    env = FormationEnv(cfg=cfg, seed=11)
    env.reset()
    return env


def _compute_reward_components(env: FormationEnv) -> dict:
    actions = np.zeros((env.cfg.n_tugs, 4), dtype=np.float32)
    _, info = env._compute_rewards(actions)
    return info["reward_components"]


def _park_unused_tugs(env: FormationEnv) -> None:
    env.tugs[2].eta = Vec3(500.0, 500.0, 0.0)
    env.tugs[2].nu = Vec3.zero()
    env.tugs[3].eta = Vec3(-500.0, 500.0, 0.0)
    env.tugs[3].nu = Vec3.zero()


def test_tug_cpa_penalizes_future_head_on_close_approach() -> None:
    env = _make_env()
    env.ship.x = 0.0
    env.ship.y = 1000.0
    env.ship.u = 0.0
    env.ship.v = 0.0

    env.tugs[0].eta = Vec3(-50.0, 0.0, 0.0)
    env.tugs[0].nu = Vec3(1.0, 0.0, 0.0)
    env.tugs[1].eta = Vec3(50.0, 0.0, 0.0)
    env.tugs[1].nu = Vec3(-1.0, 0.0, 0.0)
    _park_unused_tugs(env)

    comp = _compute_reward_components(env)

    assert comp["p_tug_cpa"][0] > 0.0
    assert comp["p_tug_cpa"][1] > 0.0
    assert comp["p_cpa_collision"][0] > 0.0
    assert np.isfinite(comp["p_collision"]).all()


def test_tug_cpa_ignores_separating_motion() -> None:
    env = _make_env()
    env.ship.x = 0.0
    env.ship.y = 1000.0
    env.ship.u = 0.0
    env.ship.v = 0.0

    env.tugs[0].eta = Vec3(-50.0, 0.0, 0.0)
    env.tugs[0].nu = Vec3(-1.0, 0.0, 0.0)
    env.tugs[1].eta = Vec3(50.0, 0.0, 0.0)
    env.tugs[1].nu = Vec3(1.0, 0.0, 0.0)
    _park_unused_tugs(env)

    comp = _compute_reward_components(env)

    assert comp["p_tug_cpa"][0] == 0.0
    assert comp["p_tug_cpa"][1] == 0.0


def test_ship_cpa_uses_future_hull_distance() -> None:
    env = _make_env()
    env.ship.x = 0.0
    env.ship.y = 0.0
    env.ship.psi = 0.0
    env.ship.u = 0.0
    env.ship.v = 0.0
    env.ship.r = 0.0

    env.tugs[0].eta = Vec3(0.0, 80.0, 0.0)
    env.tugs[0].nu = Vec3(0.0, -2.0, 0.0)
    env.tugs[1].eta = Vec3(400.0, 400.0, 0.0)
    env.tugs[1].nu = Vec3.zero()
    _park_unused_tugs(env)

    comp = _compute_reward_components(env)

    assert env.ship.distance_from_hull(env.tugs[0].eta.x, env.tugs[0].eta.y) > (
        env.cfg.reward_collision_ship_safe_m
    )
    assert comp["p_ship_cpa"][0] > 0.0
    assert comp["min_tcpa"][0] < env.cfg.reward_cpa_horizon_s
    assert np.isfinite(comp["p_collision"]).all()
