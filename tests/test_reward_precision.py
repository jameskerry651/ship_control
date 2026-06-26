import numpy as np

from config import EnvConfig
from env.formation_env import FormationEnv


def _precision_components(weight: float) -> np.ndarray:
    cfg = EnvConfig()
    cfg.reward_precision_w = weight
    cfg.reward_precision_scale_m = 100.0
    env = FormationEnv(cfg=cfg, seed=123)
    env.reset()
    actions = np.zeros((cfg.n_tugs, env.action_dim), dtype=np.float32)
    _, info = env._reward.compute_rewards(actions, env.ship.slot_positions_world())
    return info["reward_components"]["r_precision"]


def _near_hold_components(weight: float) -> np.ndarray:
    cfg = EnvConfig()
    cfg.reward_near_hold_w = weight
    cfg.reward_near_hold_scale_m = 100.0
    env = FormationEnv(cfg=cfg, seed=123)
    env.reset()
    actions = np.zeros((cfg.n_tugs, env.action_dim), dtype=np.float32)
    _, info = env._reward.compute_rewards(actions, env.ship.slot_positions_world())
    return info["reward_components"]["r_near_hold"]


def test_precision_reward_is_default_off() -> None:
    np.testing.assert_allclose(_precision_components(0.0), 0.0)


def test_precision_reward_can_be_enabled() -> None:
    precision = _precision_components(0.5)
    assert np.all(precision >= 0.0)
    assert float(np.max(precision)) > 0.0


def test_near_hold_reward_is_default_off() -> None:
    np.testing.assert_allclose(_near_hold_components(0.0), 0.0)


def test_near_hold_reward_can_be_enabled() -> None:
    near_hold = _near_hold_components(0.5)
    assert np.all(near_hold >= 0.0)
    assert float(np.max(near_hold)) > 0.0
