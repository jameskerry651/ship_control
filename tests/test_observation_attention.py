from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import EnvConfig
from env.formation_env import FormationEnv
from rl.actor import MAPPOActor


def test_attention_observation_shape_and_history_update() -> None:
    env = FormationEnv(cfg=EnvConfig(), seed=123)
    obs0 = env.reset()

    assert env.obs_dim == 71
    assert obs0.shape == (env.cfg.n_tugs, 71)
    assert np.isfinite(obs0).all()

    action = np.zeros((env.cfg.n_tugs, env.action_dim), dtype=np.float32)
    action[:, 0] = 0.25
    action[:, 1] = 0.25
    obs1, reward, done, info = env.step(action)

    assert obs1.shape == obs0.shape
    assert reward.shape == (env.cfg.n_tugs,)
    assert done.shape == (env.cfg.n_tugs,)
    assert isinstance(info, dict)
    assert np.isfinite(obs1).all()
    np.testing.assert_allclose(obs1[:, 28:32], action, atol=1e-6)


def test_attention_actor_forward_and_weights() -> None:
    actor = MAPPOActor(obs_dim=71, action_dim=4)
    obs = torch.zeros(8, 71)

    mean = actor.policy(obs)
    weights = actor.attention_weights(obs)
    action, logprob, aux = actor.act(obs)

    assert mean.shape == (8, 4)
    assert weights.shape == (8, 3)
    assert action.shape == (8, 4)
    assert logprob.shape == (8,)
    assert aux is None
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(8))


def test_mixed_initialization_covers_ready_counts_and_stays_safe() -> None:
    for ready_count in range(4):
        cfg = replace(
            EnvConfig(),
            tug_init_mode="mixed_slot_approach",
            tug_init_mixed_ready_counts=(ready_count,),
        )
        min_pair_dist = max(
            2.0 * cfg.tug_collision_dist_m,
            float(cfg.tug_init_mixed_pair_min_dist_m),
        )
        min_hull_dist = max(
            2.0 * cfg.ship_collision_dist_m,
            float(getattr(cfg, "ship_safety_dist_m", cfg.ship_collision_dist_m * 3.0)),
        )
        max_init_yaw_rate = max(
            float(cfg.tug_init_yaw_rate_noise_rads),
            float(cfg.tug_init_ready_yaw_rate_noise_rads),
        )

        for seed in range(6):
            env = FormationEnv(cfg=cfg, seed=1000 + 100 * ready_count + seed)
            env.reset()

            assert len(env._mixed_ready_tugs) == ready_count
            for tug in env.tugs:
                assert env.ship.distance_from_hull(tug.eta.x, tug.eta.y) >= min_hull_dist
                speed = math.hypot(tug.nu.x, tug.nu.y)
                assert 0.05 <= speed <= 4.0
                assert abs(tug.nu.y) <= 1.2
                assert abs(tug.nu.z) <= max_init_yaw_rate

                ctrl = tug.get_control_snapshot()
                assert 0.0 < ctrl["port_rpm_actual"] <= 0.35 * tug.rpm_limit
                assert 0.0 < ctrl["starboard_rpm_actual"] <= 0.35 * tug.rpm_limit
                assert abs(ctrl["port_azimuth_actual_deg"]) <= 1e-6
                assert abs(ctrl["starboard_azimuth_actual_deg"]) <= 1e-6

            for i in range(env.n_tugs):
                for j in range(i + 1, env.n_tugs):
                    dij = math.hypot(
                        env.tugs[i].eta.x - env.tugs[j].eta.x,
                        env.tugs[i].eta.y - env.tugs[j].eta.y,
                    )
                    assert dij >= min_pair_dist
