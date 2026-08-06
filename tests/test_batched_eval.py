from __future__ import annotations

import numpy as np
import torch

from config import EnvConfig
from env.formation_env import ACTION_DIM, FormationEnv
from rl.ppo import MAPPOActorCritic
from scripts.train import evaluate_policy


def _make_model(env_cfg: EnvConfig) -> tuple[MAPPOActorCritic, dict]:
    probe = FormationEnv(cfg=env_cfg, seed=0)
    probe.reset()
    model_kwargs = {
        "obs_dim": probe.obs_dim,
        "action_dim": ACTION_DIM,
        "n_agents": env_cfg.n_tugs,
        "global_state_dim": probe.global_state_dim,
        "actor_arch": "mlp",
        "hist_len": probe.observation_spec.history_len,
        "observation_spec": probe.observation_spec.to_dict(),
    }
    return MAPPOActorCritic(**model_kwargs), model_kwargs


def test_batched_eval_matches_sequential_eval_and_restores_train_mode() -> None:
    env_cfg = EnvConfig(max_episode_steps=4)
    torch.manual_seed(7)
    model, model_kwargs = _make_model(env_cfg)
    model.train()

    sequential = evaluate_policy(
        model,
        env_cfg,
        n_episodes=4,
        device=torch.device("cpu"),
        seed=123,
        eval_workers=1,
        model_kwargs=model_kwargs,
    )
    batched = evaluate_policy(
        model,
        env_cfg,
        n_episodes=4,
        device=torch.device("cpu"),
        seed=123,
        eval_workers=2,
        model_kwargs=model_kwargs,
    )

    assert model.training
    assert sequential.keys() == batched.keys()
    for key in sequential:
        np.testing.assert_allclose(batched[key], sequential[key], rtol=1e-6, atol=1e-6)


def test_batched_eval_rejects_empty_episode_set() -> None:
    env_cfg = EnvConfig(max_episode_steps=1)
    model, model_kwargs = _make_model(env_cfg)

    with np.testing.assert_raises_regex(ValueError, "n_episodes must be positive"):
        evaluate_policy(
            model,
            env_cfg,
            n_episodes=0,
            device=torch.device("cpu"),
            eval_workers=2,
            model_kwargs=model_kwargs,
        )
