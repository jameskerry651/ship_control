"""Actor 架构工厂与 Transformer 前向形状烟测。"""

from __future__ import annotations

import pytest
import torch

from env.obs_spec import _OWN_OBS_DIM, _NEIGHBOR_COUNT, _NEIGHBOR_OBS_DIM
from rl.actor import MAPPOActor, TransformerMAPPOActor, build_actor
from rl.ppo import MAPPOActorCritic


OBS_DIM = _OWN_OBS_DIM + _NEIGHBOR_COUNT * _NEIGHBOR_OBS_DIM
ACTION_DIM = 4
N_AGENTS = 4
GLOBAL_STATE_DIM = 2 + 17 * N_AGENTS + 3 * N_AGENTS


def test_build_actor_mlp_and_transformer_shapes() -> None:
    batch = 5
    obs = torch.randn(batch, OBS_DIM)

    mlp = build_actor("mlp", OBS_DIM, ACTION_DIM)
    tf = build_actor("transformer", OBS_DIM, ACTION_DIM)

    assert isinstance(mlp, MAPPOActor)
    assert isinstance(tf, TransformerMAPPOActor)

    for actor in (mlp, tf):
        action, logprob, hidden = actor.act(obs, deterministic=False)
        assert action.shape == (batch, ACTION_DIM)
        assert logprob.shape == (batch,)
        assert hidden is None
        assert action.abs().max() <= 1.0 + 1e-5

        lp, ent = actor.evaluate_actions(obs, action.detach())
        assert lp.shape == (batch,)
        assert ent.shape == (batch,)
        assert torch.isfinite(lp).all()
        assert torch.isfinite(ent).all()

        mean = actor.policy(obs)
        assert mean.shape == (batch, ACTION_DIM)

        weights = actor.attention_weights(obs)
        assert weights.shape == (batch, _NEIGHBOR_COUNT)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(batch), atol=1e-5)


def test_mappo_actor_critic_transformer() -> None:
    model = MAPPOActorCritic(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        n_agents=N_AGENTS,
        global_state_dim=GLOBAL_STATE_DIM,
        actor_arch="transformer",
        hist_len=4,
    )
    assert model.actor_arch == "transformer"
    assert isinstance(model.actor, TransformerMAPPOActor)

    obs = torch.randn(3, OBS_DIM)
    gs = torch.randn(3, GLOBAL_STATE_DIM)
    action, logprob, _ = model.act(obs, deterministic=True)
    values = model.get_values(gs)
    assert action.shape == (3, ACTION_DIM)
    assert logprob.shape == (3,)
    assert values.shape == (3, N_AGENTS)
    assert torch.isfinite(values).all()


def test_build_actor_reserved_and_unknown() -> None:
    with pytest.raises(NotImplementedError, match="gru"):
        build_actor("gru", OBS_DIM, ACTION_DIM)
    with pytest.raises(NotImplementedError, match="lstm"):
        build_actor("lstm", OBS_DIM, ACTION_DIM)
    with pytest.raises(ValueError, match="unknown actor_arch"):
        build_actor("cnn", OBS_DIM, ACTION_DIM)
