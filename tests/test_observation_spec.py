"""动态 ObservationSpec 的契约与 Actor 集成测试。"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from config import EnvConfig, PPOConfig
from env.formation_env import FormationEnv
from env.obs_spec import DEFAULT_OBSERVATION_SPEC, ObservationSpec
from rl.actor import MAPPOActor, TransformerMAPPOActor
from rl.ppo import MAPPOActorCritic
from scripts.train import _load_checkpoint, _save_ckpt


def test_default_observation_spec_includes_actual_thruster_state() -> None:
    spec = DEFAULT_OBSERVATION_SPEC

    assert spec.schema_version == 2
    assert spec.history_len == 4
    assert spec.preview_count == 3
    assert spec.neighbor_count == 3
    assert spec.thruster_state_dim == 4
    assert spec.own_dim == 63
    assert spec.attention_dim == 30
    assert spec.total_dim == 93

    slices = (
        spec.motion_history_slice,
        spec.action_history_slice,
        spec.ship_relative_slice,
        spec.ship_preview_slice,
        spec.slot_target_slice,
        spec.hull_clearance_slice,
        spec.thruster_state_slice,
        spec.neighbor_slice,
    )
    assert slices[0].start == 0
    assert all(left.stop == right.start for left, right in zip(slices, slices[1:]))
    assert slices[-1].stop == spec.total_dim
    assert ObservationSpec.from_dict(spec.to_dict()) == spec


def test_dynamic_spec_drives_environment_and_actor_shapes() -> None:
    cfg = EnvConfig(
        n_tugs=3,
        obs_history_k=0,
        obs_ship_preview_times_s=(2.0, 5.0),
    )
    env = FormationEnv(cfg=cfg, seed=7)
    spec = env.observation_spec
    obs = env.reset()

    assert spec.history_len == 1
    assert spec.preview_count == 2
    assert spec.neighbor_count == 2
    assert spec.own_dim == 31
    assert spec.total_dim == 51
    assert obs.shape == (cfg.n_tugs, spec.total_dim)
    commands = np.tile(
        np.asarray([1.0, -1.0, 1.0, -1.0], dtype=np.float32),
        (cfg.n_tugs, 1),
    )
    next_obs, rewards, dones, _ = env.step(commands)
    assert next_obs.shape == obs.shape
    assert rewards.shape == dones.shape == (cfg.n_tugs,)
    expected_actual = np.asarray([0.1, -0.1, 1.0 / 15.0, -1.0 / 15.0])
    np.testing.assert_allclose(
        next_obs[:, spec.thruster_state_slice],
        np.tile(expected_actual, (cfg.n_tugs, 1)),
        atol=1e-6,
    )
    assert not np.allclose(next_obs[:, spec.thruster_state_slice], commands)

    obs_t = torch.as_tensor(obs)
    for actor in (
        MAPPOActor(
            obs_dim=spec.total_dim,
            action_dim=env.action_dim,
            observation_spec=spec,
        ),
        TransformerMAPPOActor(
            obs_dim=spec.total_dim,
            action_dim=env.action_dim,
            observation_spec=spec.to_dict(),
        ),
    ):
        action, logprob, hidden = actor.act(obs_t)
        assert action.shape == (cfg.n_tugs, env.action_dim)
        assert logprob.shape == (cfg.n_tugs,)
        assert hidden is None
        assert actor.attention_weights(obs_t).shape == (
            cfg.n_tugs,
            spec.neighbor_count,
        )


def test_actor_rejects_observation_spec_dimension_mismatch() -> None:
    spec = DEFAULT_OBSERVATION_SPEC
    with pytest.raises(ValueError, match="expects obs_dim=93"):
        MAPPOActor(
            obs_dim=spec.total_dim - 1,
            action_dim=4,
            observation_spec=spec,
        )


def test_observation_spec_reports_semantic_differences() -> None:
    original = DEFAULT_OBSERVATION_SPEC
    changed = replace(original, history_len=1, preview_count=5)

    assert original.differences(changed) == {
        "history_len": (4, 1),
        "preview_count": (3, 5),
    }


def test_schema_v1_payload_migrates_to_legacy_89_dim_contract() -> None:
    payload = DEFAULT_OBSERVATION_SPEC.to_dict()
    payload["schema_version"] = 1
    payload.pop("thruster_state_dim")

    legacy = ObservationSpec.from_dict(payload)

    assert legacy.schema_version == 1
    assert legacy.thruster_state_dim == 0
    assert legacy.own_dim == 59
    assert legacy.total_dim == 89
    assert legacy.thruster_state_slice == slice(59, 59)
    assert MAPPOActor(obs_dim=89, action_dim=4).observation_spec == legacy


def test_checkpoint_persists_and_validates_observation_spec(tmp_path) -> None:
    env_cfg = EnvConfig()
    ppo_cfg = PPOConfig()
    spec = ObservationSpec.from_config(env_cfg)
    model = MAPPOActorCritic(
        obs_dim=spec.total_dim,
        action_dim=4,
        n_agents=env_cfg.n_tugs,
        global_state_dim=82,
        observation_spec=spec,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=ppo_cfg.learning_rate)
    path = tmp_path / "observation_spec.pt"

    _save_ckpt(
        path,
        model,
        optimizer,
        env_cfg,
        ppo_cfg,
        update=3,
        global_step=128,
        metric=0.5,
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    assert checkpoint["observation_spec"] == spec.to_dict()
    assert checkpoint["model_kwargs"]["observation_spec"] == spec.to_dict()

    changed = replace(spec, history_len=1)
    changed_model = MAPPOActorCritic(
        obs_dim=changed.total_dim,
        action_dim=4,
        n_agents=env_cfg.n_tugs,
        global_state_dim=82,
        observation_spec=changed,
    )
    changed_optimizer = torch.optim.Adam(
        changed_model.parameters(), lr=ppo_cfg.learning_rate
    )
    with pytest.raises(ValueError, match="ObservationSpec is incompatible"):
        _load_checkpoint(changed_model, changed_optimizer, checkpoint)


@pytest.mark.parametrize(
    ("actor_arch", "input_weight_key", "legacy_input_dim"),
    (
        ("mlp", "actor.own_encoder.0.weight", 59),
        ("transformer", "actor.context_encoder.0.weight", 19),
    ),
)
def test_v1_checkpoint_safely_appends_thruster_feedback_columns(
    actor_arch: str,
    input_weight_key: str,
    legacy_input_dim: int,
) -> None:
    current = DEFAULT_OBSERVATION_SPEC
    legacy = replace(current, schema_version=1, thruster_state_dim=0)
    common = {
        "action_dim": 4,
        "n_agents": 4,
        "global_state_dim": 82,
        "actor_arch": actor_arch,
    }
    old_model = MAPPOActorCritic(
        obs_dim=legacy.total_dim,
        observation_spec=legacy,
        **common,
    )
    with torch.no_grad():
        old_model.state_dict()[input_weight_key].fill_(1.0)
    checkpoint = {
        "algo": "mappo",
        "model": old_model.state_dict(),
        "model_kwargs": {"actor_arch": actor_arch},
        "observation_spec": legacy.to_dict(),
        "update": 7,
        "global_step": 256,
    }

    new_model = MAPPOActorCritic(
        obs_dim=current.total_dim,
        observation_spec=current,
        **common,
    )
    optimizer = torch.optim.Adam(new_model.parameters(), lr=3e-4)
    update, global_step, optimizer_loaded = _load_checkpoint(
        new_model, optimizer, checkpoint
    )
    migrated = new_model.state_dict()[input_weight_key]

    assert (update, global_step, optimizer_loaded) == (7, 256, False)
    torch.testing.assert_close(
        migrated[:, :legacy_input_dim],
        torch.ones_like(migrated[:, :legacy_input_dim]),
    )
    torch.testing.assert_close(
        migrated[:, legacy_input_dim:],
        torch.zeros_like(migrated[:, legacy_input_dim:]),
    )
