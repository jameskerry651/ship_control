"""L1: observer / global_state after GPU dynamics match CPU env."""

from __future__ import annotations

import numpy as np
import torch

from config import EnvConfig
from env.formation_env import FormationEnv
from env.gpu.observer import build_obs_for_env, get_global_state_for_env
from env.gpu.state import GpuEnvBatch, pull_env_to_gpu, push_env_from_gpu
from env.observer import Observer
from physics.batched.ship import step_ships
from physics.batched.tugboat import set_control_commands, step_tugs


def test_obs_and_global_state_match():
    cfg = EnvConfig()
    cpu = FormationEnv(cfg=cfg, seed=5)
    gpu_env = FormationEnv(cfg=cfg, seed=5)
    cpu.reset()
    gpu_env.reset()

    batch = GpuEnvBatch.create(1, cfg.n_tugs, torch.device("cpu"), torch.float64)
    pull_env_to_gpu(gpu_env, batch, 0)
    actions = np.zeros((cfg.n_tugs, 4), dtype=np.float32)
    actions[:, 0] = 0.3

    obs_c, _, _, _ = cpu.step(actions)

    prev_nu = np.asarray([[t.nu.x, t.nu.y, t.nu.z] for t in gpu_env.tugs], dtype=np.float32)
    act_t = torch.as_tensor(actions, dtype=torch.float64).unsqueeze(0)
    set_control_commands(batch.tugs, act_t, batch.tug_params)
    dt = cfg.dt_ctrl
    step_tugs(batch.tugs, batch.tug_params, dt)
    step_ships(
        batch.ships,
        dt,
        speed_min=cfg.ship_speed_min,
        speed_max=cfg.ship_speed_max,
        speed_tau=cfg.ship_speed_tau_s,
        target_resample_min_s=cfg.ship_target_resample_min_s,
        target_resample_max_s=cfg.ship_target_resample_max_s,
    )
    push_env_from_gpu(gpu_env, batch, 0)
    gpu_env.step_count += 1
    gpu_env.last_actions = actions.copy()
    Observer.append_obs_history(
        gpu_env.motion_history, gpu_env.action_history, gpu_env.tugs, actions, prev_nu
    )
    # Update prev_* like FormationEnv.step (needed for consistency of later steps only)
    obs_g = build_obs_for_env(gpu_env)
    gs_g = get_global_state_for_env(gpu_env)
    gs_c = cpu.get_global_state()

    np.testing.assert_allclose(obs_g, obs_c, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(gs_g, gs_c, rtol=1e-5, atol=1e-5)
