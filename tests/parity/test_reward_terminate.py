"""L1: reward / termination on synced GPU→CPU state match FormationEnv.step."""

from __future__ import annotations

import numpy as np
import torch

from config import EnvConfig
from env.formation_env import FormationEnv
from env.gpu.state import GpuEnvBatch, pull_env_to_gpu, push_env_from_gpu
from env.gpu.reward import compute_rewards_for_env
from env.gpu.terminate import check_termination_for_env
from env.gpu.vec_env import CudaVecEnv
from physics.batched.ship import step_ships
from physics.batched.tugboat import set_control_commands, step_tugs

RTOL = 1e-10
ATOL = 1e-12


def test_reward_and_phase_match_after_gpu_dynamics():
    cfg = EnvConfig()
    cpu = FormationEnv(cfg=cfg, seed=7)
    gpu_env = FormationEnv(cfg=cfg, seed=7)
    obs_c = cpu.reset()
    obs_g = gpu_env.reset()
    np.testing.assert_allclose(obs_c, obs_g, rtol=RTOL, atol=ATOL)

    batch = GpuEnvBatch.create(1, cfg.n_tugs, torch.device("cpu"), torch.float64)
    pull_env_to_gpu(gpu_env, batch, 0)

    rng = np.random.default_rng(11)
    actions = rng.uniform(-1, 1, size=(cfg.n_tugs, 4)).astype(np.float32)

    # CPU full step
    obs_c, rew_c, done_c, info_c = cpu.step(actions)

    # GPU dynamics + oracle reward/terminate
    prev_nu = np.asarray(
        [[t.nu.x, t.nu.y, t.nu.z] for t in gpu_env.tugs], dtype=np.float32
    )
    act_t = torch.as_tensor(actions, dtype=torch.float64).unsqueeze(0)
    set_control_commands(batch.tugs, act_t, batch.tug_params)
    dt = cfg.dt_ctrl
    need = (batch.ships.time_to_resample - dt) <= 0.0
    u_s = t_s = None
    if bool(need.any()):
        u_s = torch.tensor(
            [float(gpu_env.ship.rng.uniform(gpu_env.ship.speed_min, gpu_env.ship.speed_max))],
            dtype=torch.float64,
        )
        t_s = torch.tensor(
            [
                float(
                    gpu_env.ship.rng.uniform(
                        gpu_env.ship.target_resample_min_s, gpu_env.ship.target_resample_max_s
                    )
                )
            ],
            dtype=torch.float64,
        )
    step_tugs(batch.tugs, batch.tug_params, dt)
    step_ships(
        batch.ships,
        dt,
        speed_min=cfg.ship_speed_min,
        speed_max=cfg.ship_speed_max,
        speed_tau=cfg.ship_speed_tau_s,
        target_resample_min_s=cfg.ship_target_resample_min_s,
        target_resample_max_s=cfg.ship_target_resample_max_s,
        u_target_samples=u_s,
        resample_interval_samples=t_s,
    )
    push_env_from_gpu(gpu_env, batch, 0)
    gpu_env.step_count += 1
    rew_g, info_g = compute_rewards_for_env(gpu_env, actions)
    dones_g, term_g = check_termination_for_env(gpu_env)

    np.testing.assert_allclose(rew_g, info_c["reward_components"]["r_total"], rtol=1e-6, atol=1e-6)
    # Compare dense reward path via components
    for key in (
        "r_total",
        "r_dist",
        "p_distance",
        "r_hold",
        "r_team",
        "p_collision",
        "corridor_gate",
        "ship_soft_scale",
    ):
        np.testing.assert_allclose(
            info_g["reward_components"][key],
            info_c["reward_components"][key],
            rtol=1e-5,
            atol=1e-5,
            err_msg=key,
        )
    assert bool(term_g.get("phase")) == bool(info_c.get("phase")) or term_g["phase"] == info_c["phase"]
    assert bool(dones_g.any()) == bool(done_c.any())
    assert term_g["collision"] == info_c["collision"]
    _ = prev_nu  # captured for API symmetry with vec_env


def test_fast_batched_tug_cpa_matches_cpu():
    """The fast reward path must use the CPU relative-vector convention."""
    cfg = EnvConfig()
    cpu = FormationEnv(cfg=cfg, seed=123)
    fast = CudaVecEnv(
        cfg,
        n_envs=1,
        base_seed=123,
        device="cpu",
        dtype=torch.float32,
    )
    cpu.reset()
    fast.reset()
    rng = np.random.default_rng(2026)

    for step in range(20):
        actions = rng.uniform(-1, 1, size=(8, cfg.n_tugs, 4)).astype(np.float32)[0]
        _, reward_cpu, _, info_cpu = cpu.step(actions)
        _, reward_fast, _, info_fast, *_ = fast.step(actions[None])

        np.testing.assert_allclose(
            info_fast[0]["reward_components"]["p_tug_collision"],
            info_cpu["reward_components"]["p_tug_collision"],
            rtol=1e-4,
            atol=1e-4,
            err_msg=f"p_tug_collision@{step}",
        )
        for key in ("r_dist", "p_distance", "r_hold", "r_team"):
            np.testing.assert_allclose(
                info_fast[0]["reward_components"][key],
                info_cpu["reward_components"][key],
                rtol=1e-4,
                atol=1e-4,
                err_msg=f"{key}@{step}",
            )
        np.testing.assert_allclose(
            reward_fast[0],
            reward_cpu,
            rtol=1e-4,
            atol=1e-4,
            err_msg=f"reward@{step}",
        )
