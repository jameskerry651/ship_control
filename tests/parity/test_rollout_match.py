"""L2: short CudaVecEnv trajectories match Sync-style FormationEnv."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from config import EnvConfig
from env.formation_env import FormationEnv
from env.gpu.state import pull_env_to_gpu
from env.gpu.vec_env import CudaVecEnv
from physics.tugboat_dynamics_model import Vec3

RTOL64 = 1e-7
ATOL64 = 1e-7


def _assert_cuda_obs_close(
    actual: np.ndarray,
    expected: np.ndarray,
    gpu: CudaVecEnv,
    *,
    err_msg: str,
) -> None:
    """Compare stable features strictly and ill-conditioned TCPA separately."""
    tcpa_columns = np.asarray(
        [
            gpu.observation_spec.neighbor_item_slice(i).start + 8
            for i in range(gpu.observation_spec.neighbor_count)
        ],
        dtype=np.int64,
    )
    stable_columns = np.ones(gpu.observation_spec.total_dim, dtype=bool)
    stable_columns[tcpa_columns] = False
    np.testing.assert_allclose(
        actual[..., stable_columns],
        expected[..., stable_columns],
        rtol=1e-4,
        atol=1e-4,
        err_msg=err_msg,
    )
    np.testing.assert_allclose(
        actual[..., tcpa_columns],
        expected[..., tcpa_columns],
        rtol=1e-4,
        atol=5e-2,
        err_msg=f"{err_msg}:neighbor_tcpa",
    )


def _cpu_vec_step(envs: list[FormationEnv], actions: np.ndarray):
    """Minimal sync step matching SyncVecEnv semantics (no auto stats)."""
    obs, rews, dones, infos = [], [], [], []
    for i, env in enumerate(envs):
        o, r, d, info = env.step(actions[i])
        done = bool(d.any())
        if done:
            o = env.reset()
        obs.append(o)
        rews.append(r)
        dones.append(done)
        infos.append(info)
    return np.stack(obs), np.stack(rews), np.array(dones), infos


def test_cuda_vec_env_matches_cpu_n1_float64():
    cfg = EnvConfig()
    n_envs = 1
    seed = 42
    cpu_envs = [FormationEnv(cfg=cfg, seed=seed + i) for i in range(n_envs)]
    gpu = CudaVecEnv(cfg, n_envs=n_envs, base_seed=seed, device="cpu", dtype=torch.float64)

    for e in cpu_envs:
        e.reset()
    obs_g = gpu.reset()
    obs_c = np.stack([e._build_obs() for e in cpu_envs])
    # After reset both should match
    np.testing.assert_allclose(obs_g, obs_c, rtol=RTOL64, atol=ATOL64)

    rng = np.random.default_rng(0)
    for t in range(80):
        actions = rng.uniform(-1, 1, size=(n_envs, cfg.n_tugs, 4)).astype(np.float32)
        obs_c, rew_c, done_c, info_c = _cpu_vec_step(cpu_envs, actions)
        out = gpu.step(actions)
        obs_g, rew_g, done_g = out[0], out[1], out[2]
        info_g = out[3]

        np.testing.assert_allclose(rew_g, rew_c, rtol=RTOL64, atol=ATOL64, err_msg=f"rew@{t}")
        assert np.array_equal(done_g, done_c), f"done mismatch at t={t}"
        for i in range(n_envs):
            assert info_g[i]["collision"] == info_c[i]["collision"]
            assert info_g[i]["success"] == info_c[i]["success"]
            assert info_g[i]["phase"] == info_c[i]["phase"]
        # obs after auto-reset can diverge if reset RNG streams diverge; when not done, match
        if not done_c.any():
            np.testing.assert_allclose(obs_g, obs_c, rtol=RTOL64, atol=ATOL64, err_msg=f"obs@{t}")


def test_cuda_vec_env_matches_cpu_n4_events():
    cfg = EnvConfig()
    n_envs = 4
    seed = 3
    cpu_envs = [FormationEnv(cfg=cfg, seed=seed + i) for i in range(n_envs)]
    gpu = CudaVecEnv(cfg, n_envs=n_envs, base_seed=seed, device="cpu", dtype=torch.float64)
    for e in cpu_envs:
        e.reset()
    gpu.reset()

    rng = np.random.default_rng(1)
    for t in range(40):
        actions = rng.uniform(-1, 1, size=(n_envs, cfg.n_tugs, 4)).astype(np.float32)
        _, rew_c, done_c, info_c = _cpu_vec_step(cpu_envs, actions)
        _, rew_g, done_g, info_g, *_ = gpu.step(actions)
        np.testing.assert_allclose(rew_g, rew_c, rtol=1e-6, atol=1e-6, err_msg=f"rew@{t}")
        assert np.array_equal(done_g, done_c), f"done@{t}"
        for i in range(n_envs):
            assert info_g[i]["collision"] == info_c[i]["collision"]
            assert info_g[i]["success"] == info_c[i]["success"]


def test_cuda_vec_env_uses_device_batched_episode_state():
    """The default path must not retain per-environment Python step state."""
    gpu = CudaVecEnv(EnvConfig(), n_envs=2, base_seed=7, device="cpu", dtype=torch.float64)
    gpu.reset()

    assert gpu._fast.episode.prev_dist.shape == (2, 4)
    assert gpu._fast.episode.prev_dist.device.type == "cpu"


def test_cuda_fast_path_matches_forced_collision_event():
    """The float32 fast path preserves cooperative terminal collision events."""
    cfg = EnvConfig()
    cpu = FormationEnv(cfg=cfg, seed=19)
    fast = CudaVecEnv(cfg, n_envs=1, base_seed=19, device="cpu", dtype=torch.float32)
    cpu.reset()
    fast.reset()
    for env in (cpu, fast.envs[0]):
        tug = env.tugs[0]
        tug.set_state(
            Vec3(env.ship.x, env.ship.y, tug.eta.z),
            Vec3(tug.nu.x, tug.nu.y, tug.nu.z),
        )
    pull_env_to_gpu(fast.envs[0], fast.batch, 0)
    actions = np.zeros((1, cfg.n_tugs, 4), dtype=np.float32)

    _, _, done_cpu, info_cpu = cpu.step(actions[0])
    _, _, done_fast, info_fast, *_ = fast.step(actions)

    assert bool(done_fast[0]) == bool(done_cpu.any())
    assert info_fast[0]["collision"] == info_cpu["collision"]
    assert info_fast[0]["collision_kind"] == info_cpu["collision_kind"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_resample_preserves_cpu_rng_through_auto_reset():
    cfg = replace(
        EnvConfig(),
        ship_speed_min=0.6,
        ship_speed_max=1.4,
        max_episode_steps=1,
    )
    cpu = FormationEnv(cfg=cfg, seed=123)
    cuda = CudaVecEnv(
        cfg,
        n_envs=1,
        base_seed=123,
        device="cuda",
        dtype=torch.float32,
    )
    cpu.reset()
    cuda.reset()
    cpu.ship._time_to_resample = 0.1
    cuda.batch.ships.time_to_resample[0] = 0.1
    actions = np.zeros((1, cfg.n_tugs, 4), dtype=np.float32)

    obs_cpu_terminal, reward_cpu, done_cpu, _ = cpu.step(actions[0])
    global_cpu_terminal = cpu.get_global_state()
    obs_cpu_reset = cpu.reset()
    (
        obs_cuda_reset,
        reward_cuda,
        done_cuda,
        _,
        _,
        _,
        _,
        obs_cuda_terminal,
        global_cuda_terminal,
    ) = cuda.step(actions)

    assert bool(done_cpu.any())
    assert bool(done_cuda[0])
    np.testing.assert_allclose(reward_cuda[0], reward_cpu, rtol=1e-4, atol=1e-4)
    _assert_cuda_obs_close(
        obs_cuda_terminal,
        obs_cpu_terminal[None],
        cuda,
        err_msg="terminal_obs_after_resample",
    )
    np.testing.assert_allclose(
        global_cuda_terminal[0],
        global_cpu_terminal,
        rtol=1e-4,
        atol=1e-4,
    )
    np.testing.assert_array_equal(obs_cuda_reset[0], obs_cpu_reset)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("n_envs", (1, 4, 8), ids=("n1", "n4", "n8"))
def test_cuda_fast_rollout_matches_cpu(n_envs: int):
    cfg = EnvConfig()
    seed = 31
    cpu_envs = [FormationEnv(cfg=cfg, seed=seed + i) for i in range(n_envs)]
    cuda = CudaVecEnv(
        cfg,
        n_envs=n_envs,
        base_seed=seed,
        device="cuda",
        dtype=torch.float32,
    )
    obs_cpu = np.stack([env.reset() for env in cpu_envs])
    obs_cuda = cuda.reset()
    np.testing.assert_array_equal(obs_cuda, obs_cpu)
    rng = np.random.default_rng(2048 + n_envs)

    for step in range(40):
        actions = rng.uniform(
            -1,
            1,
            size=(n_envs, cfg.n_tugs, cuda.envs_probe.action_dim),
        ).astype(np.float32)
        obs_cpu, reward_cpu, done_cpu, info_cpu = _cpu_vec_step(cpu_envs, actions)
        obs_cuda, reward_cuda, done_cuda, info_cuda, *_ = cuda.step(actions)

        np.testing.assert_allclose(
            reward_cuda,
            reward_cpu,
            rtol=1e-4,
            atol=1e-4,
            err_msg=f"reward@{step}",
        )
        assert np.array_equal(done_cuda, done_cpu), f"done@{step}"
        _assert_cuda_obs_close(
            obs_cuda,
            obs_cpu,
            cuda,
            err_msg=f"obs@{step}",
        )
        np.testing.assert_allclose(
            cuda.get_global_state(),
            np.stack([env.get_global_state() for env in cpu_envs]),
            rtol=1e-4,
            atol=1e-4,
            err_msg=f"global@{step}",
        )
        for i in range(n_envs):
            for key in (
                "collision",
                "success",
                "phase",
                "terminated",
                "truncated",
            ):
                assert info_cuda[i][key] == info_cpu[i][key], f"{key}@{step}/env{i}"
