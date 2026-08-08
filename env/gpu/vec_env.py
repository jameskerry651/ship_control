"""CudaVecEnv: device-batched dynamics, rewards, observation and termination.

API matches ``SyncVecEnv`` in ``scripts/train.py`` so training can switch backends.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from config import EnvConfig
from env.formation_env import FormationEnv
from env.gpu.batched_step import FastBatchedStep
from env.gpu.reset import reset_env
from env.gpu.state import GpuEnvBatch, pull_env_to_gpu
from env.obs_spec import ObservationSpec


class _EnvDimProbe:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        global_state_dim: int,
        observation_spec: ObservationSpec,
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.global_state_dim = global_state_dim
        self.observation_spec = observation_spec


class CudaVecEnv:
    """Vectorized env with batched torch dynamics on ``device``."""

    def __init__(
        self,
        env_cfg: EnvConfig,
        n_envs: int,
        base_seed: int = 0,
        device: str | torch.device = "cuda",
        dtype: torch.dtype | None = None,
    ) -> None:
        self.env_cfg = env_cfg
        self.n_envs = int(n_envs)
        self.n_tugs = int(env_cfg.n_tugs)
        self.base_seed = int(base_seed)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CudaVecEnv requested CUDA but torch.cuda.is_available() is False")
        # Parity / CPU device defaults to float64; CUDA training uses float32.
        if dtype is None:
            dtype = torch.float64 if self.device.type == "cpu" else torch.float32
        self.dtype = dtype

        self.envs: list[FormationEnv] = [
            FormationEnv(cfg=env_cfg, seed=base_seed + i) for i in range(self.n_envs)
        ]
        self.batch = GpuEnvBatch.create(self.n_envs, self.n_tugs, self.device, self.dtype)
        self.episode_returns = np.zeros(self.n_envs, dtype=np.float64)
        self.episode_lengths = np.zeros(self.n_envs, dtype=np.int32)

        probe_env = self.envs[0]
        self.envs_probe = _EnvDimProbe(
            probe_env.obs_dim,
            probe_env.action_dim,
            probe_env.global_state_dim,
            probe_env.observation_spec,
        )
        # Compatibility with train.py reading vec_env.envs[0].*
        # Keep real FormationEnv list as self.envs.
        self._fast = FastBatchedStep(
            env_cfg, self.batch, self.n_envs, self.n_tugs, probe_env.observation_spec
        )

    @property
    def observation_spec(self) -> ObservationSpec:
        return self.envs[0].observation_spec

    def reset(self) -> np.ndarray:
        obs_list = []
        for i, env in enumerate(self.envs):
            obs_list.append(reset_env(env, self.batch, i, seed=None))
            self._fast.reset_from_env(env, i)
        self.episode_returns[:] = 0.0
        self.episode_lengths[:] = 0
        return np.stack(obs_list, axis=0)

    def reset_at(self, indices: np.ndarray | list[int], seeds: list[int]) -> list[np.ndarray]:
        """Reset a subset of envs with explicit seeds (for GPU eval scheduling)."""
        if len(indices) != len(seeds):
            raise ValueError("indices and seeds must have the same length")
        out: list[np.ndarray] = []
        for env_idx_raw, episode_seed in zip(indices, seeds):
            env_idx = int(env_idx_raw)
            obs = reset_env(self.envs[env_idx], self.batch, env_idx, seed=int(episode_seed))
            self._fast.reset_from_env(self.envs[env_idx], env_idx)
            self.episode_returns[env_idx] = 0.0
            self.episode_lengths[env_idx] = 0
            out.append(obs)
        return out

    def get_global_state(self) -> np.ndarray:
        return self._fast.build_global_state_batched().cpu().numpy()

    def get_global_state_tensor(self) -> torch.Tensor:
        return self._fast.build_global_state_batched()

    def step(self, actions: np.ndarray | torch.Tensor, *, auto_reset: bool = True):
        # The float64 CPU backend is an explicit oracle mode used by parity
        # tests.  It intentionally keeps the old scalar FormationEnv semantics;
        # CUDA (and float32 CPU experiments) always take the device fast path.
        if self.device.type == "cpu" and self.dtype == torch.float64:
            return self._step_parity_cpu(actions, auto_reset=auto_reset)
        if isinstance(actions, torch.Tensor):
            actions_t = actions.to(device=self.device, dtype=self.dtype).clamp(-1.0, 1.0)
        else:
            actions_t = torch.as_tensor(actions, device=self.device, dtype=self.dtype).clamp(-1.0, 1.0)
        expected = (self.n_envs, self.n_tugs, self.envs_probe.action_dim)
        if tuple(actions_t.shape) != expected:
            raise ValueError(f"actions must have shape {expected}, got {tuple(actions_t.shape)}")

        prev_nu = self._fast.dynamics_step(actions_t, rng_envs=self.envs)
        rewards, components, derived = self._fast.compute_rewards_batched(actions_t)
        term = self._fast.check_termination_batched(derived)
        self._fast.update_episode_state(actions_t, prev_nu, derived)
        obs_t = self._fast.build_obs_batched(derived)
        global_t = self._fast.build_global_state_batched()
        total_rewards = rewards + term["terminal_reward"]

        # One bulk conversion synchronizes GPU work; Python dicts are then just
        # API packaging and never participate in simulation calculations.
        all_obs = obs_t.cpu().numpy()
        all_rew = total_rewards.to(torch.float32).cpu().numpy()
        all_done = term["done"].cpu().numpy().astype(bool)
        terminal_obs_local = all_obs.copy()
        terminal_global = global_t.cpu().numpy()
        terminated_arr = term["terminated"].cpu().numpy().astype(bool)
        truncated_arr = term["truncated"].cpu().numpy().astype(bool)
        term_np = {key: value.detach().cpu().numpy() for key, value in term.items()}
        component_np = {key: value.detach().cpu().numpy() for key, value in components.items()}
        all_info: list[dict[str, Any]] = []
        ep_infos: list[dict] = []
        for i in range(self.n_envs):
            collision = bool(term_np["collision"][i])
            ship_collision = bool(term_np["ship_collision"][i])
            info: dict[str, Any] = {
                "reward_components": {key: value[i].copy() for key, value in component_np.items()},
                "success": bool(term_np["success"][i]),
                "capture": bool(term_np["capture"][i]),
                "just_captured": bool(term_np["just_captured"][i]),
                "phase": "track" if bool(term_np["capture"][i]) else "approach",
                "track_streak_steps": int(term_np["track_streak_steps"][i]),
                "track_steps_total": int(term_np["track_steps_total"][i]),
                "track_in_zone_ratio": float(term_np["track_in_zone_ratio"][i]),
                "collision": collision,
                "timeout": bool(term_np["timeout"][i]),
                "terminated": bool(term_np["terminated"][i]),
                "truncated": bool(term_np["truncated"][i]),
                "terminal_reward": term_np["terminal_reward"][i].astype(np.float32, copy=True),
            }
            if collision:
                if ship_collision:
                    info["collision_kind"] = "tug_vs_ship"
                    info["collision_tug"] = int(term_np["ship_culprit"][i])
                else:
                    info["collision_kind"] = "tug_vs_tug"
                    info["collision_pair"] = (
                        int(term_np["pair_a"][i]), int(term_np["pair_b"][i])
                    )
            all_info.append(info)
            self.episode_returns[i] += float(all_rew[i].mean())
            self.episode_lengths[i] += 1
            if all_done[i]:
                ep_infos.append(
                    {
                        "episode_return": float(self.episode_returns[i]),
                        "episode_length": int(self.episode_lengths[i]),
                        "success": bool(term_np["success"][i]),
                        "capture": bool(term_np["capture"][i]),
                        "track_in_zone_ratio": float(term_np["track_in_zone_ratio"][i]),
                        "collision": collision,
                        "timeout": bool(term_np["timeout"][i]),
                        "final_dist_mean": float(component_np["dist_to_slot"][i].mean()),
                    }
                )
                if auto_reset:
                    # CPU reset sampling is intentional; it is outside the hot path.
                    all_obs[i] = reset_env(self.envs[i], self.batch, i, seed=None)
                    self._fast.reset_from_env(self.envs[i], i)
                    self.episode_returns[i] = 0.0
                    self.episode_lengths[i] = 0

        return (
            all_obs,
            all_rew,
            all_done,
            all_info,
            ep_infos,
            terminated_arr,
            truncated_arr,
            terminal_obs_local,
            terminal_global,
        )

    def _step_parity_cpu(
        self, actions: np.ndarray | torch.Tensor, *, auto_reset: bool = True
    ):
        """Reference implementation retained for deterministic L2 parity."""
        actions_np = (
            actions.detach().cpu().numpy() if isinstance(actions, torch.Tensor) else np.asarray(actions)
        )
        actions_np = np.clip(actions_np.astype(np.float32, copy=False), -1.0, 1.0)
        obs_dim, gdim = self.envs_probe.obs_dim, self.envs_probe.global_state_dim
        all_obs = np.empty((self.n_envs, self.n_tugs, obs_dim), dtype=np.float32)
        all_rew = np.empty((self.n_envs, self.n_tugs), dtype=np.float32)
        all_done = np.zeros(self.n_envs, dtype=bool)
        terminal_obs_local = np.zeros_like(all_obs)
        terminal_global = np.zeros((self.n_envs, gdim), dtype=np.float32)
        terminated_arr = np.zeros(self.n_envs, dtype=bool)
        truncated_arr = np.zeros(self.n_envs, dtype=bool)
        all_info: list[dict[str, Any]] = []
        ep_infos: list[dict] = []
        for i, env in enumerate(self.envs):
            obs, reward, dones, info = env.step(actions_np[i])
            done = bool(dones.any())
            self.episode_returns[i] += float(reward.mean())
            self.episode_lengths[i] += 1
            terminated_arr[i] = bool(info.get("terminated", False))
            truncated_arr[i] = bool(info.get("truncated", False))
            if done:
                ep_infos.append(
                    {
                        "episode_return": float(self.episode_returns[i]),
                        "episode_length": int(self.episode_lengths[i]),
                        "success": bool(info.get("success", False)),
                        "capture": bool(info.get("capture", False)),
                        "track_in_zone_ratio": float(info.get("track_in_zone_ratio", 0.0)),
                        "collision": bool(info.get("collision", False)),
                        "timeout": bool(info.get("timeout", False)),
                        "final_dist_mean": float(
                            info.get("reward_components", {}).get("dist_to_slot", np.array([np.nan])).mean()
                        ),
                    }
                )
                terminal_obs_local[i] = obs
                terminal_global[i] = env.get_global_state()
                if auto_reset:
                    obs = reset_env(env, self.batch, i, seed=None)
                    self._fast.reset_from_env(env, i)
                    self.episode_returns[i] = 0.0
                    self.episode_lengths[i] = 0
            else:
                pull_env_to_gpu(env, self.batch, i)
            all_obs[i], all_rew[i], all_done[i] = obs, reward, done
            all_info.append(info)
        return (
            all_obs, all_rew, all_done, all_info, ep_infos, terminated_arr,
            truncated_arr, terminal_obs_local, terminal_global,
        )

    def close(self) -> None:
        pass
