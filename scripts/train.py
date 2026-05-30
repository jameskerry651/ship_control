"""MAPPO 多智能体拖轮编队训练脚本。

用法：
    python scripts/train.py --total-steps 5000000 --num-envs 8 --rollout-steps 256

特性：
- 去中心化 actor：每艘拖轮只看自己的局部观察
- 集中式 critic：使用 canonical global state，输出每艘拖轮的 value
- 向量化环境：SyncVecEnv（单进程）或 SubprocVecEnv（多进程 rollout）
- 控制台 + tensorboard 双日志
- 自动按 success 优先保存 best.pt；0% success 阶段用 final_dist/collision 预成功指标避免早停在远距离模型
- 周期性保存 last.pt（用于断点续训或观察当前训练状态）
- 学习率 CosineAnnealingLR 余弦退火（PyTorch lr_scheduler）
- Ctrl+C 安全退出，保存当前 last.pt
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from multiprocessing import Pipe, get_context
from pathlib import Path
from typing import Any, Literal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.utils.tensorboard import SummaryWriter


# ---------- 奖励运行时归一化（RunningMeanStd） ----------
class RewardNormalizer:
    """用运行均值和方差对奖励做归一化，稳定 value network 的训练目标。

    采用 Welford 在线算法，不需要存储历史数据。
    归一化后奖励 ≈ N(0, 1)，value loss 量级稳定，不受奖励函数绝对值影响。
    """

    def __init__(self, clip: float = 10.0, epsilon: float = 1e-8) -> None:
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        self.clip = clip
        self.epsilon = epsilon

    def update_and_normalize(self, rewards: np.ndarray) -> np.ndarray:
        """更新统计量并返回归一化后的奖励。"""
        batch = rewards.flatten()
        n = len(batch)
        if n == 0:
            return rewards
        batch_mean = float(batch.mean())
        batch_var = float(batch.var()) if n > 1 else 0.0
        # Welford 合并更新
        total = self.count + n
        delta = batch_mean - self.mean
        self.mean += delta * n / max(total, 1)
        self.var = (self.var * self.count + batch_var * n +
                    delta ** 2 * self.count * n / max(total, 1)) / max(total, 1)
        self.count = total
        # 归一化
        std = math.sqrt(max(self.var, self.epsilon))
        normed = (rewards - self.mean) / std
        return np.clip(normed, -self.clip, self.clip).astype(np.float32)


def _mean_var(arrays: list[np.ndarray]) -> tuple[float, float]:
    """Flatten rollout arrays and return population mean/variance."""
    if not arrays:
        return float("nan"), float("nan")
    flat = np.concatenate([np.asarray(a, dtype=np.float64).reshape(-1) for a in arrays])
    if flat.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(flat)), float(np.var(flat))


def _add_finite_scalar(writer: SummaryWriter, tag: str, value: float, step: int) -> None:
    if math.isfinite(float(value)):
        writer.add_scalar(tag, float(value), step)

from config import EnvConfig, PPOConfig
from env.formation_env import ACTION_DIM, FormationEnv
from rl.ppo import MAPPOActorCritic, MAPPORolloutBuffer, mappo_update


# ---------- 顺序执行的向量化环境 ----------
class SyncVecEnv:
    """把 N 个 FormationEnv 顺序串起来，提供与单环境相似的接口。

    step() 返回：
      obs:        (N, K, obs_dim)
      rewards:    (N, K)
      dones:      (N,)        每个环境是否已经在这一步结束
      infos:      list[dict]，长度 N
      ep_infos:   list[dict]，已完成 episode 的统计（不一定每步都有）
    """

    def __init__(self, env_cfg: EnvConfig, n_envs: int, base_seed: int = 0) -> None:
        self.envs: list[FormationEnv] = [
            FormationEnv(cfg=env_cfg, seed=base_seed + i) for i in range(n_envs)
        ]
        self.n_envs = n_envs
        self.n_tugs = env_cfg.n_tugs
        self.episode_returns = np.zeros(n_envs, dtype=np.float64)
        self.episode_lengths = np.zeros(n_envs, dtype=np.int32)

    def reset(self) -> np.ndarray:
        obs = np.stack([e.reset() for e in self.envs], axis=0)
        self.episode_returns[:] = 0.0
        self.episode_lengths[:] = 0
        return obs

    def step(self, actions: np.ndarray):
        all_obs, all_rew, all_done, all_info = [], [], [], []
        ep_infos: list[dict] = []
        # 在每个边界步上，记录 reset 之前的终态局部 obs 与全局 state，
        # 供 GAE 用 V(terminal_state_pre_reset) 做 truncated bootstrap。
        global_state_dim = self.envs[0].global_state_dim
        terminal_obs_local = np.zeros(
            (self.n_envs, self.n_tugs, self.envs[0].obs_dim), dtype=np.float32
        )
        terminal_global = np.zeros(
            (self.n_envs, global_state_dim), dtype=np.float32
        )
        terminated_arr = np.zeros(self.n_envs, dtype=bool)
        truncated_arr = np.zeros(self.n_envs, dtype=bool)
        for i, env in enumerate(self.envs):
            obs, rew, done, info = env.step(actions[i])
            self.episode_returns[i] += float(rew.mean())   # 取所有 agent 平均
            self.episode_lengths[i] += 1
            terminated_arr[i] = bool(info.get("terminated", False))
            truncated_arr[i] = bool(info.get("truncated", False))
            if done.any():
                ep_infos.append({
                    "episode_return": float(self.episode_returns[i]),
                    "episode_length": int(self.episode_lengths[i]),
                    "success": bool(info.get("success", False)),
                    "collision": bool(info.get("collision", False)),
                    "timeout": bool(info.get("timeout", False)),
                    "final_dist_mean": float(
                        info.get("reward_components", {}).get("dist_to_slot", np.array([np.nan])).mean()
                    ),
                })
                # 记录 reset 之前的终态 obs/global state，再做 reset。
                terminal_obs_local[i] = obs.astype(np.float32, copy=False)
                terminal_global[i] = env.get_global_state()
                obs = env.reset()
                self.episode_returns[i] = 0.0
                self.episode_lengths[i] = 0
            all_obs.append(obs)
            all_rew.append(rew)
            all_done.append(bool(done.any()))
            all_info.append(info)
        return (
            np.stack(all_obs, axis=0),
            np.stack(all_rew, axis=0),
            np.array(all_done, dtype=bool),
            all_info,
            ep_infos,
            terminated_arr,
            truncated_arr,
            terminal_obs_local,
            terminal_global,
        )

    def close(self) -> None:
        pass


class _EnvDimProbe:
    """仅暴露维度信息，供 train.py 读取 obs/action/global_state 大小。"""

    def __init__(self, obs_dim: int, action_dim: int, global_state_dim: int) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.global_state_dim = global_state_dim


def _subproc_env_worker(conn: Any, env_cfg: EnvConfig, seed: int) -> None:
    """子进程环境 worker；在 spawn 模式下于独立进程中 import FormationEnv。"""
    from env.formation_env import FormationEnv

    env = FormationEnv(cfg=env_cfg, seed=seed)
    try:
        while True:
            cmd, payload = conn.recv()
            if cmd == "close":
                conn.send(("ok", None))
                break
            if cmd == "reset":
                conn.send(("ok", env.reset()))
            elif cmd == "step":
                obs, rew, done, info = env.step(payload)
                conn.send(("ok", (obs, rew, done, info, env.get_global_state())))
            elif cmd == "get_global_state":
                conn.send(("ok", env.get_global_state()))
            else:
                conn.send(("err", f"unknown cmd: {cmd}"))
    except Exception:
        import traceback

        conn.send(("err", traceback.format_exc()))
        raise


def _subproc_recv(conn: Any) -> Any:
    status, payload = conn.recv()
    if status != "ok":
        raise RuntimeError(f"subproc env worker failed:\n{payload}")
    return payload


class SubprocVecEnv:
    """多进程并行 step 的向量化环境；接口与 SyncVecEnv 一致。"""

    def __init__(
        self,
        env_cfg: EnvConfig,
        n_envs: int,
        base_seed: int = 0,
        *,
        n_workers: int | None = None,
        start_method: str = "spawn",
    ) -> None:
        self.env_cfg = env_cfg
        self.n_envs = n_envs
        self.n_tugs = env_cfg.n_tugs
        self.n_workers = min(n_envs, n_workers or n_envs)
        self.episode_returns = np.zeros(n_envs, dtype=np.float64)
        self.episode_lengths = np.zeros(n_envs, dtype=np.int32)

        ctx = get_context(start_method)
        self._conns: list[Any] = []
        self._processes: list[Any] = []
        for i in range(n_envs):
            parent_conn, child_conn = Pipe()
            proc = ctx.Process(
                target=_subproc_env_worker,
                args=(child_conn, env_cfg, base_seed + i),
                daemon=True,
            )
            proc.start()
            child_conn.close()
            self._conns.append(parent_conn)
            self._processes.append(proc)

        obs0 = self.reset()
        gs0 = self._request_all("get_global_state")[0]
        self._obs_dim = int(obs0.shape[-1])
        self.envs = [
            _EnvDimProbe(self._obs_dim, ACTION_DIM, int(np.asarray(gs0).shape[0]))
        ]

    def _request_all(self, cmd: str, payloads: list[Any] | None = None) -> list[Any]:
        payloads = payloads if payloads is not None else [None] * self.n_envs
        for conn, payload in zip(self._conns, payloads):
            conn.send((cmd, payload))
        return [_subproc_recv(conn) for conn in self._conns]

    def reset(self) -> np.ndarray:
        obs_list = self._request_all("reset")
        self.episode_returns[:] = 0.0
        self.episode_lengths[:] = 0
        return np.stack(obs_list, axis=0)

    def step(self, actions: np.ndarray):
        results = self._request_all("step", [actions[i] for i in range(self.n_envs)])
        all_obs: list[np.ndarray] = []
        all_rew: list[np.ndarray] = []
        all_done: list[bool] = []
        all_info: list[dict] = []
        ep_infos: list[dict] = []
        global_state_dim = self.envs[0].global_state_dim
        terminal_obs_local = np.zeros(
            (self.n_envs, self.n_tugs, self._obs_dim), dtype=np.float32
        )
        terminal_global = np.zeros(
            (self.n_envs, global_state_dim), dtype=np.float32
        )
        terminated_arr = np.zeros(self.n_envs, dtype=bool)
        truncated_arr = np.zeros(self.n_envs, dtype=bool)
        reset_indices: list[int] = []

        for i, (obs, rew, done, info, global_state) in enumerate(results):
            self.episode_returns[i] += float(rew.mean())
            self.episode_lengths[i] += 1
            terminated_arr[i] = bool(info.get("terminated", False))
            truncated_arr[i] = bool(info.get("truncated", False))
            if done.any():
                ep_infos.append({
                    "episode_return": float(self.episode_returns[i]),
                    "episode_length": int(self.episode_lengths[i]),
                    "success": bool(info.get("success", False)),
                    "collision": bool(info.get("collision", False)),
                    "timeout": bool(info.get("timeout", False)),
                    "final_dist_mean": float(
                        info.get("reward_components", {}).get(
                            "dist_to_slot", np.array([np.nan])
                        ).mean()
                    ),
                })
                terminal_obs_local[i] = obs.astype(np.float32, copy=False)
                terminal_global[i] = global_state.astype(np.float32, copy=False)
                reset_indices.append(i)
            all_obs.append(obs)
            all_rew.append(rew)
            all_done.append(bool(done.any()))
            all_info.append(info)

        if reset_indices:
            for i in reset_indices:
                self._conns[i].send(("reset", None))
            for j, i in enumerate(reset_indices):
                all_obs[i] = _subproc_recv(self._conns[i])
                self.episode_returns[i] = 0.0
                self.episode_lengths[i] = 0

        return (
            np.stack(all_obs, axis=0),
            np.stack(all_rew, axis=0),
            np.array(all_done, dtype=bool),
            all_info,
            ep_infos,
            terminated_arr,
            truncated_arr,
            terminal_obs_local,
            terminal_global,
        )

    def close(self) -> None:
        for conn in self._conns:
            try:
                conn.send(("close", None))
                _subproc_recv(conn)
            except (BrokenPipeError, EOFError, OSError):
                pass
            finally:
                conn.close()
        for proc in self._processes:
            proc.join(timeout=1.0)
            if proc.is_alive():
                proc.terminate()


def make_vec_env(
    env_cfg: EnvConfig,
    n_envs: int,
    base_seed: int,
    backend: Literal["sync", "subproc"],
    env_workers: int | None,
) -> SyncVecEnv | SubprocVecEnv:
    if backend == "subproc":
        return SubprocVecEnv(
            env_cfg,
            n_envs=n_envs,
            base_seed=base_seed,
            n_workers=env_workers or n_envs,
        )
    if env_workers not in (None, 1):
        print("[warn] --env-workers 仅在 --env-backend subproc 时生效，已忽略。")
    return SyncVecEnv(env_cfg, n_envs=n_envs, base_seed=base_seed)


# ---------- 评估循环（确定性策略，跑若干 episode） ----------
def _build_global_state(vec_env: SyncVecEnv | SubprocVecEnv) -> np.ndarray:
    """收集每个环境的 canonical global state，形状 (N, global_state_dim)。"""
    if isinstance(vec_env, SubprocVecEnv):
        states = vec_env._request_all("get_global_state")
        return np.stack(states, axis=0).astype(np.float32, copy=False)
    return np.stack([e.get_global_state() for e in vec_env.envs], axis=0).astype(
        np.float32, copy=False
    )


def _make_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    ppo_cfg: PPOConfig,
    n_updates: int,
) -> LRScheduler | None:
    """CosineAnnealingLR：从 learning_rate 余弦退火到 eta_min，按 PPO update 步进。"""
    if not ppo_cfg.lr_anneal:
        return None
    return CosineAnnealingLR(
        optimizer,
        T_max=max(1, n_updates - 1),
        eta_min=ppo_cfg.learning_rate * ppo_cfg.lr_min_factor,
    )


def _load_lr_scheduler(
    scheduler: LRScheduler | None,
    ckpt: dict,
    *,
    start_update: int,
    restore_from_ckpt: bool,
) -> None:
    if scheduler is None:
        return
    state = ckpt.get("lr_scheduler")
    sched_type = ckpt.get("lr_scheduler_type")
    expected_type = type(scheduler).__name__
    if restore_from_ckpt and state is not None and sched_type in (None, expected_type):
        try:
            scheduler.load_state_dict(state)
            return
        except (RuntimeError, ValueError):
            print(
                f"[resume] lr_scheduler state incompatible with {expected_type}; "
                f"fast-forward {start_update} steps."
            )
    elif restore_from_ckpt and state is not None and sched_type != expected_type:
        print(
            f"[resume] lr_scheduler type changed ({sched_type} -> {expected_type}); "
            f"fast-forward {start_update} steps."
        )
    if start_update > 0:
        reason = "legacy checkpoint" if state is None else "optimizer reset"
        print(f"[resume] lr_scheduler fast-forward {start_update} steps ({reason}).")
        for _ in range(start_update):
            scheduler.step()


def _load_checkpoint(
    model: MAPPOActorCritic,
    optimizer: torch.optim.Optimizer,
    ckpt: dict,
    reset_progress: bool = False,
) -> tuple[int, int, bool]:
    """加载 MAPPO checkpoint；输入维度增长时迁移首层权重。

    Returns:
        (start_update, global_step, optimizer_loaded)
    """
    algo = str(ckpt.get("algo", "")).lower()
    if algo != "mappo":
        raise ValueError(
            f"unsupported checkpoint algo={algo!r}; only MAPPO checkpoints are supported"
        )
    src_state = ckpt["model"]
    dst_state = model.state_dict()
    load_state: dict[str, torch.Tensor] = {}
    migrated: list[str] = []
    skipped: list[str] = []
    actor_arch_changed = "actor.own_encoder.0.weight" not in src_state

    for key, dst_tensor in dst_state.items():
        if actor_arch_changed and key.startswith("actor.") and key != "actor.log_std":
            skipped.append(f"{key}: initialized for attention actor")
            continue
        src_tensor = src_state.get(key)
        if src_tensor is None:
            skipped.append(f"{key}: missing in checkpoint")
            continue
        if tuple(src_tensor.shape) == tuple(dst_tensor.shape):
            load_state[key] = src_tensor.to(device=dst_tensor.device, dtype=dst_tensor.dtype)
            continue

        # 观测/global-state 追加特征时，只扩展 critic 第一层输入列。
        # Actor 已切换为 attention 架构；旧 MLP actor 参数按 shape mismatch 跳过。
        if (
            key in ("critic.0.weight", "critic.critic.0.weight")
            and src_tensor.ndim == 2
            and dst_tensor.ndim == 2
            and src_tensor.shape[0] == dst_tensor.shape[0]
        ):
            value = torch.zeros_like(dst_tensor)
            cols = min(int(src_tensor.shape[1]), int(dst_tensor.shape[1]))
            value[:, :cols] = src_tensor[:, :cols].to(
                device=dst_tensor.device, dtype=dst_tensor.dtype
            )
            load_state[key] = value
            migrated.append(
                f"{key}: {tuple(src_tensor.shape)} -> {tuple(dst_tensor.shape)}"
            )
            continue

        skipped.append(f"{key}: {tuple(src_tensor.shape)} -> {tuple(dst_tensor.shape)}")

    model.load_state_dict(load_state, strict=False)
    if migrated:
        print("[resume] migrated input weights: " + "; ".join(migrated))
    if skipped:
        print("[resume] skipped shape-mismatched tensors: " + "; ".join(skipped))
    if reset_progress:
        print("[resume] loaded MAPPO model weights only; optimizer/progress reset.")
        return 0, 0, False
    if migrated or skipped:
        print("[resume] optimizer state reset because model input shape changed.")
        return int(ckpt.get("update", 0)), int(ckpt.get("global_step", 0)), False
    optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt.get("update", 0)), int(ckpt.get("global_step", 0)), True


def _run_eval_episode(
    episode_seed: int,
    env_cfg: EnvConfig,
    state_dict: dict[str, torch.Tensor],
    model_kwargs: dict[str, Any],
    device_str: str,
) -> tuple[float, int, bool, bool, float]:
    """单回合评估（供多进程 eval 调用）。"""
    device = torch.device(device_str)
    model = MAPPOActorCritic(**model_kwargs)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    env = FormationEnv(cfg=env_cfg, seed=episode_seed)
    obs = env.reset()
    done_any = False
    ep_ret = 0.0
    ep_len = 0
    info_last: dict = {}
    while not done_any and ep_len < env_cfg.max_episode_steps:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            action, _, _ = model.act(obs_t, deterministic=True)
        actions_np = action.cpu().numpy()
        obs, rew, done, info = env.step(actions_np)
        ep_ret += float(rew.mean())
        ep_len += 1
        done_any = bool(done.any())
        info_last = info
    final_dist = float("nan")
    comp = info_last.get("reward_components", {}) if info_last else {}
    dist_arr = comp.get("dist_to_slot") if isinstance(comp, dict) else None
    if dist_arr is not None:
        final_dist = float(np.nanmean(dist_arr))
    else:
        slot_world = env.ship.slot_positions_world()
        dists = []
        for i, tug in enumerate(env.tugs):
            slot = slot_world[env.tug_to_slot[i]]
            dists.append(math.hypot(tug.eta.x - slot[0], tug.eta.y - slot[1]))
        final_dist = float(np.mean(dists))
    return (
        ep_ret,
        ep_len,
        bool(info_last.get("success", False)),
        bool(info_last.get("collision", False)),
        final_dist,
    )


def _run_eval_episode_task(
    task: tuple[int, EnvConfig, dict[str, torch.Tensor], dict[str, Any], str],
) -> tuple[float, int, bool, bool, float]:
    """ProcessPoolExecutor.map 的单参数包装。"""
    return _run_eval_episode(*task)


def evaluate_policy(
    model: MAPPOActorCritic,
    env_cfg: EnvConfig,
    n_episodes: int,
    device: torch.device,
    seed: int = 12345,
    *,
    eval_workers: int = 1,
    model_kwargs: dict[str, Any] | None = None,
) -> dict[str, float]:
    if model_kwargs is None:
        raise ValueError("model_kwargs is required for evaluate_policy")

    if eval_workers <= 1:
        results = [
            _run_eval_episode(seed + i, env_cfg, model.state_dict(), model_kwargs, str(device))
            for i in range(n_episodes)
        ]
    else:
        state_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        tasks = [
            (seed + i, env_cfg, state_cpu, model_kwargs, str(device))
            for i in range(n_episodes)
        ]
        results = []
        with ProcessPoolExecutor(max_workers=eval_workers) as pool:
            results = list(pool.map(_run_eval_episode_task, tasks, chunksize=1))

    returns = [r[0] for r in results]
    lengths = [r[1] for r in results]
    succ = [r[2] for r in results]
    coll = [r[3] for r in results]
    final_dists = [r[4] for r in results]
    return {
        "eval/return_mean": float(np.mean(returns)),
        "eval/return_std": float(np.std(returns)),
        "eval/length_mean": float(np.mean(lengths)),
        "eval/success_rate": float(np.mean(succ)),
        "eval/collision_rate": float(np.mean(coll)),
        "eval/final_dist_mean": float(np.mean(final_dists)),
        "eval/final_dist_std": float(np.std(final_dists)),
    }


# ---------- 主训练入口 ----------
def main() -> None:
    parser = argparse.ArgumentParser(description="多智能体拖轮编队 MAPPO 训练")
    parser.add_argument("--total-steps", type=int, default=PPOConfig.total_steps,
                        help="总环境步数（含所有 envs 与所有 tugs）")
    parser.add_argument("--num-envs", type=int, default=PPOConfig.num_envs)
    parser.add_argument("--rollout-steps", type=int, default=PPOConfig.rollout_steps)
    parser.add_argument("--minibatch-size", type=int, default=PPOConfig.minibatch_size)
    parser.add_argument("--update-epochs", type=int, default=PPOConfig.update_epochs)
    parser.add_argument("--learning-rate", type=float, default=PPOConfig.learning_rate)
    parser.add_argument("--seed", type=int, default=PPOConfig.seed)
    parser.add_argument("--device", type=str, default=PPOConfig.device,
                        choices=["cpu", "cuda", "mps"])
    parser.add_argument("--run-name", type=str, default=None,
                        help="本次训练运行名（默认用时间戳）")
    parser.add_argument("--logdir", type=str, default="runs",
                        help="tensorboard 日志根目录")
    parser.add_argument("--ckptdir", type=str, default="checkpoints",
                        help="模型权重保存根目录")
    parser.add_argument("--resume", type=str, default=None,
                        help="从已有 .pt 续训")
    parser.add_argument("--reset-progress", action="store_true",
                        help="仅加载 --resume 的模型权重，重置 optimizer、update 和 global_step")
    parser.add_argument("--init-mode", type=str, default=None,
                        choices=["mixed_slot_approach"],
                        help="拖轮初始场景（当前仅支持 mixed_slot_approach；默认使用 config.py）")
    parser.add_argument("--no-ship-size-randomize", action="store_true",
                        help="关闭 v36 大船长宽随机化，使用 config.py 中的固定 ship_length_m/ship_beam_m")
    parser.add_argument("--hold-time", type=float, default=None,
                        help="覆盖 EnvConfig.hold_time_s（课程学习用：1.0 → 5.0 → 10.0）")
    parser.add_argument("--critic-warmup-updates", type=int, default=0,
                        help=("warm-start 时前 N 个 update 只训 critic（policy_coef=0），"
                              "用于 critic 重置或维度变化后先把 EV 拉起来再放开 actor。"))
    parser.add_argument("--env-backend", type=str, default="subproc",
                        choices=["sync", "subproc"],
                        help="rollout 环境后端：subproc 多进程并行 step，sync 单进程顺序")
    parser.add_argument("--env-workers", type=int, default=None,
                        help="subproc 环境进程数（默认等于 --num-envs）")
    parser.add_argument("--eval-workers", type=int, default=0,
                        help="评估并行进程数，0 表示自动取 min(8, CPU-1)")
    parser.add_argument("--torch-threads", type=int, default=0,
                        help="PyTorch CPU 算子线程数，0 表示不修改默认")
    args = parser.parse_args()

    n_cpu = os.cpu_count() or 1
    eval_workers = int(args.eval_workers)
    if eval_workers <= 0:
        eval_workers = max(1, min(8, n_cpu - 1))
    if args.torch_threads > 0:
        torch.set_num_threads(int(args.torch_threads))

    # 配置对象
    env_cfg = EnvConfig()
    if args.init_mode is not None:
        env_cfg.tug_init_mode = args.init_mode
    if args.no_ship_size_randomize:
        env_cfg.ship_size_randomize = False
    if args.hold_time is not None:
        env_cfg.hold_time_s = float(args.hold_time)
    ppo_cfg = PPOConfig(
        total_steps=args.total_steps,
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        minibatch_size=args.minibatch_size,
        update_epochs=args.update_epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
    )

    # 随机种子
    np.random.seed(ppo_cfg.seed)
    torch.manual_seed(ppo_cfg.seed)

    # 设备
    device = torch.device(ppo_cfg.device)

    # 运行名与目录
    run_name = args.run_name or time.strftime("mappo_tug_%Y%m%d_%H%M%S")
    log_dir = _project_path(args.logdir) / run_name
    ckpt_dir = _project_path(args.ckptdir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"[init] run_name = {run_name}")
    print(f"[init] log_dir  = {log_dir}")
    print(f"[init] ckpt_dir = {ckpt_dir}")
    print(f"[init] device   = {device}")
    print(f"[init] init_mode = {env_cfg.tug_init_mode}, "
          f"obs_history_k={env_cfg.obs_history_k}, "
          f"ship_preview_times={env_cfg.obs_ship_preview_times_s}, "
          f"ship_size_randomize={env_cfg.ship_size_randomize}")
    if int(args.critic_warmup_updates) > 0:
        print(f"[init] critic_warmup_updates = {args.critic_warmup_updates} "
              "(actor frozen, eval/best skipped during warmup)")

    # 把超参数 dump 到 tensorboard 与文本
    hparams_text = "\n".join([f"env.{k} = {v}" for k, v in asdict(env_cfg).items()] +
                             [f"ppo.{k} = {v}" for k, v in asdict(ppo_cfg).items()])
    writer.add_text("hparams", hparams_text.replace("\n", "  \n"), 0)

    # 向量化环境
    vec_env = make_vec_env(
        env_cfg,
        n_envs=ppo_cfg.num_envs,
        base_seed=ppo_cfg.seed,
        backend=args.env_backend,
        env_workers=args.env_workers,
    )
    obs = vec_env.reset()
    global_state_dim = vec_env.envs[0].global_state_dim
    env_workers = (
        getattr(vec_env, "n_workers", 1)
        if args.env_backend == "subproc"
        else 1
    )
    print(
        f"[init] vec_env: backend={args.env_backend}, workers={env_workers}, "
        f"{ppo_cfg.num_envs} envs × {env_cfg.n_tugs} tugs, "
        f"obs_dim={vec_env.envs[0].obs_dim}, action_dim={vec_env.envs[0].action_dim}, "
        f"global_state_dim={global_state_dim}, eval_workers={eval_workers}, "
        f"torch_threads={torch.get_num_threads()}"
    )

    # 网络与优化器
    model_kwargs = {
        "obs_dim": vec_env.envs[0].obs_dim,
        "action_dim": vec_env.envs[0].action_dim,
        "n_agents": env_cfg.n_tugs,
        "global_state_dim": global_state_dim,
    }
    model = MAPPOActorCritic(**model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=ppo_cfg.learning_rate, eps=1e-5)

    # 续训
    start_update = 0
    global_step = 0
    resume_ckpt: dict | None = None
    if args.resume:
        resume_path = _project_path(args.resume)
        resume_ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        start_update, global_step, optimizer_loaded = _load_checkpoint(
            model, optimizer, resume_ckpt, reset_progress=args.reset_progress
        )
        print(f"[resume] loaded {resume_path}: "
              f"update={start_update}, global_step={global_step}")

    # Rollout buffer
    buffer = MAPPORolloutBuffer(
        rollout_steps=ppo_cfg.rollout_steps,
        num_envs=ppo_cfg.num_envs,
        n_tugs=env_cfg.n_tugs,
        obs_dim=vec_env.envs[0].obs_dim,
        action_dim=vec_env.envs[0].action_dim,
        global_state_dim=global_state_dim,
        device=device,
    )

    # 奖励归一化器：把所有奖励归一化到 N(0,1) 量级，防止 value loss 爆炸
    reward_normalizer = RewardNormalizer(clip=10.0)

    # 跟踪最近 episode 信息
    ep_return_window = deque(maxlen=100)
    ep_length_window = deque(maxlen=100)
    success_window = deque(maxlen=100)
    collision_window = deque(maxlen=100)
    final_dist_window = deque(maxlen=100)

    # 用于"最佳模型"判定 + 早停（best 出现后 N update 内无新 best 则停）
    # success 的小幅波动按 eval 抽样噪声处理；collision 明显变差时不保存 best。
    best_eval_succ = -1.0
    best_eval_collision = float("inf")
    best_eval_return = -float("inf")
    best_eval_dist = float("inf")
    best_eval_pre_success_score = -float("inf")
    best_update = -1
    best_success_margin = max(0.05, 2.0 / max(1, ppo_cfg.eval_episodes))
    best_collision_guard = max(0.05, 3.0 / max(1, ppo_cfg.eval_episodes))
    early_stop_patience = 200

    # 总更新次数
    samples_per_update = ppo_cfg.rollout_steps * ppo_cfg.num_envs * env_cfg.n_tugs
    n_updates = max(1, ppo_cfg.total_steps // samples_per_update)
    print(f"[init] samples_per_update = {samples_per_update}, total updates = {n_updates}")

    lr_scheduler = _make_lr_scheduler(optimizer, ppo_cfg, n_updates)
    if lr_scheduler is not None:
        eta_min = ppo_cfg.learning_rate * ppo_cfg.lr_min_factor
        print(
            f"[init] lr_scheduler = CosineAnnealingLR("
            f"T_max={max(1, n_updates - 1)}, eta_min={eta_min:.2e})"
        )
    if resume_ckpt is not None and not args.reset_progress:
        _load_lr_scheduler(
            lr_scheduler,
            resume_ckpt,
            start_update=start_update,
            restore_from_ckpt=optimizer_loaded,
        )

    # Ctrl+C 安全保存
    interrupt_flag = {"stop": False}
    def _on_sigint(signum, frame):
        if interrupt_flag["stop"]:
            print("\n[interrupt] second SIGINT, exiting hard.")
            sys.exit(1)
        print("\n[interrupt] caught SIGINT, will save and exit after current update.")
        interrupt_flag["stop"] = True
    signal.signal(signal.SIGINT, _on_sigint)

    t_start = time.time()
    last_completed_update = start_update
    for update in range(start_update, n_updates):
        # ---------- 1. 收集 rollout ----------
        buffer.reset()
        rollout_t0 = time.time()
        reward_component_sums: dict[str, float] = {}
        reward_component_counts: dict[str, int] = {}
        dense_reward_batches: list[np.ndarray] = []
        terminal_reward_batches: list[np.ndarray] = []
        buffer_reward_batches: list[np.ndarray] = []
        episode_start_t = np.zeros(ppo_cfg.num_envs, dtype=np.int32)
        collision_episode_mask = np.zeros(
            (ppo_cfg.rollout_steps, ppo_cfg.num_envs, env_cfg.n_tugs), dtype=bool
        )
        noncollision_episode_mask = np.zeros_like(collision_episode_mask)
        for t in range(ppo_cfg.rollout_steps):
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)  # (N, K, obs_dim)
                # 把 (N, K, obs_dim) 展平到 (N*K, obs_dim)，复用同一份策略
                flat_obs = obs_t.reshape(-1, obs_t.shape[-1])
                global_state_np = _build_global_state(vec_env)
                global_state_t = torch.as_tensor(global_state_np, dtype=torch.float32, device=device)
                action, logp, _ = model.act(flat_obs, deterministic=False)
                value = model.get_values(global_state_t)
                action = action.reshape(ppo_cfg.num_envs, env_cfg.n_tugs, -1)
                logp = logp.reshape(ppo_cfg.num_envs, env_cfg.n_tugs)
            actions_np = action.cpu().numpy()

            obs_next, rew, done, info, ep_infos, terminated_arr, truncated_arr, \
                terminal_obs_local, terminal_global = vec_env.step(actions_np)

            # 把稠密奖励和终端奖励分开：
            # - 稠密奖励归一化后存入 buffer（量级稳定，value network 易拟合）
            # - 终端奖励（碰撞/成功）在归一化之后直接叠加，保持原始信号强度
            terminal_rew = np.stack(
                [inf.get("terminal_reward", np.zeros(env_cfg.n_tugs, dtype=np.float32))
                 for inf in info], axis=0
            )  # (N, K)
            dense_rew = rew - terminal_rew
            dense_norm = reward_normalizer.update_and_normalize(dense_rew)
            rew_for_buffer = dense_norm + terminal_rew
            dense_reward_batches.append(dense_rew)
            terminal_reward_batches.append(terminal_rew)
            buffer_reward_batches.append(rew_for_buffer)

            # 边界步的 next_value override：terminated → 0；truncated → V(terminal_state_pre_reset)。
            # 非边界步 buffer 内部会取 self.values[t+1]，此处填 0 即可（不会被使用）。
            next_value_override = np.zeros(
                (ppo_cfg.num_envs, env_cfg.n_tugs), dtype=np.float32
            )
            if truncated_arr.any():
                trunc_idx = np.flatnonzero(truncated_arr)
                with torch.no_grad():
                    term_state_t = torch.as_tensor(
                        terminal_global[trunc_idx], dtype=torch.float32, device=device
                    )
                    v_term = model.get_values(term_state_t).cpu().numpy()  # (n_trunc, K)
                next_value_override[trunc_idx] = v_term

            buffer.add(
                obs=obs,
                global_state=global_state_np,
                actions=actions_np,
                logprobs=logp.cpu().numpy(),
                rewards=rew_for_buffer,
                values=value.cpu().numpy(),
                dones=done,
                next_value_override=next_value_override,
            )

            obs = obs_next
            global_step += ppo_cfg.num_envs * env_cfg.n_tugs

            for env_idx, done_i in enumerate(done):
                if not done_i:
                    continue
                start_t = int(episode_start_t[env_idx])
                if start_t <= t:
                    if bool(info[env_idx].get("collision", False)):
                        collision_episode_mask[start_t:t + 1, env_idx, :] = True
                    else:
                        noncollision_episode_mask[start_t:t + 1, env_idx, :] = True
                episode_start_t[env_idx] = t + 1

            if info:
                comp_arrays: dict[str, list[np.ndarray]] = {}
                for inf in info:
                    for key, value in inf.get("reward_components", {}).items():
                        arr = np.asarray(value)
                        if arr.dtype.kind in "biufc":
                            comp_arrays.setdefault(key, []).append(arr.astype(np.float32, copy=False))
                for key, arrays in comp_arrays.items():
                    if arrays:
                        reward_component_sums[key] = reward_component_sums.get(key, 0.0) + float(
                            np.nanmean(np.stack(arrays, axis=0))
                        )
                        reward_component_counts[key] = reward_component_counts.get(key, 0) + 1

            # 记录已完成 episode 的统计
            for ep in ep_infos:
                ep_return_window.append(ep["episode_return"])
                ep_length_window.append(ep["episode_length"])
                success_window.append(1.0 if ep["success"] else 0.0)
                collision_window.append(1.0 if ep["collision"] else 0.0)
                if not math.isnan(ep["final_dist_mean"]):
                    final_dist_window.append(ep["final_dist_mean"])

        rollout_dt = time.time() - rollout_t0

        # ---------- 2. 末端价值用于 GAE ----------
        with torch.no_grad():
            global_state_t = torch.as_tensor(
                _build_global_state(vec_env), dtype=torch.float32, device=device
            )
            last_value = model.get_values(global_state_t).cpu().numpy()
        buffer.compute_gae(last_value, gamma=ppo_cfg.gamma, lam=ppo_cfg.gae_lambda)
        dense_reward_mean, dense_reward_var = _mean_var(dense_reward_batches)
        terminal_reward_mean, terminal_reward_var = _mean_var(terminal_reward_batches)
        buffer_reward_mean, buffer_reward_var = _mean_var(buffer_reward_batches)
        return_target_np = buffer.returns.detach().cpu().numpy()
        return_target_mean, return_target_var = _mean_var([return_target_np])
        collision_mask_t = torch.as_tensor(
            collision_episode_mask, dtype=torch.bool, device=device
        )
        noncollision_mask_t = torch.as_tensor(
            noncollision_episode_mask, dtype=torch.bool, device=device
        )

        # ---------- 3. MAPPO 更新 ----------
        # warm-start 时前 N 个 update 只训 critic：policy_coef=0 把 actor / entropy
        # 的梯度阻断掉。等 critic 把 EV 拉起来再放开 actor，避免随机 advantage 把
        # 已经训练好的 actor 冲烂（v54 实验里观察到的失败模式）。
        in_warmup = update < int(args.critic_warmup_updates)
        policy_coef = 0.0 if in_warmup else 1.0
        update_t0 = time.time()
        stats = mappo_update(
            model,
            optimizer,
            buffer,
            clip_eps=ppo_cfg.clip_eps,
            value_clip_eps=ppo_cfg.value_clip_eps,
            entropy_coef=ppo_cfg.entropy_coef,
            value_coef=ppo_cfg.value_coef,
            max_grad_norm=ppo_cfg.max_grad_norm,
            minibatch_size=ppo_cfg.minibatch_size,
            update_epochs=ppo_cfg.update_epochs,
            target_kl=ppo_cfg.target_kl,
            policy_coef=policy_coef,
            collision_mask=collision_mask_t,
            noncollision_mask=noncollision_mask_t,
        )
        update_dt = time.time() - update_t0

        # ---------- 4. 日志 ----------
        sps = samples_per_update / max(1e-6, rollout_dt + update_dt)
        recent_ret = float(np.mean(ep_return_window)) if ep_return_window else float("nan")
        recent_len = float(np.mean(ep_length_window)) if ep_length_window else float("nan")
        succ_rate = float(np.mean(success_window)) if success_window else 0.0
        coll_rate = float(np.mean(collision_window)) if collision_window else 0.0
        final_dist = float(np.mean(final_dist_window)) if final_dist_window else float("nan")

        # TensorBoard：最近 100 个已完成 episode 的滑动均值（rollout/*）
        writer.add_scalar("rollout/ep_return_mean", recent_ret, global_step)   # 每局回报（多 agent 奖励均值累加）
        writer.add_scalar("rollout/ep_length_mean", recent_len, global_step)   # 每局步数
        writer.add_scalar("rollout/success_rate", succ_rate, global_step)      # 全部拖轮入位并保持 hold_time 的比例
        writer.add_scalar("rollout/collision_rate", coll_rate, global_step)  # 拖轮-大船或拖轮-拖轮碰撞比例
        writer.add_scalar("rollout/final_dist_mean", final_dist, global_step)  # 终局到槽位距离 dist_to_slot 均值（米）
        # PPO 更新损失与诊断（loss/*）
        writer.add_scalar("loss/policy", stats.policy_loss, global_step)         # clipped surrogate 策略损失
        writer.add_scalar("loss/value", stats.value_loss, global_step)         # clipped value 回归损失
        writer.add_scalar("loss/entropy", stats.entropy, global_step)          # 策略熵（探索强度，越大越随机）
        writer.add_scalar("loss/approx_kl", stats.approx_kl, global_step)        # 新旧策略近似 KL，过大时 target_kl 早停
        writer.add_scalar("loss/clip_frac", stats.clip_frac, global_step)      # ratio 被 PPO clip 的样本比例
        writer.add_scalar("loss/explained_variance", stats.explained_variance, global_step)  # critic 对 return 的解释度，→1 越好
        _add_finite_scalar(writer, "loss/value_collision", stats.value_loss_collision, global_step)
        _add_finite_scalar(writer, "loss/value_noncollision", stats.value_loss_noncollision, global_step)
        _add_finite_scalar(writer, "loss/explained_variance_collision", stats.explained_variance_collision, global_step)
        _add_finite_scalar(writer, "loss/explained_variance_noncollision", stats.explained_variance_noncollision, global_step)
        writer.add_scalar("loss/grad_norm", stats.grad_norm, global_step)      # 梯度裁剪前的 L2 范数
        _add_finite_scalar(writer, "reward_stats/dense_mean", dense_reward_mean, global_step)
        _add_finite_scalar(writer, "reward_stats/dense_var", dense_reward_var, global_step)
        _add_finite_scalar(writer, "reward_stats/terminal_mean", terminal_reward_mean, global_step)
        _add_finite_scalar(writer, "reward_stats/terminal_var", terminal_reward_var, global_step)
        _add_finite_scalar(writer, "reward_stats/buffer_reward_mean", buffer_reward_mean, global_step)
        _add_finite_scalar(writer, "reward_stats/buffer_reward_var", buffer_reward_var, global_step)
        _add_finite_scalar(writer, "return_target/mean", return_target_mean, global_step)
        _add_finite_scalar(writer, "return_target/var", return_target_var, global_step)
        writer.add_scalar("value_diag/collision_sample_count", int(collision_episode_mask.sum()), global_step)
        writer.add_scalar("value_diag/noncollision_sample_count", int(noncollision_episode_mask.sum()), global_step)
        # 优化器与探索尺度（opt/*）
        writer.add_scalar("opt/learning_rate", stats.learning_rate, global_step)  # 当前 Adam 学习率
        writer.add_scalar("opt/log_std_mean", stats.log_std_mean, global_step)     # 动作 log_std 均值（探索噪声尺度）
        # 吞吐与耗时（perf/*）
        writer.add_scalar("perf/sps", sps, global_step)                        # 每秒环境步数（rollout+update）
        writer.add_scalar("perf/rollout_seconds", rollout_dt, global_step)     # 采集 rollout 耗时
        writer.add_scalar("perf/update_seconds", update_dt, global_step)         # MAPPO 梯度更新耗时
        for key, total in reward_component_sums.items():
            count = reward_component_counts.get(key, 0)
            if count > 0:
                writer.add_scalar(f"reward/{key}", total / float(count), global_step)

        # 控制台日志
        if update % ppo_cfg.log_interval == 0:
            elapsed = time.time() - t_start
            phase = "warm" if in_warmup else "main"
            print(
                f"[upd {update:5d}/{n_updates}|{phase}] step={global_step:>9d} "
                f"ret={recent_ret:7.2f} len={recent_len:5.1f} "
                f"succ={succ_rate*100:5.1f}% coll={coll_rate*100:5.1f}% "
                f"d={final_dist:6.1f}m | "
                f"pl={stats.policy_loss:+.4f} vl={stats.value_loss:.4f} "
                f"ent={stats.entropy:+.3f} kl={stats.approx_kl:.4f} "
                f"clip={stats.clip_frac:.2f} ev={stats.explained_variance:+.2f} "
                f"lr={stats.learning_rate:.2e} | "
                f"sps={sps:.0f} elapsed={elapsed/60:.1f}min"
            )

        # ---------- 5. 评估 + 保存 best ----------
        if (update + 1) % ppo_cfg.eval_interval == 0 and not in_warmup:
            eval_stats = evaluate_policy(
                model, env_cfg, n_episodes=ppo_cfg.eval_episodes,
                device=device, seed=ppo_cfg.seed + 9999 + update,
                eval_workers=eval_workers,
                model_kwargs=model_kwargs,
            )
            for k, v in eval_stats.items():
                writer.add_scalar(k, v, global_step)
            print(
                f"  [eval] return={eval_stats['eval/return_mean']:.2f} "
                f"±{eval_stats['eval/return_std']:.2f}, "
                f"len={eval_stats['eval/length_mean']:.1f}, "
                f"succ={eval_stats['eval/success_rate']*100:.1f}%, "
                f"coll={eval_stats['eval/collision_rate']*100:.1f}%, "
                f"d={eval_stats['eval/final_dist_mean']:.1f}m"
            )
           
            cur_succ = eval_stats["eval/success_rate"]
            cur_coll = eval_stats["eval/collision_rate"]
            cur_ret = eval_stats["eval/return_mean"]
            cur_dist = eval_stats["eval/final_dist_mean"]
            cur_pre_success_score = (
                -cur_dist - 200.0 * cur_coll
                if np.isfinite(cur_dist)
                else -float("inf")
            )
            is_better = False
            save_skip_reason = ""
            if best_update < 0:
                is_better = True
            else:
                success_gain = cur_succ - best_eval_succ
                collision_gain = best_eval_collision - cur_coll
                collision_within_guard = cur_coll <= best_eval_collision + best_collision_guard
                success_clearly_better = success_gain > best_success_margin
                success_near_tie = abs(success_gain) <= best_success_margin

                if success_clearly_better:
                    is_better = collision_within_guard
                    if not is_better:
                        save_skip_reason = (
                            f"success improved but collision worsened beyond guard "
                            f"({cur_coll*100:.1f}% > "
                            f"{(best_eval_collision + best_collision_guard)*100:.1f}%)"
                        )
                elif success_near_tie:
                    if collision_gain > best_collision_guard:
                        is_better = True
                    elif abs(collision_gain) <= best_collision_guard:
                        if cur_succ <= 0.0 and best_eval_succ <= 0.0:
                            is_better = cur_pre_success_score > best_eval_pre_success_score
                        elif cur_succ >= best_eval_succ:
                            is_better = cur_ret > best_eval_return
            if is_better:
                best_eval_succ = cur_succ
                best_eval_collision = cur_coll
                best_eval_return = cur_ret
                best_eval_dist = cur_dist
                best_eval_pre_success_score = cur_pre_success_score
                best_update = update
                _save_ckpt(
                    ckpt_dir / "best.pt",
                    model, optimizer, env_cfg, ppo_cfg,
                    update=update + 1, global_step=global_step,
                    metric=cur_succ,
                    lr_scheduler=lr_scheduler,
                )
                print(f"  [save] best.pt updated (succ={cur_succ*100:.1f}%, "
                      f"coll={cur_coll*100:.1f}%, d={cur_dist:.1f}m, "
                      f"return={cur_ret:.2f})")
            elif save_skip_reason:
                print(f"  [save-skip] {save_skip_reason}")
            elif best_update >= 0 and (update - best_update) >= early_stop_patience:
                print(f"  [early-stop] no new best for {early_stop_patience} updates "
                      f"(best succ={best_eval_succ*100:.1f}% / "
                      f"coll={best_eval_collision*100:.1f}% / "
                      f"d={best_eval_dist:.1f}m / return={best_eval_return:.2f} "
                      f"at update {best_update}), stopping.")
                interrupt_flag["stop"] = True

        # ---------- 6. 定期保存 last.pt ----------
        if (update + 1) % ppo_cfg.save_interval == 0 or interrupt_flag["stop"]:
            _save_ckpt(
                ckpt_dir / "last.pt",
                model, optimizer, env_cfg, ppo_cfg,
                update=update + 1, global_step=global_step,
                metric=recent_ret,
                lr_scheduler=lr_scheduler,
            )

        if lr_scheduler is not None:
            lr_scheduler.step()

        last_completed_update = update + 1
        if interrupt_flag["stop"]:
            print("[interrupt] saved last.pt, exiting.")
            break

    # 训练结束最终保存
    _save_ckpt(
        ckpt_dir / "last.pt",
        model, optimizer, env_cfg, ppo_cfg,
        update=last_completed_update, global_step=global_step,
        metric=float(np.mean(ep_return_window)) if ep_return_window else 0.0,
        lr_scheduler=lr_scheduler,
    )
    vec_env.close()
    writer.close()
    if best_update >= 0:
        print(f"[done] total time = {(time.time() - t_start)/60:.1f} min, "
              f"best eval succ = {best_eval_succ*100:.1f}% / "
              f"coll = {best_eval_collision*100:.1f}% / return = {best_eval_return:.2f}")
    else:
        print(f"[done] total time = {(time.time() - t_start)/60:.1f} min, no eval was run.")


def _save_ckpt(
    path: Path,
    model: MAPPOActorCritic,
    optimizer: torch.optim.Optimizer,
    env_cfg: EnvConfig,
    ppo_cfg: PPOConfig,
    *,
    update: int,
    global_step: int,
    metric: float,
    lr_scheduler: LRScheduler | None = None,
) -> None:
    payload = {
        "algo": "mappo",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_kwargs": {
            "obs_dim": model.obs_dim,
            "action_dim": model.action_dim,
            "n_agents": model.n_agents,
            "global_state_dim": model.global_state_dim,
        },
        "env_cfg": asdict(env_cfg),
        "ppo_cfg": asdict(ppo_cfg),
        "update": update,
        "global_step": global_step,
        "metric": metric,
    }
    if lr_scheduler is not None:
        payload["lr_scheduler"] = lr_scheduler.state_dict()
        payload["lr_scheduler_type"] = type(lr_scheduler).__name__
    torch.save(payload, str(path))


if __name__ == "__main__":
    main()
