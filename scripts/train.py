"""MAPPO 多智能体拖轮编队训练脚本。

用法：
    python scripts/train.py --total-steps 5000000 --num-envs 8 --rollout-steps 256

特性：
- 去中心化 actor：每艘拖轮只看自己的局部观察
- 集中式 critic：使用 canonical global state，输出每艘拖轮的 value
- 顺序执行的向量化环境（SyncVecEnv）
- 控制台 + tensorboard 双日志
- 自动按"评估平均回报"保存最佳模型权重 best.pt
- 周期性保存 last.pt（用于断点续训或观察当前训练状态）
- 学习率线性退火
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
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


import numpy as np
import torch
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

from config import EnvConfig, PPOConfig
from env.formation_env import FormationEnv
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


# ---------- 评估循环（确定性策略，跑若干 episode） ----------
def _build_global_state(vec_env: "SyncVecEnv") -> np.ndarray:
    """收集每个环境的 canonical global state，形状 (N, global_state_dim)。

    canonical state 在大船船体系下表达，跨 4 个 agent 共享。
    """
    return np.stack([e.get_global_state() for e in vec_env.envs], axis=0).astype(
        np.float32, copy=False
    )


def _load_checkpoint(
    model: MAPPOActorCritic,
    optimizer: torch.optim.Optimizer,
    ckpt: dict,
    reset_progress: bool = False,
) -> tuple[int, int]:
    """加载 MAPPO checkpoint；输入维度增长时迁移首层权重。"""
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

    for key, dst_tensor in dst_state.items():
        src_tensor = src_state.get(key)
        if src_tensor is None:
            continue
        if tuple(src_tensor.shape) == tuple(dst_tensor.shape):
            load_state[key] = src_tensor.to(device=dst_tensor.device, dtype=dst_tensor.dtype)
            continue

        # 观测/global-state 追加特征时，只扩展 actor/critic 第一层输入列。
        # 旧列保持原权重，新列置零，warm-start 初始行为不被新增特征扰动。
        if (
            key in ("actor_trunk.0.weight", "critic.0.weight")
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
        return 0, 0
    if migrated or skipped:
        print("[resume] optimizer state reset because model input shape changed.")
        return int(ckpt.get("update", 0)), int(ckpt.get("global_step", 0))
    optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt.get("update", 0)), int(ckpt.get("global_step", 0))


def evaluate_policy(
    model: MAPPOActorCritic,
    env_cfg: EnvConfig,
    n_episodes: int,
    device: torch.device,
    seed: int = 12345,
) -> dict[str, float]:
    rng_seed = seed
    returns, lengths, succ, coll = [], [], [], []
    for _ in range(n_episodes):
        env = FormationEnv(cfg=env_cfg, seed=rng_seed)
        rng_seed += 1
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
        returns.append(ep_ret)
        lengths.append(ep_len)
        succ.append(bool(info_last.get("success", False)))
        coll.append(bool(info_last.get("collision", False)))
    return {
        "eval/return_mean": float(np.mean(returns)),
        "eval/return_std": float(np.std(returns)),
        "eval/length_mean": float(np.mean(lengths)),
        "eval/success_rate": float(np.mean(succ)),
        "eval/collision_rate": float(np.mean(coll)),
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
                        choices=["astern_approach", "mixed_slot_approach"],
                        help="拖轮初始场景（默认使用 config.py）")
    parser.add_argument("--route-planner", type=str, default=None,
                        choices=["visibility", "manual"],
                        help="waypoint 生成器：visibility 或 manual 固定模板")
    parser.add_argument("--no-route-obs", action="store_true",
                        help="关闭 v29 路线 waypoint/stage 观察特征")
    parser.add_argument("--no-ego-accel-obs", action="store_true",
                        help="关闭 actor 观察中的自身 3D 加速度特征")
    parser.add_argument("--no-ship-size-randomize", action="store_true",
                        help="关闭 v36 大船长宽随机化，使用 config.py 中的固定 ship_length_m/ship_beam_m")
    parser.add_argument("--no-simple-stage", action="store_true",
                        help="关闭 v52+ 的 simple_stage 简化奖励，回到 v50 时代的 multi-component dense reward")
    parser.add_argument("--hold-time", type=float, default=None,
                        help="覆盖 EnvConfig.hold_time_s（课程学习用：1.0 → 5.0 → 10.0）")
    parser.add_argument("--critic-warmup-updates", type=int, default=0,
                        help=("warm-start 时前 N 个 update 只训 critic（policy_coef=0），"
                              "用于 critic 重置或维度变化后先把 EV 拉起来再放开 actor。"))
    args = parser.parse_args()

    # 配置对象
    env_cfg = EnvConfig()
    if args.init_mode is not None:
        env_cfg.tug_init_mode = args.init_mode
    if args.route_planner is not None:
        env_cfg.route_planner = args.route_planner
    if args.no_route_obs:
        env_cfg.obs_include_route = False
    if args.no_ego_accel_obs:
        env_cfg.obs_include_ego_accel = False
    if args.no_ship_size_randomize:
        env_cfg.ship_size_randomize = False
        env_cfg.obs_include_ship_size = False
    if args.no_simple_stage:
        env_cfg.reward_use_simple_stage = False
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
    print(f"[init] init_mode = {env_cfg.tug_init_mode}, route_planner={env_cfg.route_planner}, "
          f"route_obs={env_cfg.obs_include_route}, "
          f"ego_accel_obs={env_cfg.obs_include_ego_accel}, "
          f"ship_size_randomize={env_cfg.ship_size_randomize}")
    if int(args.critic_warmup_updates) > 0:
        print(f"[init] critic_warmup_updates = {args.critic_warmup_updates} "
              "(actor frozen, eval/best skipped during warmup)")

    # 把超参数 dump 到 tensorboard 与文本
    hparams_text = "\n".join([f"env.{k} = {v}" for k, v in asdict(env_cfg).items()] +
                             [f"ppo.{k} = {v}" for k, v in asdict(ppo_cfg).items()])
    writer.add_text("hparams", hparams_text.replace("\n", "  \n"), 0)

    # 向量化环境
    vec_env = SyncVecEnv(env_cfg, n_envs=ppo_cfg.num_envs, base_seed=ppo_cfg.seed)
    obs = vec_env.reset()
    global_state_dim = vec_env.envs[0].global_state_dim
    print(f"[init] vec_env: {ppo_cfg.num_envs} envs × {env_cfg.n_tugs} tugs, "
          f"obs_dim={vec_env.envs[0].obs_dim}, action_dim={vec_env.envs[0].action_dim}, "
          f"global_state_dim={global_state_dim}")

    # 网络与优化器
    model = MAPPOActorCritic(
        obs_dim=vec_env.envs[0].obs_dim,
        action_dim=vec_env.envs[0].action_dim,
        n_agents=env_cfg.n_tugs,
        hidden_dims=ppo_cfg.hidden_dims,
        critic_hidden_dims=ppo_cfg.critic_hidden_dims,
        activation=ppo_cfg.activation,
        log_std_init=ppo_cfg.log_std_init,
        global_state_dim=global_state_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=ppo_cfg.learning_rate, eps=1e-5)

    # 续训
    start_update = 0
    global_step = 0
    if args.resume:
        resume_path = _project_path(args.resume)
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        start_update, global_step = _load_checkpoint(
            model, optimizer, ckpt, reset_progress=args.reset_progress
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
    # v23: best 判定改用 succ_rate（return 仅作 tiebreaker），patience 100→200
    best_eval_succ = -1.0
    best_eval_collision = float("inf")
    best_eval_return = -float("inf")
    best_update = -1
    early_stop_patience = 200

    # 总更新次数
    samples_per_update = ppo_cfg.rollout_steps * ppo_cfg.num_envs * env_cfg.n_tugs
    n_updates = max(1, ppo_cfg.total_steps // samples_per_update)
    print(f"[init] samples_per_update = {samples_per_update}, total updates = {n_updates}")

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
        # 学习率退火：v23 撤掉 v22 的 0.2 下限，回到 0.05
        # （v22 用 0.2 起步 lr=1.64e-4 把 v21 末段精细化打散，导致真退步）
        if ppo_cfg.lr_anneal:
            frac = 1.0 - update / max(1, n_updates - 1)
            for pg in optimizer.param_groups:
                pg["lr"] = ppo_cfg.learning_rate * max(frac, 0.05)

        # ---------- 1. 收集 rollout ----------
        buffer.reset()
        rollout_t0 = time.time()
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
        )
        update_dt = time.time() - update_t0

        # ---------- 4. 日志 ----------
        sps = samples_per_update / max(1e-6, rollout_dt + update_dt)
        recent_ret = float(np.mean(ep_return_window)) if ep_return_window else float("nan")
        recent_len = float(np.mean(ep_length_window)) if ep_length_window else float("nan")
        succ_rate = float(np.mean(success_window)) if success_window else 0.0
        coll_rate = float(np.mean(collision_window)) if collision_window else 0.0
        final_dist = float(np.mean(final_dist_window)) if final_dist_window else float("nan")

        writer.add_scalar("rollout/ep_return_mean", recent_ret, global_step)
        writer.add_scalar("rollout/ep_length_mean", recent_len, global_step)
        writer.add_scalar("rollout/success_rate", succ_rate, global_step)
        writer.add_scalar("rollout/collision_rate", coll_rate, global_step)
        writer.add_scalar("rollout/final_dist_mean", final_dist, global_step)
        writer.add_scalar("loss/policy", stats.policy_loss, global_step)
        writer.add_scalar("loss/value", stats.value_loss, global_step)
        writer.add_scalar("loss/entropy", stats.entropy, global_step)
        writer.add_scalar("loss/approx_kl", stats.approx_kl, global_step)
        writer.add_scalar("loss/clip_frac", stats.clip_frac, global_step)
        writer.add_scalar("loss/explained_variance", stats.explained_variance, global_step)
        writer.add_scalar("loss/grad_norm", stats.grad_norm, global_step)
        writer.add_scalar("opt/learning_rate", stats.learning_rate, global_step)
        writer.add_scalar("opt/log_std_mean", stats.log_std_mean, global_step)
        writer.add_scalar("perf/sps", sps, global_step)
        writer.add_scalar("perf/rollout_seconds", rollout_dt, global_step)
        writer.add_scalar("perf/update_seconds", update_dt, global_step)

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
        # warmup 期间 actor 没有任何梯度更新；eval 结果与 update 0 一致，
        # 跑也是浪费，且容易把"v50 actor + 随机 critic"的 0% 成绩当成 best 写盘。
        if (update + 1) % ppo_cfg.eval_interval == 0 and not in_warmup:
            eval_stats = evaluate_policy(
                model, env_cfg, n_episodes=ppo_cfg.eval_episodes,
                device=device, seed=ppo_cfg.seed + 9999 + update,
            )
            for k, v in eval_stats.items():
                writer.add_scalar(k, v, global_step)
            print(
                f"  [eval] return={eval_stats['eval/return_mean']:.2f} "
                f"±{eval_stats['eval/return_std']:.2f}, "
                f"len={eval_stats['eval/length_mean']:.1f}, "
                f"succ={eval_stats['eval/success_rate']*100:.1f}%, "
                f"coll={eval_stats['eval/collision_rate']*100:.1f}%"
            )
            # v23: 按 succ_rate 优先选 best，return 仅作同 succ 时的 tiebreaker
            # （v21 best 4-ep return=565 但 succ=0%，证明 return 会被"陪走 1200 步"骗）
            # v34: success 相同时优先选择 collision 更低的模型，return 只做最后 tie-breaker。
            cur_succ = eval_stats["eval/success_rate"]
            cur_coll = eval_stats["eval/collision_rate"]
            cur_ret = eval_stats["eval/return_mean"]
            is_better = (
                cur_succ > best_eval_succ
                or (
                    cur_succ == best_eval_succ
                    and (
                        cur_coll < best_eval_collision
                        or (cur_coll == best_eval_collision and cur_ret > best_eval_return)
                    )
                )
            )
            if is_better:
                best_eval_succ = cur_succ
                best_eval_collision = cur_coll
                best_eval_return = cur_ret
                best_update = update
                _save_ckpt(
                    ckpt_dir / "best.pt",
                    model, optimizer, env_cfg, ppo_cfg,
                    update=update + 1, global_step=global_step,
                    metric=cur_succ,
                )
                print(f"  [save] best.pt updated (succ={cur_succ*100:.1f}%, "
                      f"coll={cur_coll*100:.1f}%, return={cur_ret:.2f})")
            elif best_update >= 0 and (update - best_update) >= early_stop_patience:
                print(f"  [early-stop] no new best for {early_stop_patience} updates "
                      f"(best succ={best_eval_succ*100:.1f}% / "
                      f"coll={best_eval_collision*100:.1f}% / return={best_eval_return:.2f} "
                      f"at update {best_update}), stopping.")
                interrupt_flag["stop"] = True

        # ---------- 6. 定期保存 last.pt ----------
        if (update + 1) % ppo_cfg.save_interval == 0 or interrupt_flag["stop"]:
            _save_ckpt(
                ckpt_dir / "last.pt",
                model, optimizer, env_cfg, ppo_cfg,
                update=update + 1, global_step=global_step,
                metric=recent_ret,
            )

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
    )
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
) -> None:
    payload = {
        "algo": "mappo",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "env_cfg": asdict(env_cfg),
        "ppo_cfg": asdict(ppo_cfg),
        "update": update,
        "global_step": global_step,
        "metric": metric,
    }
    torch.save(payload, str(path))


if __name__ == "__main__":
    main()
