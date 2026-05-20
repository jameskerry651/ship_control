"""MAPPO 算法实现：去中心化 Actor + 集中式 Critic + Rollout Buffer + 训练步。

设计要点：
- Actor 只看单 agent 局部观察；Critic 看 canonical global state。
- 高斯策略经 tanh 压到 [-1, 1]，log_std 为可学习参数。
- 4 个智能体共享同一份 actor（参数共享），观察都是"以自身为参考系"的相对量。
- 训练时把 (T, N, n_tugs, *) 展平到 (T*N*n_tugs, *) 处理。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


_ACTION_SQUASH_EPS = 1e-6


def _atanh(x: torch.Tensor) -> torch.Tensor:
    """Stable inverse tanh for values in (-1, 1)."""
    x = torch.clamp(x, -1.0 + _ACTION_SQUASH_EPS, 1.0 - _ACTION_SQUASH_EPS)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def _squash_log_det(action: torch.Tensor) -> torch.Tensor:
    """Log absolute det of tanh Jacobian, summed over action dims."""
    action = torch.clamp(action, -1.0 + _ACTION_SQUASH_EPS, 1.0 - _ACTION_SQUASH_EPS)
    return torch.log(torch.clamp(1.0 - action.pow(2), min=_ACTION_SQUASH_EPS)).sum(dim=-1)


class _SquashedDiagonalGaussian:
    """Diagonal Gaussian policy with tanh squash to [-1, 1]."""

    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor) -> None:
        self.mean = mean
        self.std = log_std.exp().expand_as(mean)
        self.base = Normal(mean, self.std)

    def sample(self, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pre_tanh = self.mean if deterministic else self.base.rsample()
        action = torch.tanh(pre_tanh)
        logprob = self.base.log_prob(pre_tanh).sum(dim=-1) - _squash_log_det(action)
        entropy = self.base.entropy().sum(dim=-1)
        return action, logprob, entropy

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        action = torch.clamp(action, -1.0 + _ACTION_SQUASH_EPS, 1.0 - _ACTION_SQUASH_EPS)
        pre_tanh = _atanh(action)
        return self.base.log_prob(pre_tanh).sum(dim=-1) - _squash_log_det(action)

    def entropy(self) -> torch.Tensor:
        # Exact tanh-Gaussian entropy is not analytic here; use the base Gaussian entropy
        # as a stable surrogate for PPO regularization.
        return self.base.entropy().sum(dim=-1)


class MAPPOActorCritic(nn.Module):
    """MAPPO 网络：去中心化 actor + 集中式 critic。

    Actor 只看单个拖轮的局部观察 o_i，执行时不依赖其他 agent。
    Critic 看一份与 agent 无关的 canonical global state（由 env.get_global_state()
    给出，量都在大船船体系下表达），并输出每个 agent 的 V_i(s)。

    Critic 输入为 env.get_global_state() 给出的 canonical global state。
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_agents: int,
        hidden_dims: Sequence[int] = (256, 256),
        critic_hidden_dims: Sequence[int] | None = None,
        activation: str = "tanh",
        log_std_init: float = -0.5,
        global_state_dim: int | None = None,
    ) -> None:
        super().__init__()
        act_cls = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU}[activation]
        critic_hidden_dims = tuple(critic_hidden_dims or hidden_dims)

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        if global_state_dim is None:
            raise ValueError("global_state_dim is required for MAPPO critic")
        self.global_state_dim = int(global_state_dim)

        actor_layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in hidden_dims:
            actor_layers.append(nn.Linear(in_dim, h))
            actor_layers.append(act_cls())
            in_dim = h
        self.actor_trunk = nn.Sequential(*actor_layers)
        self.policy_mean = nn.Linear(in_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))

        critic_layers: list[nn.Module] = []
        in_dim = self.global_state_dim
        for h in critic_hidden_dims:
            critic_layers.append(nn.Linear(in_dim, h))
            critic_layers.append(act_cls())
            in_dim = h
        critic_layers.append(nn.Linear(in_dim, n_agents))
        self.critic = nn.Sequential(*critic_layers)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.actor_trunk.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.policy_mean.weight, gain=0.01)
        nn.init.zeros_(self.policy_mean.bias)

        for m in self.critic.modules():
            if isinstance(m, nn.Linear):
                gain = 1.0 if m.out_features == self.n_agents else math.sqrt(2.0)
                nn.init.orthogonal_(m.weight, gain=gain)
                nn.init.zeros_(m.bias)

    def policy(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.actor_trunk(obs)
        return self.policy_mean(h)

    def get_values(self, global_state: torch.Tensor) -> torch.Tensor:
        """返回集中式 critic 的 value，形状 (..., n_agents)。"""
        leading_shape = global_state.shape[:-1]
        flat = global_state.reshape(-1, self.global_state_dim)
        values = self.critic(flat)
        return values.reshape(*leading_shape, self.n_agents)

    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        mean = self.policy(obs)
        dist = _SquashedDiagonalGaussian(mean, self.log_std)
        action, logprob, _ = dist.sample(deterministic=deterministic)
        return action, logprob, None

    def evaluate_for_agents(
        self,
        obs: torch.Tensor,
        global_state: torch.Tensor,
        agent_ids: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.policy(obs)
        dist = _SquashedDiagonalGaussian(mean, self.log_std)
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        values_all = self.get_values(global_state)
        value = values_all.gather(1, agent_ids.long().unsqueeze(1)).squeeze(1)
        return logprob, entropy, value


@dataclass
class MAPPORolloutBatch:
    obs: torch.Tensor              # (B, obs_dim)
    global_state: torch.Tensor     # (B, global_state_dim)
    agent_ids: torch.Tensor        # (B,)
    actions: torch.Tensor          # (B, action_dim)
    old_logprobs: torch.Tensor     # (B,)
    advantages: torch.Tensor       # (B,)
    returns: torch.Tensor          # (B,)
    old_values: torch.Tensor       # (B,)


class MAPPORolloutBuffer:
    """MAPPO rollout buffer，额外保存 critic 使用的 canonical global state。

    `dones` 字段记录 episode 边界（含 timeout）。在边界步上，下一个状态的 V
    用 `next_value_override` 显式传入：碰撞/成功为 0；timeout 为 V(terminal_obs_pre_reset)。
    非边界步上 next_value 仍取 self.values[t+1]。这样 GAE 既不会在 timeout 处把
    V 截断为 0（避免长 horizon 系统性低估），也不会把 V(reset_state) 误当成
    truncated episode 的延续。
    """

    def __init__(
        self,
        rollout_steps: int,
        num_envs: int,
        n_tugs: int,
        obs_dim: int,
        action_dim: int,
        global_state_dim: int,
        device: torch.device,
    ) -> None:
        self.T = rollout_steps
        self.N = num_envs
        self.K = n_tugs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.global_state_dim = global_state_dim
        self.device = device

        shape_per_agent = (rollout_steps, num_envs, n_tugs)
        self.obs = torch.zeros((*shape_per_agent, obs_dim), dtype=torch.float32, device=device)
        self.global_states = torch.zeros(
            (*shape_per_agent, global_state_dim), dtype=torch.float32, device=device
        )
        self.agent_ids = torch.arange(n_tugs, dtype=torch.long, device=device).view(1, 1, n_tugs)
        self.actions = torch.zeros((*shape_per_agent, action_dim), dtype=torch.float32, device=device)
        self.logprobs = torch.zeros(shape_per_agent, dtype=torch.float32, device=device)
        self.rewards = torch.zeros(shape_per_agent, dtype=torch.float32, device=device)
        self.values = torch.zeros(shape_per_agent, dtype=torch.float32, device=device)
        self.dones = torch.zeros(shape_per_agent, dtype=torch.float32, device=device)
        # 边界步上覆盖 next_value：terminated 写 0，truncated 写 V(terminal_obs_pre_reset)。
        self.next_value_override = torch.zeros(shape_per_agent, dtype=torch.float32, device=device)

        self.advantages = torch.zeros_like(self.rewards)
        self.returns = torch.zeros_like(self.rewards)
        self.ptr = 0

    def reset(self) -> None:
        self.ptr = 0

    def add(
        self,
        obs: np.ndarray,              # (N, K, obs_dim)
        global_state: np.ndarray,   # (N, global_state_dim)
        actions: np.ndarray,          # (N, K, action_dim)
        logprobs: np.ndarray,         # (N, K)
        rewards: np.ndarray,          # (N, K)
        values: np.ndarray,           # (N, K)
        dones: np.ndarray,            # (N,) bool；episode 边界（含 timeout）
        next_value_override: np.ndarray | None = None,  # (N, K) 边界步的 V_next
    ) -> None:
        t = self.ptr
        self.obs[t] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        gstate = torch.as_tensor(global_state, dtype=torch.float32, device=self.device)
        self.global_states[t] = gstate.unsqueeze(1).expand(-1, self.K, -1)
        self.actions[t] = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        self.logprobs[t] = torch.as_tensor(logprobs, dtype=torch.float32, device=self.device)
        self.rewards[t] = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        self.values[t] = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        d = torch.as_tensor(dones.astype(np.float32), dtype=torch.float32, device=self.device)
        self.dones[t] = d.unsqueeze(-1).expand(-1, self.K)
        if next_value_override is None:
            self.next_value_override[t] = 0.0
        else:
            self.next_value_override[t] = torch.as_tensor(
                next_value_override, dtype=torch.float32, device=self.device
            )
        self.ptr += 1

    def compute_gae(self, next_values: np.ndarray, gamma: float, lam: float) -> None:
        next_v = torch.as_tensor(next_values, dtype=torch.float32, device=self.device)
        adv = torch.zeros((self.N, self.K), dtype=torch.float32, device=self.device)
        for t in reversed(range(self.T)):
            boundary = self.dones[t]
            if t == self.T - 1:
                # buffer 末端：非边界用调用方传入的 last value；边界用 override
                #（terminated=0, truncated=V(terminal)）。
                continuing_next = next_v
            else:
                continuing_next = self.values[t + 1]
            next_value = boundary * self.next_value_override[t] + (1.0 - boundary) * continuing_next
            delta = self.rewards[t] + gamma * next_value - self.values[t]
            adv = delta + gamma * lam * (1.0 - boundary) * adv
            self.advantages[t] = adv
        self.returns = self.advantages + self.values

    def iter_minibatches(self, batch_size: int):
        B = self.T * self.N * self.K
        obs = self.obs.reshape(B, self.obs_dim)
        global_state = self.global_states.reshape(B, self.global_state_dim)
        agent_ids = self.agent_ids.expand(self.T, self.N, self.K).reshape(B)
        act = self.actions.reshape(B, self.action_dim)
        logp = self.logprobs.reshape(B)
        adv = self.advantages.reshape(B)
        ret = self.returns.reshape(B)
        val = self.values.reshape(B)

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        indices = torch.randperm(B, device=self.device)
        for start in range(0, B, batch_size):
            idx = indices[start : start + batch_size]
            yield MAPPORolloutBatch(
                obs=obs[idx],
                global_state=global_state[idx],
                agent_ids=agent_ids[idx],
                actions=act[idx],
                old_logprobs=logp[idx],
                advantages=adv[idx],
                returns=ret[idx],
                old_values=val[idx],
            )


# ---------- PPO 更新步 ----------
@dataclass
class PPOUpdateStats:
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_frac: float
    explained_variance: float
    grad_norm: float
    learning_rate: float
    log_std_mean: float


def mappo_update(
    model: MAPPOActorCritic,
    optimizer: torch.optim.Optimizer,
    buffer: MAPPORolloutBuffer,
    *,
    clip_eps: float,
    value_clip_eps: float,
    entropy_coef: float,
    value_coef: float,
    max_grad_norm: float,
    minibatch_size: int,
    update_epochs: int,
    target_kl: float,
    policy_coef: float = 1.0,
) -> PPOUpdateStats:
    """MAPPO 更新：actor 用局部 obs，critic 用 centralized global obs。

    policy_coef 默认 1.0；warm-start 场景下，前若干个 update 把它和 entropy_coef
    一起设为 0，让 critic 在 random init / shape-changed 之后先稳到合理 EV，
    避免随机 advantage 把已经训练好的 actor 冲烂。
    target_kl 早停只在 policy_coef > 0 时生效（warmup 阶段 KL 应恒为 0）。
    """
    pls, vls, ents, kls, clip_fracs, gnorms = [], [], [], [], [], []

    all_returns = buffer.returns.reshape(-1)
    all_values = buffer.values.reshape(-1)
    var_y = all_returns.var(unbiased=False).clamp_min(1e-8)
    explained_variance = float(1.0 - (all_returns - all_values).var(unbiased=False) / var_y)

    stop_early = False
    for _ in range(update_epochs):
        if stop_early:
            break
        for batch in buffer.iter_minibatches(minibatch_size):
            new_logp, entropy, new_value = model.evaluate_for_agents(
                batch.obs, batch.global_state, batch.agent_ids, batch.actions
            )

            log_ratio = new_logp - batch.old_logprobs
            ratio = log_ratio.exp()
            surr1 = ratio * batch.advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * batch.advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_pred_clipped = batch.old_values + torch.clamp(
                new_value - batch.old_values, -value_clip_eps, value_clip_eps
            )
            value_loss_unclipped = (new_value - batch.returns).pow(2)
            value_loss_clipped = (value_pred_clipped - batch.returns).pow(2)
            value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

            entropy_loss = -entropy.mean()
            loss = (
                policy_coef * policy_loss
                + value_coef * value_loss
                + policy_coef * entropy_coef * entropy_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean().item()

            pls.append(policy_loss.item())
            vls.append(value_loss.item())
            ents.append(entropy.mean().item())
            kls.append(approx_kl)
            clip_fracs.append(clip_frac)
            gnorms.append(float(grad_norm))

            if policy_coef > 0.0 and target_kl > 0.0 and approx_kl > target_kl * 1.5:
                stop_early = True
                break

    lr = optimizer.param_groups[0]["lr"]
    return PPOUpdateStats(
        policy_loss=float(np.mean(pls)) if pls else 0.0,
        value_loss=float(np.mean(vls)) if vls else 0.0,
        entropy=float(np.mean(ents)) if ents else 0.0,
        approx_kl=float(np.mean(kls)) if kls else 0.0,
        clip_frac=float(np.mean(clip_fracs)) if clip_fracs else 0.0,
        explained_variance=explained_variance,
        grad_norm=float(np.mean(gnorms)) if gnorms else 0.0,
        learning_rate=lr,
        log_std_mean=float(model.log_std.detach().mean().item()),
    )
