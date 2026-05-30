"""MAPPO 算法实现：去中心化 Actor + 集中式 Critic + Rollout Buffer + 训练步。

设计要点：
- Actor 只看单 agent 局部观察；Critic 看 canonical global state。
- 4 个智能体共享同一份 actor（参数共享），观察都是"以自身为参考系"的相对量。
- 训练时把 (T, N, n_tugs, *) 展平到 (T*N*n_tugs, *) 处理。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from rl.actor import MAPPOActor
from rl.critic import MAPPOCritic


class MAPPOActorCritic(nn.Module):
    """MAPPO 网络：去中心化 actor + 集中式 critic。

    Actor 只看单个拖轮的局部观察 o_i，执行时不依赖其他 agent。
    Critic 看一份与 agent 无关的 canonical global state（由 env.get_global_state()
    给出，量都在大船船体系下表达），并输出每个 agent 的 V_i(s)。
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_agents: int,
        global_state_dim: int | None = None,
    ) -> None:
        super().__init__()
        if global_state_dim is None:
            raise ValueError("global_state_dim is required for MAPPO critic")

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.global_state_dim = int(global_state_dim)

        self.actor = MAPPOActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
        )
        self.critic = MAPPOCritic(
            global_state_dim=self.global_state_dim,
            n_agents=n_agents,
        )

    @property
    def log_std(self) -> torch.Tensor:
        return self.actor.log_std

    def policy(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor.policy(obs)

    def get_values(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.critic.get_values(global_state)

    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        return self.actor.act(obs, deterministic=deterministic)

    def evaluate_for_agents(
        self,
        obs: torch.Tensor,
        global_state: torch.Tensor,
        agent_ids: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logprob, entropy = self.actor.evaluate_actions(obs, action)
        values_all = self.critic.get_values(global_state)
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
    value_loss_collision: float
    value_loss_noncollision: float
    explained_variance_collision: float
    explained_variance_noncollision: float
    grad_norm: float
    learning_rate: float
    log_std_mean: float


def _value_diagnostics(
    returns: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor | None,
) -> tuple[float, float]:
    """Return raw value MSE loss and EV for a masked rollout subset."""
    if mask is None:
        return float("nan"), float("nan")
    mask = mask.reshape(-1).bool()
    if int(mask.sum().item()) < 2:
        return float("nan"), float("nan")
    y = returns[mask]
    y_pred = values[mask]
    value_loss = float(0.5 * (y_pred - y).pow(2).mean().item())
    var_y = y.var(unbiased=False)
    if float(var_y.item()) < 1e-8:
        return value_loss, float("nan")
    ev = float(1.0 - (y - y_pred).var(unbiased=False) / var_y)
    return value_loss, ev


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
    collision_mask: torch.Tensor | None = None,
    noncollision_mask: torch.Tensor | None = None,
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
    value_loss_collision, ev_collision = _value_diagnostics(
        all_returns, all_values, collision_mask
    )
    value_loss_noncollision, ev_noncollision = _value_diagnostics(
        all_returns, all_values, noncollision_mask
    )

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
        value_loss_collision=value_loss_collision,
        value_loss_noncollision=value_loss_noncollision,
        explained_variance_collision=ev_collision,
        explained_variance_noncollision=ev_noncollision,
        grad_norm=float(np.mean(gnorms)) if gnorms else 0.0,
        learning_rate=lr,
        log_std_mean=float(model.log_std.detach().mean().item()),
    )
