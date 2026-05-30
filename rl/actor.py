"""MAPPO 去中心化 Actor：历史观测 + 邻居 attention + tanh 高斯策略。"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# tanh 动作压缩时的数值稳定项，避免 log(0) 与 atanh 边界溢出
_ACTION_SQUASH_EPS = 1e-6
# 本船历史观测维度（不含邻居信息）
_OWN_OBS_DIM = 56
# 参与 attention 的邻居数量
_NEIGHBOR_COUNT = 3
# 单个邻居观测维度
_NEIGHBOR_OBS_DIM = 5
# 邻居观测总维度，供 attention 模块输入
_ATTENTION_OBS_DIM = _NEIGHBOR_COUNT * _NEIGHBOR_OBS_DIM
# Actor 网络结构固定在本模块，避免从训练 config 动态改结构。
_ACTOR_HIDDEN_DIMS = (256, 256)
_ACTOR_ACTIVATION = nn.Tanh
_LOG_STD_INIT = -0.5


def _atanh(x: torch.Tensor) -> torch.Tensor:
    """Stable inverse tanh for values in (-1, 1)."""
    x = torch.clamp(x, -1.0 + _ACTION_SQUASH_EPS, 1.0 - _ACTION_SQUASH_EPS)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def _squash_log_det(action: torch.Tensor) -> torch.Tensor:
    """Log absolute det of tanh Jacobian, summed over action dims."""
    action = torch.clamp(action, -1.0 + _ACTION_SQUASH_EPS, 1.0 - _ACTION_SQUASH_EPS)
    return torch.log(torch.clamp(1.0 - action.pow(2), min=_ACTION_SQUASH_EPS)).sum(dim=-1)


class SquashedDiagonalGaussian:
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


class AttentionCollisionAvoidance(nn.Module):
    """Single-head scaled dot-product attention over the three neighbour tugs."""

    def __init__(self, own_feat_dim: int, neigh_feat_dim: int, embed_dim: int = 64) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.w_q = nn.Linear(own_feat_dim, embed_dim)
        self.w_k = nn.Linear(neigh_feat_dim, embed_dim)
        self.w_v = nn.Linear(neigh_feat_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, embed_dim)
        self.scale = math.sqrt(float(embed_dim))

    def forward(
        self,
        e_own: torch.Tensor,
        e_neighbors: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.w_q(e_own).unsqueeze(1)
        key = self.w_k(e_neighbors)
        value = self.w_v(e_neighbors)
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, value).squeeze(1)
        agg_feat = self.fc_out(context)
        return agg_feat, attn_weights.squeeze(1)


class MAPPOActor(nn.Module):
    """去中心化 Actor：只看单个拖轮的局部观察 o_i。"""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
    ) -> None:
        super().__init__()
        act_cls = _ACTOR_ACTIVATION

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.own_obs_dim = _OWN_OBS_DIM
        self.neighbor_count = _NEIGHBOR_COUNT
        self.neighbor_obs_dim = _NEIGHBOR_OBS_DIM
        expected_obs_dim = _OWN_OBS_DIM + _ATTENTION_OBS_DIM
        if self.obs_dim != expected_obs_dim:
            raise ValueError(
                f"attention actor expects obs_dim={expected_obs_dim}, got {self.obs_dim}"
            )

        self.own_encoder = nn.Sequential(
            nn.Linear(self.own_obs_dim, 128),
            act_cls(),
            nn.Linear(128, 64),
            act_cls(),
        )
        self.neigh_encoder = nn.Sequential(
            nn.Linear(self.neighbor_obs_dim, 64),
            act_cls(),
            nn.Linear(64, 64),
            act_cls(),
        )
        self.attention_block = AttentionCollisionAvoidance(
            own_feat_dim=64,
            neigh_feat_dim=64,
            embed_dim=64,
        )

        actor_layers: list[nn.Module] = []
        in_dim = 128
        for h in _ACTOR_HIDDEN_DIMS:
            actor_layers.append(nn.Linear(in_dim, h))
            actor_layers.append(act_cls())
            in_dim = h
        self.actor_head = nn.Sequential(*actor_layers)
        self.policy_mean = nn.Linear(in_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), _LOG_STD_INIT))

        self._init_weights()

    def _init_weights(self) -> None:
        for module in (self.own_encoder, self.neigh_encoder, self.attention_block, self.actor_head):
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=math.sqrt(2.0))
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.policy_mean.weight, gain=0.01)
        nn.init.zeros_(self.policy_mean.bias)

    def _split_obs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        own_obs = obs[..., :self.own_obs_dim]
        neigh_flat = obs[..., self.own_obs_dim:]
        neighbors_obs = neigh_flat.reshape(
            *obs.shape[:-1], self.neighbor_count, self.neighbor_obs_dim
        )
        return own_obs, neighbors_obs

    def _features(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        leading_shape = obs.shape[:-1]
        flat_obs = obs.reshape(-1, self.obs_dim)
        own_obs, neighbors_obs = self._split_obs(flat_obs)
        e_own = self.own_encoder(own_obs)
        n = self.neighbor_count
        e_neigh = self.neigh_encoder(
            neighbors_obs.reshape(-1, self.neighbor_obs_dim)
        ).reshape(-1, n, 64)
        env_threat_feat, weights = self.attention_block(e_own, e_neigh)
        combined = torch.cat([e_own, env_threat_feat], dim=-1)
        return combined.reshape(*leading_shape, -1), weights.reshape(*leading_shape, n)

    def policy(self, obs: torch.Tensor) -> torch.Tensor:
        features, _ = self._features(obs)
        h = self.actor_head(features)
        return self.policy_mean(h)

    def attention_weights(self, obs: torch.Tensor) -> torch.Tensor:
        _, weights = self._features(obs)
        return weights

    def _dist(self, obs: torch.Tensor) -> SquashedDiagonalGaussian:
        return SquashedDiagonalGaussian(self.policy(obs), self.log_std)

    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        dist = self._dist(obs)
        action, logprob, _ = dist.sample(deterministic=deterministic)
        return action, logprob, None

    def evaluate_actions(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self._dist(obs)
        return dist.log_prob(action), dist.entropy()
