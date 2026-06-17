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
_LOG_STD_INIT = -0.5


def _atanh(x: torch.Tensor) -> torch.Tensor:
    """对 (-1, 1) 区间内的值做数值稳定的反 tanh。"""
    x = torch.clamp(x, -1.0 + _ACTION_SQUASH_EPS, 1.0 - _ACTION_SQUASH_EPS)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def _squash_log_det(action: torch.Tensor) -> torch.Tensor:
    """根据概率论变量变换定理，计算tanh 动作压缩（squash）带来的对数雅可比修正项，用于稳定训练。"""
    action = torch.clamp(action, -1.0 + _ACTION_SQUASH_EPS, 1.0 - _ACTION_SQUASH_EPS)
    return torch.log(torch.clamp(1.0 - action.pow(2), min=_ACTION_SQUASH_EPS)).sum(dim=-1)


class SquashedDiagonalGaussian:
    """对角高斯策略，经 tanh 压缩到 [-1, 1]。"""

    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor) -> None:
        self.mean = mean
        self.std = log_std.exp().expand_as(mean)
        self.base = Normal(mean, self.std)

    def sample(self, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        
        1. 如果 deterministic 为 True，则直接使用均值作为动作。(确定模式，用于评估和可视化)
        2. 如果 deterministic 为 False，则使用基础高斯分布采样得到动作。（随机模式，用于训练）
        3. 对动作进行 tanh 压缩。
        4. 计算修正后的动作对数概率。
        5. 计算基础高斯熵。
        6. 返回动作、对数概率和熵。
        """
        pre_tanh = self.mean if deterministic else self.base.rsample()
        action = torch.tanh(pre_tanh)
        logprob = self.base.log_prob(pre_tanh).sum(dim=-1) - _squash_log_det(action) # 计算修正后的动作对数概率
        entropy = self.base.entropy().sum(dim=-1)
        return action, logprob, entropy

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        """
        1. 对动作进行 clamp，确保在 [-1, 1] 区间内。
        2. 计算修正后的动作对数概率。
        3. 返回动作对数概率。
        """
        action = torch.clamp(action, -1.0 + _ACTION_SQUASH_EPS, 1.0 - _ACTION_SQUASH_EPS)
        pre_tanh = _atanh(action)
        return self.base.log_prob(pre_tanh).sum(dim=-1) - _squash_log_det(action)

    def entropy(self) -> torch.Tensor:
        # 此处 tanh-高斯熵无解析闭式；用基础高斯熵作为 PPO 正则化的稳定代理。
        # 计算智能体动作分布的熵，衡量动作的随机性。熵越大，动作越随机。
        return self.base.entropy().sum(dim=-1)


class AttentionCollisionAvoidance(nn.Module):
    """对三艘邻居拖轮做单头缩放点积注意力。"""

    def __init__(self, own_feat_dim: int, neigh_feat_dim: int, embed_dim: int = 64) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.w_q = nn.Linear(own_feat_dim, embed_dim)
        self.w_k = nn.Linear(neigh_feat_dim, embed_dim)
        self.w_v = nn.Linear(neigh_feat_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, embed_dim)
        self.scale = math.sqrt(float(embed_dim))

    def forward(self, e_own: torch.Tensor, e_neighbors: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        1. 计算自身特征的查询向量。
        2. 计算邻居特征的键向量和值向量。
        3. 计算注意力分数。
        4. 计算注意力权重。
        5. 计算聚合特征。
        6. 返回聚合特征和注意力权重。
        """
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

    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()

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
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
        )
        self.neigh_encoder = nn.Sequential(
            nn.Linear(self.neighbor_obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.attention_block = AttentionCollisionAvoidance(
            own_feat_dim=64,
            neigh_feat_dim=64,
            embed_dim=64,
        )

        self.actor_head = nn.Sequential(
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
        )
        self.policy_mean = nn.Linear(256, action_dim)
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

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:    
        dist = self._dist(obs)
        action, logprob, _ = dist.sample(deterministic=deterministic)
        return action, logprob, None

    def evaluate_actions(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self._dist(obs)
        return dist.log_prob(action), dist.entropy()
        