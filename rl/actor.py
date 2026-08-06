"""MAPPO 去中心化 Actor：历史观测 + 邻居 attention + tanh 高斯策略。"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from env.obs_spec import (
    DEFAULT_OBSERVATION_SPEC,
    LEGACY_OBSERVATION_SPEC_V1,
    ObservationSpec,
    _NEIGHBOR_COUNT,
    _NEIGHBOR_OBS_DIM,
)
from rl.temporal import TemporalTransformerEncoder

# tanh 动作压缩时的数值稳定项，避免 log(0) 与 atanh 边界溢出
_ACTION_SQUASH_EPS = 1e-6
_LOG_STD_INIT = -0.5
# log_std 的硬裁剪边界，防止 std 指数爆炸（上界）或熵塌缩（下界）导致训练不稳定
_LOG_STD_MIN = -5.0
_LOG_STD_MAX = 2.0


def _resolve_observation_spec(
    observation_spec: ObservationSpec | Mapping[str, Any] | None,
    *,
    obs_dim: int,
    hist_len: int | None = None,
) -> ObservationSpec:
    """解析并校验 Actor 使用的观测契约。

    无显式规格时识别默认 93 维和旧 89 维布局；为旧调用传入 ``hist_len`` 时，可从
    ``obs_dim`` 反推出预瞄点数量。新训练代码应始终显式传入规格。
    """
    if isinstance(observation_spec, ObservationSpec):
        spec = observation_spec
    elif observation_spec is not None:
        spec = ObservationSpec.from_dict(observation_spec)
    elif hist_len is None and int(obs_dim) == DEFAULT_OBSERVATION_SPEC.total_dim:
        spec = DEFAULT_OBSERVATION_SPEC
    elif (
        int(obs_dim) == LEGACY_OBSERVATION_SPEC_V1.total_dim
        and (hist_len is None or int(hist_len) == LEGACY_OBSERVATION_SPEC_V1.history_len)
    ):
        spec = LEGACY_OBSERVATION_SPEC_V1
    else:
        resolved_hist_len = (
            DEFAULT_OBSERVATION_SPEC.history_len if hist_len is None else int(hist_len)
        )
        fixed_without_preview = (
            resolved_hist_len * DEFAULT_OBSERVATION_SPEC.history_token_dim
            + DEFAULT_OBSERVATION_SPEC.ship_relative_dim
            + DEFAULT_OBSERVATION_SPEC.thruster_state_dim
            + DEFAULT_OBSERVATION_SPEC.slot_target_dim
            + DEFAULT_OBSERVATION_SPEC.hull_clearance_dim
            + DEFAULT_OBSERVATION_SPEC.attention_dim
        )
        preview_size = int(obs_dim) - fixed_without_preview
        point_dim = DEFAULT_OBSERVATION_SPEC.preview_point_dim
        if preview_size < 0 or preview_size % point_dim != 0:
            raise ValueError(
                "cannot infer ObservationSpec from "
                f"obs_dim={obs_dim}, hist_len={resolved_hist_len}; "
                "pass observation_spec explicitly"
            )
        spec = ObservationSpec(
            history_len=resolved_hist_len,
            preview_count=preview_size // point_dim,
            neighbor_count=DEFAULT_OBSERVATION_SPEC.neighbor_count,
        )

    if hist_len is not None and int(hist_len) != spec.history_len:
        raise ValueError(
            f"hist_len={hist_len} disagrees with ObservationSpec.history_len="
            f"{spec.history_len}"
        )
    if int(obs_dim) != spec.total_dim:
        raise ValueError(
            f"ObservationSpec expects obs_dim={spec.total_dim}, got {obs_dim}"
        )
    return spec


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
        # 裁剪 log_std 到安全区间，避免 std 过大（梯度爆炸）或过小（熵塌缩、过早收敛）
        log_std = torch.clamp(log_std, _LOG_STD_MIN, _LOG_STD_MAX)
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
    """对动态数量的邻居拖轮做单头缩放点积注意力。"""

    def __init__(self, own_feat_dim: int, neigh_feat_dim: int, embed_dim: int = 64) -> None:
        """
        初始化单头缩放点积注意力模块。

        Parameters
        ----------
        own_feat_dim : int
            本船特征维度（查询向量输入维度）。
        neigh_feat_dim : int
            单个邻居特征维度（键/值向量输入维度）。
        embed_dim : int, optional
            注意力嵌入维度，即 Q/K/V 投影后的统一维度，默认 64。
        """
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.w_q = nn.Linear(own_feat_dim, embed_dim)
        self.w_k = nn.Linear(neigh_feat_dim, embed_dim)
        self.w_v = nn.Linear(neigh_feat_dim, embed_dim)
        # 输出投影：对聚合后的上下文向量做线性变换
        self.fc_out = nn.Linear(embed_dim, embed_dim)
        # 缩放因子 sqrt(d_k)，防止点积过大导致 softmax 梯度消失
        self.scale = math.sqrt(float(embed_dim))

    def forward(self, e_own: torch.Tensor, e_neighbors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, value).squeeze(1)
        agg_feat = self.fc_out(context)
        """
        attn_weights 是注意力权重，表示每个邻居对本船的影响程度。
        agg_feat 是 attention 聚合后的邻居上下文向量，可以理解为：本船根据当前状态，从 3 艘邻居里「加权汇总」出的一份 64 维环境威胁摘要。
        """
        return agg_feat, attn_weights.squeeze(1)


class MAPPOActor(nn.Module):
    """去中心化 Actor：只看单个拖轮的局部观察 o_i。"""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        observation_spec: ObservationSpec | Mapping[str, Any] | None = None,
    ) -> None:
        """
        初始化去中心化 Actor 网络。

        整体结构：
            观测 → [本船编码器 + 邻居编码器] → Attention 聚合 → 拼接特征 → Actor Head → 策略均值

        Parameters
        ----------
        obs_dim : int
            总观测维度，必须与 ``observation_spec.total_dim`` 一致。
        action_dim : int
            动作空间维度，即策略输出的连续动作分量数。
        """
        super().__init__()

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.observation_spec = _resolve_observation_spec(
            observation_spec, obs_dim=self.obs_dim
        )
        self.own_obs_dim = self.observation_spec.own_dim
        self.neighbor_count = self.observation_spec.neighbor_count
        self.neighbor_obs_dim = self.observation_spec.neighbor_dim

        # 本船特征编码器：输入维度由 ObservationSpec 派生。
        self.own_encoder = nn.Sequential(
            nn.Linear(self.own_obs_dim, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.Tanh(),
        )
        # 邻居特征编码器：将单个邻居风险特征 (10维) 编码为 64 维特征向量，供 Attention 模块使用
        self.neigh_encoder = nn.Sequential(
            nn.Linear(self.neighbor_obs_dim, 64),
            nn.LayerNorm(64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        # 注意力碰撞规避模块：邻居数量由 ObservationSpec 派生。
        self.attention_block = AttentionCollisionAvoidance(own_feat_dim=64,neigh_feat_dim=64,embed_dim=64,)

        # Actor 头部网络：将拼接后的 128 维特征 (64本船 + 64威胁摘要) 映射到 256 维隐藏层
        self.actor_head = nn.Sequential(
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
        )
        # 策略均值头：将 256 维隐藏层映射为 action_dim 维的动作均值
        self.policy_mean = nn.Linear(256, action_dim)
        # 可学习的对数标准差参数，初始化为 -0.5，用于控制策略的探索幅度
        self.log_std = nn.Parameter(torch.full((action_dim,), _LOG_STD_INIT))

        self._init_weights()

    def _init_weights(self) -> None:
        """
        自定义权重初始化策略。

        - 隐藏层（编码器 + Attention + Actor Head）使用正交初始化 (gain=√2)，
          配合 Tanh 激活函数，有助于缓解梯度消失/爆炸问题。
        - 策略均值输出层使用较小增益 (gain=0.1) 的正交初始化，
          使训练初期策略接近均匀随机，避免过早收敛到次优动作。
        - 所有偏置初始化为零。
        """
        for module in (self.own_encoder, self.neigh_encoder, self.attention_block, self.actor_head):
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=math.sqrt(2.0))
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.policy_mean.weight, gain=0.1)
        nn.init.zeros_(self.policy_mean.bias)

    def _split_obs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        将原始观测张量拆分为本船观测和邻居观测。

        观测布局：[own_obs | neigh_0 | ... | neigh_n]，各段由 ObservationSpec 定义。

        Parameters
        ----------
        obs : torch.Tensor
            原始观测张量，shape=(..., obs_dim)。

        Returns
        -------
        own_obs : torch.Tensor
            本船观测，shape=(..., observation_spec.own_dim)。
        neighbors_obs : torch.Tensor
            邻居观测，shape=(..., neighbor_count, neighbor_obs_dim)。
        """
        own_obs = obs[..., self.observation_spec.own_slice]
        neigh_flat = obs[..., self.observation_spec.neighbor_slice]
        neighbors_obs = neigh_flat.reshape(
            *obs.shape[:-1], self.neighbor_count, self.neighbor_obs_dim
        )
        return own_obs, neighbors_obs

    def _features(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        特征提取主流程：编码 → Attention 聚合 → 拼接。

        处理步骤：
            1. 将输入展平为二维张量以适配网络层。
            2. 拆分本船观测与邻居观测。
            3. 分别通过本船编码器和邻居编码器得到特征向量。
            4. 通过 Attention 模块聚合邻居特征，得到环境威胁摘要。
            5. 将本船特征与环境威胁摘要拼接为 128 维组合特征。
            6. 恢复原始 leading shape 后返回。

        Parameters
        ----------
        obs : torch.Tensor
            原始观测张量，shape=(..., obs_dim)。支持任意 leading shape（如 batch、时间步等）。

        Returns
        -------
        combined : torch.Tensor
            组合特征，shape=(..., 128)，由 64 维本船特征和 64 维威胁摘要拼接而成。
        weights : torch.Tensor
            注意力权重，shape=(..., neighbor_count=3)，反映每个邻居对当前决策的影响程度。
        """
        leading_shape = obs.shape[:-1]
        flat_obs = obs.reshape(-1, self.obs_dim)
        own_obs, neighbors_obs = self._split_obs(flat_obs)
        e_own = self.own_encoder(own_obs)
        n = self.neighbor_count
        e_neigh = self.neigh_encoder(
            neighbors_obs.reshape(-1, self.neighbor_obs_dim)
        ).reshape(-1, n, self.attention_block.embed_dim)
        env_threat_feat, weights = self.attention_block(e_own, e_neigh)
        combined = torch.cat([e_own, env_threat_feat], dim=-1)
        return combined.reshape(*leading_shape, -1), weights.reshape(*leading_shape, n)

    def policy(self, obs: torch.Tensor) -> torch.Tensor:
        """
        计算策略均值（未经 tanh 压缩的 pre-tanh 动作均值）。

        前向流程：obs → 特征提取 → Actor Head → policy_mean

        Parameters
        ----------
        obs : torch.Tensor
            原始观测张量，shape=(..., obs_dim)。

        Returns
        -------
        torch.Tensor
            策略均值，shape=(..., action_dim)。
        """
        features, _ = self._features(obs)
        h = self.actor_head(features)
        return self.policy_mean(h)

    def attention_weights(self, obs: torch.Tensor) -> torch.Tensor:
        """
        提取当前观测下的注意力权重，用于可视化和分析智能体的碰撞规避决策。

        Parameters
        ----------
        obs : torch.Tensor
            原始观测张量，shape=(..., obs_dim)。

        Returns
        -------
        torch.Tensor
            注意力权重，shape=(..., neighbor_count=3)，
            每个元素表示对应邻居在决策中的权重占比。
        """
        _, weights = self._features(obs)
        return weights

    def _dist(self, obs: torch.Tensor) -> SquashedDiagonalGaussian:
        """
        构造 tanh 压缩的对角高斯策略分布。

        Parameters
        ----------
        obs : torch.Tensor
            原始观测张量。

        Returns
        -------
        SquashedDiagonalGaussian
            经 tanh 压缩到 [-1, 1] 的对角高斯分布，均值由 policy() 计算，
            标准差由可学习参数 log_std 决定。
        """
        return SquashedDiagonalGaussian(self.policy(obs), self.log_std)

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:    
        """
        采样动作，供环境交互时调用。

        Parameters
        ----------
        obs : torch.Tensor
            当前时刻的观测，shape=(..., obs_dim)。
        deterministic : bool, optional
            是否使用确定性模式（直接输出均值），默认 False。
            - True：用于评估/可视化，输出确定性动作。
            - False：用于训练，从策略分布中采样以鼓励探索。

        Returns
        -------
        action : torch.Tensor
            采样得到的动作，shape=(..., action_dim)，值域 [-1, 1]。
        logprob : torch.Tensor
            动作的对数概率，shape=(...,)，用于 PPO 的重要性比率计算。
        None
            占位符（预留隐藏状态接口，当前架构不使用 RNN，故返回 None）。
        """
        dist = self._dist(obs)
        action, logprob, _ = dist.sample(deterministic=deterministic)
        return action, logprob, None

    def evaluate_actions(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        评估给定观测-动作对的对数概率和熵，供 PPO 更新时调用。

        在 PPO 的 on-policy 更新中，需要用当前策略重新评估 rollout 阶段采集的 (obs, action) 对，
        以计算新旧策略的重要性比率 (ratio = exp(new_logprob - old_logprob)) 和熵正则项。

        Parameters
        ----------
        obs : torch.Tensor
            观测张量，shape=(..., obs_dim)。
        action : torch.Tensor
            历史采集的动作，shape=(..., action_dim)，值域应在 [-1, 1] 内。

        Returns
        -------
        log_prob : torch.Tensor
            当前策略下动作的对数概率，shape=(...,)。
        entropy : torch.Tensor
            当前策略的熵（使用基础高斯熵近似），shape=(...,)，
            用于 PPO 的熵正则化损失，鼓励探索。
        """
        dist = self._dist(obs)
        return dist.log_prob(action), dist.entropy()


class TransformerMAPPOActor(nn.Module):
    """去中心化 Actor：对本船历史帧做 Transformer，邻居仍用 Attention。"""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        hist_len: int | None = None,
        observation_spec: ObservationSpec | Mapping[str, Any] | None = None,
        tf_d_model: int = 64,
        tf_nhead: int = 4,
        tf_num_layers: int = 2,
        tf_ffn_dim: int = 128,
        tf_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.observation_spec = _resolve_observation_spec(
            observation_spec,
            obs_dim=self.obs_dim,
            hist_len=hist_len,
        )
        self.own_obs_dim = self.observation_spec.own_dim
        self.neighbor_count = self.observation_spec.neighbor_count
        self.neighbor_obs_dim = self.observation_spec.neighbor_dim
        self.hist_len = self.observation_spec.history_len
        self.token_dim = self.observation_spec.history_token_dim
        self.context_dim = self.observation_spec.own_context_dim
        self.motion_dim = self.observation_spec.motion_dim
        self.action_hist_dim = self.observation_spec.action_history_dim

        self.temporal_encoder = TemporalTransformerEncoder(
            token_dim=self.token_dim,
            hist_len=self.hist_len,
            d_model=int(tf_d_model),
            nhead=int(tf_nhead),
            num_layers=int(tf_num_layers),
            ffn_dim=int(tf_ffn_dim),
            dropout=float(tf_dropout),
            out_dim=64,
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(self.context_dim, 64),
            nn.LayerNorm(64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.own_fuse = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.Tanh(),
        )
        self.neigh_encoder = nn.Sequential(
            nn.Linear(self.neighbor_obs_dim, 64),
            nn.LayerNorm(64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.attention_block = AttentionCollisionAvoidance(
            own_feat_dim=64, neigh_feat_dim=64, embed_dim=64
        )
        self.actor_head = nn.Sequential(
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
        )
        self.policy_mean = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), _LOG_STD_INIT))
        self._init_weights()

    def _init_weights(self) -> None:
        for module in (
            self.context_encoder,
            self.own_fuse,
            self.neigh_encoder,
            self.attention_block,
            self.actor_head,
        ):
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=math.sqrt(2.0))
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.policy_mean.weight, gain=0.1)
        nn.init.zeros_(self.policy_mean.bias)

    def _split_obs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        own_obs = obs[..., self.observation_spec.own_slice]
        neigh_flat = obs[..., self.observation_spec.neighbor_slice]
        neighbors_obs = neigh_flat.reshape(
            *obs.shape[:-1], self.neighbor_count, self.neighbor_obs_dim
        )
        return own_obs, neighbors_obs

    def _own_history_tokens_and_context(
        self, own_obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """own → (tokens K×10, context)。观测布局为全部 motion 再全部 action。"""
        motion = own_obs[..., self.observation_spec.motion_history_slice].reshape(
            *own_obs.shape[:-1], self.hist_len, self.motion_dim
        )
        action = own_obs[..., self.observation_spec.action_history_slice].reshape(
            *own_obs.shape[:-1], self.hist_len, self.action_hist_dim
        )
        tokens = torch.cat([motion, action], dim=-1)
        context = own_obs[..., self.observation_spec.action_history_slice.stop :]
        if context.shape[-1] != self.context_dim:
            raise ValueError(
                f"expected context_dim={self.context_dim}, got {context.shape[-1]}"
            )
        return tokens, context

    def _features(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        leading_shape = obs.shape[:-1]
        flat_obs = obs.reshape(-1, self.obs_dim)
        own_obs, neighbors_obs = self._split_obs(flat_obs)
        tokens, context = self._own_history_tokens_and_context(own_obs)
        e_temporal = self.temporal_encoder(tokens)
        e_context = self.context_encoder(context)
        e_own = self.own_fuse(torch.cat([e_temporal, e_context], dim=-1))
        n = self.neighbor_count
        e_neigh = self.neigh_encoder(
            neighbors_obs.reshape(-1, self.neighbor_obs_dim)
        ).reshape(-1, n, self.attention_block.embed_dim)
        env_threat_feat, weights = self.attention_block(e_own, e_neigh)
        combined = torch.cat([e_own, env_threat_feat], dim=-1)
        return combined.reshape(*leading_shape, -1), weights.reshape(*leading_shape, n)

    def policy(self, obs: torch.Tensor) -> torch.Tensor:
        features, _ = self._features(obs)
        return self.policy_mean(self.actor_head(features))

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


def build_actor(
    arch: str,
    obs_dim: int,
    action_dim: int,
    observation_spec: ObservationSpec | Mapping[str, Any] | None = None,
    **arch_kwargs: Any,
) -> nn.Module:
    """按架构名构造 Actor；gru/lstm 预留接口。"""
    name = str(arch).strip().lower()
    if name == "mlp":
        return MAPPOActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            observation_spec=observation_spec,
        )
    if name == "transformer":
        tf_keys = (
            "hist_len",
            "tf_d_model",
            "tf_nhead",
            "tf_num_layers",
            "tf_ffn_dim",
            "tf_dropout",
        )
        kwargs = {k: arch_kwargs[k] for k in tf_keys if k in arch_kwargs}
        return TransformerMAPPOActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            observation_spec=observation_spec,
            **kwargs,
        )
    if name in ("gru", "lstm"):
        raise NotImplementedError(
            f"actor_arch={name!r} is reserved for the ablation study but not implemented yet"
        )
    raise ValueError(
        f"unknown actor_arch={arch!r}; expected one of: mlp, transformer, gru, lstm"
    )
