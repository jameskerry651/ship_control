"""MAPPO 集中式 Critic：看 canonical global state + agent one-hot，输出 V_i(s)。

输入为 canonical global state 拼接被估值 agent 的 one-hot 身份。trunk 因此能学习
「以 agent_i 为中心」的特征，而非像纯多头那样仅靠输出层区分。前向时对 K 个 agent
各拼一次 one-hot 过同一 trunk，输出标量并拼成 (..., n_agents)，对外形状不变。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class MAPPOCritic(nn.Module):
    """集中式 Critic：输入为 global state ⊕ agent one-hot，输出该 agent 的标量 value。"""

    def __init__(self, global_state_dim: int, n_agents: int, popart_beta: float = 3e-4) -> None:
        super().__init__()

        self.global_state_dim = int(global_state_dim)
        self.n_agents = n_agents
        # trunk 实际输入维度 = global state + agent one-hot
        self.input_dim = self.global_state_dim + n_agents

        # 固定的 agent one-hot 身份表，注册为 buffer 随模型迁移设备/保存
        self.register_buffer("_agent_onehot", torch.eye(n_agents))

        # ---- PopArt 归一化统计量 ----
        # critic 内部在归一化空间回归（value loss 量级稳定），对外反归一化输出
        # 真实尺度的 value，使 GAE/采样不受影响。每次 update 用 returns 的运行
        # 均值/方差更新 mu/sigma，并同步 rescale 输出层权重以保持输出不变。
        # 拖轮同质、共享 reward 尺度，故 mu/sigma 用单一标量。
        self.popart_beta = float(popart_beta)
        self.register_buffer("popart_mu", torch.zeros(1))
        self.register_buffer("popart_sigma", torch.ones(1))
        # 二阶矩，用于在线估计方差：sigma^2 = nu - mu^2
        self.register_buffer("popart_nu", torch.ones(1))
        self.register_buffer("popart_initialized", torch.zeros(1))

        self.critic = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            # 对首层输出做 LayerNorm，吸收 global state 里 ship/tug/加速度
            # 不同量纲特征的尺度差异，避免 Tanh 饱和、稳定早期训练
            nn.LayerNorm(512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            # agent-centric：输出单标量 value，agent 身份由输入 one-hot 提供
            nn.Linear(512, 1),
        )

        self._init_weights()

    @property
    def _out_layer(self) -> nn.Linear:
        """输出层（critic Sequential 的最后一个 Linear），PopArt rescale 的对象。"""
        return self.critic[-1]

    def normalized_values(self, global_state: torch.Tensor) -> torch.Tensor:
        """critic 原始输出，处于归一化空间，形状 (..., n_agents)。用于计算 value loss。

        对 K 个 agent 各拼一次 one-hot 身份过同一 trunk，得到各自的标量 value 后
        拼成 (..., n_agents)，对外形状与纯多头版本一致。
        """
        leading_shape = global_state.shape[:-1]
        flat = global_state.reshape(-1, self.global_state_dim)  # (B, G)
        b = flat.shape[0]
        k = self.n_agents
        # 每个样本复制 K 份，分别拼接 agent one-hot：(B, K, G+K) -> (B*K, G+K)
        state_rep = flat.unsqueeze(1).expand(b, k, self.global_state_dim)
        onehot = self._agent_onehot.unsqueeze(0).expand(b, k, k)
        critic_in = torch.cat([state_rep, onehot], dim=-1).reshape(b * k, self.input_dim)
        values = self.critic(critic_in).reshape(b, k)  # (B, K)
        return values.reshape(*leading_shape, self.n_agents)

    def get_values(self, global_state: torch.Tensor) -> torch.Tensor:
        """返回真实尺度的 value（反归一化），形状 (..., n_agents)。"""
        norm = self.normalized_values(global_state)
        return norm * self.popart_sigma + self.popart_mu

    def normalize_returns(self, returns: torch.Tensor) -> torch.Tensor:
        """把真实尺度的 return target 映射到归一化空间，与 normalized_values 对齐。"""
        return (returns - self.popart_mu) / self.popart_sigma

    @torch.no_grad()
    def update_popart(self, returns: torch.Tensor) -> None:
        """用本轮 returns 更新 PopArt 统计量，并 rescale 输出层以保持反归一化输出不变。

        ART（Adaptively Rescaling Targets）：mu/sigma 用一阶/二阶矩的 EMA 估计。
        POP（Preserving Outputs Precisely）：mu/sigma 改变后，按
            w_new = w_old * sigma_old / sigma_new
            b_new = (b_old * sigma_old + mu_old - mu_new) / sigma_new
        调整输出层，使 sigma*f(x)+mu 在更新前后对任意输入保持一致。
        """
        flat = returns.reshape(-1).float()
        if flat.numel() == 0:
            return

        old_mu = self.popart_mu.clone()
        old_sigma = self.popart_sigma.clone()

        batch_mu = flat.mean()
        batch_nu = (flat * flat).mean()
        if self.popart_initialized.item() < 1.0:
            # 首个 batch 直接用其统计量初始化，避免从 (0,1) 缓慢爬升
            new_mu = batch_mu.reshape(1)
            new_nu = batch_nu.reshape(1)
            self.popart_initialized.fill_(1.0)
        else:
            beta = self.popart_beta
            new_mu = (1.0 - beta) * self.popart_mu + beta * batch_mu
            new_nu = (1.0 - beta) * self.popart_nu + beta * batch_nu

        new_sigma = torch.sqrt(torch.clamp(new_nu - new_mu * new_mu, min=1e-4))

        self.popart_mu = new_mu
        self.popart_nu = new_nu
        self.popart_sigma = new_sigma

        # POP：rescale 输出层，保持 get_values 输出在统计量更新前后不变
        w = self._out_layer.weight
        b = self._out_layer.bias
        w.mul_(old_sigma / new_sigma)
        b.copy_((b * old_sigma + old_mu - new_mu) / new_sigma)

    def _init_weights(self) -> None:
        out_layer = self._out_layer
        for m in self.critic.modules():
            if isinstance(m, nn.Linear):
                # 输出层（标量 value）用 gain=1.0，隐藏层用 sqrt(2)（配 Tanh）
                gain = 1.0 if m is out_layer else math.sqrt(2.0)
                nn.init.orthogonal_(m.weight, gain=gain)
                nn.init.zeros_(m.bias)
