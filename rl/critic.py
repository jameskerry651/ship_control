"""MAPPO 集中式 Critic：看 canonical global state，输出每个 agent 的 V_i(s)。"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


_CRITIC_HIDDEN_DIMS = (512, 512, 512)
_CRITIC_ACTIVATION = nn.Tanh


class MAPPOCritic(nn.Module):
    """集中式 Critic：输入为 env.get_global_state() 给出的 canonical global state。"""

    def __init__(
        self,
        global_state_dim: int,
        n_agents: int,
    ) -> None:
        super().__init__()
        act_cls = _CRITIC_ACTIVATION

        self.global_state_dim = int(global_state_dim)
        self.n_agents = n_agents

        critic_layers: list[nn.Module] = []
        in_dim = self.global_state_dim
        for h in _CRITIC_HIDDEN_DIMS:
            critic_layers.append(nn.Linear(in_dim, h))
            critic_layers.append(act_cls())
            in_dim = h
        critic_layers.append(nn.Linear(in_dim, n_agents))
        self.critic = nn.Sequential(*critic_layers)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.critic.modules():
            if isinstance(m, nn.Linear):
                gain = 1.0 if m.out_features == self.n_agents else math.sqrt(2.0)
                nn.init.orthogonal_(m.weight, gain=gain)
                nn.init.zeros_(m.bias)

    def get_values(self, global_state: torch.Tensor) -> torch.Tensor:
        """返回集中式 critic 的 value，形状 (..., n_agents)。"""
        leading_shape = global_state.shape[:-1]
        flat = global_state.reshape(-1, self.global_state_dim)
        values = self.critic(flat)
        return values.reshape(*leading_shape, self.n_agents)
