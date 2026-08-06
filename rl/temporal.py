"""Actor 时序编码器：Transformer（本期）；GRU/LSTM 后续接入同一接口。"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class TemporalTransformerEncoder(nn.Module):
    """把 K 个历史 token 编码为固定维特征向量。

    输入 tokens 形状 ``(..., K, token_dim)``，输出 ``(..., out_dim)``。
    池化取最新帧（index 0，与观测「从新到旧」一致）。
    """

    def __init__(
        self,
        token_dim: int,
        hist_len: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 128,
        dropout: float = 0.0,
        out_dim: int = 64,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.hist_len = int(hist_len)
        self.d_model = int(d_model)
        self.out_dim = int(out_dim)

        self.input_proj = nn.Linear(self.token_dim, self.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.hist_len, self.d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(nhead),
            dim_feedforward=int(ffn_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(num_layers),
            enable_nested_tensor=False,
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.out_dim),
            nn.Tanh(),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2.0))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        tokens : torch.Tensor
            ``(..., K, token_dim)``，K 必须等于 ``hist_len``。
        """
        leading = tokens.shape[:-2]
        k = tokens.shape[-2]
        if k != self.hist_len:
            raise ValueError(f"expected hist_len={self.hist_len}, got K={k}")
        flat = tokens.reshape(-1, self.hist_len, self.token_dim)
        x = self.input_proj(flat) + self.pos_embed
        x = self.encoder(x)
        latest = x[:, 0, :]  # 最新帧
        out = self.out_proj(latest)
        return out.reshape(*leading, self.out_dim)
