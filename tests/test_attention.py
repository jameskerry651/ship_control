"""AttentionCollisionAvoidance 最简调用示例。"""

import torch

from rl.actor import AttentionCollisionAvoidance


def demo_attention_collision_avoidance() -> None:
    # 与 MAPPOActor 中一致：本船特征 64 维，3 个邻居各 64 维
    own_feat_dim = 64
    neigh_feat_dim = 64
    embed_dim = 64
    n_neighbors = 3

    attn = AttentionCollisionAvoidance(own_feat_dim, neigh_feat_dim, embed_dim=embed_dim)

    batch = 2
    e_own = torch.randn(batch, own_feat_dim)                    # 本船编码特征
    e_neighbors = torch.randn(batch, n_neighbors, neigh_feat_dim)  # 邻居编码特征

    agg_feat, attn_weights = attn(e_own, e_neighbors)

    print("agg_feat shape     :", agg_feat.shape)       # [batch, embed_dim]
    print("attn_weights shape :", attn_weights.shape)   # [batch, n_neighbors]
    print("attn_weights[0]    :", attn_weights[0].tolist())  # 3 个邻居的注意力权重，和为 1

    # 可选：mask 标记无效邻居槽位（0 = 无邻居）
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    _, masked_weights = attn(e_own, e_neighbors, mask=mask)
    print("masked_weights[0]  :", masked_weights[0].tolist())


def test_attention_forward() -> None:
  attn = AttentionCollisionAvoidance(64, 64, embed_dim=64)
  e_own = torch.randn(1, 64)
  e_neighbors = torch.randn(1, 3, 64)

  agg_feat, weights = attn(e_own, e_neighbors)

  assert agg_feat.shape == (1, 64)
  assert weights.shape == (1, 3)
  assert torch.allclose(weights.sum(dim=-1), torch.ones(1), atol=1e-5)


if __name__ == "__main__":
  demo_attention_collision_avoidance()
