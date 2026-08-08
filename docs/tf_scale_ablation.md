# Transformer Actor 规模消融协议

设计规格：[superpowers/specs/2026-08-07-tf-scale-ablation-design.md](superpowers/specs/2026-08-07-tf-scale-ablation-design.md)

在固定任务/奖励/init 下，用 `S` / `M` / `L` 三档 actor Transformer 容量做快筛，判断单纯扩模是否改善靠近与 capture。

## 固定条件

| 项 | 值 |
|----|-----|
| `--arch` | `transformer` |
| `--tf-size` | `S` / `M` / `L` |
| `--init-radius` | `120` |
| `--slot-assignment` | `minimax` |
| 奖励 | `EnvConfig` 现行默认（无 `--reward-preset`） |
| `--seed` | `42` |
| `--env-backend` | `cuda` |
| `--num-envs` | `12288` |
| `--rollout-steps` | `128` |
| `--minibatch-size` | `65536` |
| `--total-steps` | `50000000`（约 7–8 次 PPO update） |
| `--device` | `cuda` |

若某档（尤其 `L`）显存不足，三档同步下调同一 `num_envs`（如 `8192`），禁止只改一档。

## 规模表

| id | d_model | nhead | layers | ffn | actor 参数量（约） |
|----|--------:|------:|-------:|----:|------------------:|
| `S` | 64 | 4 | 2 | 128 | 0.21M |
| `M` | 128 | 4 | 3 | 256 | 0.54M |
| `L` | 256 | 8 | 4 | 512 | 2.3M |

Preset 定义：`config.TF_SIZE_PRESETS`。只改 actor `tf_*`，不改 critic / 奖励。

Run-name：`tf_scale_S_r120` / `tf_scale_M_r120` / `tf_scale_L_r120`。

## 推荐命令

串行三档（runner 显式传并行/预算，防默认漂移）：

```bash
python scripts/run_tf_scale_ablation.py --dry-run
python scripts/run_tf_scale_ablation.py
```

单档：

```bash
python scripts/train.py --arch transformer --tf-size M \
  --init-radius 120 --slot-assignment minimax \
  --env-backend cuda --num-envs 12288 --rollout-steps 128 \
  --minibatch-size 65536 --total-steps 50000000 \
  --run-name tf_scale_M_r120 --seed 42
```

汇总：

```bash
python scripts/summarize_tf_scale.py --runs \
  tf_scale_S_r120 tf_scale_M_r120 tf_scale_L_r120
```

## 读结果

| 优先级 | 指标 | 方向 |
|--------|------|------|
| 主 | `eval/final_dist_mean` | 越低越好 |
| 主 | `eval/capture_rate` | 越高越好（快筛可为 0） |
| 旁 | `eval/collision_rate` | 越低越好；躲远换低碰撞 → 假阳性 |
| 旁 | `loss/explained_variance` | 训练是否健康 |
| 参考 | `eval/return_mean` | 勿与旧奖励协议横比 |

假阳性：return 改善但 `final_dist` 仍远，或 collision↓ 而距离不变/变差。

晋级：选距离最好且训练健康的一档加长训；三档无趋势则停扩模，优先奖励/init。
