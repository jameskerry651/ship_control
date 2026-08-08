# Transformer-S rollout_steps 消融设计

日期：2026-08-07  
状态：已批准  
范围：固定约 **1M steps** 快筛，扫更短 `rollout_steps`；模型固定 `tf_S`。不改奖励、不改 num_envs。

## 1. 目标

在现行 cuda 吞吐栈上快速检验：缩短 rollout（同预算下更多 PPO update）对靠近 / capture 的短期影响。

## 2. 固定条件

| 项 | 值 |
|----|-----|
| `--arch` | `transformer` |
| `--tf-size` | `S` |
| `--init-radius` | `120` |
| `--slot-assignment` | `minimax` |
| `--env-backend` | `cuda` |
| `--num-envs` | `256` |
| `--minibatch-size` | `8192` |
| `--total-steps` | `1000000` |
| `--seed` | `42` |
| `--device` | `cuda` |
| `--eval-workers` | `1` |
| 奖励 | EnvConfig 默认；无 `--reward-preset` |

## 3. 网格

| `--rollout-steps` | samples/update（N=256） | 约 PPO updates / 1M |
|------------------:|------------------------:|--------------------:|
| 32 | 32,768 | ~30 |
| 64 | 65,536 | ~15 |
| 128 | 131,072 | ~7 |

> `n_updates = max(1, total_steps // (rollout × num_envs × 4))`。降 `num_envs` 是为在 ~1M 预算内拉高 update 次数。

run-name：`tf_S_roll32_r120` / `tf_S_roll64_r120` / `tf_S_roll128_r120`。

## 4. 读结果

主：`eval/final_dist_mean` ↓、`eval/capture_rate` ↑。  
旁：collision（躲远假阳性）、EV/KL、墙钟与 sps。

## 5. 运行

```bash
PYTHONUNBUFFERED=1 python -u scripts/run_tf_rollout_ablation.py
```
