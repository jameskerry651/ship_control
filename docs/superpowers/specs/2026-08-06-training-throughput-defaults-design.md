# 训练吞吐默认超参调整设计

日期：2026-08-06  
状态：已批准，待实现  
范围：仅调整 `PPOConfig` 默认值与相关文档/脚本文案；不改训练算法、网络或环境逻辑。

## 1. 背景与目标

在 RTX 3090（24GB）+ 40 核 CPU 上，对当前默认训练路径做短诊断（`mlp` / `num_envs=16` / `subproc`）得到：

| 指标 | 实测 |
|------|------|
| 训练段 `sps`（不含 eval） | ~5800–6700 |
| Rollout vs Update | ~78% / ~22% |
| GPU SM 利用率 | 均值 ~5%，峰值 <10% |
| 显存 | ~62 MB / 24 GB |
| 模型规模 | ~0.71M 参数 |
| Eval（64 eps，`eval_workers=1`，每 2 update） | ~110 s，墙钟占比约 90%+ |

结论：瓶颈是 **CPU 环境 rollout** 与 **顺序 eval**，不是 GPU 算力。路线选择为 **A：只改超参/默认值**（不做 AMP / compile / 异步 pipeline / 性能埋点）。

目标：在保持 MAPPO 更新形态基本不变的前提下，提高墙钟吞吐（训练 `sps` + 降低 eval 占用），并为弱机器保留 CLI 下调路径。

## 2. 目标默认值

| 参数 | 基线（诊断时） | 新默认 | 理由 |
|------|----------------|--------|------|
| `num_envs` | 16 | **32** | 提高 rollout 并行；40 核上留余量，先不冲 64 |
| `minibatch_size` | 2048 | **4096** | 每 update 样本翻倍（65536），加大 GPU batch，更新步数大致不变 |
| `eval_interval` | 2 | **5** | 样本密度升高后，保持相近的「每多少样本评一次」 |
| `eval_episodes` | 64 | **32** | 墙钟约减半；`best` 判据已按 episode 数缩放 |
| `eval_workers` | 1 | **8** | 利用多核并行 eval（CUDA 下 spawn+CPU，代码已支持） |
| `rollout_steps` | 512 | 不变 | 避免额外改变 GAE horizon |
| `update_epochs` | 4 | 不变 | 避免额外改变 PPO 更新形态 |

弱机器：继续用 CLI 下调，例如 `--device cpu --env-backend sync --num-envs 2 --eval-workers 1`。

## 3. 改动范围

**修改**

- `config.py`：`PPOConfig` 上述字段与注释
- `README.md`：默认训练说明中的 env / eval 默认
- `docs/architecture.md`：向量化与 eval 默认描述
- `scripts/run_reward_preset_ablation.py`、`docs/reward_presets.md`：文案中的默认 `num_envs=16` 同步为 32（CLI 仍可覆盖）

**明确不做**

- AMP、`torch.compile`、推理与 env 异步重叠
- 新增 `rollout_dt` / `update_dt` / `eval_dt` 埋点（属路线 B）
- 改 `total_steps`、奖励、网络结构、`env-backend` 默认值以外的训练逻辑

## 4. 验收

短跑命令：

```bash
python scripts/train.py --total-steps 262144 --run-name diag_throughput_a32 --seed 0
```

（新默认下 `samples_per_update = 512×32×4 = 65536`，约 4 个 update。）

相对诊断基线（16 env / `eval_workers=1` / `eval_episodes=64` / `eval_interval=2`）：

1. 训练段 `sps` 明显上升（目标方向：接近 **≥1.5×**，不硬卡）
2. 单次 eval 墙钟显著短于 ~110 s（目标方向：约 **15–30 s** 量级，非严格线性）
3. 跑完无 subproc / eval 挂死；`best.pt` 仍按原逻辑更新

## 5. 风险与回退

- **学习动态变化**：更大 `num_envs` 提高每 update 样本量，等效 batch 变大；若曲线变差，可用 CLI 回到 16 env / 原 eval 设置对照。
- **eval 并行稳定性**：历史上 CUDA 后 fork 易挂；实现依赖现有 spawn+CPU 路径。若 `eval_workers=8` 不稳定，回退到 1 或 2。
- **CPU 过订阅**：`num_envs=32` + `eval_workers=8` 在 eval 窗口可能抢核；若 eval 反而变慢，优先降 `eval_workers`。
