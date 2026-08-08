# Transformer Actor 规模消融设计

日期：2026-08-07  
状态：已实现  
范围：增加 `tf_S/M/L` 规模 preset 与训练 CLI；在 **cuda 吞吐默认栈** 上对照；汇总读结果。不改奖励、不扩 critic、不调 LR 网格。

## 1. 背景与目标

当前默认 Transformer actor（`d_model=64`，`layers=2`，`ffn=128`）约 **0.21M** 参数；整模 MAPPO（含 critic）约 0.78M。主跑 `tf_minimax_r120` 显示 PPO 健康但 `capture=0`、`final_dist` 卡在 ~320 m，行为像「躲远避碰」。

本轮用**可复现快筛**检验：单纯加大 actor Transformer 容量，在固定任务/奖励/init 下是否改善靠近与 capture。

成功标准（快筛）：

1. 三档均可训满预算步数，日志/ckpt 记录规模超参与参数量；
2. 汇总表可对比 `final_dist` / collision / capture；
3. 人工判定：距离随规模明显下降或出现非零 capture → 优胜档可加长训；三档仍远且 capture=0 → 容量非主瓶颈，回到奖励/init。

## 2. 决策摘要

| 决策 | 选择 |
|------|------|
| 预算 | 快筛：各 50M steps（与现行 `PPOConfig.total_steps` 对齐，约 7–8 次 PPO update） |
| 扩参方式 | 宽+深阶梯（S/M/L） |
| 固定条件 | r120 + minimax + 新奖励默认；并行对齐现行 cuda 吞吐默认 |
| 扩谁 | 仅 actor TF；critic 默认不动 |
| 对照 | 同 seed、同并行默认；每档独立 run |

## 3. 固定条件

| 项 | 值 |
|----|-----|
| `--arch` | `transformer` |
| `--init-radius` | `120` |
| `--slot-assignment` | `minimax` |
| 奖励 | EnvConfig 现行默认；无 `--reward-preset` |
| `--seed` | `42` |
| `--total-steps` | `50000000` |
| `--env-backend` | `cuda` |
| `--num-envs` | `12288` |
| `--rollout-steps` | `128` |
| `--minibatch-size` | `65536` |
| `--device` | `cuda` |
| 每 update 样本 | `128 × 12288 × 4 = 6,291,456`（约 7–8 次 update / 50M） |
| Checkpoint | 不 resume；每档独立 `--run-name` |

> 说明：若某档（尤其 `tf_L`）显存不足，三档同步下调同一 `num_envs`（如 8192），禁止只改一档。弱机器勿用本协议默认，应另开 `subproc` 小并行协议。

## 4. 规模表

只覆盖 `PPOConfig` 的 `tf_d_model` / `tf_nhead` / `tf_num_layers` / `tf_ffn_dim`。

| id | d_model | layers | ffn | nhead | actor 参数量（约） |
|----|--------:|-------:|----:|------:|------------------:|
| `tf_S` | 64 | 2 | 128 | 4 | 0.21M（现行默认） |
| `tf_M` | 128 | 3 | 256 | 4 | 0.54M（~2.6×） |
| `tf_L` | 256 | 4 | 512 | 8 | 2.3M（~11×） |

`tf_dropout` 保持 `0.0`。`nhead` 必须整除 `d_model`（上表已满足）。

建议 run-name：`tf_scale_S_r120` / `tf_scale_M_r120` / `tf_scale_L_r120`。

## 5. 实现约定

1. 在 `config.py` 定义：
   - `TF_SIZE_PRESETS: dict[str, dict[str, int]]`，键固定为 `S` / `M` / `L`
   - `list_tf_size_presets() -> list[str]`
   - `apply_tf_size_preset(ppo_cfg: PPOConfig, size_id: str | None) -> str | None`
2. `scripts/train.py` 增加 `--tf-size {S,M,L}`；在构建 actor 前应用到 `PPOConfig`；非法 id 报错并列出合法 id。大小写规范化为大写单字母。
3. 启动打印：`tf_size`、各 `tf_*`、actor 参数量（可 `sum(p.numel())`）。
4. Checkpoint / TensorBoard hparams 写入 `tf_size` 与 `tf_*` 字段。
5. 可选 runner：`scripts/run_tf_scale_ablation.py` 串行三档；结束后调用汇总脚本。
6. 汇总：`scripts/summarize_tf_scale.py` 从 `runs/<run-name>` 读末期（或最后 N 个 eval）指标表。
7. 单测：preset 覆盖；非法 id；三档 `build_actor` 可构造；参数量相对 S 的倍率落在合理区间（允许实现细节误差，用上下界断言）。

不要求本轮增加细粒度 `--tf-d-model` CLI（可后续加）；preset 足够快筛。

## 6. 读结果

| 优先级 | 指标 | 方向 |
|--------|------|------|
| 主 | `eval/final_dist_mean` | 越低越好 |
| 主 | `eval/capture_rate` | 越高越好（快筛可为 0） |
| 旁 | `eval/collision_rate` | 越低越好；但若靠躲远换低碰撞 → 假阳性 |
| 旁 | `loss/explained_variance`、`loss/approx_kl` | 训练是否健康 |
| 参考 | `eval/return_mean` | stall 尺度下勿与旧奖励 run 横比 |

假阳性：return 改善但 `final_dist` 仍远、或 collision↓ 而距离不变/变差。

晋级：选距离最好且训练健康的 1 档，另开 2M–5M；若三档无趋势，停止扩模，优先奖励/init。

## 7. 文档交付

- `docs/tf_scale_ablation.md`：协议 + 命令 + 读结果
- README / `docs/architecture.md` 链到该页
- 本设计文档

## 8. 非目标

- 不改 `env/reward.py` 与奖励默认
- 不扩大 critic 宽度/深度
- 不做学习率 / batch 随规模联动网格
- 不做 XL（~5M actor）与四档密扫
- 不强制打断正在进行的 `tf_minimax_r120` 长跑（消融用新 run-name）

## 9. 示例命令

```bash
python scripts/train.py \
  --arch transformer \
  --tf-size M \
  --init-radius 120 \
  --slot-assignment minimax \
  --env-backend cuda \
  --num-envs 12288 \
  --rollout-steps 128 \
  --minibatch-size 65536 \
  --total-steps 50000000 \
  --run-name tf_scale_M_r120 \
  --seed 42
```

推荐串行：

```bash
python scripts/run_tf_scale_ablation.py
python scripts/run_tf_scale_ablation.py --dry-run
```

## 10. 修订（2026-08-07）

并行栈从 `subproc` + `num_envs=16` + 1M 步，改为现行 cuda 吞吐默认（`cuda` / `12288` / `minibatch=65536` / `rollout=128` / `50M` steps），以保证大 batch 下仍有多次 PPO update，并与主训练路径一致。
