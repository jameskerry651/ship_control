# Actor 时序架构对比实验

对比轴：在相同 MAPPO/CTDE、相同奖励与任务目标（Approach → Capture → Track，见 [task_spec.md](task_spec.md)）下，**只更换 Actor 的时序编码器**。Critic、环境、观测布局与 PPO 超参保持不变。

## 1. 模型矩阵

| 模型 | 观测 | Actor 时序 | 状态 |
|------|------|------------|------|
| MLP-stack（基线） | `obs_history_k=3`（4 帧堆叠） | 扁平 MLP 编码 own obs | 已有（`--arch mlp`） |
| MLP-1frame | `obs_history_k=0` | 扁平 MLP | 配置可开，作消融 |
| GRU | 同 4 帧（网络内 reshape 为序列） | GRU over tokens | 接口预留 |
| LSTM | 同 4 帧 | LSTM over tokens | 接口预留 |
| Transformer | 同 4 帧 | Transformer Encoder over tokens | `--arch transformer` |

训练示例：

```bash
python scripts/train.py --arch mlp --run-name mlp_stack
python scripts/train.py --arch transformer --run-name tf_baseline
```

## 2. 公平性协议

共享（所有模型相同）：

- 环境与奖励（默认 `EnvConfig` / `FormationRewardComputer`）
- PPO 超参（`PPOConfig` 默认值，除非对比轴明确要求改动）
- 评估协议（确定性策略、`eval_episodes`）
- 随机种子集合（建议至少 3 个：`42, 43, 44`）

仅允许改动：

- `actor_arch`（`mlp` / `transformer` / 后续 `gru` / `lstm`）
- 各架构自身小超参（如 Transformer 的 `tf_*`）
- 参数量尽量同量级：Transformer 默认 `d_model=64, nhead=4, layers=2, ffn=128`

## 3. 评价指标

主指标：

- `success_rate`（完整 Track）
- `capture_rate`
- `track_in_zone_ratio`
- `collision_rate`
- `final_dist_mean`

辅指标：

- episode return
- 训练墙钟时间 / 达到目标 success 的环境步数

## 4. Transformer Actor 结构

环境仍返回扁平观测（默认 93 维）。Actor 内部切片：

1. own / neighbors 拆分（与 MLP 相同）
2. own 内：运动历史 `K×6`、动作历史 `K×4` → 逐帧拼接为 token `K×10`；剩余 context（实际推进器/船/预瞄/槽位/船体间隙）23 维
3. `TemporalTransformerEncoder`：`Linear(10→d_model)` + 可学习位置编码 + `TransformerEncoder`，取最新帧（index 0）投影到 64 维
4. Context MLP：`23→64→64`，与时序特征 `concat` 后 `Linear(128→64)` 得到 `own_feat`
5. 邻居编码器 + 现有 `AttentionCollisionAvoidance` 不变
6. Actor head + tanh-高斯策略不变

不引入跨 step hidden state；`act()` 隐状态占位仍为 `None`，rollout buffer / GAE 无需修改。

## 5. 后续 GRU / LSTM 接入点

1. 在 [`rl/temporal.py`](../rl/temporal.py) 增加 `TemporalGRUEncoder` / `TemporalLSTMEncoder`（输入同为 `K×10` tokens）
2. 增加对应 Actor 类（或与 Transformer 共享邻居 Attention / 策略头）
3. 在 `build_actor()` 中去掉 `NotImplementedError`，接入 `"gru"` / `"lstm"`
4. 用同一对比协议训练并汇总指标

## 6. Checkpoint 约定

`model_kwargs` 必须包含 `actor_arch` 及所用 `tf_*` 字段，以便 `train --resume` 与 `visualize` 重建正确网络。旧 checkpoint 缺省 `actor_arch="mlp"`。
