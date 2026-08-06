# 系统架构

## 1. 概述

基于 **MAPPO** 的多拖轮编队控制：4 艘拖轮 CTDE 训练，共享 Actor 参数，Critic 看 canonical global state。任务为 Approach → Capture → Track（见 [task_spec.md](task_spec.md)）。

| 决策 | 选择 |
|------|------|
| 范式 | CTDE：训练时 Critic 见全局，执行时 Actor 仅局部观测 |
| 算法 | 自定义 MAPPO（纯 PyTorch，无 SB3/RLlib） |
| Actor | 参数共享；邻居 Scaled Dot-Product Attention；时序可选 MLP / Transformer |
| 物理 | 拖轮 3DOF MMG；大船简化运动学（默认匀速直航） |
| 配置 | `config.py` 中 dataclass |

## 2. 分层与依赖

```
scripts/          train.py · visualize.py · export_maneuver_videos.py
rl/               Actor / Critic / PPO / temporal
env/              FormationEnv · Observer · Reward · init · obs_spec · state
physics/          TugboatDynamicsModel · LargeShipModel
config.py         EnvConfig · PPOConfig · VizConfig · REWARD_PRESETS（骨架）
simulator/        独立手动仿真（仅依赖 physics）
```

依赖方向（无环）：

```
config  → stdlib
physics → numpy
env     → physics + config
rl      → torch + env.obs_spec（不依赖 FormationEnv 运行时）
scripts → config + env + rl
simulator → physics only
```

## 3. 环境 `env/`

`FormationEnv` 组合：

- `sample_tug_init_states`（`init.py`）：安全圆环初始化 `tug_init_schema=safe_circle_v2`，默认半径 `tug_init_radius_m=120`；随机顺序逐艇拒绝采样保证船体间隙和艇间距，失败时显式报错
- `assign_tugs_to_slots`（`init.py`）：默认 minimax 唯一匹配，优先降低最远单艇初始距离；匹配后按 slot 规范化 agent 顺序，从而保持 canonical critic 的固定角色语义；`fixed` 模式用于旧实验复现
- `Observer`：局部观测与 global state
- `FormationRewardComputer`：稠密奖励
- `ObservationSpec`（`obs_spec.py`）：观测维度单一真相源（默认 **93** 维 / agent）

接近策略为反应式：slot 相对量 + 船体间隙 + 邻居风险（无外部航路模块）。

### 终止（摘要）

| 事件 | 行为 |
|------|------|
| 碰撞 | terminate，culprit/bystander 终端惩罚 |
| Capture（全体 in-zone ≥ `hold_time_s`） | 发一次 `reward_arrival_bonus`，进入 Track，**不结束** |
| Track 成功（Capture 后连续 in-zone ≥ `track_horizon_s`） | success terminate（不再发到位奖励） |
| `max_episode_steps` | truncate（timeout） |

### 大船默认

`ship_speed_min = ship_speed_max = 1.0` m/s，`ship_yaw_rate_max = 0`（匀速直航）。

## 4. 算法 `rl/`

```
训练：Critic ← global state (82 维默认) → V_i
执行：Actor  ← local obs (93 维) → π(a|o)
```

| 模块 | 职责 |
|------|------|
| `rl/actor.py` | `MAPPOActor`（MLP）、`TransformerMAPPOActor`、`build_actor()` |
| `rl/temporal.py` | `TemporalTransformerEncoder`（GRU/LSTM 预留） |
| `rl/critic.py` | 集中式 Critic + PopArt + agent one-hot |
| `rl/ppo.py` | Rollout buffer、GAE、`mappo_update` |

架构切换：`--arch mlp|transformer`（见 [arch_ablation.md](arch_ablation.md)）。

## 5. 训练入口 `scripts/train.py`

- 向量化：默认 `cuda` + `num_envs=12288` + `minibatch_size=65536`（RTX 3090 扫参峰值；可用 `--env-backend sync|subproc` 覆盖）
- `cuda`：[`env/gpu/CudaVecEnv`](../env/gpu/vec_env.py) + [`batched_step`](../env/gpu/batched_step.py)。设计见 [gpu-parallel-env-design](superpowers/specs/2026-08-07-gpu-parallel-env-design.md)
- 设备：默认 `cuda`；评估默认 `eval_workers=32`（CPU env + 主进程策略推理）
- 奖励 RunningMeanStd 归一化（稠密）+ 终端奖罚原尺度叠加
- TensorBoard 核心曲线（见 [tensorboard_metrics.md](tensorboard_metrics.md)）
- 按 eval `success_rate` 优先存 `best.pt`；定期 `last.pt`
- CLI：`--arch`、`--init-radius`、`--reward-preset`、`--env-backend`（见 [arch_ablation.md](arch_ablation.md)；preset 映射见 `config.REWARD_PRESETS`）

## 6. 测试（与文档相关）

| 测试 | 覆盖 |
|------|------|
| `tests/parity/` | CPU vs batched/CudaVecEnv 数值与事件对照（L0–L2） |
| `tests/test_actor_arch.py` | mlp / transformer 工厂与形状 |
| `tests/test_track_phase.py` | Capture / Track 终止 |
| `tests/test_observation_spec.py` | ObservationSpec 契约 |
| `tests/test_attention.py` | 邻居 Attention |
| `tests/test_init_radius.py` | init 半径配置 |
| `tests/test_reward_presets.py` | `REWARD_PRESETS` 骨架 / `apply_reward_preset` |
| `tests/test_reward_cpa.py` | CPA 风险项 |
| `tests/test_maneuvers.py` | 动力学操纵性 |
