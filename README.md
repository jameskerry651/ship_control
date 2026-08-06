# 多智能体拖轮编队强化学习

4 艘全回转拖轮协同接近移动大船周围的固定 slot，避碰入位后持续跟随。算法为自定义 **MAPPO**（CTDE，纯 PyTorch）。

## 任务目标

三段式规格见 [docs/task_spec.md](docs/task_spec.md)：

1. **Approach**：自行接近分配 slot，靠近大船时避碰；
2. **Capture**：全体同时 in-zone 满 `hold_time_s`（默认 2 s）→ 发捕获奖励，进入 Track（不结束）；
3. **Track**：再连续全体 in-zone 满 `track_horizon_s`（默认 30 s）→ **success**。

主指标：`success_rate`；辅指标：`capture_rate`、`track_in_zone_ratio`、`collision_rate`、`final_dist`。

## 文档索引

| 文档 | 内容 |
|------|------|
| [docs/task_spec.md](docs/task_spec.md) | 任务阶段与成功判定 |
| [docs/architecture.md](docs/architecture.md) | 系统架构与模块依赖 |
| [docs/observation_space.md](docs/observation_space.md) | 观测 / 全局状态 |
| [docs/reward_function.md](docs/reward_function.md) | 奖励函数 |
| [docs/reward_presets.md](docs/reward_presets.md) | 奖励超参 preset 消融 |
| [docs/arch_ablation.md](docs/arch_ablation.md) | Actor 时序架构对比实验 |
| [docs/tensorboard_metrics.md](docs/tensorboard_metrics.md) | TensorBoard 指标含义 |

## 对比实验（Actor 时序）

协议详见 [docs/arch_ablation.md](docs/arch_ablation.md)：

- MLP + 4 帧堆叠（`--arch mlp`，默认）
- 单帧 MLP（`obs_history_k=0` 消融）
- GRU / LSTM（接口预留）
- Transformer（`--arch transformer`）

```bash
# 默认 init 半径 100 m；远距复现加 --init-radius 200
python scripts/train.py --arch transformer --run-name tf_r100
python scripts/train.py --arch mlp --run-name mlp_r100
```

奖励超参 preset 消融（协议见 [docs/reward_presets.md](docs/reward_presets.md)）：

```bash
python scripts/train.py --arch transformer --init-radius 100 \
  --reward-preset rw_combo --run-name rw_combo --total-steps 1000000 --seed 42
```

## 目录结构

```
config.py            EnvConfig / PPOConfig / VizConfig
physics/             拖轮 3DOF MMG + 大船运动学
env/                 FormationEnv、观测、奖励、初始化
rl/                  MAPPO Actor / Critic / PPO / 时序编码器
utils/               通用工具
simulator/           罗技 G29 手动驾驶仿真
scripts/             train.py / visualize.py / export_maneuver_videos.py
tests/               测试
docs/                设计文档
```

## 快速开始

```bash
pip install torch numpy pygame tensorboard scipy

# 训练（默认 init 半径 100 m、5M env-steps）
python scripts/train.py --arch transformer --run-name tf_r100
# 远距：python scripts/train.py --arch transformer --run-name tf_r200 --init-radius 200

# TensorBoard（指标说明见 docs/tensorboard_metrics.md）
tensorboard --logdir runs

# 可视化策略
python scripts/visualize.py --ckpt checkpoints/<run_name>/best.pt

# 动力学操纵性试验
python tests/test_maneuvers.py

# 导出操纵性视频
python scripts/export_maneuver_videos.py
```

### 手动驾驶仿真（罗技 G29）

方向盘转角控制左右舵方位角；油门控左桨、刹车控右桨；`+/-` 调倍速。默认有一艘匀速直线大船，HUD 显示相对纵/横向偏移与速度差。

```bash
python -m simulator
python -m simulator --speed 4 --mpp 0.4
python -m simulator --ship-speed 2.0
python -m simulator --no-ship
python -m simulator --no-wheel
python -m simulator --throttle-axis 1 --brake-axis 2
```

拨片：左/右拨片反转左/右桨。快捷键：`Space` 暂停 / `R` 重置 / `+``-` 倍速 / `D` 轴调试 / `C` 校准踏板 / `Esc` 退出。键盘：方向键转向，`Q`/`Z` 左桨，`E`/`C` 右桨，`Shift` 反转对应桨。
