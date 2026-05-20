# 多智能体拖轮编队强化学习

4 艘拖轮（双桨双舵）从大船船尾后方带初速追赶，沿对应舷侧绕行并驶入移动大船周围的 4 个 slot，使用参数共享 MAPPO 训练。
Actor 仍然是去中心化执行，每艘拖轮只看自己的局部观察；critic 使用 canonical global state。

## 文件结构

```
config.py                   全部超参数
physics/
  tugboat_dynamics_model.py   拖轮 3DOF 动力学模型
  large_ship_model.py         大船运动学模型
env/
  formation_env.py            多智能体编队环境
rl/
  ppo.py                      MAPPO 算法
utils/
  mpl_fonts.py                matplotlib 中文字体
scripts/
  train.py                    MAPPO 训练脚本
  visualize.py                pygame 可视化
tests/
  test_maneuvers.py           拖轮 Z 字 / S 形操纵性试验
docs/                         奖励、观测等设计文档
memory/                       训练实验记录
```

## 快速开始

```bash
# 安装依赖
pip install torch numpy pygame tensorboard scipy

# 训练（默认 500 万步）
python scripts/train.py

# 从 v28 MAPPO 权重 warm start 新的船尾追赶场景
python scripts/train.py --resume checkpoints/v28_mappo_slotonehot_v25init/best.pt \
  --run-name v29_mappo_astern_v28init

# 查看 tensorboard 日志
tensorboard --logdir runs

# 可视化（自动加载最新 best.pt）
python scripts/visualize.py

# 指定权重
python scripts/visualize.py --ckpt checkpoints/<run_name>/best.pt
```

## 训练参数

常用命令行参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--total-steps` | 5000000 | 总环境步数 |
| `--num-envs` | 8 | 并行环境数 |
| `--rollout-steps` | 256 | 每次 rollout 步数 |
| `--learning-rate` | 3e-4 | 初始学习率 |
| `--device` | cpu | 设备（cpu/cuda/mps） |
| `--run-name` | 时间戳 | 本次运行名 |
| `--resume` | — | 从 .pt 续训 |
| `--init-mode` | config.py | `astern_approach` / `mixed_slot_approach` / `near_slot` / `ring` |
| `--no-route-obs` | false | 关闭 v29 路线观察特征 |
| `--no-ship-size-randomize` | false | 关闭 v36 大船长宽随机化与尺度观测 |

所有超参数集中在 `config.py`，修改后重新训练即可。

## 可视化操作

| 按键 | 功能 |
|------|------|
| Space | 暂停/继续 |
| R | 重置 episode |
| + / - | 加速 / 减速 |
| Q / Esc | 退出 |

## 奖励函数设计

**训练默认**（`reward_use_simple_stage=True`）：每步稠密奖励由少量组合项构成，详见 [`docs/reward_function.md`](reward_function.md)。

| 阶段 | 稠密组合（简化） |
|------|------------------|
| 非 final（绕行/追赶） | route progress + chase speed − 超速/限速风险 − 聚合安全 − lane − 动作惩罚 |
| final（就位/伴航） | escort + hold 累进 + 速度匹配 − 聚合安全 − 动作惩罚 |

环境仍计算全部原子分项（进度、朝向、CPA、预测船体安全等）并写入 `reward_components`，便于 TensorBoard 诊断。

**终端奖励**（不归一化，终止步叠加）：
- 碰撞：仅肇事/涉事拖轮 `-20`（v56）
- 成功：全员 `+80`（`reward_arrival_bonus`）
- 超时：`0`

就位判定（三项同时满足并连续 `hold_time_s`，当前默认 **10s**）：
- 到 slot 距离 < `pos_tol_m`（60m）
- 航向误差 < `heading_tol_rad`（30°）
- 世界系速度差 < `speed_tol_ms`（3 m/s）

## 动作空间

每个拖轮的动作是 4 维连续向量，归一化到 `[-1, 1]`：

```
[port_rpm_norm, starboard_rpm_norm, port_azimuth_norm, starboard_azimuth_norm]
```

映射到物理量：RPM ∈ [-240, 240]，方位角 ∈ [-45°, 45°]。

## 观察空间（64 维，以自身为参考系）

| 索引 | 内容 |
|------|------|
| 0-2 | 自身体系速度 (u, v, r) |
| 3-4 | 自身在 slot 局部系下的偏移 |
| 5-7 | 到 slot 的极坐标 (log_dist, sinθ, cosθ) |
| 8-9 | 航向误差 (sin, cos) |
| 10-11 | 大船相对自身位置（自身体系） |
| 12-13 | 大船航向相对自身 |
| 14-16 | 大船体系速度 |
| 17-20 | 自身执行器实际值 |
| 21-24 | 上一步动作 |
| 25-30 | 其他 3 个拖轮的相对位置 |
| 31-34 | slot one-hot |
| 35-42 | 路线 waypoint / stage / 左右舷 / 剩余路线距离 |
| 43-54 | 拖轮间 CPA 特征 |
| 55-58 | 拖轮-大船 CPA 特征 |
| 59-60 | 大船尺度特征 |
| 61-63 | 自身体系加速度 (u_dot, v_dot, r_dot) |

centralized critic 额外在 canonical global state 末尾为每条拖轮追加 6 维加速度 tail：自身 3D 加速度（线加速度投影到大船船体系 + 偏航角加速度）和大船 3D 加速度。默认 4 条拖轮时 critic 输入维度为 121。

## 混合初始化鲁棒性课程

`mixed_slot_approach` 用于在主场景基础上增加初始条件多样性：每个 episode 随机 2 或 3 艘拖轮已经以合理航向、速度和推进器状态占据 slot 外侧保持点，其余拖轮从船尾/舷侧路线的不同阶段随机起步，继续学习绕行到自己的 slot。该模式保留固定 slot 角色和 route 观察/奖励，适合从 `astern_approach` 的 best checkpoint 低学习率 warm start。

## 大船尺度随机化

v36 起默认在每个 episode reset 时随机采样大船长宽，默认范围为 `ship_length_min_m=180m` 到 `ship_length_max_m=240m`、`ship_beam_min_m=26m` 到 `ship_beam_max_m=40m`。slot、路线 waypoint、船体距离、可视化船体轮廓都会使用当前 episode 的实际尺寸。观测末尾新增 2 维 `ship_size`，表示船长/船宽相对基准 `ship_length_m=200m`、`ship_beam_m=30m` 的偏差。

## 算法式 waypoint planner

默认 `route_planner="visibility"`：环境在大船船体系下把船体矩形按 `route_hull_clearance_m` 膨胀，生成同舷 stern gate、outer holding、final entry、final slot 等语义 anchor，并用 visibility graph + Dijkstra 在膨胀船体外连接 anchor。输出仍是原来的 waypoint 数组，观测维度和 MAPPO 接口不变。

船尾 slot 不再从 stern staging 直接切向 final slot，而是：

```
stern gate -> outer holding -> final entry -> final slot
```

这用于降低 `stern:final_slot` 阶段贴近船尾角和右舷 T3 船体碰撞。`route_planner="manual"` 可回到旧固定模板，便于复现实验或 ablation。

## v30+ 船尾追赶几何（当前 `config.py` 默认）

`astern_approach` 使用内外双航道，避免同侧拖轮共用入口点：
- 船首 slot：外侧 lane `route_bow_lane_lat_m=100m`，初始距船尾 `260~380m`
- 船尾 slot：内侧 lane `route_stern_lane_lat_m=60m`，初始距船尾 `120~180m`
- 同侧初始间距下限 `tug_init_pair_min_dist_m=110m`，间距惩罚阈值 `route_tug_spacing_dist_m=90m`

更完整的观测/奖励说明见 [`docs/observation_space.md`](observation_space.md)、[`docs/reward_function.md`](reward_function.md)。
