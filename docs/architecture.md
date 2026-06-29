# 多智能体拖轮编队强化学习 — 架构文档

## 1. 项目概述

本项目实现基于 **MAPPO (Multi-Agent Proximal Policy Optimization)** 的多拖轮编队控制强化学习系统。任务场景：4 艘全回转拖轮协同伴航一艘移动的大型船舶，各拖轮需平滑驶入并维持在母船周围 4 个预设槽位（船首左/右、船尾左/右）。

### 核心设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 强化学习范式 | CTDE (集中训练、分拆执行) | 训练时 Critic 可见全局状态，执行时 Actor 仅需局部观测 |
| 算法 | 自定义 MAPPO | 无第三方 RL 库依赖（非 SB3、非 RLlib），纯 PyTorch 实现 |
| 策略架构 | 参数共享 Actor + Attention 碰撞避免 | 4 艇共享同一网络，通过目标槽位观测区分角色 |
| 物理模型 | 3DOF MMG 拖轮 + 简化为运动学的大船 | 拖轮需要高保真回转/横移动力学，大船仅需位姿参考 |
| 配置管理 | dataclass 单文件 | 所有超参数类型化、有默认值、集中可查 |
| 课程学习 | 渐进式难度 + 消融实验 | 从 3-ready 逐步到 0-ready，最终收紧位置容差 |

---

## 2. 系统分层架构

```
┌──────────────────────────────────────────────────────────────┐
│  scripts/           入口脚本层                                │
│  train.py / visualize.py / reproduce_c04_curriculum.py ...    │
│  依赖：全部模块（编排层）                                      │
├──────────────────────────────────────────────────────────────┤
│  curricula/         课程定义层（纯数据）                       │
│  c01..c04 / strict_pos / ablations  依赖：config（仅校验）     │
├──────────────────────┬───────────────────────────────────────┤
│  rl/                 │  simulator/                           │
│  MAPPO 算法层        │  手动驾驶仿真器（pygame + G29）         │
│  Actor / Critic /    │  依赖：physics only                    │
│  PPO / Buffer        │  完全独立于 RL 训练                    │
│  依赖：config + torch│                                       │
├──────────────────────┴───────────────────────────────────────┤
│  env/                环境层                                   │
│  FormationEnv 组合：InitSampler + Observer +                  │
│  RoutePlanner + FormationRewardComputer                      │
│  新增：obs_spec（维度常量）/ state（状态快照）                  │
│  依赖：physics + config                                      │
├──────────────────────────────────────────────────────────────┤
│  physics/            物理模型层                               │
│  TugboatDynamicsModel (3DOF MMG) + LargeShipModel (运动学)    │
│  依赖：仅 numpy — 零 RL 概念                                  │
├──────────────────────────────────────────────────────────────┤
│  config.py           配置层（单一真相源）                       │
│  EnvConfig / PPOConfig / VizConfig — 仅依赖 stdlib            │
└──────────────────────────────────────────────────────────────┘
```

### 2.1 依赖方向

```
config.py ──→ (stdlib only)
physics/  ──→ numpy
env/      ──→ physics/ + config/ + env/state + env/obs_spec
rl/       ──→ config/ + torch + env/obs_spec（不依赖 env/）
simulator/──→ physics/ only（不依赖 env/ 和 rl/）
scripts/  ──→ config + env + rl + curricula
tests/    ──→ 各被测模块
```

**无循环依赖。** 依赖图是干净的 DAG。

---

## 3. 各模块详解

### 3.1 `config.py` — 全局配置

| 类 | 字段数 | 职责 |
|----|--------|------|
| `EnvConfig` | 122 | 环境参数（dt、船舶尺寸/随机化、槽位几何、拖轮初始化、路径规划、奖励权重/阈值、观测维度、碰撞距离等） |
| `PPOConfig` | 14 | PPO 超参数（γ=0.99, λ=0.98, ε=0.2, rollout=512, lr 余弦退火 1e-4→5e-6, 总步数 5M） |
| `VizConfig` | 3 | 可视化参数（mpp、跟船开关、推力显示） |

**默认设备**：CPU（注释说明对小网络 CPU 比 MPS 快）。

### 3.2 `physics/` — 物理模型

#### `TugboatDynamicsModel` — 高保真拖轮动力学

| 特性 | 参数 |
|------|------|
| 自由度 | 3DOF（纵荡/横荡/艏摇） |
| 质量 | 699,000 kg，长 36m，宽 11m，吃水 2.5m |
| 附加质量 | 纵荡 8%，横荡 30%，艏摇惯量 55% |
| 阻尼 | 线性 (20%) + 二次 (Cd_x=0.70, Cd_y=4.0) |
| 横流阻力 | 转弯时纵荡速度损失 20-40%（真实的 ASD 拖轮特性） |
| 尾鳍升力 | 抗 Munk 力矩，提供航向稳定性 |
| 推进器 | 2× 全回转导管桨，推力模型 T = Kt·ρ·D⁴·n·\|n\|，前向 Kt=0.40，倒车 0.60 |
| 执行器动态 | 转速率限 ±120 RPM/s，方位角率限 ±30°/s，子步积分 |
| 积分 | Euler 积分，内步长 ≤0.02s |

#### `LargeShipModel` — 简化大船运动学

| 特性 | 参数 |
|------|------|
| 尺寸 | 200m × 30m（可随机化 180-240m × 26-40m） |
| 运动 | 一阶低通跟随随机目标速度 0.5-2.0 m/s |
| 航向 | 当前强制直航（r=0） |
| 槽位 | 4 个固定位置（首左/右、尾左/右），带配置偏移量 |
| 碰撞 | 矩形船体外廓 + 距离函数 |

### 3.3 `rl/` — MAPPO 算法

采用 **CTDE (Centralized Training with Decentralized Execution)** 范式：

```
训练时：                     执行时：
┌──────────┐                ┌──────────┐
│ Critic   │ ← 90维全局状态  │ Actor    │ ← 93维局部观测
│ (PopArt) │                │ (共享参数) │
│ V(s)     │                │ π(a|o)    │
└──────────┘                └──────────┘
      ↑                          ↑
  全局状态                   自身+邻居观测
```

#### `MAPPOActor` (387 行)

- **观测编码器**：63 维自身特征（运动历史 4 帧×6 + 动作历史 4 帧×4 + 大船/槽位/路径/间隙）
- **邻居 Attention**：Scaled Dot-Product Attention 覆盖 3 邻居 × 10 维风险特征，输出 64 维环境威胁摘要
- **策略头**：3 层 MLP（512 维）→ tanh 压缩对角高斯分布，输出 4 维连续动作 ∈ [-1,1]
- **数值稳定**：log_std 裁剪 [-5, 2]，tanh log-det 雅可比修正

#### `MAPPOCritic` (137 行)

- **全局状态编码**：90 维规范状态（大船 2 维 + 4×19 艇 + 4×3 加速度）
- **Agent 标识**：one-hot 拼接（4 位）区分不同 agent 的 V 值
- **PopArt 归一化**：内部归一化空间运行，自适应缩放输出层跟踪变化回报幅度
- **网络**：3 层 MLP（512 维）

#### `MAPPOActorCritic` + `mappo_update()` (375 行)

- **Rollout Buffer**：(T, N, K, *) 形状存储，正确处理 episode 边界
- **GAE**：terminated → next_value=0，truncated → V(terminal_obs_pre_reset)
- **PPO 更新**：Clipped Surrogate + PopArt 归一化值损失 + 熵正则 + 可选成功 BC
- **NaN 防护**：MPS 设备 NaN 检测与恢复

### 3.4 `env/` — 多智能体环境

#### `FormationEnv` (675 行)

遵循类 Gymnasium 接口：

```python
obs = env.reset()                              # → (4, 93) float32
obs, rewards, dones, info = env.step(actions)  # actions ∈ [-1,1]^(4×4)
global_state = env.get_global_state()           # → (90,) float32 (Critic 用)
```

**动作空间**（4 维连续 ∈ [-1,1]）：`[port_rpm_norm, stbd_rpm_norm, port_az_norm, stbd_az_norm]`

**观测空间**（93 维/艇）：

| 分量 | 维度 | 说明 |
|------|------|------|
| 运动历史 | 4×6=24 | u,v,r,du,dv,dr（归一化） |
| 动作历史 | 4×4=16 | 归一化动作 |
| 大船相对状态 | 5 | dx,dy,u,sin(Δψ),cos(Δψ) |
| 大船预瞄点 | 3×2=6 | 未来 5/10/15s 位置 |
| 目标槽位 | 5 | dx,dy,dist,sin(Δψ),cos(Δψ) |
| 路径目标 | 4 | dx,dy,stage_norm,remaining |
| 船体间隙 | 3 | 最近边界 dx,dy,d_hull |
| 邻居特征 | 3×10=30 | dx,dy,dist,bearing,du,dv,range_rate,TCPA,DCPA |

**动作空间维度** (ACTION_DIM): 4

**全局状态**（90 维，Critic 专用）：

| 分量 | 维度 |
|------|------|
| 大船状态 | 2 (u, u_dot) |
| 每艇状态 | 4×19=76 |
| 每艇加速度 | 4×3=12 |

**终止条件**：
- 碰撞（拖轮-大船 或 拖轮-拖轮）→ terminated（惩罚 -80 肇事艇 / -15 旁艇）
- 4 艇全在位 > hold_time_s → success（奖励 +80）
- step ≥ max_episode_steps → truncated

#### 子模块组合

```
FormationEnv (拥有所有状态)
  ├── InitSampler        初始布放采样（5 区域 mixed_slot_approach）
  ├── RoutePlanner       A* 路径规划 + LOS 简化 + B 样条平滑
  ├── Observer           观测构造（build_obs + get_global_state）
  └── FormationRewardComputer  稠密奖励计算（6 项 + 2 可选项 + 2 终端项）
```

**2026-06 重构**：子模块不再持有 `FormationEnv` 引用，改为通过 `SimState`（不可变快照）和 `MutableEpisodeState`（可变追踪）接收数据。所有维度常量集中到 `env/obs_spec.py`。

#### 观测与奖励文档

- `docs/observation_space.md` — 93 维观测向量的数学规格
- `docs/reward_function.md` — 稠密奖励函数的数学规格

### 3.5 `curricula/` — 课程学习

#### 课程体系

```
c01_three_ready (600k)     3/4 ready, hold=1.0s
c02_two_ready   (800k)     2/4 ready, hold=1.0s
c03_one_ready   (1M)       1/4 ready, hold=1.5s
c04_zero_ready  (1.5M)     0/4 ready, hold=2.0s
  ├── strict_pos/         位置容差收紧（post-c04）
  │   pos140m → pos120m → pos100m → pos80m → pos60m → pos40m → pos20m → pos10m
  └── ablations/          消融实验（增量组件分析）
      P1 → P1+P2 → P1+P2+P3 → P1+P2+P3+P4 → P1..P5 (full)
```

**设计特性**：

- **数据驱动**：课程文件是纯 Python dict (`COURSE`)，通过 `loader.py` 的 `CourseSpec` + `apply_course()` 动态加载
- **不可变**：`MappingProxyType` 防止意外修改
- **校验**：`load_course()` 在加载时校验 `env_overrides` 的 key 是否属于 `EnvConfig` 字段
- **渐进维度**：ready 数量 ↓、hold 时间 ↑、生成区域 ↑、噪声 ↑、速度范围 ↑

### 3.6 `simulator/` — 手动驾驶仿真器

| 文件 | 行数 | 职责 |
|------|------|------|
| `app.py` | 232 | 主循环：输入 → 动力学 → 渲染 |
| `render.py` | 350 | pygame 渲染（海面/网格/船/推力矢量/HUD/罗盘/轴调试） |
| `wheel.py` | 231 | 输入抽象（G29 方向盘/键盘） |
| `config.py` | 61 | SimConfig（窗口/相机/G29 轴映射） |
| `__main__.py` | 67 | CLI 入口 |

**特点**：
- 使用与 RL 训练**相同的** `TugboatDynamicsModel`
- 油门踏板 → 左桨转速，刹车踏板 → 右桨转速
- 方向盘换挡拨片可反转单侧推进器（增强原地回转/横移）
- 伴航练习模式：显示拖轮相对大船的纵/横向偏移、速度差
- 快捷键：Space 暂停、R 重置、+/- 倍速、D 调试面板、C 踏板校准
- 完全独立于 RL 训练代码（不导入 `env/`、`rl/`）

### 3.7 `scripts/` — 入口脚本

| 脚本 | 行数 | 职责 |
|------|------|------|
| `train.py` | 1320 | 主训练循环：vec env（Sync/Subproc）、RewardNormalizer、评估、checkpoint、TensorBoard |
| `train_strict_pos_curriculum.py` | 392 | pos_tol 阶梯训练到 10 m（spawn train.py 子进程） |
| `reproduce_c04_curriculum.py` | 367 | c01→c02→c03→c04 全流程复现 |
| `visualize.py` | 913 | PyGame 可视化：9 通道图表面板、CPA 预警、PNG 导出 |
| `export_maneuver_videos.py` | 283 | 机动试验动画导出（MP4/GIF） |

### 3.8 `utils/` — 工具模块

| 文件 | 行数 | 职责 |
|------|------|------|
| `mpl_fonts.py` | 36 | 中文字体配置（PingFang → Hiragino → … → DejaVu Sans） |

### 3.9 `tests/` — 测试套件

| 文件 | 测试数 | 覆盖范围 |
|------|--------|----------|
| `test_maneuvers.py` | 11 | 拖轮动力学 3 层验证（代码正确性→物理守恒→标准机动） |
| `test_curricula.py` | 2 | 课程加载+环境创建 |
| `test_reward_precision.py` | 4 | 精度奖励开关 |
| `test_reward_route_progress.py` | 2 | 路径进度奖励梯度 |
| `test_reward_cpa.py` | 3 | CPA 碰撞惩罚：接近/远离/船体 |
| `test_attention.py` | 1 | AttentionCollisionAvoidance 形状 |

---

## 4. 数据流

### 4.1 训练循环 (train.py)

```
┌─────────────────────────────────────────────────────────────┐
│  for episode in range(num_episodes):                       │
│    obs = env.reset()           # (num_envs, n_tugs, 93)    │
│    for t in range(rollout_steps):                          │
│      actions = actor.act(obs)  # (num_envs, n_tugs, 4)     │
│      next_obs, rewards, dones, info = env.step(actions)    │
│      buffer.store(obs, actions, rewards, dones, values,    │
│                   global_state)                            │
│      obs = next_obs                                        │
│    advantages = GAE(rewards, values, dones)                │
│    for epoch in range(update_epochs):                      │
│      mappo_update(actor, critic, buffer, advantages)       │
│      │  ├─ Actor loss: clipped surrogate + entropy         │
│      │  ├─ Critic loss: PopArt-normalized MSE              │
│      │  └─ Optional: success BC loss                       │
│    eval_score = evaluate_policy(eval_env, actor)           │
│    if early_stop: break                                    │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 环境 step 内部流程

```
actions [-1,1]^(4×4)
  │
  ├─ denormalize: actions[i,k] * tug.rpm_limit
  ├─ route_speed_governor: 限制接近速度
  │
  ├─ for each tug:
  │    tug.set_control_commands()
  │    tug.step(dt_ctrl)          # 3DOF 动力学积分
  │
  ├─ ship.step(dt_ctrl)           # 大船运动学更新
  ├─ route.advance_stage()        # 航点推进
  │
  ├─ build SimState snapshot      # 不可变状态快照
  │
  ├─ reward_computer.compute_rewards(state, episode)  # 稠密奖励
  ├─ _check_termination()         # 碰撞/成功/超时
  │
  ├─ observer.append_obs_history()
  ├─ observer.build_obs(state, history)  # → (4, 93)
  │
  └─ return obs, rewards+terminal, dones, info
```

### 4.3 奖励结构

```
R_total = w_target * R_target
        + w_velocity * R_velocity
        - w_collision * P_collision
        + w_route * R_route
        + w_shape * R_shape
        + w_team * R_team
        + R_precision + R_near_hold + R_hold_streak
        + terminal_reward (±80)

其中：
  R_target = R_chase (进度 + 接近速度) + R_hold (位置×航向×速度)
  R_chase  = far_gate * (0.6 * progress + 0.4 * closing_speed)
  R_hold   = hold_gate * pos_score * (0.5 + 0.25*heading + 0.25*speed)
  P_collision = barrier(dist) + cpa_w * CPA_risk(dcpa, tcpa)
  R_shape = w_shape * clip(γ·Φ(s') - Φ(s))
  R_team  = w_team * softmin_β(z_1, z_2, z_3, z_4)
```

---

## 5. 关键设计模式

### 5.1 组合优于继承

`FormationEnv` 不使用深层类继承，而是通过组合 4 个功能模块：

```python
self._route = RoutePlanner()    # 路径规划
self._obs = Observer()          # 观测构造
self._init = InitSampler()      # 初始采样
self._reward = FormationRewardComputer()  # 奖励计算
```

每个模块是无状态的（不持有 env 引用），通过 `SimState` 快照和 `MutableEpisodeState` 接收数据。

### 5.2 配置覆盖链

```
CLI args > course file overrides > dataclass defaults
```

`apply_course()` 通过 `setattr()` 修改 `EnvConfig` 实例，CLI 参数再覆盖选定字段。

### 5.3 Episode 边界处理

GAE 计算区分两种终止：
- **terminated**（碰撞/成功）→ `next_value = 0`（无后续状态）
- **truncated**（超时）→ `next_value = V(terminal_obs_pre_reset)`（环境可能继续）

### 5.4 P1 按责分配碰撞惩罚

碰撞时不是全员均摊惩罚，而是：
- 肇事艇：全额惩罚（-80）
- 旁艇：小额共担（-15，可配置）

消除"全员共担"导致的 Critic 归因困难。

---

## 6. 运行时目录

| 目录 | 内容 |
|------|------|
| `checkpoints/` | 模型权重（按 run_name 子目录，含 best.pt / last.pt） |
| `runs/` | TensorBoard 日志 |
| `outputs/` | 静态输出（如 init_scenes.png） |
| `exports/` | visualize.py 导出的轨迹/曲线图 |
| `memory/` | 实验记录（MEMORY.md） |
| `maneuver_tests/` | 机动验证图（PNG） |
| `docs/` | 设计文档 |

---

## 7. 当前技术债务

| 严重度 | 问题 | 位置 |
|--------|------|------|
| 🟡 | `train.py` 1320 行过大，内含 VecEnv、RewardNormalizer 等 | `scripts/train.py` |
| 🟡 | `formation_env.py` 674 行，TerminationChecker/SpeedGovernor 可提取 | `env/formation_env.py` |
| 🟡 | 缺少 `requirements.txt` / `pyproject.toml` | 根目录 |
| 🟡 | 无 CI/CD 配置（无 GitHub Actions） | 根目录 |
| 🟡 | RL 核心模块（PPO update、Actor、Critic）无测试 | `rl/` |
| 🟢 | `env/` 初始化/路由规划器无独立测试 | `env/init.py`, `env/route_planner.py` |
| 🟢 | 坐标变换函数曾分散在 `env/observer.py` 和 `env/formation_env.py`（已收敛到 `env/state.py`） | ✅ 已修复 |
| 🟢 | 维度常量曾重复定义在 `observer.py` 和 `actor.py`（已收敛到 `env/obs_spec.py`） | ✅ 已修复 |
| 🟢 | env 子模块曾紧耦合持有 FormationEnv 引用（已解耦为 SimState + MutableEpisodeState） | ✅ 已修复 |

---

## 8. 快速开始

```bash
pip install torch numpy pygame tensorboard scipy

# 训练
python scripts/train.py

# 可视化
python scripts/visualize.py --ckpt checkpoints/<run_name>/best.pt

# 动力学验证
python tests/test_maneuvers.py

# 手动驾驶仿真
python -m simulator
```

---

## 9. 版本历史

| Commit | 日期 | 变更 |
|--------|------|------|
| `a0bb173` | 2026-06-26 | 修改动力学参数符合 ASD 真实结果 |
| `6dba723` | 2026-06 | 为拖轮动力学模型添加尾鳍恢复力 + 螺旋试验 |
| `2b28284` | 2026-06 | 引入课程学习、扩展奖励与观测、新增模拟器与测试 |
| `4dbc81e` | 2026-06 | 修改 Actor 设计，优化邻居观测状态 |
| `cdf50d9` | 2026-06 | 将 formation_env 拆分为 init/observer/route_planner 模块 |
| `273de67` | 2026-06 | Refactor RL stack for simplified reward and attention-based actor |
| `ce459f8` | 2026-05 | Initial commit |
| — | 2026-06 | 架构重构：obs_spec 维度常数统一 + SimState 子模块解耦 |
