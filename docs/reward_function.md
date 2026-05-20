# 拖轮编队强化学习 — 奖励函数文档

> **版本**: v52 (simple staged) + v56 (差异化终端)  
> **架构**: 参数共享 MAPPO，多智能体同构，集中式 critic  
> **更新日期**: 2026-05-20  
> **实现**: `env/formation_env.py` → `_compute_rewards()`，`config.py` → `EnvConfig`

---

## 目录

1. [整体结构](#整体结构)
2. [训练用稠密奖励（simple staged，默认）](#训练用稠密奖励simple-staged默认)
3. [原子稠密奖励（诊断与 ablation）](#原子稠密奖励诊断与-ablation)
4. [稠密奖励 — 追赶阶段](#稠密奖励--追赶阶段)
5. [稠密奖励 — 伴航阶段](#稠密奖励--伴航阶段)
6. [稠密奖励 — 操纵品质](#稠密奖励--操纵品质)
7. [稠密奖励 — 安全约束](#稠密奖励--安全约束)
8. [终端奖励](#终端奖励)
9. [奖励归一化](#奖励归一化)
10. [到位判定与终止条件](#到位判定与终止条件)
11. [参数速查表](#参数速查表)
12. [设计原则与演进历史](#设计原则与演进历史)

---

## 整体结构

$$
r_i^{\text{buffer}} =
\text{clip}\!\left(\frac{r_i^{\text{dense}} - \mu}{\sqrt{\sigma^2 + \epsilon}},\; -10,\; 10\right)
+ r_i^{\text{term}}
$$

- **稠密** $r_i^{\text{dense}}$：默认由 [simple staged](#训练用稠密奖励simple-staged默认) 组合；`reward_use_simple_stage=False` 时改为下方全部原子项之和。
- **终端** $r_i^{\text{term}}$：不归一化，在 Welford 归一化之后直接叠加。

环境每步仍计算全部原子分项并写入 `reward_components`（TensorBoard/诊断用），但 `reward_use_simple_stage=True`（当前默认）时，**只有组合后的 $r_i^{\text{dense}}$ 进入 PPO buffer**。

---

## 训练用稠密奖励（simple staged，默认）

`reward_use_simple_stage=True` 时，按 route 是否进入 **final stage**（`route_stage >= len(waypoints)-1`）分两档：

### 非 final stage（追赶 / 绕行）

$$
r_i^{\text{dense}} =
w_{\text{prog}} \, r_{\text{progress}}
+ w_{\text{chase}} \, r_{\text{chase\_speed}}
+ w_{\text{spd}} \, r_{\text{speed\_risk}}
+ w_{\text{saf}} \, r_{\text{safety\_risk}}
+ w_{\text{lane}} \, r_{\text{lane}}
+ w_{\text{act}} \, r_{\text{action\_pen}}
$$

| 组合项 | 默认权重 | 构成 |
|--------|----------|------|
| $r_{\text{progress}}$ | `reward_simple_nonfinal_progress_w` = 1.0 | route 模式下为 $r_{\text{route\_progress}}$（带每步 clip） |
| $r_{\text{speed\_risk}}$ | `reward_simple_speed_risk_w` = 1.0 | $\min(0,\, r_{\text{chase\_overspeed}},\, r_{\text{route\_speed\_limit}})$ |
| $r_{\text{safety\_risk}}$ | `reward_simple_safety_risk_w` = 1.0 | 见下式，总惩罚上限 `reward_simple_safety_risk_cap` = 1.8 |
| $r_{\text{action\_pen}}$ | `reward_simple_action_w` = 1.0 | $r_{\text{smooth}} + 0.5\, r_{\text{jerk}} + 0.25\, r_{\text{mag}} + 0.25\, r_{\text{yaw\_rate}}$ |

**安全风险聚合（v53 capped-sum）**：

$$
r_{\text{hull\_risk}} = \max\!\left(-c_{\text{hull}},\; r_{\text{hull\_safety}} + r_{\text{ship\_future\_safety}}\right)
$$

$$
r_{\text{tug\_risk}} = \max\!\left(-c_{\text{tug}},\; r_{\text{tug\_safety}} + r_{\text{spacing}} + r_{\text{cpa}}\right)
$$

$$
r_{\text{safety\_risk}} = \max\!\left(-c_{\text{saf}},\; r_{\text{hull\_risk}} + r_{\text{tug\_risk}}\right)
$$

默认 $c_{\text{hull}}=1.4$，$c_{\text{tug}}=1.2$，$c_{\text{saf}}=1.8$。相对 v52 的 `min()`，capped-sum 在多种风险并存时惩罚更强，减轻“只躲一种风险”的漂移。

### Final stage（就位 / 伴航）

$$
r_i^{\text{dense}} =
w_{\text{esc}} \, r_{\text{escort}}
+ r_{\text{hold}}
+ w_{\text{vm}} \, r_{\text{speed\_match}}
+ w_{\text{saf}} \, r_{\text{safety\_risk}}
+ w_{\text{act}} \, r_{\text{action\_pen}}
$$

| 组合项 | 默认权重 | 说明 |
|--------|----------|------|
| $r_{\text{escort}}$ | `reward_simple_final_escort_w` = 1.0 | 乘在原子 $r_{\text{escort}}$ 上（含 `reward_escort_final_multiplier`） |
| $r_{\text{hold}}$ | `reward_simple_hold_w` = 0.2 | 仅在 `in_zone` 时：$w_{\text{hold}} \cdot \min(1,\, \text{in\_zone\_steps}/\text{hold\_steps})$ |
| $r_{\text{speed\_match}}$ | `reward_simple_final_speed_match_w` = 0.25 | final 阶段才启用速度匹配 |

**关闭 staged 模式**：设 `reward_use_simple_stage=False`，则 $r_i^{\text{dense}}$ 为下文全部原子项之和（不含 $r_{\text{hold}}$）。

---

## 原子稠密奖励（诊断与 ablation）

以下分项每步都会计算并记入 `reward_components`；仅在 `reward_use_simple_stage=False` 时直接相加得到 $r^{\text{dense}}$。

---

## 稠密奖励 — 追赶阶段

### 1. 进度奖励 $r_{\text{progress}}$

鼓励拖轮持续逼近目标 slot。

**Route 模式**（基于路线剩余距离差分，带每步 clip）：

$$
\Delta d_{\text{route}} = \text{clip}\!\left( d_{\text{route}}^{(t-1)} - d_{\text{route}}^{(t)},\; -\Delta_{\max},\; \Delta_{\max} \right)
$$

$$
r_{\text{route\_progress}} = w_{\text{route\_progress}} \cdot \Delta d_{\text{route}}
$$

其中 $\Delta_{\max} =$ `route_progress_step_clip_m`（默认 0.45m/步），防止单步“跳 waypoint”刷进度。

**Direct 模式**（基于直线距离差分）：

$$
r_{\text{progress}} = w_{\text{progress}} \cdot \Big( d_{\text{slot}}^{(t-1)} - d_{\text{slot}}^{(t)} \Big)
$$

其中 $d_{\text{slot}}$ 为拖轮到目标 slot 的三维欧氏距离（含航向），$d_{\text{route}}$ 为沿路线到终点的剩余距离。

> **设计选择**：不用 $\log$ 势能——$\log$ 在 $d > 200\text{m}$ 时梯度极小 ($< 0.005$)，策略会停在远处不靠近。线性差分在全距离范围内梯度恒定。安全惩罚负责抑制冲撞。

**参数**：

| 参数 | 值 | 说明 |
|------|-----|------|
| $w_{\text{progress}}$ | 0.05 | Direct 模式权重 |
| $w_{\text{route\_progress}}$ | 0.06 | Route 模式权重（略高，补偿路线长度） |
| `route_progress_step_clip_m` | 0.45 | 每步 route progress 差分上限 (m) |

**典型值**：远距离 (600m) 靠近 1m/步 → $r \approx 0.05$；近距离 (30m) 靠近 1m/步 → $r \approx 0.05$。

---

### 2. 朝向奖励 $r_{\text{heading}}$

鼓励拖轮船首与期望航向（slot 方向/大船航向）对齐。

$$
r_{\text{heading}} = w_{\text{heading}} \cdot \cos(\Delta\psi)
$$

其中 $\Delta\psi = \psi_{\text{slot}} - \psi_{\text{tug}}$ 为拖轮当前航向与目标 slot 期望航向的偏差。

- $\cos(\Delta\psi) \in [-1, 1]$
- 精确对齐时 $r = +0.1$；反向时 $r = -0.1$

| 参数 | 值 |
|------|-----|
| $w_{\text{heading}}$ | 0.1 |

---

### 3. 速度匹配惩罚 $r_{\text{speed\_match}}$

在 **final approach 阶段**（route 模式最后一站），惩罚拖轮与大船之间的世界系速度差异。

$$
r_{\text{speed\_match}} = - w_{\text{speed\_match}} \cdot e^{-d / 80} \cdot \Big( \Delta u_{\text{world}} \cdot \tanh\big(\frac{\Delta u_{\text{world}}}{3}\big) \Big)
$$

其中 $\Delta u_{\text{world}} = \| \mathbf{v}_{\text{tug}} - \mathbf{v}_{\text{ship}} \|$ 为世界系速度差的模。

**门控设计**：
- $e^{-d/80}$：距离越近权重越大。$d=80\text{m}$ 时权重 $\approx 0.37$；$d=20\text{m}$ 时 $\approx 0.78$。
- $\tanh$ 压缩：大误差 ($\Delta u > 3\text{m/s}$) 封顶在 $3$，防止单步惩罚过大压制进度奖励。
- **追赶阶段（非 final）不激活**，避免拖轮在远处被迫降速。

| 参数 | 值 | 说明 |
|------|-----|------|
| $w_{\text{speed\_match}}$ | 0.15 | v25 从 0.2 降低 |
| 距离衰减尺度 | 80m | 扩大适用范围 |
| 误差压缩尺度 | 3m/s | tanh 饱和点 |

**典型值**：$d=30\text{m}$, $\Delta u=1\text{m/s}$ → $r \approx -0.037$；$d=10\text{m}$, $\Delta u=0.5\text{m/s}$ → $r \approx -0.047$。

---

### 4. 追赶速度奖励 $r_{\text{chase\_speed}}$

在 **追赶阶段**（非 final），鼓励拖轮相对于大船保持正向 closing speed。

$$
r_{\text{chase\_speed}} = w_{\text{chase\_speed}} \cdot \max\left(0,\; 1 - \frac{|\Delta u_x^{\text{ship}} - u_{\text{target}}|}{\max(u_{\text{target}}, 10^{-3})}\right)
$$

其中 $\Delta u_x^{\text{ship}}$ 为速度差 $\mathbf{v}_{\text{tug}} - \mathbf{v}_{\text{ship}}$ 在大船体系 $\hat{x}$ 方向（前进方向）的投影。$u_{\text{target}} =$ `route_chase_speed_target_ms`（默认 **0.35 m/s**）。

- 追赶值精确匹配目标时 $r = +0.08$；偏差超过目标时 $r = 0$。
- **Final stage 后不激活**，切回速度匹配惩罚。

| 参数 | 值 |
|------|-----|
| $w_{\text{chase\_speed}}$ | 0.08 |
| `route_chase_speed_target_ms` | 0.35 m/s |

---

### 4a. 追赶超速惩罚 $r_{\text{chase\_overspeed}}$

非 final 阶段，相对大船前进方向 closing speed 超过软上限时二次惩罚：

$$
r_{\text{chase\_overspeed}} = - w_{\text{os}} \cdot \left(\frac{\max(0,\, \Delta u_x^{\text{ship}} - u_{\max})}{u_{\max}}\right)^2
$$

其中 $u_{\max} =$ `route_chase_speed_max_ms`（默认 0.9 m/s）。进入 simple staged 的 $r_{\text{speed\_risk}}$。

| 参数 | 值 |
|------|-----|
| `reward_chase_overspeed_w` | 0.05 |

---

### 4b. 拖轮世界速度软上限 $r_{\text{route\_speed\_limit}}$

非 final 阶段，拖轮世界速度模超过软上限时二次惩罚，抑制远距离满油门：

$$
r_{\text{route\_speed\_limit}} = - w_{\text{lim}} \cdot \left(\frac{\max(0,\, \|\mathbf{v}_{\text{tug}}\| - v_{\lim})}{v_{\lim}}\right)^2
$$

$v_{\lim} =$ `route_tug_speed_soft_limit_ms`（默认 3.0 m/s）。

| 参数 | 值 |
|------|-----|
| `reward_route_speed_limit_w` | 0.02 |

---

## 稠密奖励 — 伴航阶段

### 5. 伴航累进奖励 $r_{\text{escort}}$ ⭐ v38

**v38 核心改动**：原 $r_{\text{zone}}$ 为三项乘积，任意维度略超阈值即整项归零（梯度悬崖）；改为 **位置门控加权和**。final stage 可乘 `reward_escort_final_multiplier`（默认 1.5×）加强持续伴航。

**软评分**（对每艘拖轮独立计算）：

$$
s_{\text{pos}} = \max\left(0,\; 1 - \frac{d}{D_{\text{pos}}}\right)
$$

$$
s_{\text{hdg}} = \max\left(0,\; 1 - \frac{|\Delta\psi|}{\Theta_{\text{hdg}}}\right)
$$

$$
s_{\text{spd}} = \max\left(0,\; 1 - \frac{\Delta u}{U_{\text{spd}}}\right)
$$

其中：
- $d$：拖轮到目标 slot 的直线距离
- $\Delta\psi$：航向偏差
- $\Delta u$：世界系速度差

**奖励公式**：

$$
r_{\text{escort}} = w_{\text{escort}} \cdot s_{\text{pos}} \cdot \Big( 0.4 + 0.3 \cdot s_{\text{hdg}} + 0.3 \cdot s_{\text{spd}} \Big)
$$

**门控机制**：
- $s_{\text{pos}}$ 作为主门控：$d > D_{\text{pos}}$ 时 $s_{\text{pos}} = 0$，整个 $r_{\text{escort}} = 0$，防止"远处对齐领奖励"。
- $s_{\text{pos}}$ 线性衰减从 1 到 0，提供全距离范围的连续梯度。
- $s_{\text{hdg}}$ 和 $s_{\text{spd}}$ 在位置接近后提供增量奖励（而非乘积消灭）。

| 场景 | $s_{\text{pos}}$ | $r_{\text{escort}}$ |
|------|------------------|---------------------|
| 完美伴航 ($d<10\text{m}$, $\Delta\psi<5^\circ$, $\Delta u<0.5$) | $\approx 0.85$ | $\approx 0.35$ |
| 位置达标但航向偏 | $\approx 0.5$ | $\approx 0.10$ |
| 远距离 ($d > 60\text{m}$) | $0$ | $0$ |
| 满分（理论上界） | $1.0$ | $0.50$ |

**参数**：

| 参数 | 值 | 说明 |
|------|-----|------|
| $D_{\text{pos}}$ (`pos_tol_m`) | 60m | 软评分参考距离 |
| $\Theta_{\text{hdg}}$ (`heading_tol_rad`) | $30^\circ = \pi/6$ | 航向容差 |
| $U_{\text{spd}}$ (`speed_tol_ms`) | 3.0 m/s | 速度容差 |
| $w_{\text{escort}}$ | 0.5 | 伴航奖励权重 |
| `reward_escort_final_multiplier` | 1.5× | final stage 权重倍率 |

### 5a. 在位累进奖励 $r_{\text{hold}}$（v52，仅 simple staged final）

当 `in_zone` 为真时，按已连续在位步数比例给小额正奖励：

$$
r_{\text{hold}} = w_{\text{hold}} \cdot \min\!\left(1,\; \frac{\text{in\_zone\_steps}}{\text{hold\_steps}}\right)
$$

默认 $w_{\text{hold}} =$ `reward_simple_hold_w` = 0.2。与 $r_{\text{escort}}$ 配合，鼓励稳定保持而非瞬时蹭线。

**与旧版对比**：

| | v37 (r_zone) | v38 (r_escort) | 当前默认 |
|---|---|---|---|
| 公式 | 三项乘积 | 位置门控加权和 | 同 v38 + staged 组合 |
| 终端成功 | $+20$ | 曾移除 | **+80 全员**（v57，`reward_arrival_bonus`） |
| hold_time | 1s | 曾 30s | **10s**（课程，可逐步提到 20s/30s） |

---

## 稠密奖励 — 操纵品质

### 6. 动作平滑性惩罚 $r_{\text{smooth}}$

抑制控制信号的剧烈跳变，鼓励平顺操控。

$$
r_{\text{smooth}} = - w_{\text{smooth}} \cdot \sum_{k=1}^{4} \big( a_k^{(t)} - a_k^{(t-1)} \big)^2
$$

其中 $\mathbf{a}^{(t)} \in [-1, 1]^4$ 为当前步的归一化动作向量，4 维对应 `[port_rpm, stbd_rpm, port_az, stbd_az]`。

**参数**：$w_{\text{smooth}} = 0.1$。完全不变的动作得 $r=0$；相邻步满量程反转得 $r \approx -1.6$。

---

### 7. 动作急动度惩罚 $r_{\text{jerk}}$ ⭐ v37

二阶平滑，抑制高频振荡模式（如 $+1, -1, +1, -1$ 交替）。

$$
\Delta\mathbf{a}^{(t)} = \mathbf{a}^{(t)} - \mathbf{a}^{(t-1)}
$$

$$
r_{\text{jerk}} = - w_{\text{jerk}} \cdot \sum_{k=1}^{4} \big( \Delta a_k^{(t)} - \Delta a_k^{(t-1)} \big)^2
$$

其中 $\Delta\mathbf{a}^{(t-1)}$ 为上一步缓存的动作变化量。$r_{\text{smooth}}$ 仅惩罚相邻变化，但在交替振荡模式下 $\Delta a^{(t)} \approx -\Delta a^{(t-1)}$，$\text{dda} \approx 2\Delta a$——jerk 惩罚恰好捕获这种模式。

**参数**：$w_{\text{jerk}} = 0.05$，设为平滑权重的一半，避免过度惩罚合法的快速机动调整。

---

### 8. 动作幅度惩罚 $r_{\text{mag}}$

防止无谓使用满舵满油门。

$$
r_{\text{mag}} = - w_{\text{mag}} \cdot \sum_{k=1}^{4} \big( a_k^{(t)} \big)^2
$$

**参数**：$w_{\text{mag}} = 0.01$。权重极小——只在没有其他约束时才鼓励零油门，不影响正常操控。

---

### 9. 偏航角速度惩罚 $r_{\text{yaw\_rate}}$

防止拖轮在原地高速自转，浪费能源且制造碰撞风险。

$$
r_{\text{yaw\_rate}} = - w_{\text{yaw\_rate}} \cdot r^2
$$

其中 $r = \dot{\psi}_{\text{tug}}$ 为拖轮当前偏航角速度 (rad/s)。

**参数**：$w_{\text{yaw\_rate}} = 0.1$。

**典型值**：$r = 0.1\text{rad/s} (\approx 5.7^\circ/\text{s})$ → $r \approx -0.001$；$r = 0.5\text{rad/s}$ → $r \approx -0.025$。

---

## 稠密奖励 — 安全约束

### 10. 安全距离惩罚 $r_{\text{safety}}$

连续的碰撞预警惩罚，在碰撞发生前就学会规避风险。

**对船体**（拖轮→大船）：

$$
r_{\text{safety}}^{\text{hull}} = - w_{\text{safety}} \cdot \exp\!\left( -\frac{d_{\text{hull}} - D_{\text{coll}}^{\text{ship}}}{\max(D_{\text{safe}}^{\text{hull}} - D_{\text{coll}}^{\text{ship}}, 10^{-3})} \right) \quad \text{if } d_{\text{hull}} < D_{\text{safe}}^{\text{hull}}
$$

**对拖轮间**：

$$
r_{\text{safety}}^{\text{tug}} = - w_{\text{safety}} \cdot \exp\!\left( -\frac{d_{\text{pair}} - D_{\text{coll}}^{\text{tug}}}{\max(D_{\text{safe}}^{\text{tug}} - D_{\text{coll}}^{\text{tug}}, 10^{-3})} \right) \quad \text{if } d_{\text{pair}} < D_{\text{safe}}^{\text{tug}}
$$

**指数势能设计**：
- $d \to D_{\text{coll}}$ 时 → $\exp(0) = 1$，惩罚 $\approx -w_{\text{safety}}$
- $d \to D_{\text{safe}}$ 时 → $\exp(-1) \approx 0.37$，惩罚平滑衰减
- $d \ge D_{\text{safe}}$ 时 → 不惩罚 = 0

**Final approach 阶段加强**：`ship_safety_final_dist_m` 提升到 **40m**，`reward_hull_safety_final_multiplier` = **3.0×**。

拖轮间安全预警距离 $D_{\text{safe}}^{\text{tug}} = 2 \times D_{\text{coll}}^{\text{tug}}$（默认 40m）。

**参数**：

| 参数 | 值 | 说明 |
|------|-----|------|
| $w_{\text{safety}}$ | 0.3 | 基础权重 |
| $D_{\text{coll}}^{\text{ship}}$ (`ship_collision_dist_m`) | 6m | 船体碰撞阈值 |
| $D_{\text{coll}}^{\text{tug}}$ (`tug_collision_dist_m`) | 20m | 拖轮间碰撞阈值 |
| $D_{\text{safe}}^{\text{hull}}$ (`ship_safety_dist_m`) | 18m (非final) | 船体安全预警距离 |
| $D_{\text{safe}}^{\text{hull}}$ (`ship_safety_final_dist_m`) | 40m (final) | final 阶段扩大 |
| 倍率 (`reward_hull_safety_final_multiplier`) | 3.0× | final 阶段权重倍率 |

---

### 10a. 预测船体安全惩罚 $r_{\text{ship\_future\_safety}}$

在短时未来窗口内，按常速外推大船与拖轮轨迹，检查到**膨胀船体**的最小距离，提前压制“当前尚安全、数秒后会贴船”的高速切入。

对采样时刻 $\tau \in \{ \Delta t,\, \ldots,\, T_h \}$（默认 $T_h=14$s，5 个样本）：

$$
\text{score}_\tau = \left(\frac{D_{\text{future}} - d_\tau}{D_{\text{future}} - D_{\text{coll}}^{\text{ship}}}\right)^2 \cdot \max\!\left(0,\, 1 - \frac{\tau}{T_h}\right)
$$

$$
r_{\text{ship\_future\_safety}} = -\min\!\left( w_f \cdot \sum_\tau \text{score}_\tau,\; P_{\max} \right)
$$

仅当 $d_\tau < D_{\text{future}}$ 时计入；final stage 使用更大的 `ship_future_safety_final_dist_m` 与 `reward_ship_future_safety_final_multiplier`（1.8×）。

| 参数 | 值 |
|------|-----|
| `reward_ship_future_safety_w` | 0.25 |
| `ship_future_safety_dist_m` | 26m |
| `ship_future_safety_final_dist_m` | 45m |
| `reward_ship_future_safety_max_penalty` | 0.9 |

---

### 11. 航道约束惩罚 $r_{\text{lane}}

船尾追赶任务中（route 模式），左舷/右舷角色应沿对应舷侧绕行，不应从船体后方直接穿越到对侧。

**原理**：
- 将拖轮位置投影到**大船体坐标系**下
- 用 `y_body` 表示横距：正 = 右舷，负 = 左舷
- 根据 slot 身份（船首左/右、船尾左/右）确定应在舷侧

$$
\text{side\_coord} = \text{sign}_{\text{slot}} \cdot y_{\text{body}}
$$

$$
r_{\text{lane}} = - w_{\text{lane}} \cdot \frac{\max(0,\; D_{\text{lane}} - \text{side\_coord})}{D_{\text{lane}}}
$$

其中 $y_{\text{body}}$ 为大船体坐标系横向坐标（右为正），$\text{sign}_{\text{slot}} \in \{-1,+1\}$ 为 slot 舷侧符号；$\text{side\_coord}$ 过小表示未保持在对应舷侧外侧。$D_{\text{lane}} =$ `route_lane_min_lat_m`（32m）。

- 仅在纵向走廊范围内生效（大船尾部 ± 限幅）
- 同侧条件判断：船尾左舷和船首左舷共享左侧走廊

**参数**：

| 参数 | 值 |
|------|-----|
| $w_{\text{lane}}$ | 0.2 |
| `route_lane_min_lat_m` | 32m |

---

### 12. 同侧拖轮间距惩罚 $r_{\text{spacing}}$

比硬碰撞阈值更早给梯度，抑制同侧拖轮（如船首左+船尾左）追尾或并线挤压。

$$
r_{\text{spacing}} = - w_{\text{spacing}} \cdot \left( \frac{D_{\text{spacing}} - d_{\text{pair}}}{D_{\text{spacing}}} \right)^2 \quad \text{if } d_{\text{pair}} < D_{\text{spacing}}
$$

其中 $D_{\text{spacing}} = 90\text{m}$，仅对同侧角色生效。

- 二次增长：间距越近惩罚加速加大
- 间距 $\ge D_{\text{spacing}}$ 时 = 0

**参数**：

| 参数 | 值 |
|------|-----|
| $w_{\text{spacing}}$ | 0.25 |
| `route_tug_spacing_dist_m` | 90m |

---

### 13. CPA 风险惩罚 $r_{\text{cpa}}$

短视距离惩罚只看当前距离，容易漏掉"现在还远但正在快速会遇"的风险。CPA 使用未来短时间窗口内的**最近会遇距离 (DCPA)** 和 **最近会遇时间 (TCPA)**，鼓励拖轮提前错峰/减速。

**CPA 计算**（对每对拖轮 $(i,j)$）：

相对速度 $\mathbf{v}_r = \mathbf{v}_j - \mathbf{v}_i$，相对位置 $\mathbf{p}_r = \mathbf{p}_j - \mathbf{p}_i$：

$$
t_{\text{cpa}} = -\frac{\mathbf{p}_r \cdot \mathbf{v}_r}{\|\mathbf{v}_r\|^2} \quad (\text{若 } t_{\text{cpa}} < 0 \text{ 则取 } 0)
$$

$$
\text{DCPA} = \| \mathbf{p}_r + \mathbf{v}_r \cdot t_{\text{cpa}} \|
$$

**风险评分**（对每对进入预警窗口的拖轮）：

$$
s_{\text{dcpa}} = \min\left(1,\; \frac{D_{\text{alert}} - \text{DCPA}}{D_{\text{alert}} - D_{\text{coll}}^{\text{tug}}}\right)
$$

$$
s_{\text{tcpa}} = \max\left(0,\; 1 - \frac{\text{TCPA}}{T_{\text{horizon}}}\right)
$$

$$
r_{\text{cpa}}^{(i)} = - \min\!\left( w_{\text{cpa}} \cdot \sum_{j \neq i} s_{\text{dcpa}}^2 \cdot s_{\text{tcpa}},\; P_{\text{max}} \right)
$$

- $s_{\text{dcpa}}^2$ 使 DCPA 越接近碰撞阈值惩罚增长越快
- $s_{\text{tcpa}}$ 过滤"时间尚早"的会遇——TCPA 超过 $T_{\text{horizon}}$ 不惩罚
- 单 agent 单步惩罚上限 $P_{\text{max}} = 0.75$

**Final approach 阶段加强**：权重重 $\times 2.0$。

**参数**：

| 参数 | 值 | 说明 |
|------|-----|------|
| $w_{\text{cpa}}$ | 0.18 | 基础权重 |
| $D_{\text{alert}}$ (`cpa_alert_dist_m`) | 70m | DCPA 低于此值开始预警 |
| $T_{\text{horizon}}$ (`cpa_time_horizon_s`) | 45s | TCPA 超此时限不惩罚 |
| $P_{\text{max}}$ (`reward_cpa_max_penalty`) | 0.75 | 单步 CPA 惩罚上限 |
| 倍率 (`reward_cpa_final_multiplier`) | 2.0× | final 阶段翻倍 |

---

## 终端奖励

终端奖励 **不参与稠密奖励归一化**，在 episode 终止步写入 `info["terminal_reward"]`，由训练脚本在 Welford 归一化之后叠加到 buffer。

**v56 差异化碰撞（当前实现）**：

| 事件 | 受影响 agent | 奖励 |
|------|----------------|------|
| 拖轮-大船碰撞 | 肇事拖轮 $i$ | $-20$ |
| 拖轮-拖轮碰撞 | 涉事对 $(i,j)$ | 各 $-20$ |
| 成功 | **全部 4 艘** | $+80$（`reward_arrival_bonus`） |
| 超时 | 全部 | $0$ |

$$
r_i^{\text{term}} = \begin{cases}
-\texttt{reward\_collision\_pen} & \text{仅肇事/涉事拖轮} \\
+\texttt{reward\_arrival\_bonus} & \text{成功时全员} \\
0 & \text{超时或未触发}
\end{cases}
$$

> v27 曾在**稠密**奖励里做差异化碰撞惩罚且共享策略，导致 succ 下降；v56 把差异化仅用于**终端**碰撞，成功仍全员同奖，以拉开 success vs timeout 的 return 差距（v50 audit：timeout 累积 return 常高于 success）。

| 参数 | 值 | 说明 |
|------|-----|------|
| `reward_collision_pen` | 20.0 | 碰撞终端惩罚（按肇事分配） |
| `reward_arrival_bonus` | 80.0 | 成功终端奖励（全员） |

---

## 奖励归一化

训练使用 **Welford 在线算法** 维护稠密奖励的运行均值 $\mu$ 和方差 $\sigma^2$：

$$
\mu_{n+1} = \mu_n + \delta \cdot \frac{m}{n+m}
$$

$$
\sigma^2_{n+1} = \frac{\sigma^2_n \cdot n + \sigma^2_{\text{batch}} \cdot m + \delta^2 \cdot \frac{n \cdot m}{n+m}}{n + m}
$$

其中 $\delta = \mu_{\text{batch}} - \mu_n$。归一化后裁剪到 $[-10, 10]$：

$$
r_i^{\text{norm}} = \text{clip}\left( \frac{r_i^{\text{dense}} - \mu}{\sqrt{\sigma^2 + \epsilon}},\; -10,\; 10 \right)
$$

最终进入 PPO buffer 的奖励：

$$
r_i^{\text{buffer}} = r_i^{\text{norm}} + r_i^{\text{term}}
$$

> **设计原则**：稠密奖励归一化保证 value loss 量级稳定（不受奖励绝对值影响），再独立叠加终端奖励保证碰撞/成功信号的强度不被方差压缩。

---

## 到位判定与终止条件

### 硬阈值判定（用于 hold_time 累计和成功判定）

$$
\text{in\_zone} = \Big( d < D_{\text{pos}} \Big) \;\land\; \Big( |\Delta\psi| < \Theta_{\text{hdg}} \Big) \;\land\; \Big( \Delta u < U_{\text{spd}} \Big)
$$

三项必须**同时满足**。任一不满足则 `in_zone_steps` 重置为 0。

### 成功终止

所有 4 艘拖轮的 `in_zone_steps` 连续 $\ge T_{\text{hold}} / \Delta t_{\text{ctrl}}$ 步时判定成功：

$$
\text{hold\_steps} = \left\lfloor \frac{\texttt{hold\_time\_s}}{\texttt{dt\_ctrl}} \right\rfloor
$$

当前默认 `hold_time_s=10.0`，`dt_ctrl=0.2` → **50 步**。课程设计为 5s → 10s → 20s → 30s 逐步加长。

### 终止事件优先级

1. **碰撞**（最高优先）—— episode 立即终止，`terminated=True`，肇事/涉事拖轮 $-20$
2. **成功**—— 4 艘拖轮均连续 `in_zone` ≥ hold_steps，全员 $+80$
3. **超时**—— `max_episode_steps` 达到，`truncated=True`（GAE 可 bootstrap）

---

## 参数速查表

### Simple staged 组合权重（`reward_use_simple_stage=True`，默认）

| 参数 | 值 | 阶段 |
|------|-----|------|
| `reward_simple_nonfinal_progress_w` | 1.0 | 非 final |
| `reward_simple_route_chase_w` | 0.5 | 非 final |
| `reward_simple_speed_risk_w` | 1.0 | 非 final |
| `reward_simple_safety_risk_w` | 1.0 | 全程 |
| `reward_simple_lane_w` | 1.0 | 非 final |
| `reward_simple_action_w` | 1.0 | 全程 |
| `reward_simple_final_escort_w` | 1.0 | final |
| `reward_simple_hold_w` | 0.2 | final |
| `reward_simple_final_speed_match_w` | 0.25 | final |
| `reward_simple_hull_risk_cap` | 1.4 | 安全聚合 |
| `reward_simple_tug_risk_cap` | 1.2 | 安全聚合 |
| `reward_simple_safety_risk_cap` | 1.8 | 安全聚合 |

### 原子奖励权重

| 参数 | 值 | 类别 |
|------|-----|------|
| `reward_progress_w` | 0.05 | 追赶 |
| `reward_route_progress_w` | 0.06 | 追赶 |
| `reward_heading_w` | 0.1 | 追赶 |
| `reward_speed_match_w` | 0.15 | final |
| `reward_chase_speed_w` | 0.08 | 非 final |
| `reward_chase_overspeed_w` | 0.05 | 非 final |
| `reward_route_speed_limit_w` | 0.02 | 非 final |
| `reward_escort_w` | 0.5 | final |
| `reward_escort_final_multiplier` | 1.5 | final |
| `reward_ship_future_safety_w` | 0.25 | 安全 |
| `reward_smooth_w` / `reward_jerk_w` / `reward_mag_w` / `reward_yaw_rate_w` | 0.1 / 0.05 / 0.01 / 0.1 | 品质 |
| `reward_safety_w` | 0.3 | 安全 |
| `reward_lane_w` | 0.2 | 安全 |
| `reward_spacing_w` | 0.25 | 安全 |
| `reward_cpa_w` | 0.18 | 安全 |
| `reward_collision_pen` | 20.0 | 终端 |
| `reward_arrival_bonus` | 80.0 | 终端 |

### 安全/判定阈值

| 参数 | 值 | 说明 |
|------|-----|------|
| `pos_tol_m` | 60m | 到位距离阈值 |
| `heading_tol_rad` | $\pi/6$ (30°) | 到位航向阈值 |
| `speed_tol_ms` | 3.0 | 到位速度阈值 |
| `hold_time_s` | 10.0 | 成功所需连续在位时间（课程） |
| `dt_ctrl` | 0.2s | 控制周期 |
| `ship_collision_dist_m` | 6m | 拖轮-大船 碰撞判定 |
| `tug_collision_dist_m` | 20m | 拖轮间 碰撞判定 |
| `ship_safety_dist_m` | 18m | 船体安全预警（非final） |
| `ship_safety_final_dist_m` | 40m | 船体安全预警（final） |
| `route_chase_speed_target_ms` | 0.35 | 追赶目标相对速度 |
| `route_chase_speed_max_ms` | 0.9 | 追赶超速软上限 |
| `route_tug_spacing_dist_m` | 90m | 同侧间距惩罚阈值 |
| `cpa_alert_dist_m` | 70m | CPA 预警 DCPA 阈值 |
| `cpa_time_horizon_s` | 45s | CPA 预警时间窗口 |

---

## 设计原则与演进历史

### 核心设计原则

1. **稠密归一化 + 终端独立**：稠密奖励经 Welford 归一化到 $\mathcal{N}(0,1)$，终端奖励直接叠加——碰撞信号不被方差压缩。
2. **线性进度（不用势能）**：$\log(d)$ 在远距离梯度极小，线性差分在全距离范围梯度恒定。
3. **指数安全势能**：$-\exp(-(d-D_{\text{coll}})/\alpha)$ 在靠近碰撞阈值时急剧放大惩罚。
4. **乘积 → 加权和 (v38)**：$r_{\text{escort}}$ 用位置门控加权和，消除 zone 乘积悬崖。
5. **Staged 组合 (v52)**：训练目标改为少量组合项，原子项仅诊断；非 final / final 分档。
6. **安全 capped-sum (v53)**：多种风险并存时惩罚可叠加（带上限），避免 `min()` 只保留最严重一项。
7. **终端差异化碰撞 (v56)**：仅肇事/涉事拖轮 $-20$；成功仍全员 $+80$。
8. **CPA + 预测船体安全**：DCPA/TCPA 与短时轨迹外推，显式化碰撞风险。

### 关键版本演进

| 版本 | 改动 | 动机 |
|------|------|------|
| v4 | 终端奖励分离 | dense/terminal 独立归一化 |
| v7 | 线性进度替换 log 势能 | 远距离梯度消失 |
| v25 | zone soft + tanh 速度匹配 | 梯度悬崖 + 速度惩罚过强 |
| v32-34 | CPA 观测/奖励 + 船体 CPA | 显式碰撞风险 |
| v37 | jerk 惩罚 | 抑制高频振荡 |
| v38 | stable escort（加权和） | 消除 zone 乘积悬崖 |
| v52-53 | simple staged + capped safety | 简化训练信号、加强复合风险 |
| v56-57 | 差异化终端碰撞 + arrival 80 | 拉开 success/timeout return |
