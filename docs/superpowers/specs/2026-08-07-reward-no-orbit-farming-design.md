# 防外围刷分奖励重构设计

## 1. 背景与根因

当前奖励已经包含停滞惩罚和进槽走廊软化，但近期训练仍长期为
`capture_rate=0`，评估末距常停留在约 200–350 m。

现实现从 `reward_hold_start_m=150` 开始用 `hold_gate` 逐渐关闭距离接近
奖励；与此同时，`r_hold` 的位置分数只有进入 `pos_tol_m=10` 后才非零。
因此 10–150 m 之间缺少连续、明确的到位激励。目标 slot 距船体外廓约
32 m，又仍在 80 m 船体软碰范围内，进一步放大了在外围规避风险的局部
最优。

现有 `dist_bonus` 是可逐步累积的绝对距离正奖励，需要额外的滑动窗停滞
惩罚来抑制刷分；现有负势函数配合折扣形式也可能向静止状态泄漏正奖励。
本设计从奖励结构上移除这些激励冲突，而不是继续叠加补丁惩罚。

## 2. 目标与非目标

### 2.1 目标

- 外围静止或等半径绕圈的长期平均稠密奖励严格小于 0。
- 沿安全进槽走廊持续接近的奖励高于静止和远离。
- 接近奖励在进入 10 m 目标区前始终有效，不在 150 m 附近衰减。
- 目标区内的位置、航向和速度保持获得稳定正奖励。
- 团队项重点约束最落后的拖轮，避免部分智能体留在外围。
- CPU 与 GPU 奖励公式、分量和总奖励保持一致。

### 2.2 非目标

- 不修改 observation、actor、PPO 或初始化策略。
- 不修改 Capture / Track 判定。
- 不修改船体 6 m、拖轮间 20 m 的硬碰撞阈值。
- 不修改硬碰撞终止及 culprit/bystander 终端惩罚。
- 不引入离散里程碑状态或奖励权重课程。

## 3. 总体奖励结构

每艘拖轮的稠密奖励为：

\[
R_i =
w_p R_{\mathrm{progress},i}
- w_d C_{\mathrm{distance},i}
+ w_h R_{\mathrm{hold},i}
- w_c P_{\mathrm{collision},i}
+ w_v R_{\mathrm{velocity},i}
+ w_t R_{\mathrm{team}}.
\]

`R_velocity` 保留现有可选接口且默认权重为 0。旧 `dist_bonus`、停滞滑动
窗和负势函数不再参与有效奖励。

## 4. 目标门控

令 \(d_i\) 为拖轮到分配 slot 的平面距离，
\(d_{\mathrm{tol}}=\texttt{pos_tol_m}=10\,\mathrm m\)：

\[
x_i=\operatorname{clip}\left(1-\frac{d_i}{d_{\mathrm{tol}}},0,1\right),
\qquad
g_i=x_i^2(3-2x_i).
\]

- \(d_i\ge d_{\mathrm{tol}}\) 时 \(g_i=0\)，接近项完全生效。
- 进入 10 m 后，接近项平滑退出，保持项平滑进入。
- 该门控替代奖励中的旧 20–150 m `hold_gate`；150 m 范围只继续服务于
  船体软碰走廊。

## 5. 单艇奖励分量

### 5.1 接近进度

\[
R_{\mathrm{progress},i}
=(1-g_i)\,
\operatorname{clip}\left(
\frac{d_{i,t-1}-d_{i,t}}{d_{\mathrm{clip}}},-1,1
\right),
\qquad d_{\mathrm{clip}}=1\,\mathrm m.
\]

接近为正、远离为负、静止或等半径运动为 0。默认进度归一化尺度从
5 m 改为 1 m，使正常航速产生可辨识的单步学习信号。

### 5.2 距离代价

\[
C_{\mathrm{distance},i}
=(1-g_i)\,
\operatorname{clip}\left(\frac{d_i}{d_{\mathrm{ref}}},0,1\right),
\qquad d_{\mathrm{ref}}=200\,\mathrm m.
\]

任何目标区外位置静止都会持续扣分；200 m 外封顶，防止尺度随初始化或失控
距离无限增长。它直接替代绝对距离正奖励和停滞补丁。

### 5.3 目标保持

保持方向和速度分数沿用现有线性饱和定义：

\[
s_{\mathrm{heading},i}=\max\left(
0,1-\frac{|\Delta\psi_i|}{\psi_{\mathrm{tol}}}
\right),
\qquad
s_{\mathrm{speed},i}=\max\left(
0,1-\frac{e_{v,i}}{v_{\mathrm{tol}}}
\right).
\]

新的保持项为：

\[
R_{\mathrm{hold},i}
=g_i\left(0.5+0.25s_{\mathrm{heading},i}
+0.25s_{\mathrm{speed},i}\right).
\]

位置精度由 \(g_i\) 编码；目标中心且航向、速度匹配时该项为 1。

## 6. 团队最落后代价

令 \(c_i=C_{\mathrm{distance},i}\)，使用 softmax 权重构造连续的软最大值：

\[
C_{\mathrm{team}}
=\frac{\sum_i c_i e^{\beta c_i}}{\sum_i e^{\beta c_i}},
\qquad
R_{\mathrm{team}}=-C_{\mathrm{team}}.
\]

该共享项加到每艘拖轮的奖励中。一个拖轮明显落后时，它主导团队代价；
全部进入目标中心时该项为 0。默认 `reward_team_softmin_beta=4.0` 保留，但
语义更新为上述软最大距离代价的锐度。

实现需使用减去最大指数输入的稳定 softmax 形式，避免极端配置下溢出或
溢出。

## 7. 碰撞安全边界

现有碰撞结构原样保留：

- 船体和拖轮间近距势垒；
- CPA 预测风险；
- `reward_collision_cap`；
- 走廊内仅软化船体软碰项；
- 拖轮间软碰不随走廊软化；
- 船体距离小于 6 m 或拖轮间距离小于 20 m 时立即终止；
- culprit/bystander 终端惩罚保持现值。

`reward_hold_start_m=150.0` 继续作为进槽走廊的启用距离，不再参与
`R_progress`、`C_distance` 或 `R_hold` 的门控。

## 8. 默认参数与兼容策略

| 配置项 | 新默认 | 语义 |
|---|---:|---|
| `reward_dist_w` | 3.0 | 纯接近进度权重 |
| `reward_distance_cost_w` | 0.2 | 单艇距离代价权重 |
| `reward_dist_progress_clip_m` | 1.0 | 单步进度归一化尺度 |
| `reward_dist_scale_m` | 200.0 | 距离代价封顶尺度 |
| `reward_hold_w` | 2.0 | 目标保持权重 |
| `reward_team_w` | 0.2 | 共享最落后代价权重 |
| `reward_shape_w` | 0.0 | 旧势函数关闭 |

碰撞相关权重和终端奖励全部保持现值。

`reward_hold_full_m`、`reward_stall_*` 和距离历史环形缓冲退出有效路径并从
代码、配置及文档中清理。旧负势函数实现删除；`reward_shape_w` 暂留为默认
0 的兼容字段且不再参与计算。`reward_dist_w` 保留字段名以降低接口破坏；
其文档语义更新为接近进度权重。

所有距离尺度在计算时限制到正数下限，权重和门控限制到合法范围。

## 9. 诊断与数据流

每步动力学更新后，CPU 或 GPU 奖励路径按以下顺序计算：

1. slot 距离、航向误差、相对速度和船体距离；
2. 目标门控、接近进度、距离代价和保持奖励；
3. 船体/拖轮近距与 CPA 风险及走廊软化；
4. 团队软最大距离代价；
5. 加权总奖励；
6. Capture / Track 与硬碰撞终止；
7. 更新 `prev_dist` 等单步历史。

`reward_components` 调整为：

- `r_dist`：乘权重前的纯接近进度；
- `p_distance`：乘权重前的非负距离代价；
- `r_hold`：乘权重前的目标保持奖励；
- `r_team`：实际加入总奖励的加权负团队项；
- 保留 `r_total`、`r_velocity`、`p_collision`、`p_ship_collision`、
  `p_tug_collision`、`dist_to_slot`、`heading_err_deg`、`speed_err`、
  `hull_dist`、`in_zone`、`corridor_gate` 和 `ship_soft_scale`；
- 删除 `p_stall`、`stall_scale`、`r_shape` 和旧奖励 `hold_gate` 诊断。

TensorBoard 至少记录 `r_dist`、`p_distance`、`r_hold` 和 `p_collision`。

## 10. 实现落点

| 文件 | 变更 |
|---|---|
| `config.py` | 新距离代价权重、默认尺度和废弃字段清理 |
| `env/reward.py` | CPU 新公式、团队项和诊断 |
| `env/gpu/batched_step.py` | GPU 同构公式与诊断 |
| `env/state.py` | 删除距离历史环形缓冲 |
| `env/formation_env.py` | 删除距离历史初始化与写入 |
| `scripts/train.py` | TensorBoard 奖励键更新 |
| `tests/` | 奖励关系、碰撞不变量及 CPU/GPU parity |
| `docs/reward_function.md` | 奖励公式与默认值同步 |
| `docs/tensorboard_metrics.md` | 指标语义同步 |

## 11. 测试与验收

### 11.1 单元与一致性测试

- 200 m、100 m、25 m 静止时总稠密奖励均小于 0。
- 等半径绕圈时 `r_dist=0` 且总奖励小于 0。
- 同状态安全接近的奖励高于静止，远离的奖励低于静止。
- 150 m 两侧相同进度得到连续的 `r_dist`。
- 10 m 内位置、航向和速度越准确，`r_hold` 越高。
- 目标中心稳定保持的总稠密奖励明显为正。
- 单艇滞后时全体收到负 `r_team`，滞后距离下降后其绝对值减小。
- 硬碰撞阈值、终止类型和终端奖励逐值不变。
- 走廊只软化船体软碰，不软化拖轮间风险。
- CPU/GPU 在确定性轨迹上的各分量和总奖励一致。

### 11.2 脚本化轨迹

比较相同时长的三类轨迹：外围静止、外围等半径绕圈、沿走廊接近并保持。
接近并保持的累计回报必须严格最高；延长前两类轨迹不能增加累计收益。

### 11.3 固定配置短训

实现与单测通过后，以当前固定初始化、slot assignment 和 seed 做短训对照：

- `capture_rate` 从 0 提升为正；
- `final_dist` 明显低于现有约 200–350 m；
- 碰撞率不高于对应旧奖励基线 10 个百分点以上；
- `p_distance`、`r_dist`、`r_hold` 曲线与接近、入位阶段一致。

若短训仍为 `capture_rate=0`，必须先依据奖励分量和轨迹诊断定位原因，不能
继续叠加未经验证的权重修改。
