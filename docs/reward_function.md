# 拖轮编队奖励函数

> 实现：`env/reward.py` → `FormationRewardComputer.compute_rewards()`  
> 配置：`config.py` → `EnvConfig`  
> 终端奖罚：`env/formation_env.py` → `step()`（不参与稠密奖励归一化）

## 1. 稠密奖励

每艘拖轮独立计算：

$$
R_{i,t}
=
w_{\mathrm{dist}} R_{\mathrm{dist},i}
+
w_{\mathrm{hold}} R_{\mathrm{hold},i}
+
w_{\mathrm{vel}} R_{\mathrm{vel},i}
-
w_{\mathrm{coll}} P_{\mathrm{coll},i}
+
R_{\mathrm{shape},i}
+
R_{\mathrm{team}}.
$$

默认权重（`EnvConfig`）：

| 符号 | 配置项 | 默认 |
|------|--------|-----:|
| \(w_{\mathrm{dist}}\) | `reward_dist_w` | 3.0 |
| \(w_{\mathrm{hold}}\) | `reward_hold_w` | 2.0 |
| \(w_{\mathrm{vel}}\) | `reward_velocity_w` | 0.0（关闭） |
| \(w_{\mathrm{coll}}\) | `reward_collision_w` | 1.0 |
| shaping | `reward_shape_w` | 0.3 |
| team | `reward_team_w` | 0.2 |

\(P_{\mathrm{coll}}\) 非负，总奖励中减去；并经 `reward_collision_cap`（默认 2.0）截断。

## 2. 距离与保持

记 \(d_i\) 为到目标 slot 的平面距离，\(\Delta\psi_i\) 为航向误差，\(e_{v,i}\) 为相对大船速度误差。

**Hold gate**（距离平滑切换，`reward_hold_start_m=150`，`reward_hold_full_m=20`）：

- \(d \le 20\,\mathrm{m}\) → \(\mathrm{gate}=1\)
- \(d \ge 150\,\mathrm{m}\) → \(\mathrm{gate}=0\)
- 中间 smoothstep

**距离项**（远距主导）：

\[
R_{\mathrm{dist}}
=
(1-\mathrm{gate})
\cdot
\bigl(0.4\cdot \mathrm{progress} + 0.6\cdot \mathrm{dist\_bonus}\bigr)
\]

- \(\mathrm{progress}\)：相对上一步距离缩短，clip 到 \(\pm\) `reward_dist_progress_clip_m`（默认 5 m）后归一化到 \([-1,1]\)
- \(\mathrm{dist\_bonus}=1-d/\texttt{reward\_dist\_scale\_m}\)（默认 scale 500 m），再 clip 到 \([-0.5,1]\)

**保持项**（近距主导）：

\[
R_{\mathrm{hold}}
=
\mathrm{gate}
\cdot
s_{\mathrm{pos}}
\cdot
(0.5 + 0.25\,s_{\mathrm{head}} + 0.25\,s_{\mathrm{speed}})
\]

其中 \(s_{\mathrm{pos}},s_{\mathrm{head}},s_{\mathrm{speed}}\) 分别为相对 `pos_tol_m` / `heading_tol_rad` / `speed_tol_ms` 的线性饱和分数。

**速度匹配**（默认权重 0，公式仍计算）：近距 gate 下对相对速度与偏航差的高斯型惩罚，记为 \(R_{\mathrm{vel}}\le 0\)。

## 3. 碰撞风险 \(P_{\mathrm{coll}}\)

\[
P_{\mathrm{coll}}
=
\min\bigl(P_{\mathrm{ship}} + P_{\mathrm{tug}},\; \texttt{reward\_collision\_cap}\bigr)
\]

对大船与每艘邻居拖轮：

- **近距势垒**：距离低于 safe 半径时线性升至 1（船 safe 默认 100 m，拖轮 120 m；碰撞硬阈值见 `ship_collision_dist_m` / `tug_collision_dist_m`）
- **CPA**：在 `reward_cpa_horizon_s`（60 s）内若接近，用未来最近点距离再加权势垒；系数 `reward_collision_cpa_w`（默认 2.0）

## 4. Shaping 与团队项

**势函数 shaping**（`reward_shape_w>0`）：

\[
\phi
=
-\bigl(
0.6\,d/d_{\mathrm{ref}}
+
0.25\,e_v/v_{\mathrm{tol}}
+
0.15\,|\Delta\psi|/\psi_{\mathrm{tol}}
\bigr)
\]

\[
R_{\mathrm{shape}}
=
w_{\mathrm{shape}}
\cdot
\mathrm{clip}(\gamma\phi_t - \phi_{t-1},\;\pm\texttt{reward\_shape\_clip})
\]

**团队 softmin**（`reward_team_w>0`）：对全体「到位软分数」做 softmin，鼓励同步入位；加到每个 agent 的奖励上。

## 5. in-zone 计数

当且仅当

\[
d < \texttt{pos\_tol\_m}
\;\wedge\;
|\Delta\psi| < \texttt{heading\_tol\_rad}
\;\wedge\;
e_v < \texttt{speed\_tol\_ms}
\]

时 `in_zone_steps += 1`，否则 `max(0, steps-2)`。该计数驱动 Capture / Track（见 [task_spec.md](task_spec.md)）。

## 6. 终端奖罚

| 事件 | 奖励 |
|------|------|
| **刚 Capture**（`just_captured`） | 全体 `+reward_arrival_bonus`（默认 80），只发一次 |
| **Track success** | 无额外到位奖励 |
| 碰撞 | culprit `−reward_collision_pen_culprit`（80）；其余 bystander `−reward_collision_pen_bystander`（15） |

训练时：稠密奖励经 RunningMeanStd 归一化后写入 buffer；终端项在归一化后按原尺度叠加。

## 7. `reward_components` 诊断字段

| 键 | 含义 |
|----|------|
| `r_total` / `r_dist` / `r_hold` / `r_velocity` / `r_shape` / `r_team` | 各分量 |
| `p_collision` / `p_ship_collision` / `p_tug_collision` | 风险项 |
| `dist_to_slot` / `heading_err_deg` / `speed_err` / `hull_dist` | 几何误差 |
| `in_zone` / `hold_gate` | 到位与 gate |

TensorBoard 默认只记录 `r_dist`、`r_hold`、`p_collision`（见 [tensorboard_metrics.md](tensorboard_metrics.md)）。

## Preset 消融

筛选用超参组合见 [reward_presets.md](reward_presets.md)（`--reward-preset`）。
