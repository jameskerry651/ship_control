# 拖轮编队奖励函数

> 实现：`env/reward.py` → `FormationRewardComputer.compute_rewards()`  
> 配置：`config.py` → `EnvConfig`  
> 终端奖罚：`env/formation_env.py` → `step()`（不参与稠密奖励归一化）  
> 结构设计：`docs/superpowers/specs/2026-08-07-reward-redesign-design.md`

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
-
w_{\mathrm{stall}} P_{\mathrm{stall},i}
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
| \(w_{\mathrm{stall}}\) | `reward_stall_w` | 0.5 |
| shaping | `reward_shape_w` | 0.3 |
| team | `reward_team_w` | 0.2 |

\(P_{\mathrm{coll}}\)、\(P_{\mathrm{stall}}\) 非负，总奖励中减去；\(P_{\mathrm{coll}}\) 经 `reward_collision_cap`（默认 2.0）截断。

阶段由到己方 slot 的距离 \(d\) 经 `hold_gate`（smoothstep）调制：Approach（远）主攻靠近与完整船软碰；Near/Hold 打开 hold，并在进槽走廊内软化船软碰。

## 2. 距离与保持

记 \(d_i\) 为到目标 slot 的平面距离，\(\Delta\psi_i\) 为航向误差，\(e_{v,i}\) 为相对大船速度误差。

**Hold gate**（`reward_hold_start_m=150`，`reward_hold_full_m=20`）：

- \(d \le 20\,\mathrm{m}\) → \(\mathrm{gate}=1\)
- \(d \ge 150\,\mathrm{m}\) → \(\mathrm{gate}=0\)
- 中间 smoothstep；`approach_gate = 1 - gate`

**距离项**（远距主导，含停滞缩放）：

\[
R_{\mathrm{dist}}
=
(1-\mathrm{gate})
\cdot
\bigl(\alpha\cdot\mathrm{progress} + (1-\alpha)\cdot\mathrm{dist\_bonus}\bigr)
\cdot
\mathrm{stall\_scale}
\]

- \(\alpha=\) `reward_dist_progress_frac`（默认 0.7）
- \(\mathrm{progress}\)：相对上一步距离缩短，clip 到 \(\pm\) `reward_dist_progress_clip_m`（默认 5 m）后归一化到 \([-1,1]\)
- \(\mathrm{dist\_bonus}=1-d/\texttt{reward\_dist\_scale\_m}\)（默认 scale 500 m），再 clip 到 \([-0.5,1]\)
- \(\mathrm{stall\_scale}\)：见 §4；Hold 区为 1

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

## 3. 碰撞风险与进槽走廊软化

\[
P_{\mathrm{coll}}
=
\min\bigl(P_{\mathrm{ship}} + P_{\mathrm{tug}},\; \texttt{reward\_collision\_cap}\bigr)
\]

对大船与每艘邻居拖轮：

- **近距势垒**：距离低于 safe 半径时线性升至 1（船 safe 默认 **80 m**，拖轮 120 m；硬碰撞阈值见 `ship_collision_dist_m` / `tug_collision_dist_m`）
- **CPA**：在 `reward_cpa_horizon_s`（60 s）内若接近，用未来最近点距离再加权势垒；系数 `reward_collision_cpa_w`（默认 2.0）

**走廊软化（仅船项）：** 轴为船心→己方 slot 的单位向量 \(\mathbf{e}\)；\(\mathbf{r}\) 为 slot→拖轮，\(a=\mathbf{r}\cdot\mathbf{e}\)，横向 \(\ell=\|\mathbf{r}-a\mathbf{e}\|\)。

- \(d < \) `reward_hold_start_m` 且 \(a \ge -\texttt{reward\_corridor\_axial\_slack\_m}\)（默认 30 m）时，用 \(\ell/\texttt{half\_width}\)（默认 half_width 40 m）的 smoothstep 得 `corridor_gate`
- \(\mathrm{ship\_soft\_scale}=1-(1-s_{\min})\cdot\mathrm{corridor\_gate}\)，\(s_{\min}=\) `reward_ship_soft_min_scale`（0.15）
- \(P_{\mathrm{ship}}\leftarrow P_{\mathrm{ship}}\cdot\mathrm{ship\_soft\_scale}\)（近距势垒 + CPA）
- **硬碰撞与终端惩罚不变**；**拖轮间软碰不软化**

## 4. 停滞惩罚

防外围刷弱距离分。维护 slot 距离环形历史（`MutableEpisodeState.dist_hist`；在 `step` 中于稠密奖励**之后**写入）。

- 窗长 `reward_stall_window_s`（默认 5 s），净接近 \(\Delta d_{\mathrm{net}}=d_{t-T}-d_t\)
- 若 `hold_gate < 0.99` 且 \(\Delta d_{\mathrm{net}} < \) `reward_stall_min_progress_m`（默认 2 m）：
  - \(P_{\mathrm{stall}}=\mathrm{clip}((d_{\mathrm{stall}}-\Delta d_{\mathrm{net}})/d_{\mathrm{stall}},0,1)\)
  - \(\mathrm{stall\_scale}=1-(1-\texttt{reward\_stall\_floor})\cdot P_{\mathrm{stall}}\)（floor 默认 0.2）
- Hold 区：\(P_{\mathrm{stall}}=0\)，`stall_scale=1`
- 历史不足一个窗：不触发

## 5. Shaping 与团队项

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

## 6. in-zone 计数

当且仅当

\[
d < \texttt{pos\_tol\_m}
\;\wedge\;
|\Delta\psi| < \texttt{heading\_tol\_rad}
\;\wedge\;
e_v < \texttt{speed\_tol\_ms}
\]

时 `in_zone_steps += 1`，否则 `max(0, steps-2)`。该计数驱动 Capture / Track（见 [task_spec.md](task_spec.md)）。

## 7. 终端奖罚

| 事件 | 奖励 |
|------|------|
| **刚 Capture**（`just_captured`） | 全体 `+reward_arrival_bonus`（默认 80），只发一次 |
| **Track success** | 无额外到位奖励 |
| 碰撞 | culprit `−reward_collision_pen_culprit`（80）；其余 bystander `−reward_collision_pen_bystander`（15） |

训练时：稠密奖励经 RunningMeanStd 归一化后写入 buffer；终端项在归一化后按原尺度叠加。

## 8. `reward_components` 诊断字段

| 键 | 含义 |
|----|------|
| `r_total` / `r_dist` / `r_hold` / `r_velocity` / `r_shape` / `r_team` | 各分量 |
| `p_collision` / `p_ship_collision` / `p_tug_collision` | 风险项（船项已乘 soft scale） |
| `p_stall` / `stall_scale` | 停滞项（乘权重前）与距离项缩放 |
| `corridor_gate` / `ship_soft_scale` | 走廊门控与船软碰缩放 |
| `dist_to_slot` / `heading_err_deg` / `speed_err` / `hull_dist` | 几何误差 |
| `in_zone` / `hold_gate` | 到位与 gate |

TensorBoard 默认记录 `r_dist`、`r_hold`、`p_collision`、`p_stall`（见 [tensorboard_metrics.md](tensorboard_metrics.md)）。

## Preset 消融

旧超参组合见 [reward_presets.md](reward_presets.md)（`--reward-preset`）。结构性重设计后，部分 preset 语义可能过时，需另开筛选协议。
