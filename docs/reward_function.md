# 拖轮编队奖励函数

> 实现：`env/reward.py` → `FormationRewardComputer.compute_rewards()`  
> 配置：`config.py` → `EnvConfig`  
> 终端奖罚：`env/formation_env.py` → `step()`（不参与稠密奖励归一化）  
> 结构设计：`docs/superpowers/specs/2026-08-07-reward-redesign-design.md`

## 1. 稠密奖励

\[
R_i = 3R_{\mathrm{progress},i}
-0.2C_{\mathrm{distance},i}
+2R_{\mathrm{hold},i}
-P_{\mathrm{collision},i}
+w_vR_{\mathrm{velocity},i}
+R_{\mathrm{team}}.
\]

## 2. 目标门控、接近与距离代价

\[
x_i=\operatorname{clip}(1-d_i/\texttt{pos_tol_m},0,1),
\qquad g_i=x_i^2(3-2x_i)
\]

\[
R_{\mathrm{progress},i}=(1-g_i)\operatorname{clip}(
(d_{i,t-1}-d_{i,t})/1\mathrm m,-1,1)
\]

\[
C_{\mathrm{distance},i}=(1-g_i)\operatorname{clip}(
d_i/200\mathrm m,0,1)
\]

## 3. 目标保持

\[
R_{\mathrm{hold},i}=g_i(0.5+0.25s_{\mathrm{heading},i}
+0.25s_{\mathrm{speed},i})
\]

## 4. 团队最落后代价

\[
R_{\mathrm{team}}=-w_t
\frac{\sum_i c_i e^{\beta c_i}}{\sum_i e^{\beta c_i}},
\qquad c_i=C_{\mathrm{distance},i}
\]

接近进度在 10 m 目标区外持续生效；静止或等半径绕圈时接近进度为零，但距离代价仍为非零。`reward_hold_start_m=150` 仅控制进槽走廊软化；总奖励不包含停滞窗口或势函数 shaping。

## 5. 碰撞风险与进槽走廊软化

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
| `r_total` / `r_dist` / `p_distance` / `r_hold` / `r_team` | 奖励及团队分量 |
| `p_collision` | 碰撞风险项（船项已乘 soft scale） |
| `corridor_gate` / `ship_soft_scale` | 走廊门控与船软碰缩放 |
| `dist_to_slot` / `heading_err_deg` / `speed_err` / `hull_dist` / `in_zone` | 几何字段 |

TensorBoard 默认记录 `r_dist`、`p_distance`、`r_hold`、`p_collision`（见 [tensorboard_metrics.md](tensorboard_metrics.md)）。

## Preset 接口

训练 CLI 保留 `--reward-preset`，映射表为 `config.REWARD_PRESETS`（当前为空骨架）。新一轮超参筛选在此注册后再写实验协议。
