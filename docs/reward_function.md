# 拖轮编队奖励函数

> 实现位置：`env/reward.py` 的 `FormationRewardComputer.compute_rewards()`  
> 配置位置：`config.py` 的 `EnvConfig` 奖励参数  
> 终端成功奖励和终端碰撞惩罚在 `env/formation_env.py` 的 `step()` 中单独处理。

## 1. 公式渲染约定

本文档中的公式使用 Markdown LaTeX 块：

```tex
$$
R_{\mathrm{total}} = w_1 R_{\mathrm{target}}
$$
```

为保证 GitHub、VS Code、Typora 等常见渲染器稳定显示：

- 下标使用 `\mathrm{...}`，例如 `R_{\mathrm{target}}`，避免写成 `R_{t a r g e t}`。
- 向量范数使用 `\lVert x \rVert_2`。
- 分段函数使用 `cases` 环境。
- 中文说明放在公式外，不放进数学模式。

## 2. 总奖励

每个拖轮独立计算 dense reward。第 `i` 艘拖轮在时刻 `t` 的奖励为：

$$
R_{i,t}
=
w_{\mathrm{target}} R_{\mathrm{target},i}
+
w_{\mathrm{velocity}} R_{\mathrm{velocity},i}
-
w_{\mathrm{collision}} P_{\mathrm{collision},i}.
$$

对应默认配置为：

| 符号 | 配置项 | 默认值 |
|---|---|---:|
| `w_target` | `reward_target_w` | 1.0 |
| `w_velocity` | `reward_velocity_w` | 0.25 |
| `w_collision` | `reward_collision_w` | 5.0 |

`R_target` 越大越好；`R_velocity` 是非正惩罚项；`P_collision` 是非负风险项，因此在总奖励中显式减去。

## 3. 符号定义

拖轮状态：

$$
\eta_i = (x_i, y_i, \psi_i),
\qquad
\nu_i = (u_i, v_i, r_i).
$$

SLOT 状态：

$$
s_i = (x_{\mathrm{slot},i}, y_{\mathrm{slot},i}, \psi_{\mathrm{slot},i}).
$$

距离和航向误差：

$$
d_i
=
\sqrt{
(x_i - x_{\mathrm{slot},i})^2
+
(y_i - y_{\mathrm{slot},i})^2
},
\qquad
\Delta \psi_i
=
\operatorname{wrap}_{[-\pi,\pi]}
(\psi_{\mathrm{slot},i} - \psi_i).
$$

自身体系速度转世界系：

$$
\begin{aligned}
v_{x,i}^{\mathrm{world}}
&=
\cos \psi_i \cdot u_i
-
\sin \psi_i \cdot v_i, \\
v_{y,i}^{\mathrm{world}}
&=
\sin \psi_i \cdot u_i
+
\cos \psi_i \cdot v_i.
\end{aligned}
$$

大船速度也按同样方式转到世界系，记为：

$$
\mathbf{v}_{\mathrm{ship}}^{\mathrm{world}}
=
\left[
v_{x,\mathrm{ship}}^{\mathrm{world}},
v_{y,\mathrm{ship}}^{\mathrm{world}}
\right].
$$

## 4. 航向与位置奖励：远场追赶 + 近场保持

设计目的：远离 SLOT 时优先追赶目标 SLOT，靠近 SLOT 后转为稳定 hold，让位置、航向和速度持续满足伴航条件。

距离进度分数：

$$
q_i
=
\operatorname{clip}
\left(
\frac{d_{i,t-1} - d_{i,t}}{D_{\mathrm{progress}}},
-1,
1
\right).
$$

远场追赶还使用相对 SLOT 的 closing speed。令从拖轮指向 SLOT 的单位向量为：

$$
\mathbf{e}_{\mathrm{slot},i}
=
\frac{
\left[
x_{\mathrm{slot},i} - x_i,\;
y_{\mathrm{slot},i} - y_i
\right]
}
{d_i}.
$$

closing speed 定义为：

$$
c_i
=
\left(
\mathbf{v}_i^{\mathrm{world}}
-
\mathbf{v}_{\mathrm{ship}}^{\mathrm{world}}
\right)
\cdot
\mathbf{e}_{\mathrm{slot},i}.
$$

closing speed 分数：

$$
k_i
=
\operatorname{clip}
\left(
\frac{c_i}{V_{\mathrm{chase}}},
-1,
1
\right).
$$

近场 hold gate 使用平滑插值。若 `D_hold_full <= d_i <= D_hold_start`：

$$
z_i
=
\frac{
D_{\mathrm{hold,start}} - d_i
}
{
D_{\mathrm{hold,start}} - D_{\mathrm{hold,full}}
},
\qquad
G_{\mathrm{hold},i}
=
z_i^2(3 - 2z_i).
$$

完整分段定义为：

$$
G_{\mathrm{hold},i}
=
\begin{cases}
1,
& d_i \le D_{\mathrm{hold,full}}, \\
z_i^2(3 - 2z_i),
& D_{\mathrm{hold,full}} < d_i < D_{\mathrm{hold,start}}, \\
0,
& d_i \ge D_{\mathrm{hold,start}}.
\end{cases}
$$

远场 gate：

$$
G_{\mathrm{far},i}
=
1 - G_{\mathrm{hold},i}.
$$

远场 chase reward：

$$
R_{\mathrm{chase},i}
=
G_{\mathrm{far},i}
\left(
0.6 q_i
+
0.4 k_i
\right),
$$

近场 hold 的位置、航向、速度分数：

$$
s_{p,i}
=
\max
\left(
0,
1 - \frac{d_i}{D_{\mathrm{pos,tol}}}
\right),
$$

$$
s_{\psi,i}
=
\max
\left(
0,
1 - \frac{|\Delta \psi_i|}{\psi_{\mathrm{tol}}}
\right),
$$

其中 `e_{v,i}` 是第 5 节定义的世界系速度误差。

$$
s_{v,i}
=
\max
\left(
0,
1 - \frac{e_{v,i}}{V_{\mathrm{tol}}}
\right).
$$

近场 hold reward：

$$
R_{\mathrm{hold},i}
=
G_{\mathrm{hold},i}
\,
s_{p,i}
\left(
0.5
+
0.25 s_{\psi,i}
+
0.25 s_{v,i}
\right)
.
$$

最终 target reward：

$$
R_{\mathrm{target},i}
=
R_{\mathrm{chase},i}
+
R_{\mathrm{hold},i}.
$$

严格位置课程还可额外打开两个默认关闭的近场项。`R_precision` 在靠近 slot 时提供连续精度梯度；`R_near_hold` 不会在 `d_i > pos_tol_m` 立刻归零，用于把确定性 mean policy 从 60 m 继续拉向 10 m：

$$
R_{\mathrm{near\_hold},i}
=
w_{\mathrm{near}}
\, 
\exp\left[-\left(\frac{d_i}{D_{\mathrm{near}}}\right)^2\right]
\, 
\left(0.4 + 0.3s_{\psi,i} + 0.3s_{v,i}\right).
$$

默认配置：

| 参数 | 配置项 | 默认值 |
|---|---|---:|
| `D_progress` | `reward_target_progress_clip_m` | 1.5 m |
| `V_chase` | `reward_chase_speed_target_ms` | 0.8 m/s |
| `D_hold_start` | `reward_hold_start_m` | 140.0 m |
| `D_hold_full` | `reward_hold_full_m` | 20.0 m |
| `D_pos_tol` | `pos_tol_m` | 140.0 m default; strict curriculum target 10.0 m |
| `w_precision` | `reward_precision_w` | 0.0 default; strict curriculum enables it |
| `D_precision` | `reward_precision_scale_m` | 40.0 m |
| `w_near` | `reward_near_hold_w` | 0.0 default; strict curriculum enables it |
| `D_near` | `reward_near_hold_scale_m` | 80.0 m |
| `psi_tol` | `heading_tol_rad` | 30 deg |
| `V_tol` | `speed_tol_ms` | 3.0 m/s |

## 5. 速度与姿态控制

设计目的：拖轮靠近 SLOT 时，逐步把速度和偏航角速度调整到与大船一致，降低惯性撞船风险。

速度误差：

$$
e_{v,i}
=
\left\lVert
\mathbf{v}_i^{\mathrm{world}}
-
\mathbf{v}_{\mathrm{ship}}^{\mathrm{world}}
\right\rVert_2.
$$

速度匹配门控在近场 hold 时为 1，远场只保留弱约束：

$$
G_{v,i}
=
G_{\mathrm{hold},i}
+
\left(
1 - G_{\mathrm{hold},i}
\right)
\exp
\left(
-
\frac{d_i}{D_{\mathrm{gate}}}
\right).
$$

速度惩罚和偏航角速度惩罚：

$$
b_{v,i}
=
1
-
\exp
\left(
-
\left(
\frac{e_{v,i}}{V_{\mathrm{scale}}}
\right)^2
\right),
$$

$$
b_{r,i}
=
1
-
\exp
\left(
-
\left(
\frac{|r_i - r_{\mathrm{ship}}|}{R_{\mathrm{scale}}}
\right)^2
\right).
$$

最终 velocity reward：

$$
R_{\mathrm{velocity},i}
=
-
G_{v,i}
\left(
0.8 b_{v,i}
+
0.2 b_{r,i}
\right).
$$

默认配置：

| 参数 | 配置项 | 默认值 |
|---|---|---:|
| `D_gate` | `reward_velocity_gate_m` | 120.0 m |
| `V_scale` | `reward_velocity_speed_scale_ms` | 3.0 m/s |
| `R_scale` | `reward_velocity_yaw_scale_rads` | 0.05 rad/s |

## 6. 碰撞惩罚

设计目的：避免拖轮之间碰撞，以及拖轮与大船船体碰撞。实际碰撞会触发 episode 终止，并额外施加终端惩罚。

连续 barrier 函数：

$$
B(d; d_{\mathrm{collision}}, d_{\mathrm{safe}})
=
\begin{cases}
0,
& d \ge d_{\mathrm{safe}}, \\
\operatorname{clip}
\left(
\frac{d_{\mathrm{safe}} - d}
{d_{\mathrm{safe}} - d_{\mathrm{collision}}},
0,
1
\right),
& d < d_{\mathrm{safe}}.
\end{cases}
$$

CPA 时间权重在短时会遇时更强，超过预判窗口则不计入 CPA 风险：

$$
T(t_{\mathrm{cpa}})
=
\operatorname{clip}
\left(
1 - \frac{t_{\mathrm{cpa}}}{T_{\mathrm{horizon}}},
0,
1
\right).
$$

只有当相对速度正在接近且 `TCPA` 位于预判窗口内时，CPA 项才生效。相对静止、正在远离或窗口外的会遇不产生 CPA 惩罚。

拖轮与大船船体风险：

$$
P_{\mathrm{ship},i}
=
B
\left(
d_{\mathrm{hull},i};
D_{\mathrm{ship,collision}},
D_{\mathrm{ship,safe}}
\right)
+
c_{\mathrm{cpa}}
B
\left(
D^{\mathrm{CPA}}_{\mathrm{hull},i};
D_{\mathrm{ship,collision}},
D_{\mathrm{ship,safe}}
\right)
T(t^{\mathrm{CPA}}_{\mathrm{ship},i}).
$$

第一项表示当前进入船体安全区后的持续距离惩罚；第二项表示预测会遇风险。`D^{CPA}_{hull}` 不是到大船中心的距离，而是在 CPA 时刻按未来大船位姿计算拖轮到矩形船体边界的距离。

拖轮之间风险：

$$
P_{\mathrm{tug},i}
=
\sum_{j \ne i}
\left[
B
\left(
d_{ij};
D_{\mathrm{tug,collision}},
D_{\mathrm{tug,safe}}
\right)
+
c_{\mathrm{cpa}}
B
\left(
D^{\mathrm{CPA}}_{ij};
D_{\mathrm{tug,collision}},
D_{\mathrm{tug,safe}}
\right)
T(t^{\mathrm{CPA}}_{ij})
\right].
$$

总碰撞风险：

$$
P_{\mathrm{collision},i}
=
P_{\mathrm{ship},i}
+
P_{\mathrm{tug},i}.
$$

默认配置：

| 参数 | 配置项 | 默认值 |
|---|---|---:|
| `D_ship_collision` | `ship_collision_dist_m` | 6.0 m |
| `D_ship_safe` | `reward_collision_ship_safe_m` | 60.0 m |
| `D_tug_collision` | `tug_collision_dist_m` | 20.0 m |
| `D_tug_safe` | `reward_collision_tug_safe_m` | 80.0 m |
| `T_horizon` | `reward_cpa_horizon_s` | 60.0 s |
| `c_cpa` | `reward_collision_cpa_w` | 2.0 |

## 7. 终端信号

Dense reward 不直接包含成功 bonus 和硬碰撞终端惩罚。`FormationEnv.step()` 在终止判定后额外处理：

$$
R_{\mathrm{final},i}
=
R_{i,t}
+
R_{\mathrm{terminal},i}.
$$

成功时：

$$
R_{\mathrm{terminal},i}
=
\mathrm{reward\_arrival\_bonus}.
$$

碰撞时：

$$
R_{\mathrm{terminal},i}
=
-
\mathrm{reward\_collision\_pen}.
$$

当前合作式环境中，发生碰撞时 episode 终止，所有拖轮都会收到同量级终端碰撞惩罚。

## 8. 诊断字段

`info["reward_components"]` 输出以下字段，用于训练日志和可视化：

| 字段 | 含义 |
|---|---|
| `r_total` | dense reward 总和，不含 terminal reward |
| `r_target` | 航向与位置奖励 |
| `r_chase` | 远场追赶 SLOT 奖励 |
| `r_hold` | 近场稳定保持奖励 |
| `r_velocity` | 速度与偏航角速度匹配奖励 |
| `p_collision` | 总连续碰撞风险 |
| `p_ship_collision` | 拖轮-大船连续碰撞风险 |
| `p_tug_collision` | 拖轮-拖轮连续碰撞风险 |
| `p_cpa_collision` | CPA 预判风险加权后的总贡献 |
| `p_ship_cpa` | 拖轮-大船 CPA 预判风险，未乘 CPA 权重 |
| `p_tug_cpa` | 拖轮-拖轮 CPA 预判风险，未乘 CPA 权重 |
| `min_tcpa` | 当前拖轮最小有效 TCPA，单位 s；无有效 CPA 风险时为预判窗口 |
| `min_dcpa` | 当前拖轮最小有效 DCPA，单位 m；无有效 CPA 风险时为安全距离上界 |
| `dist_to_slot` | 拖轮到目标 SLOT 的距离 |
| `heading_err_deg` | 航向误差，单位 degree |
| `speed_err` | 世界系速度误差，单位 m/s |
| `hull_dist` | 拖轮到大船船体的最短距离 |
| `chase_closing_speed` | 拖轮相对 SLOT 的 closing speed |
| `hold_gate` | 近场 hold gate |
| `far_gate` | 远场 chase gate |
| `in_zone` | 是否满足成功保持区间的单步条件 |
