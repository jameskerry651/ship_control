# 拖轮编队奖励函数

> 实现：`env/reward.py` → `FormationRewardComputer.compute_rewards()`  
> 配置：`config.py` → `EnvConfig`  
> 终端奖罚：`env/formation_env.py` → `step()`（不参与稠密奖励归一化）  
> 结构设计：`docs/superpowers/specs/2026-08-08-safe-slot-approach-reward-design.md`

## 1. 稠密奖励

\[
R_i = 3\tilde R_{\mathrm{progress},i}
-0.2C_{\mathrm{distance},i}
+2R_{\mathrm{safe},i}
+3R_{\mathrm{hold},i}
-2P_{\mathrm{collision},i}
+w_vR_{\mathrm{velocity},i}
+R_{\mathrm{team}}.
\]

## 2. 目标门控、风险门控接近与距离代价

\[
x_i=\operatorname{clip}(1-d_i/\texttt{pos\_tol\_m},0,1),
\qquad g_i=x_i^2(3-2x_i)
\]

\[
R_{\mathrm{progress},i}=(1-g_i)\operatorname{clip}(
(d_{i,t-1}-d_{i,t})/1\mathrm m,-1,1)
\]

\[
\rho_i=\operatorname{clip}(P_{\mathrm{ship},i}^{\mathrm{raw}} / 0.5,\,0,1)
\]

\[
\tilde R_{\mathrm{progress},i}=
\begin{cases}
R_{\mathrm{progress},i}(1-\rho_i) & R_{\mathrm{progress},i}>0 \\
R_{\mathrm{progress},i} & \text{otherwise}
\end{cases}
\]

\(P_{\mathrm{ship}}^{\mathrm{raw}}\) 为走廊软化**之前**的船体近距+CPA 风险。

\[
C_{\mathrm{distance},i}=(1-g_i)\operatorname{clip}(
d_i/200\mathrm m,0,1)
\]

## 3. 走廊安全项 \(R_{\mathrm{safe}}\)

复用进槽 `corridor_gate` \(c_i\)：

\[
R_{\mathrm{safe},i}=c_i(1-g_i)\big(
0.5 s_{\mathrm{axial},i}+0.3 s_{\mathrm{lat},i}+0.2 s_{\mathrm{approach},i}
\big)
\]

- \(s_{\mathrm{axial}}=\operatorname{clip}((d_{t-1}-d)/1\mathrm m,0,1)\)（只奖靠近）
- \(s_{\mathrm{lat}}=1-\mathrm{smoothstep}(\ell/w_{\mathrm{half}})\)
- \(s_{\mathrm{approach}}\)：相对船心闭合速度越高越低（`reward_safe_closing_speed_mps=1`）

走廊外或目标区内为 0。

## 4. 目标保持

\[
R_{\mathrm{hold},i}=g_i(0.5+0.25s_{\mathrm{heading},i}
+0.25s_{\mathrm{speed},i})
\]

## 5. 团队最落后代价

\[
R_{\mathrm{team}}=-w_t
\frac{\sum_i c_i e^{\beta c_i}}{\sum_i e^{\beta c_i}},
\qquad c_i=C_{\mathrm{distance},i}
\]

## 6. 碰撞风险与进槽走廊软化

\[
P_{\mathrm{coll}}
=
\min\bigl(P_{\mathrm{ship}} + P_{\mathrm{tug}},\; 4\bigr)
\]

- 船 safe 默认 **80 m**，拖轮 120 m；硬碰撞阈值不变（船 6 m / 拖轮 20 m）
- CPA 系数默认 2.0，地平线 60 s
- 走廊内船体软碰：`ship_soft_min_scale=0.70`（不再压到 0.15）
- **拖轮间软碰不软化**；硬碰撞与终止几何不变

## 7. in-zone 计数

当且仅当 \(d < \texttt{pos\_tol\_m}\) 且航向/速度在容差内时累加，否则 `max(0, steps-2)`。驱动 Capture / Track（见 [task_spec.md](task_spec.md)）。

## 8. 终端奖罚

| 事件 | 奖励 |
|------|------|
| **刚 Capture** | 全体 `+120`（`reward_arrival_bonus`），只发一次 |
| **Track success** | 无额外到位奖励 |
| 碰撞 | culprit `−160`；bystander `−30` |

训练时：稠密奖励经 RunningMeanStd 归一化后写入 buffer；终端项在归一化后按原尺度叠加。

## 9. `reward_components` 诊断字段

| 键 | 含义 |
|----|------|
| `r_dist` | 门控后的 \(\tilde R_{\mathrm{progress}}\) |
| `r_safe` | 走廊安全项 |
| `r_hold` | 保持项 |
| `p_distance` | 距离代价 |
| `p_collision` | 封顶后总碰撞风险 |
| `progress_risk` | \(\rho_i\) |
| `corridor_gate` / `ship_soft_scale` | 走廊门控与软化系数 |
