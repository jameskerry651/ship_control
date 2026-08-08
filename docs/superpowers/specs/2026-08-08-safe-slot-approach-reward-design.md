# 安全进槽奖励设计

日期：2026-08-08  
状态：已批准  
范围：在保留硬碰撞几何与 Capture/Track 判定的前提下，用风险门控接近、走廊安全正奖励、加重稠密/终端碰撞，消除「冲撞式靠近」。CPU/GPU 公式一致。5M 同栈验证过关前不宣称问题解决。

## 1. 背景与失败证据

| Run | 最终末距 | 最终碰撞 | 捕获 |
|---|---:|---:|---:|
| `reward_no_orbit_smoke` / `rsc_5m_baseline` | 137.5 m | 81.2% | 0% |
| `rsc_5m_balanced`（dist_w=1.5, cap=4） | 120.9 m | 93.8% | 0% |

1M 尺度扫描中，压低 `dist_w` 曾暂时降低碰撞但伴随躲远；5M 后该策略失效并更撞。`reward/r_hold` 在训练中几乎始终 ≈0。根因不是单一权重，而是：接近正信号在近场仍可与碰撞并存，且 10–80 m 缺少安全进槽正路径。

## 2. 目标与非目标

### 2.1 目标

- 消除「用碰撞换靠近」的局部最优。
- 在走廊/近场提供可学习的安全进槽稠密正信号。
- 5M 验证过关：`capture_rate > 0`，`final_dist_mean < 200`，`collision_rate ≤ 0.40`。
- CPU 与 GPU 奖励公式、分量、总奖励一致。

### 2.2 非目标

- 不修改 `ship_collision_dist_m=6` / `tug_collision_dist_m=20` 硬阈值与硬碰撞终止几何。
- 不修改 observation、actor、PPO、初始化策略。
- 不修改 Capture/Track 的 in-zone 几何与时间阈值（可增加 shaping / 终端奖励尺度）。
- 验证未过关前，不以「已解决」为由合并为唯一叙事默认；过关后再把新默认写入文档主路径。

## 3. 决策摘要

| 决策 | 选择 |
|---|---|
| 改动范围 | C：可改稠密项、走廊软化、终端碰撞/到达奖励 |
| 主方案 | 风险门控接近 + 新 `R_safe` + 碰撞加重 |
| 成功标准 | A：捕获>0 且末距<200 m 且碰撞≤40% |
| 验证 | 与 smoke 同栈 5M，`safe_slot_v1_r120` |

## 4. 总奖励

每艇稠密奖励：

\[
R_i=
w_p\,\tilde R_{\mathrm{progress},i}
-w_d\,C_{\mathrm{distance},i}
+w_s\,R_{\mathrm{safe},i}
+w_h\,R_{\mathrm{hold},i}
-w_c\,P_{\mathrm{collision},i}
+w_v\,R_{\mathrm{velocity},i}
+R_{\mathrm{team}}.
\]

`R_velocity` 默认权重仍为 0。团队项保持现有软最大距离代价。目标门控 \(g_i\) 仍由 `pos_tol_m=10` 的 smoothstep 定义。

## 5. 风险门控接近

现有：

\[
R_{\mathrm{progress},i}=(1-g_i)\,
\operatorname{clip}\!\left(\frac{d_{i,t-1}-d_{i,t}}{d_{\mathrm{clip}}},-1,1\right),
\quad d_{\mathrm{clip}}=1\,\mathrm m.
\]

新：

\[
\tilde R_{\mathrm{progress},i}
=R_{\mathrm{progress},i}\cdot(1-\rho_i),
\quad
\rho_i=\operatorname{clip}\!\left(
\frac{P_{\mathrm{ship},i}^{\mathrm{raw}}}{p_{\mathrm{gate}}},0,1\right).
\]

- \(P_{\mathrm{ship}}^{\mathrm{raw}}\)：对船体的近距势垒 + CPA 风险，在乘走廊 `ship_soft_scale` **之前**计算。
- \(p_{\mathrm{gate}}=\) `reward_progress_risk_gate`（默认 **0.5**）。
- 高船体风险时接近**正**进度被抑制；负进度（远离）仍可经 \(R_{\mathrm{progress}}<0\) 表达（门控只乘 \((1-\rho)\)，不翻转符号约定：实现时对正进度乘门控、负进度保持，避免「高风险远离也被关掉」。**规范：仅当 \(R_{\mathrm{progress}}>0\) 时乘 \((1-\rho)\)；\(R_{\mathrm{progress}}\le 0\) 不乘风险门控。**）

## 6. 走廊安全项 \(R_{\mathrm{safe}}\)

复用现有 `corridor_gate` \(c_i\in[0,1]\)（船心→slot 轴、半宽、轴向松弛、`reward_hold_start_m`）。

\[
R_{\mathrm{safe},i}
=c_i\,(1-g_i)\,
\big(
0.5\,s_{\mathrm{axial},i}
+0.3\,s_{\mathrm{lat},i}
+0.2\,s_{\mathrm{approach},i}
\big).
\]

定义：

- \(s_{\mathrm{axial},i}=\operatorname{clip}((d_{i,t-1}-d_{i,t})/d_{\mathrm{clip}},0,1)\)  
  （只奖励靠近，不奖励远离；与 progress 同 clip 尺度）
- 令 \(\ell_i\) 为相对 slot 的横向偏移（与走廊门控同一几何）。  
  \(s_{\mathrm{lat},i}=1-\operatorname{smoothstep}(\operatorname{clip}(\ell_i/w_{\mathrm{half}},0,1))\)
- 令 \(\mathbf{u}\) 为拖轮位置指向最近船体点（或船心，若实现更简）的单位向量，\(\mathbf{v}_{\mathrm{rel}}\) 为拖轮相对大船的世界速度。  
  \(s_{\mathrm{approach},i}=\operatorname{clip}\big(1-\max(0,-\mathbf{v}_{\mathrm{rel}}\cdot\mathbf{u})/v_{\mathrm{ref}},0,1\big)\)，  
  \(v_{\mathrm{ref}}=\) `reward_safe_closing_speed_mps`（默认 **1.0** m/s）。  
  即：朝船体快速闭合时该项下降。

权重：`reward_safe_w` 默认 **2.0**。走廊外或已在目标区内（\(g_i\to 1\)）该项为 0，避免外围刷分与 hold 双计。

## 7. 碰撞、走廊软化与终端

稠密碰撞结构保持 \(P=\min(P_{\mathrm{ship}}+P_{\mathrm{tug}},\mathrm{cap})\)，但默认尺度变更：

| 配置项 | 旧默认 | 新默认 |
|---|---:|---:|
| `reward_collision_w` | 1.0 | **2.0** |
| `reward_collision_cap` | 2.0 | **4.0** |
| `reward_ship_soft_min_scale` | 0.15 | **0.70** |
| `reward_collision_pen_culprit` | 80 | **160** |
| `reward_collision_pen_bystander` | 15 | **30** |
| `reward_dist_w` | 3.0 | 3.0（不变；靠风险门控） |
| `reward_hold_w` | 2.0 | **3.0** |
| `reward_arrival_bonus` | 80 | **120** |
| `reward_safe_w` | — | **2.0** |
| `reward_progress_risk_gate` | — | **0.5** |
| `reward_safe_closing_speed_mps` | — | **1.0** |

硬碰撞阈值与终止逻辑不变。拖轮间软碰仍不随走廊软化。

## 8. 诊断字段

在 `reward_components` 中：

| 键 | 含义 |
|---|---|
| `r_dist` | 门控后的 \(\tilde R_{\mathrm{progress}}\)（写入与 TB 一致） |
| `r_safe` | \(R_{\mathrm{safe}}\) |
| `r_hold` | 保持项 |
| `p_collision` | 封顶后总碰撞风险 |
| `p_ship_collision` | 软化后船体项 |
| `progress_risk` | \(\rho_i\) |
| `corridor_gate` | 现有 \(c_i\) |

训练日志启动摘要须打印有效 `reward_safe_w`、`reward_progress_risk_gate`、`reward_collision_w/cap`、`reward_ship_soft_min_scale`。

## 9. 验证协议

| 项 | 值 |
|---|---|
| 栈 | transformer S，r120，minimax，cuda，N=256，roll=64，mb=8192，eval-workers=32，seed=42 |
| 步数 | 5_000_000 |
| run-name | `safe_slot_v1_r120` |
| 主对照 | `rsc_5m_baseline`（已有产物） |

过关（须同时满足）：

1. `capture_rate > 0`（最终或程序 best 评估任一）
2. 最终 `final_dist_mean < 200`
3. 最终 `collision_rate ≤ 0.40`

未过关：输出分量诊断与下一刀建议；不声称奖励问题已解决。

## 10. 实现顺序

1. EnvConfig 新字段与默认值；夹紧/文档。
2. CPU：风险门控 progress + 碰撞/走廊/终端默认；单测「高 \(P_{\mathrm{ship}}^{\mathrm{raw}}\) 时正进度被抑制」。
3. CPU：`R_safe` + 诊断；单测「走廊外≈0；走廊内居中靠近 > 横向偏离；快速闭合船体时 approach 分更低」。
4. GPU 对齐 + parity。
5. `docs/reward_function.md` 更新；跑 5M 验证并写报告。

## 11. 风险

- 惩罚过重导致躲远：监控末距是否长期 >250 m 且 `r_safe`≈0。
- `R_safe` 刷分：已要求走廊门控 + 仅奖励轴向靠近 + 目标区外。
- 终端惩罚加倍在 reward 归一化之后叠加，相对稠密更大——有意提高碰撞终局代价。
- \(P_{\mathrm{ship}}^{\mathrm{raw}}\) 与软化后 `p_ship` 必须同时可测，避免门控用错量。

## 12. 决策出口

| 结果 | 动作 |
|---|---|
| 5M 过关 | 将本节默认视为新 EnvConfig 默认，更新主文档 |
| 末距好但仍高碰撞 | 再抬 `collision_w` / 降走廊软化，或收紧 `s_approach` |
| 低碰撞但末距远 | 提高 `reward_safe_w` 或放宽 progress 门控 |
| 捕获仍 0 但碰撞/距离过关 | 检查 hold/arrival 与 in-zone 时间，另开小任务 |
