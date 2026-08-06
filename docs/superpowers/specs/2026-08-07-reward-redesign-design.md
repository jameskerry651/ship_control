# 奖励函数结构性重设计

日期：2026-08-07  
状态：已批准，待实现  
范围：改 `env/reward.py` 公式结构 + `EnvConfig` 新字段；更新 `docs/reward_function.md`；补单测。不改任务成功判定与终端奖罚默认值。

## 1. 背景与目标

现策略易学会「不靠近大船、在外围刷稠密奖励」：

- 绝对距离分 `dist_bonus` 在中远距仍给正分；
- 船软障默认 `reward_collision_ship_safe_m=100`，而 slot 约在船体侧 ~25 m 外，进槽路径几乎全程在软罚区；
- `r_hold` / `capture_rate` 长期接近 0，`final_dist` 常停在 ~200 m。

上一轮权重/几何 preset 消融未能打破「躲远刷 return」；本轮做结构性重设计（该消融协议与 `rw_*` 表已移除）。

**成功标准（同时满足，相对「躲远刷分」基线，同 `init_radius=100` + transformer、可比步数）：**

1. `eval/final_dist_mean` 明显下降（进入 Near/Hold，而非长期 ~200 m）；
2. `eval/collision_rate` 不明显恶化（不以长期 ~90%+ 碰撞换靠近）；
3. `eval/capture_rate` 稳定 > 0，且 `reward/r_hold` 明显非零；
4. 人工否决：return 高但距离仍远，或 `p_stall` 长期打满却不靠近 → 假阳性。

## 2. 设计决策摘要

| 决策 | 选择 |
|------|------|
| 改造范围 | 结构性重设计（非纯调权） |
| 靠近 vs 避碰 | 阶段门控 + 进槽走廊软化船软碰 |
| 外围刷分 | 保留弱绝对距离分 + 停滞衰减/惩罚 |
| 走廊几何 | 径向椭球/胶囊（拖轮→己方 slot） |
| 终端奖罚 | 默认不变 |
| `R_vel` | 默认权重仍为 0，本轮不作为主杠杆 |

## 3. 总结构与阶段

每艘拖轮独立稠密奖励：

\[
R_i=
w_{\mathrm{dist}}R_{\mathrm{dist},i}
+
w_{\mathrm{hold}}R_{\mathrm{hold},i}
-
w_{\mathrm{coll}}P_{\mathrm{coll},i}
-
w_{\mathrm{stall}}P_{\mathrm{stall},i}
+
R_{\mathrm{shape},i}
+
R_{\mathrm{team}}.
\]

阶段由到己方 slot 的平面距离 \(d_i\) 平滑调制（不硬切公式）：

| 阶段 | 距离（默认） | 角色 |
|------|-------------|------|
| Approach | \(d \ge 150\,\mathrm{m}\) | 靠近为主；完整船软碰；弱绝对距离 + 进度；停滞生效 |
| Near | \(20\,\mathrm{m} < d < 150\,\mathrm{m}\) | 进度加强；hold 渐入；**走廊内**船软碰软化 |
| Hold | \(d \le 20\,\mathrm{m}\) | hold 主导；走廊内船软碰保留下限，防贴壳 |

门控：

- `hold_gate`：沿用现有 smoothstep（\(d\le20\to1\)，\(d\ge150\to0\)）；
- `approach_gate = 1 - hold_gate`；
- `corridor_gate ∈ [0,1]`：见 §4，用于船软碰软化。

## 4. 径向椭球走廊与船软碰软化

**走廊定义（己方 slot）：**

- 以当前位置到 slot 的连线为轴向；
- 横向半宽 `reward_corridor_half_width_m`（默认 40 m）；
- 轴向允许越过 slot 的余量 `reward_corridor_axial_slack_m`（默认 30 m），避免到位瞬间掉出走廊；
- 轴 \(\mathbf{e}\) 为**船心→己方 slot** 的固定单位向量（不可用瞬时 tug→slot，否则横向恒为 0）。
- \(\mathbf{r}\) 为 slot→拖轮；轴向 \(a=\mathbf{r}\cdot\mathbf{e}\)（slot 外侧为正）、横向 \(\ell=\|\mathbf{r}-a\mathbf{e}\|\)。
- 走廊内条件：\(a \ge -\texttt{axial\_slack}\) 且 \(\ell\) 经 half_width 归一化后的 smoothstep > 0，且 \(d < d_{\mathrm{near}}\)（默认 150 m，与 `reward_hold_start_m` 一致）。
- `corridor_gate`：满足上式时升高；否则为 0。

**软化：**

\[
\mathrm{ship\_soft\_scale}
=
1 - (1 - s_{\min})\cdot\mathrm{corridor\_gate},
\quad s_{\min}=\texttt{reward\_ship\_soft\_min\_scale}\ (0.15).
\]

- 仅缩放**船**的近距势垒与 CPA 贡献：\(P_{\mathrm{ship}} \leftarrow P_{\mathrm{ship}}\cdot\mathrm{ship\_soft\_scale}\)；
- **硬碰撞** `ship_collision_dist_m`、碰撞终止与终端惩罚不变；
- **拖轮间**软碰/CPA **不**软化；
- Hold 区内走廊内仍保留 \(s_{\min}\) 量级近船体势垒。

完整软障半径默认由 100 m 收到 `reward_collision_ship_safe_m=80`，走廊外仍挡横穿船体。

## 5. 距离、停滞与 Hold

**距离项：**

\[
R_{\mathrm{dist}}
=
\mathrm{approach\_gate}
\cdot
\bigl(
\alpha\cdot\mathrm{progress}
+
(1-\alpha)\cdot\mathrm{dist\_bonus}
\bigr)
\cdot
\mathrm{stall\_scale}
\]

- \(\alpha=\) `reward_dist_progress_frac`（默认 0.7）；
- `progress` / `dist_bonus` 计算方式与现实现一致（progress clip、`reward_dist_scale_m`），仅混合比与 `stall_scale` 变化。

**停滞（Approach/Near，Hold 关闭）：**

- 滑动窗 `reward_stall_window_s`（默认 5 s）内净接近 \(\Delta d_{\mathrm{net}}=d_{t-T}-d_t\)；
- 若 \(\Delta d_{\mathrm{net}} <\) `reward_stall_min_progress_m`（默认 2 m）：
  - `stall_scale` 从 1 降至 `reward_stall_floor`（默认 0.2）；
  - \(P_{\mathrm{stall}}=\mathrm{clip}((d_{\mathrm{stall}}-\Delta d_{\mathrm{net}})/d_{\mathrm{stall}},\,0,\,1)\)，权重 `reward_stall_w`（默认 0.5）；
- `hold_gate` 高时：`stall_scale=1`，\(P_{\mathrm{stall}}=0\)（到位保持不算刷分）；
- 明显后退仍由负 `progress` 体现，不另开项。

**Hold：** 现公式不变，仅由 `hold_gate` 打开。

**Shaping / team：** 保留默认权重；公式不变。

**终端：** `reward_arrival_bonus`、culprit/bystander 碰撞惩罚默认不变。

## 6. 配置默认

| 字段 | 默认 | 含义 |
|------|-----:|------|
| `reward_dist_progress_frac` | 0.7 | \(R_{\mathrm{dist}}\) 中 progress 占比 |
| `reward_stall_w` | 0.5 | 停滞惩罚权重 |
| `reward_stall_window_s` | 5.0 | 净接近窗口 |
| `reward_stall_min_progress_m` | 2.0 | 窗内最少净接近 |
| `reward_stall_floor` | 0.2 | `stall_scale` 下限 |
| `reward_corridor_half_width_m` | 40.0 | 椭球横向半宽 |
| `reward_corridor_axial_slack_m` | 30.0 | 轴向越过 slot 余量 |
| `reward_ship_soft_min_scale` | 0.15 | 走廊内船软碰下限 |
| `reward_collision_ship_safe_m` | 80.0 | 完整船软障（原 100） |

未列出的奖励字段保持现默认（`reward_dist_w=3`、`reward_hold_w=2`、`reward_collision_w=1`、`reward_shape_w=0.3`、`reward_team_w=0.2` 等）。

旧 `rw_*` preset 消融表已清空；保留 `REWARD_PRESETS` / `--reward-preset` 骨架，待新筛选协议注册。

## 7. 诊断字段

`reward_components` 保留现有键；新增：

| 键 | 含义 |
|----|------|
| `p_stall` | 停滞项（乘 `reward_stall_w` 之前，与 `p_collision` 相同风格） |
| `corridor_gate` | 走廊门控 |
| `ship_soft_scale` | 船软碰缩放 |
| `stall_scale` | 距离项停滞缩放 |

TensorBoard：在现有 `r_dist` / `r_hold` / `p_collision` 基础上增加 `p_stall`（若训练脚本白名单需同步）。

## 8. 实现落点与测试

| 落点 | 工作 |
|------|------|
| `env/reward.py` | 走廊、软化、停滞、\(R_{\mathrm{dist}}\) 混合比 |
| `env/state.py`（若需） | 停滞窗所需历史距离缓冲 |
| `config.py` | 新字段与 `ship_safe` 默认 |
| `scripts/train.py` | TB 白名单（如需要） |
| `docs/reward_function.md` | 与实现对齐 |
| `tests/` | 见下 |

**单测最少覆盖：**

1. 走廊内：`ship_soft_scale < 1`，走廊外：`= 1`（同 hull 距离）；
2. 停滞：窗内无净接近时 `p_stall>0` 且 `stall_scale` 下降；Hold 区停滞关闭；
3. 硬碰撞距离阈值与终端惩罚路径不受软化影响；
4. 拖轮间 `p_tug` 不随 `corridor_gate` 变化。

## 9. 非目标

- 不改 observation / actor 架构；
- 不改 Capture / Track 成功判定；
- 不在本轮重跑完整 preset 消融矩阵；
- 不引入离散里程碑奖励（曾评估为方案 3，未采用）。

## 10. 验收跑法（实现后）

建议短跑对照：新默认 vs 旧行为（可用 git 旧奖励或 feature flag），固定 `--arch transformer --init-radius 100 --seed 42`，步数至少 1M，按 §1 四条标准人工 + 指标复核。
