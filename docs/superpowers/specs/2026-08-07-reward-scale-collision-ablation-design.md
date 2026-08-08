# 奖励尺度扫描：抑制「冲撞式靠近」设计

日期：2026-08-07  
状态：已批准  
范围：在固定训练栈上，用稀疏奖励预设扫描接近权重、碰撞稠密上限与走廊软化；1M 粗扫晋级后 5M 复验。不改硬碰撞阈值、终止逻辑、观测、网络或 PPO 超参。默认权重是否改写由 5M 过关结果决定，本设计不自动改默认。

## 1. 背景与失败模式

`reward_no_orbit_smoke`（5M 步，固定种子 CUDA）行为门禁失败：

| 指标 | 结果 |
|---|---:|
| 捕获率 | 0%（38/38 eval） |
| 最终末距 | 137.5 m（过 200 m 门槛） |
| 最终碰撞率 | 81.2% |
| `reward/r_hold` | 几乎始终 ≈0 |

诊断假设：单步接近奖励封顶约 +3（`reward_dist_w=3`），稠密碰撞风险封顶 -2（`reward_collision_cap=2`），进槽走廊再将船体软碰压到 `ship_soft_min_scale=0.15`，使「快速接近后碰撞」在早期具有竞争力。上一版设计曾约束「碰撞相关权重保持不变」；本轮明确用数据扫描接近侧与碰撞侧，再决定是否改默认。

## 2. 目标与非目标

### 2.1 目标

- 用可复现协议检验：减弱接近拉力、抬高碰撞稠密上限、减弱走廊软化，是否降低碰撞并保留靠近能力。
- 1M 粗筛淘汰无效点；最多 2 个幸存者做 5M 复验。
- 仅当 5M 达到捕获与碰撞门槛时，才提议改写 `EnvConfig` 默认。

### 2.2 非目标

- 不修改 `ship_collision_dist_m` / `tug_collision_dist_m` 硬阈值与硬碰撞终止。
- 不修改 culprit/bystander 终端惩罚数值（本轮扫描不包含它们）。
- 不修改 observation、actor、PPO、初始化或 Capture/Track 判定。
- 不在本设计执行阶段自动改仓库默认奖励权重。
- 不扫 `reward_hold_w`、`reward_collision_w`（本轮固定 `collision_w=1`，只动 `cap`）。

## 3. 决策摘要

| 决策 | 选择 |
|---|---|
| 扫描哲学 | 因子扫描（接近 + 碰撞两侧），默认改写看数据 |
| 预算 | 混合：全表 1M 粗扫 → ≤2 个 5M 复验 |
| 因子集 | 双轴最小集 + 走廊对照（6 点） |
| 训练栈 | 对齐 `reward_no_orbit_smoke` |
| 成功标准 | 分层：1M 筛碰撞/末距；5M 必须 capture > 0 |
| 默认改写 | 仅 5M 过关后人工确认另开任务 |

## 4. 固定协议

与 `reward_no_orbit_smoke` 对齐，保证可比：

| 项 | 值 |
|---|---|
| `--arch` | `transformer` |
| `--tf-size` | `S` |
| `--init-radius` | `120` |
| `--slot-assignment` | `minimax` |
| `--seed` | `42` |
| `--device` | `cuda` |
| `--env-backend` | `cuda` |
| `--eval-backend` | `cuda` |
| `--num-envs` | `256` |
| `--rollout-steps` | `64` |
| `--minibatch-size` | `8192` |
| `--eval-workers` | `32` |
| 粗扫 `--total-steps` | `1000000` |
| 复验 `--total-steps` | `5000000` |

未列出的奖励字段保持 `EnvConfig` 现行默认。每个 preset 独立 `--run-name`，不 resume。

## 5. 预设表

写入 `config.REWARD_PRESETS`（前缀 `rsc_` = reward scale collision）：

| preset id | `reward_dist_w` | `reward_collision_cap` | `reward_ship_soft_min_scale` | 意图 |
|---|---:|---:|---:|---|
| `rsc_baseline` | 3.0 | 2.0 | 0.15 | 与现行默认一致 |
| `rsc_dist_soft` | 1.5 | 2.0 | 0.15 | 减弱接近拉力 |
| `rsc_coll_mid` | 3.0 | 4.0 | 0.15 | 抬高碰撞稠密上限 |
| `rsc_coll_hi` | 3.0 | 6.0 | 0.15 | 更强碰撞抑制 |
| `rsc_balanced` | 1.5 | 4.0 | 0.15 | 接近×碰撞交互 |
| `rsc_corridor_hard` | 3.0 | 2.0 | 0.50 | 减弱走廊船体软化 |

`reward_collision_w` 固定为 1.0，避免与 `cap` 混淆。

**run 命名**（`<short>` = preset id 去掉 `rsc_` 前缀）

| preset id | 1M run-name | 5M run-name |
|---|---|---|
| `rsc_baseline` | `rsc_1m_baseline` | `rsc_5m_baseline` |
| `rsc_dist_soft` | `rsc_1m_dist_soft` | `rsc_5m_dist_soft` |
| `rsc_coll_mid` | `rsc_1m_coll_mid` | `rsc_5m_coll_mid` |
| `rsc_coll_hi` | `rsc_1m_coll_hi` | `rsc_5m_coll_hi` |
| `rsc_balanced` | `rsc_1m_balanced` | `rsc_5m_balanced` |
| `rsc_corridor_hard` | `rsc_1m_corridor_hard` | `rsc_5m_corridor_hard` |

## 6. 晋级与过关规则

### 6.1 1M 晋级（相对同协议 `rsc_1m_baseline` 最终 eval）

须同时满足：

1. **末距不崩**：`eval/final_dist_mean < 200`，或相对基线恶化不超过 +20 m。
2. **碰撞改善**：`eval/collision_rate` 较基线下降 ≥ 0.15（15 个百分点）。
3. **捕获加分**：若 `eval/capture_rate > 0`，自动晋级并优先排序。

排序键：碰撞降幅降序，其次末距升序。取最多 **2** 个进入 5M。若无人过关：停止，输出尺度敏感性结论，**不改默认**。

### 6.2 5M 过关（视为解决本次「冲撞式靠近」问题）

须同时满足（最终评估；捕获允许看 best 或 final 任一 > 0）：

1. `capture_rate > 0`
2. `final_dist_mean < 200`
3. `collision_rate ≤ 0.40`

未过关：只记线索，不改默认。多人过关：选最终碰撞更低者；并列看捕获率与末距。

## 7. 实现交付

| 交付物 | 职责 |
|---|---|
| `config.py` | 注册 §5 六个 `REWARD_PRESETS` |
| `scripts/run_reward_scale_ablation.py` | 串行 1M 全表；`--presets` / `--total-steps` / `--promote` / `--dry-run`；日志追加 |
| `scripts/summarize_reward_scale.py` | 读 `runs/rsc_*` TB，表格式输出 final/best 的 capture、dist、coll、return |
| `docs/reward_scale_ablation.md` | 命令、读表方式、与本 spec 链接 |
| 本文件 | 设计真源 |

实现阶段验证（非完整训练）：

- preset 应用后三字段覆盖正确；未知 id 报错。
- dry-run 命令行含正确 `--reward-preset`、`--run-name`、步数。
- 可选极短 smoke（数个 update）确认 TB 标量标签仍存在。

训练证据（实验阶段）：`training_logs/` / `outputs/logs/`、`runs/rsc_*`、`checkpoints/rsc_*`（保持 gitignore）。

壁钟粗估（RTX 3090 级、对齐 smoke ~48 min / 5M）：1M×6 ≈ 1 h；5M×≤2 ≈ 1.5–2 h。

## 8. 决策出口

| 结果 | 动作 |
|---|---|
| 某 preset 5M 过关 | 提议将该三字段写入 `EnvConfig` 默认，并更新 `docs/reward_function.md`（另任务） |
| 多人 5M 过关 | 选碰撞更低；并列看捕获与末距 |
| 无人 5M 过关 | 不改默认；建议下一轮（hold 轴或动 `collision_w`） |
| 1M 无人晋级 | 停止；报告哪条轴最敏感 |

## 9. 风险与注意事项

- 1M 信号比 5M 更噪：晋级阈值用相对基线，避免绝对碰撞阈值误杀。
- 程序 `best.pt` 在 capture=0 时可能不是「最近且少撞」的点；汇总脚本须同时报告 final 与观测最优距离/碰撞。
- 已有 `reward_no_orbit_smoke` 可作定性对照，但正式晋级基线以本协议重跑的 `rsc_1m_baseline` 为准（同 eval-workers / 同步数）。
- 走廊对照只改 `reward_ship_soft_min_scale`；不改变硬碰撞与拖轮间软碰不软化的规则。
