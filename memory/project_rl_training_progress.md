---
name: rl-training-progress
description: Multi-agent tugboat formation RL training progress — best checkpoints, reward design lessons, and what to try next
metadata:
  type: project
---

## 训练进展总结（截至 v32 CPA 观测训练）

**最佳检查点（统一口径：16-ep × 3 seed 平均，2026/05/15 重测）：**

| 版本 | path | succ% | coll% | len | 备注 |
|---|---|---|---|---|---|
| v19 | `checkpoints/v19_serial_init_straight/best.pt` | 0.0% | 43.8% | 964 | 串行初始化 + 大船直行 |
| v20 | `checkpoints/v20_loose_tol/best.pt` | 12.5% | 18.8% | 996 | pos_tol/speed_tol 放宽 |
| v21 | `checkpoints/v21_close_far_speedm/best.pt` | 27.1% | 12.5% | 931 | 距离档位 + speed zone /40→/80 |
| v22 | `checkpoints/v22_eval16_lr02_earlystop/best.pt` | 6.2% | 27.1% | 979 | **退步**（lr 下限 0.2 打散末段精细化）|
| v23 | `checkpoints/v23_resume_v21_lr05_succbest/best.pt` | 25.0% | 10.4% | 901 | 续训流程修正，无奖励变化（同档）|
| v24 | `checkpoints/v24_heading_gate_arrival80/best.pt` | 10.4% | 10.4% | 1052 | **退步**（heading 门控+arrival 80 误判）|
| **v25** | `checkpoints/v25_softzone_speedtanh/best.pt` | **43.8%** | **10.4%** | 779 | **大突破**（speed_err tanh + soft zone）|
| v26 | `checkpoints/v26_resume_v25/best.pt` | 39.6% | 12.5% | 769 | v25 续训，同档（无实质改进）|
| v27 | `checkpoints/v27_per_tug_collpen/best.pt` | 25.0% | 10.4% | 985 | **退步**（碰撞惩罚只给惹事 tug，共享策略下反而削弱避撞）|

**v25 是项目首次成功率突破：27.1% → 43.8%（相对 +62%）**，三个 seed 全部 ≥ 31%（最高 56.2%），coll 不变，len 从 931 减到 779（更快到位）。训练时单 seed 16-ep 出现 succ=68.8% 的 best，证明策略已学会"激进+成功"模式。

**v28 MAPPO 首轮训练（尚未纳入上表统一口径）：**
- 运行：`v28_mappo_slotonehot_v25init`
- checkpoint：`checkpoints/v28_mappo_slotonehot_v25init/best.pt`
- 改动：PPO actor 迁移到 MAPPO actor，新增 centralized critic；当前观测为 35D（31D 旧观测 + 4D slot one-hot），旧 v25 actor 第一层权重迁移时保留前 31D，新 4D 输入权重置零。
- 训练窗口 best：update=20, global_step=163840, succ=56.2%, coll=6.2%, len=875, return=159.63。
- 训练后期漂移：last.pt 到 global_step=1802240 时 eval succ=0%, coll=6.2%, len=1134, return=201.75，说明 return 仍会偏向长时间陪走，best 必须按 success 保存并重测。
- 结论：MAPPO 代码链路已跑通，且 warm-start 早期能达到比 v25 统一重测更高的训练窗口 success；但 v25 曾有训练窗口 68.8% 而统一重测只有 43.8%，所以 v28 必须做同 seed 多轮评估后才能宣称突破。

**v29 场景重构（2026/05/15，尚未正式训练）：**
- 目标：把任务从"slot 附近静止起步"改为更真实的"拖轮从大船船尾后方带初速追赶，沿左右舷绕行，最后进入 4 个 slot"。
- 代码改动：
  - `EnvConfig.tug_init_mode="astern_approach"` 作为默认主场景；保留 `near_slot` 和 `ring`。
  - 拖轮初始位置在大船船体系船尾后方，左/右 slot 固定走对应左/右舷航道。
  - 拖轮 reset 初始速度为 `ship.u + 0.2~0.8m/s`，航向接近大船航向，并同步初始前进推进器动作。
  - 观察从 35D 扩为 43D：新增 next waypoint 相对位置/极坐标、route stage、side sign、route remaining。
  - 奖励新增 route progress、lane penalty、chase speed；船尾追赶阶段不做速度匹配，到 final stage 后切回 speed match。
  - `train.py` MAPPO checkpoint 迁移支持 35D→43D：actor 第一层复制旧列，critic 第一层按 agent observation block 复制旧列，新 route 列置零。
- 已验证：
  - `python3 -m py_compile config.py formation_env.py train.py visualize.py`
  - 默认环境 reset/step smoke：obs `(4,43)`，slot 固定 `[0,1,2,3]`，初始 tug 速度高于大船。
  - `near_slot`/`ring` 模式仍能 reset。
  - v28 MAPPO 和 v25 PPO checkpoint 均可 warm start 到 43D。
  - 极小训练回路 `smoke_v29_astern` 可完成 rollout/update/save。
- 下一步正式训练建议：
  - `python3 train.py --resume checkpoints/v28_mappo_slotonehot_v25init/best.pt --total-steps 5000000 --run-name v29_mappo_astern_v28init --device cpu`
  - v29 是新任务分布，不能直接和 v25/v28 near-slot 成绩横比；需要单独建立 astern 场景评估口径。

**v29 首轮正式训练（2026/05/15）：**
- 运行：`v29_mappo_astern_v28init`
- 命令：`python3 train.py --resume checkpoints/v28_mappo_slotonehot_v25init/best.pt --total-steps 5000000 --run-name v29_mappo_astern_v28init --device cpu`
- warm start：v28 MAPPO 35D -> v29 43D，迁移 13 个 tensor，新 route 输入列置零；optimizer 重新初始化。
- early stop：update 210 / global_step 1,720,320 停止，原因是 200 个 update 没有超过早期 best。
- best：`checkpoints/v29_mappo_astern_v28init/best.pt`，update=10, global_step=81,920, eval succ=43.8%, coll=50.0%, len=307.2, return=1740.03。
- last：`checkpoints/v29_mappo_astern_v28init/last.pt`，update=210, global_step=1,720,320，最后 eval succ=25.0%, coll=75.0%, len=419.9, return=2181.24。
- 曲线判断：return 后期升高但 success 没跟上，仍有"长时间陪走/绕行但碰撞或不到位"倾向；best 仍应按 success 使用，不要按 return 选 last。
- 训练脚本修复：early-stop 后 final save 原本会把 `last.pt.update` 覆盖为计划总 update=610；已改为保存实际 `last_completed_update`，并把本次 `last.pt` 元数据修正为 update=210。

**v30 几何修正：同侧拖轮间距（2026/05/15，尚未正式训练）：**
- 问题：v29 虽然从船尾后方出发，但同侧的船首/船尾拖轮共用 `route_lane_lat_m=70m` 和早期 stern gate，且初始纵向距离都从同一范围采样，容易在入口/航道内追尾或并线互撞。
- 改动：
  - 船首 slot 走外侧 lane：`route_bow_lane_lat_m=90m`，初始距船尾 `150~230m`。
  - 船尾 slot 走内侧 lane：`route_stern_lane_lat_m=55m`，初始距船尾 `60~100m`。
  - 初始横向位置围绕对应 lane 加 `±8m` jitter，不再从统一 lateral range 采样。
  - 初始 pair 最小距离提高为 `60m`。
  - 新增同侧 spacing reward：`route_tug_spacing_dist_m=60m`、`reward_spacing_w=0.25`，比硬碰撞阈值更早惩罚同侧拖轮过近。
- 验证：
  - 200 seed reset：最小初始 pair distance `63.2m`，开局碰撞数 0；同侧平均最小间距约 `100.2m`。
  - 100 seed 使用初始前进动作连续跑 80 步：碰撞数 0，最小 pair distance `58.3m`。
  - `python3 -m py_compile config.py formation_env.py train.py visualize.py`
  - `python3 train.py --total-steps 64 --num-envs 2 --rollout-steps 4 --minibatch-size 8 --update-epochs 1 --run-name smoke_v30_spacing --device cpu`
- 下一步建议：从 `checkpoints/v29_mappo_astern_v28init/best.pt` 低学习率 warm start 训练，例如 `--learning-rate 1e-4 --run-name v30_astern_spacing_lr1e4`。

**v30 正式训练（2026/05/15）：**
- 运行：`v30_astern_spacing_lr1e4`
- 命令：`python3 -u train.py --resume checkpoints/v29_mappo_astern_v28init/best.pt --total-steps 3000000 --learning-rate 1e-4 --run-name v30_astern_spacing_lr1e4 --device cpu`
- 运行时长：约 9.1 分钟，完成计划训练，没有 early-stop。
- best：`checkpoints/v30_astern_spacing_lr1e4/best.pt`
  - update=310, global_step=2,539,520
  - eval succ=43.8%, coll=56.2%, len=560.2, return=4115.25
  - env 几何：bow lane=90m，stern lane=55m
- last：`checkpoints/v30_astern_spacing_lr1e4/last.pt`
  - update=366, global_step=2,998,272
  - metric/rollout return=2333.89
- 训练曲线：
  - 早期新几何下旧 v29 policy 不适配，eval 多次 succ=0%, coll=100%。
  - update 149 首次明显改善：succ=25.0%, coll=56.2%。
  - update 199: succ=31.2%, coll=68.8%。
  - update 309: succ=43.8%, coll=56.2%，为本轮 best。
  - 后期 return 多次升高但 success 波动/回落，仍应使用 best 而不是 last。
- 结论：内外双航道和初始间距修正有效，能在新 astern 场景恢复到 v29 best 级别 success，并将碰撞从早期 100% 压到 best 窗口 56.2%；但碰撞仍是主要瓶颈，下一步应做碰撞类型/stage 诊断和更强 hull/tug clearance 门控。

**v31 初始距离放大（2026/05/15，尚未正式训练）：**
- 动机：用户观察到拖轮之间初始化仍偏近，且希望整体离大船更远。
- 配置改动：
  - 船尾 slot 初始距船尾：`60~100m` -> `120~180m`
  - 船首 slot 初始距船尾：`150~230m` -> `260~380m`
  - 船首 lane：`90m` -> `100m`
  - 船尾 lane：`55m` -> `60m`
  - 初始 pair 最小距离：`60m` -> `110m`
  - 同侧 spacing 惩罚阈值：`60m` -> `90m`
  - 初始 lateral jitter：`8m` -> `6m`
  - astern 初始化 fallback 改为使用各 slot 距离区间中点，避免 fallback 破坏最小间距。
- 验证：
  - 500 seed reset：最小初始 pair distance `110.1m`，低于新阈值次数 0；同侧平均最小间距 `153.7m`；到大船 hull 最小距离 `126.4m`。
  - 100 seed 用初始前进动作连续 120 步：碰撞 0；过程最小 pair distance `107.4m`，最小 hull 距离 `96.5m`。
  - `python3 -m py_compile config.py formation_env.py train.py visualize.py`
  - smoke：`python3 train.py --total-steps 64 --num-envs 2 --rollout-steps 4 --minibatch-size 8 --update-epochs 1 --run-name smoke_v31_farinit --device cpu`
- 下一步训练建议：由于任务分布明显变远，优先从 v30 best 低学习率 warm start：`python3 -u train.py --resume checkpoints/v30_astern_spacing_lr1e4/best.pt --total-steps 3000000 --learning-rate 1e-4 --run-name v31_farinit_lr1e4 --device cpu`。

**v31 正式训练（2026/05/15）：**
- 运行：`v31_farinit_reset_lr1e4`
- 命令：`python3 -u train.py --resume checkpoints/v30_astern_spacing_lr1e4/best.pt --reset-progress --total-steps 3000000 --learning-rate 1e-4 --run-name v31_farinit_reset_lr1e4 --device cpu`
- 关键流程修正：新增并使用 `--reset-progress`，只加载模型权重，重置 optimizer/update/global_step。若直接从 v30 `best.pt` 普通 resume，会继承 update=310 和接近末段的低学习率，导致新远距离分布适应很差。
- 运行时长：约 10.2 分钟，完成计划训练，没有 early-stop。
- best：`checkpoints/v31_farinit_reset_lr1e4/best.pt`
  - update=260, global_step=2,129,920
  - eval succ=43.8%, coll=56.2%, len=564.4, return=4963.09
  - env 几何：pair min=110m，stern init=120~180m，bow init=260~380m
- last：`checkpoints/v31_farinit_reset_lr1e4/last.pt`
  - update=366, global_step=2,998,272
  - rollout return=3145.50，后期评估 succ 仅 6.2%、coll 93.8%，不建议使用。
- 训练曲线：
  - 早期远距离分布适应困难，多次 eval succ=0%，coll=100%。
  - 约 573k 步首次恢复到 succ=6.2%。
  - 约 1.15M 步 eval coll 一度降到 43.8%，但 succ=0%，说明策略先学会延迟/避免碰撞，尚未完成作业。
  - 约 1.80M 步 succ=25.0%，1.97M 步 succ=31.2%。
  - 约 2.13M 步达到本轮 best：succ=43.8%，coll=56.2%。
  - 后期学习率降低后 success 波动回落，`best.pt` 明显优于 `last.pt`。
- 结论：把初始化进一步拉远并把拖轮最小初始间距提高到 110m 是可训练的，能恢复到 v30 同级别成功率；当前瓶颈已从“初始化过近”转为“进入作业区后的碰撞控制”。下一步不建议继续盲目拉远初始化，应做碰撞类型/stage 诊断，并强化 final approach 的 hull/tug clearance 或用课程学习逐步收紧靠泊阶段。

**v32 CPA 观测训练（2026/05/15）：**
- 用户改动：观测空间新增 CPA 特征，`obs_include_cpa=True`，每艘拖轮对其他 3 艘拖轮增加 `dcpa/tcpa/bearing sin/bearing cos` 共 12 维；局部 obs 从 43D -> 55D，central critic 输入从 172D -> 220D。
- 验证：
  - `python3 -m py_compile config.py formation_env.py train.py visualize.py ppo.py`
  - 环境 reset：`obs_shape=(4,55)`。
  - smoke：`python3 train.py --resume checkpoints/v31_farinit_reset_lr1e4/best.pt --reset-progress --total-steps 64 --num-envs 2 --rollout-steps 4 --minibatch-size 8 --update-epochs 1 --run-name smoke_v32_cpa_obs --device cpu`
  - v31 43D MAPPO checkpoint 成功迁移到 55D，迁移 13 个 tensor；actor 新 CPA 输入列置零，critic 每个 agent block 的新 CPA 输入列置零。
- 正式训练：
  - 运行：`v32_cpa_obs_v31init_lr1e4`
  - 命令：`python3 -u train.py --resume checkpoints/v31_farinit_reset_lr1e4/best.pt --reset-progress --total-steps 3000000 --learning-rate 1e-4 --run-name v32_cpa_obs_v31init_lr1e4 --device cpu`
  - 运行时长：约 11.8 分钟，early-stop 于 update=360 / global_step=2,949,120。
- best：`checkpoints/v32_cpa_obs_v31init_lr1e4/best.pt`
  - update=160, global_step=1,310,720
  - eval succ=43.8%, coll=56.2%, len=472.2, return=5776.14
  - 模型输入：actor 55D，critic 220D
- last：`checkpoints/v32_cpa_obs_v31init_lr1e4/last.pt`
  - update=360, global_step=2,949,120
  - 后期评估 succ=0%, coll=100%，不建议使用。
- 训练曲线：
  - 初始 eval：81,920 步 succ=37.5%, coll=50.0%，说明 CPA 观测没有破坏 warm start。
  - 655k 步：succ=25.0%, coll=25.0%。
  - 819k 与 901k 步：succ=25.0%, coll=0.0%，说明 CPA 特征确实能帮助避碰。
  - 983k~1.06M 步：coll 低但 success 降到 0~6.2%，episode 长到 1460~1490 步，出现“安全陪走”倾向。
  - 1.31M 步达到本轮 best：succ=43.8%, coll=56.2%，与 v31 best 持平但没有降低 best 窗口碰撞率。
  - 后半程漂移到高碰撞低成功，early-stop 保存中段 best。
- 结论：CPA 观测本身有效，能产生 0% collision 的评估窗口；但当前 reward/best 选择仍把“到位成功”和“低碰撞”分离成两个局部最优。下一步不应只继续加观测，应该把 CPA 风险接入奖励或 curriculum：例如 final approach 阶段保留到位激励，同时对小 DCPA/短 TCPA 给连续惩罚，并用 success 优先、collision 作为 tie-breaker 或复合 score 保存 best。

**v33 CPA reward 接入（2026/05/15，已实现，尚未正式训练）：**
- 目标：把 v32 的 CPA 观测从“只给网络看”改为“直接约束训练目标”，解决低碰撞策略和到位策略分裂的问题。
- 配置新增：
  - `reward_cpa_w=0.18`
  - `reward_cpa_final_multiplier=2.0`
  - `reward_cpa_max_penalty=0.75`
  - `cpa_alert_dist_m=70.0`
  - `cpa_time_horizon_s=45.0`
- 奖励新增 `r_cpa`：
  - 对每艘拖轮和其他 3 艘拖轮计算 DCPA/TCPA。
  - 只惩罚 `dcpa < cpa_alert_dist_m` 且 `tcpa <= cpa_time_horizon_s` 的未来会遇风险。
  - 风险项为 `dcpa_score^2 * tcpa_score`，用平方降低远距离轻微风险的影响。
  - final route stage 使用 `reward_cpa_final_multiplier` 加强惩罚。
  - 单 agent 单步 CPA 惩罚用 `reward_cpa_max_penalty` 截断，避免重新学成“安全陪走”。
- reward_components 新增：
  - `r_cpa`
  - `cpa_risk`
  - `cpa_min_dcpa`
  - `cpa_min_tcpa`
- 验证：
  - `python3 -m py_compile config.py formation_env.py train.py visualize.py ppo.py`
  - 正常 reset/step 下 reward_components 包含 CPA 分量，且不对非风险状态无谓惩罚。
  - 手工构造两艘拖轮迎面会遇时 `r_cpa` 触发。
  - smoke：`python3 train.py --resume checkpoints/v32_cpa_obs_v31init_lr1e4/best.pt --reset-progress --total-steps 64 --num-envs 2 --rollout-steps 4 --minibatch-size 8 --update-epochs 1 --run-name smoke_v33_cpa_reward --device cpu`
- 下一步正式训练建议：
  - `python3 -u train.py --resume checkpoints/v32_cpa_obs_v31init_lr1e4/best.pt --reset-progress --total-steps 3000000 --learning-rate 1e-4 --run-name v33_cpa_reward_v32init_lr1e4 --device cpu`

**v25 的关键改动（"评估奖励机制是否可以再优化"分析后）：**
- **P3：speed_err 用 `speed_err * tanh(speed_err/3)` 替代 speed_err_sq**，权重 0.2→0.15
  - **Why：** 原 speed_err_sq 在 d→0 时单步惩罚可达 -0.4~-0.8，**比 r_zone=+1 还接近**，导致策略到 zone 边缘时净奖励为负，反而退回。tanh 压缩封顶在 ~3。
- **P4：r_zone 从二值 0/+1 改为 soft 评分** `pos_score × hdg_score × spd_score`（满分仍 1.0）
  - **Why：** 原硬阈值让 30~60m 区间梯度信号几乎为零，soft 评分提供从远到近的连续梯度。终止判定（hold_time 累计）继续用硬阈值不变。
- **How to apply:** 这俩改动是迄今最显著的奖励层突破，下次类似"靠近 zone 净奖励反而是负"的问题应该首先想到 tanh 压缩 + soft 评分。

**v24 失败教训（先做错的奖励改动）：**
- 试过 P1（heading 加 exp(-d/30) 距离门控）+ P2（arrival_bonus 20→80），结果 succ 27.1% → 10.4% 退步。
- **Why：** P1 错诊——陪走解的根因不是"远处朝向对齐拿正奖励"，而是 `cos(dpsi)` 是 ±0.1 对称项，门控削掉了远处的姿态调整奖励反而让靠近前的转向缺信号；P2 单独提高 arrival_bonus 在没解决"zone 边缘净奖励为负"的情况下只是放大了"激进失败"的相对成本，策略反而更保守。
- **How to apply:** 改奖励前先做"靠近 zone 边缘单步净奖励"的量级估算，不要凭直觉门控。

**重大教训：4-ep eval 不可信。**
v21 best 在它自己的 4-ep 评估窗口（seed=10680）下：return=565、**succ=0%**、coll=0%、len=1200——全部"陪走满 1200 步"；4-ep return std=±200，比信号大。把同一个 best 改用 16-ep 多 seed 评估：return 在 296~480 间波动、succ ≈ 18~31%。结论：
- 之前各代 best 之间用 4-ep return 横向比是被噪声驱动的
- best 判定口径必须统一，至少 eval_episodes=16，best 判定改用 succ 而不是 return（v23 起已实施）

**v22 退步根因：**
lr_anneal 下限 0.05→0.2 让续训起步 lr=1.64e-4（v21 末段是 1.5e-5），策略被打散重练。**lr 下限 0.2 是有害的**——v23 起改回 0.05。


## 训练曲线特征（v19/v20 共同观察）

- 早期 (~30 update)：coll 攀升到 60-67%，ret 大负，策略学避碰
- 中期：ret 稳步上升到 50-200，coll 降到 25%，episode 长度涨到接近 1200
- 中后期：ret 第一次冲到峰值（v19=274, v20=451），出现 coll=0% / len=1200 的"陪走"轨迹
- 收尾：lr 退到 1.5e-5，ret 在大区间内剧烈波动（v19: -46~+160; v20: 30~400），峰值难复现

**Why:** 收尾大幅波动是 lr 太低 + 稀有高奖励轨迹推动 value 偏移导致策略漂移。
**How to apply:** 下次训练 lr_anneal 下限从 0.05 提到 0.2，或在 best 出现后早停；监控 succ 而不是 return 作为 best 判定。

## 关键设计决策

**有效的改动：**
1. 终端奖励分离归一化（v4）：碰撞/成功奖励不混入稠密奖励的归一化，保持信号强度
2. Slot 附近初始化（v10+）：每个拖轮从自己 slot 外侧 20-60m 出发
3. 就位阈值放宽（v17/v20）：pos_tol 5→30→50→60m，speed_tol 1.5→2.0→3.0 m/s
4. 连续安全惩罚（v5+）：进入碰撞预警区每步给负奖励，抑制冲撞
5. 分级初始距离（v19/v21）：4 档 60/80/100/150 → 60/70/90/120，自然形成先后到达 + 远档位可达
6. 大船直行简化任务（v19）：去掉转向，让策略先学纯追逐 + 避碰
7. 续训时配大 total_steps（v20）：续训传 --total-steps 必须 > 已用步数才能真正训练
8. r_speed_match zone_factor 放宽（v21）：exp(-d/40) → exp(-d/80)，让更远处也能学速度匹配
9. **speed_err tanh 压缩（v25）**：speed_err_sq → speed_err * tanh(speed_err/3)，避免靠近 zone 边缘惩罚 > 奖励
10. **r_zone soft 评分（v25）**：硬阈值 0/+1 → pos × hdg × spd 连续 0~1，30~60m 区间有梯度信号

**无效或有害的改动：**
- 势能奖励（log 势能）：远距离梯度太小，策略停在远处不靠近
- entropy_coef=0.02：太高，策略无法收敛，entropy 反而上升
- 碰撞不终止（v14-v15）：多次碰撞累积惩罚导致 value loss 爆炸
- 速度匹配权重 0.5（v11）：speed_err_sq 未压缩，vl 爆炸到 271
- **lr_anneal 下限 0.2（v22）**：续训起步 lr 太高，打散末段精细化（应保持 0.05~0.1）
- **r_heading 加 exp(-d/30) 距离门控（v24 P1）**：错诊陪走解原因，门控反而让靠近前的姿态调整缺信号
- **arrival_bonus 单独提到 80（v24 P2）**：在没解决"zone 边缘净奖励为负"前只放大失败成本，策略反而更保守
- **碰撞惩罚只给惹事 tug（v27 P5）**：在参数共享 PPO 下退步，succ 43.8% → 25.0%。原以为可以让梯度更干净（无辜 agent 不受连坐），实际：(1) 总碰撞惩罚减少 75%，全局避撞约束削弱；(2) 共享策略下"惹事 vs 旁观"在前向传播时无法区分（同一份参数），梯度差异比想象小，反而损失了"集体责任"对避撞的强约束。Lesson: **参数共享多智能体下，全局事件应该全员同奖励**——独立 actor 才适合做差异化奖励。

**奖励量级教训：**
- 所有稠密奖励每步量级应控制在 [-0.5, +0.5]
- 终端奖励（碰撞/成功）应与稠密奖励分离归一化
- speed_err_sq 需要 tanh 压缩或降低权重，否则会主导梯度
- r_progress 必须是主导信号

## 当前状态（v32 CPA 观测训练后）

- astern far-init 场景当前可用 checkpoint：
  - `checkpoints/v31_farinit_reset_lr1e4/best.pt`：succ=43.8%, coll=56.2%, len=564.4。
  - `checkpoints/v32_cpa_obs_v31init_lr1e4/best.pt`：succ=43.8%, coll=56.2%, len=472.2，使用 55D CPA 观测。
- v32 证明 CPA 观测有信息价值：训练窗口中出现过 succ=25.0%, coll=0.0%。
- 但 best 选择仍未突破 v31：最高 success 仍为 43.8%，对应 collision 仍为 56.2%。
- 当前瓶颈不是观测缺失，而是奖励/选择口径没有把“安全避碰”和“最终到位”绑定到同一个最优策略。
- **续训规律仍成立**：新观测/新奖励常在中段出现局部最优，后期容易漂移；实际使用必须选 `best.pt`，不要用 `last.pt`。

## 下一步建议（按优先级）

1. **把 CPA 从观测接入奖励**：对短 TCPA + 小 DCPA 的风险给连续惩罚，尤其是 final approach 阶段；避免策略只“看见风险”但 reward 不要求它在完成任务时避开风险。
2. **修改 best 选择口径**：保持 success 优先，但 success 相同时用 lower collision 作为 tie-breaker；或引入复合 score，例如 `succ - 0.25*collision`，防止 25% succ / 0% coll 这种有价值策略被完全忽略。
3. **做 collision/stage 诊断**：按 tug-tug / tug-ship、route stage、slot id 统计碰撞，确认高碰撞来自追赶航道、绕行入口还是 final approach。
4. **课程训练**：先用 CPA reward 训练低碰撞绕行，再逐步提高到位权重或收紧 hold 条件，让安全策略向完成任务收敛。
5. 若继续 warm start，优先从 `v32_cpa_obs_v31init_lr1e4/best.pt` 或 0-collision 中间 checkpoint 分支，而不是 `last.pt`。

## v33 CPA reward 正式训练结果（2026-05-15）

**训练配置：**
- 命令：`python3 -u train.py --resume checkpoints/v32_cpa_obs_v31init_lr1e4/best.pt --reset-progress --total-steps 3000000 --learning-rate 1e-4 --run-name v33_cpa_reward_v32init_lr1e4 --device cpu`
- 起点：`checkpoints/v32_cpa_obs_v31init_lr1e4/best.pt`
- CPA reward：`reward_cpa_w=0.18`，`cpa_alert_dist_m=70.0`，`cpa_time_horizon_s=45.0`
- 训练用时：约 11.6 min，`366` updates，`2,998,272` env steps

**结果：**
- `checkpoints/v33_cpa_reward_v32init_lr1e4/best.pt`
  - update `270`，global_step `2,211,840`
  - eval：succ `81.2%`，coll `18.8%`，return `10293.12`，len `739.1`
- `checkpoints/v33_cpa_reward_v32init_lr1e4/last.pt`
  - update `366`，global_step `2,998,272`
  - 末段 rollout 已退化到 succ `1%左右`、coll `98%左右`
  - 不应作为后续起点或部署模型使用

**关键曲线窗口：**
- step `1,638,400`：succ `68.8%`，coll `31.2%`，第一次明显超过 v31/v32。
- step `2,211,840`：succ `81.2%`，coll `18.8%`，刷新 best。
- step `2,293,760` 之后策略快速回摆，eval 多次回到 `0% success / 100% collision`。

**结论：**
- CPA reward 接入是有效的：v32 best `43.8% / 56.2%` 提升到 v33 best `81.2% / 18.8%`。
- 当前主要问题变为训练稳定性和 checkpoint 选择：最优策略出现在中后段窗口，继续训练会漂移退化。
- 后续必须从 `v33_cpa_reward_v32init_lr1e4/best.pt` 继续，不要从 `last.pt` 继续。

**下一步建议：**
1. 先用 v33 best 做轨迹可视化和碰撞类型统计，确认剩余 `18.8%` 碰撞发生在追赶航道、绕行入口还是 final approach。
2. 修改 best 选择口径：success 优先，相同 success 时选择更低 collision；或者使用 `succ - 0.25 * collision` 作为保存分数。
3. 下一轮训练建议从 v33 best 出发，缩短训练步数到 `1M~1.5M`，并启用早停，避免 2.3M 后的策略漂移。

## v34 final hull safety 诊断与训练结果（2026-05-15）

**v33 best 诊断：**
- 128 回合确定性诊断：succ `75.0%`，coll `25.0%`，timeout `0.0%`。
- 剩余碰撞高度集中：`32/32` 都是 `tug_vs_ship`，全部为 `T3` 在右舷 `stern:final_slot` 阶段撞船。
- 失败轨迹显示 T3 不是拖轮互撞，而是在 final stage 沿右舷贴近船体，hull distance 从约 `40m` 继续下降到 `6m` 阈值。

**代码改动：**
- `config.py`：新增 final stage 船体安全参数：
  - `ship_safety_dist_m=18.0`
  - `ship_safety_final_dist_m=30.0`
  - `reward_hull_safety_final_multiplier=2.0`
- `formation_env.py`：final route stage 下提前到 `30m` 开始船体 clearance 惩罚，并将船体安全权重加倍；`reward_components` 增加 `hull_dist` 便于诊断。
- `train.py`：best 选择改为 success 优先；success 相同时优先选择更低 collision；return 只作为最后 tie-breaker。
- `diagnose_checkpoint.py`：新增 checkpoint 离线诊断和单 episode 轨迹图导出。

**训练配置：**
- 命令：`python3 -u train.py --resume checkpoints/v33_cpa_reward_v32init_lr1e4/best.pt --reset-progress --total-steps 1500000 --learning-rate 5e-5 --run-name v34_final_hull_safety_v33init_lr5e5 --device cpu`
- 起点：`checkpoints/v33_cpa_reward_v32init_lr1e4/best.pt`
- 训练用时：约 `5.6 min`，`183` updates，`1,499,136` env steps。

**结果：**
- `checkpoints/v34_final_hull_safety_v33init_lr5e5/best.pt`
  - update `20`，global_step `163,840`
  - 训练期 eval：succ `93.8%`，coll `6.2%`，return `7455.42`，len `588.9`
  - 128 回合离线诊断：succ `93.0%`，coll `7.0%`，timeout `0.0%`
- `checkpoints/v34_final_hull_safety_v33init_lr5e5/last.pt`
  - update `183`，global_step `1,499,136`
  - 末段 rollout 已退化到 succ `13%左右`、coll `79%左右`
  - 不应作为后续起点或部署模型使用

**剩余瓶颈：**
- v34 将 T3 船体碰撞从 v33 的 `32/128` 降到 `7/128`，说明 final hull safety 改动有效。
- 剩余 `9/128` 碰撞中：
  - `7/9` 仍是 `T3` 右舷 `stern:final_slot` 船体碰撞。
  - `2/9` 是 `T0-T2` 左舷同侧碰撞，发生在 `stern_gate/stern_side` 附近。
- 当前不建议继续从 `last.pt` 训练；后续必须从 `v34_final_hull_safety_v33init_lr5e5/best.pt` 出发。

**下一步建议：**
1. 若继续压低剩余 7% collision，优先做几何/路线层修正，而不是单纯加训练步数：
   - 给右舷/左舷 stern final approach 增加一个外侧 holding waypoint，避免 final stage 后从 alongside 切向船体。
   - 或将 final stage 的 `ship_safety_final_dist_m` 小幅提高到 `35m`，但要检查是否压低成功率。
2. 对 `T0-T2` 少量同舷碰撞，可把 `route_tug_spacing_dist_m` 从 `90m` 提到 `100~110m`，或只在 stern gate/stern_side 阶段加强同侧 spacing。
3. 下一轮训练建议更短：从 v34 best 出发，`0.5M~0.8M` steps、`lr=2e-5~3e-5`，并在首个 `succ>=93.8%, coll<=6.2%` 的窗口后停止。

## v35 mixed slot approach 鲁棒性初始化（2026-05-15）

**目标：**
- 增加初始条件多样性，避免策略只适应 4 艘拖轮全部从船尾同分布起步。
- 支持“3 艘已合理就位，1 艘从周围绕行到 slot”和“2 艘已就位，2 艘从别的位置绕行”的 episode 分布。

**代码改动：**
- `EnvConfig.tug_init_mode` 新增 `mixed_slot_approach`。
- 每个 episode 从 `tug_init_mixed_ready_counts=(2,3)` 随机选择 ready 数量。
- 已就位拖轮：
  - 固定对应 slot 角色；
  - 初始位置在 slot 外侧保持点，默认 outward offset `18m`、位置扰动 `±2m`；
  - 航向接近大船航向、速度接近大船速度，执行器用小前进指令 `0.22`。
- 未就位拖轮：
  - 从对应路线的船尾/舷侧不同 route stage 随机起步；
  - 保留 route 观察和 route progress/lane/spacing/CPA reward；
  - 初始速度为大船速度 + `0.2~0.8m/s`，避免静止起步。
- `train.py --init-mode` 支持 `mixed_slot_approach`。

**验证：**
- `python3 -m py_compile config.py formation_env.py train.py diagnose_checkpoint.py visualize.py ppo.py` 通过。
- 100 seed reset smoke：
  - obs shape `(4, 59)`；
  - ready counts `{2: 48, 3: 52}`；
  - 最小初始拖轮间距 `110.9m`，均值 `119.7m`；
  - 最小船体距离 `43.4m`，均值 `51.2m`。
- 50 seed 使用初始动作连续 80 step：
  - 碰撞 `0/50`；
  - 运行中最小拖轮间距 `111.5m`；
  - 运行中最小船体距离 `47.5m`。

**训练建议：**
- 从 `checkpoints/v34_final_hull_safety_v33init_lr5e5/best.pt` 启动，不要从 last。
- 首轮用短训低学习率，目标是适应新初态分布而不打散已有绕行/避碰策略：
  `python3 -u train.py --resume checkpoints/v34_final_hull_safety_v33init_lr5e5/best.pt --reset-progress --total-steps 500000 --learning-rate 2e-5 --run-name v35_mixed_slot_approach_v34init_lr2e5 --init-mode mixed_slot_approach --device cpu`

**训练结果更新：**
- v35 (`lr=2e-5`, `500k`)：
  - `checkpoints/v35_mixed_slot_approach_v34init_lr2e5/best.pt`
  - 训练 eval best：update `10`，succ `100.0%`，coll `0.0%`（16 回合，偏乐观）
  - 128 回合诊断：succ `81.2%`，coll `18.0%`，timeout `0.8%`
  - 结论：学习率/步数仍偏激进，后期 rollout 明显漂移，不建议作为后续起点。
- v34 best 在 mixed 初始化下的基线：
  - 128 回合诊断：succ `88.3%`，coll `11.7%`，timeout `0.0%`
  - 碰撞全为 tug_vs_ship，主要仍是 `T3` 的 `stern:final_slot`。
- v35b (`lr=5e-6`, `200k`)：
  - `checkpoints/v35b_mixed_slot_approach_v34init_lr5e6/best.pt`
  - 训练 eval best：update `10`，succ `100.0%`，coll `0.0%`（16 回合）
  - 128 回合诊断：succ `88.3%`，coll `11.7%`，timeout `0.0%`
  - 与 v34 mixed 基线持平，但 v35b 是当前 59D 观测空间 checkpoint，可作为后续 59D 分支起点。
- `diagnose_checkpoint.py` 新增 `--init-mode`，便于直接比较旧 checkpoint 在 mixed/astern/ring/near_slot 场景下的表现。

## v36 大船尺度随机化（2026-05-15）

**目标：**
- 将大船几何从固定 `200m × 30m` 扩展为 episode 级随机采样，降低策略对特定船长/船宽、slot 几何和路线距离的过拟合。

**代码改动：**
- `EnvConfig` 新增：
  - `ship_size_randomize=True`
  - `ship_length_min_m=180.0`，`ship_length_max_m=240.0`
  - `ship_beam_min_m=26.0`，`ship_beam_max_m=40.0`
  - `obs_include_ship_size=True`
- `FormationEnv.reset()` 每个 episode 先采样 `ship.length_m / ship.beam_m`，再 reset 大船运动状态。
- slot、route waypoint、船体距离、可视化船体轮廓全部使用当前 episode 实际尺寸。
- 观测空间从 `59D -> 61D`：末尾新增 `(length/base_length - 1, beam/base_beam - 1)`。
- `train.py` 新增 `--no-ship-size-randomize` 便于固定几何 ablation。
- `diagnose_checkpoint.py` / `visualize.py` 对旧 checkpoint 做兼容：旧权重默认保持固定尺寸并关闭 2D 尺度观测。

**验证：**
- `python3 -m py_compile config.py formation_env.py train.py diagnose_checkpoint.py visualize.py ppo.py` 通过。
- 20 seed reset smoke：
  - obs shape `(4, 61)`；
  - 船长范围 `185.1~237.4m`；
  - 船宽范围 `28.3~39.8m`。
- 旧 `v35b` 在随机船型但无显式尺度观测下的 128 回合基线：
  - succ `86.7%`，coll `13.3%`，timeout `0.0%`
  - 说明尺寸扰动略增加难度，但不是完全分布外。

**训练结果：**
- v36 (`lr=5e-6`, `200k`, 61D, mixed + ship size randomization)：
  - `checkpoints/v36_ship_size_randomized_v35binit_lr5e6/best.pt`
  - 训练 eval best：succ `93.8%`，coll `6.2%`（16 回合，偏乐观）
  - 128 回合诊断：succ `84.4%`，coll `15.6%`
  - 不建议作为后续起点。
- v36b (`lr=1e-6`, `100k`, 61D, mixed + ship size randomization)：
  - `checkpoints/v36b_ship_size_randomized_v35binit_lr1e6/best.pt`
  - 训练 eval best：succ `87.5%`，coll `12.5%`
  - 128 回合诊断：succ `86.7%`，coll `13.3%`
  - 与旧 v35b 随机船型基线持平，但 v36b 是当前 61D 观测空间 checkpoint，可作为后续尺寸随机化分支起点。

**结论：**
- 大船尺度随机化已接入环境和观测。
- 当前主要瓶颈仍不是尺寸随机化本身，而是 `T3` 右舷 `stern:final_slot` 船体碰撞；v36/v36b 剩余碰撞仍集中在该阶段。
- 下一步如果要真正提升随机船型鲁棒性，应先改 stern final approach 几何（外侧 holding waypoint / final slot 前的 clearance waypoint），再继续在 v36b 上低学习率训练。

## v37 算法式 waypoint planner（2026-05-15）

**目标：**
- 用算法生成 waypoint 替代固定手写模板，先解决 `stern:final_slot` 过于直接切入导致的船体碰撞。
- 保持 MAPPO 观测维度、动作空间和 reward 接口不变，便于从 v36b 继续 warm start。

**代码改动：**
- `EnvConfig.route_planner="visibility"` 作为默认路线生成器；`"manual"` 可回到旧模板做 ablation。
- `FormationEnv._route_waypoints_body()` 改为：
  - 在大船船体系下按 `route_hull_clearance_m` 膨胀船体矩形；
  - 生成 stern gate、同舷 lane、outer holding、final entry、final slot 等语义 anchor；
  - 用 visibility graph + Dijkstra 连接穿不过膨胀船体的 anchor；
  - 船尾 slot 路线变为 `stern gate -> outer holding -> final entry -> final slot`。
- 新增 planner 参数：
  - `route_hull_clearance_m`
  - `route_outer_holding_extra_m`
  - `route_final_entry_lat_extra_m`
  - `route_final_entry_lon_offset_m`
  - `route_visibility_node_margin_m`
  - `route_min_waypoint_spacing_m`
- `train.py` 新增 `--route-planner visibility/manual`。
- `diagnose_checkpoint.py` 新增 `--route-planner visibility/manual`；旧 checkpoint 若没有 `route_planner` 字段，默认保持 `manual`，避免历史基线被新路线改变。
- `diagnose_checkpoint.py` 的 stage label 更新为 `outer_holding` / `final_entry`。

**验证：**
- `python3 -m py_compile config.py formation_env.py train.py diagnose_checkpoint.py ppo.py large_ship_model.py tugboat_dynamics_model.py` 通过。
- reset smoke：
  - `visibility` 和 `manual` 均保持 obs shape `(4, 61)`；
  - 随机船型下 visibility 路线长度：bow slot `6` 个 waypoint，stern slot `4` 个 waypoint；
  - 每条路线最后一个 waypoint 与当前 episode 的 slot body 坐标一致。

**下一步训练建议：**
- 从当前推荐 61D 随机船型分支启动：
  `python3 -u train.py --resume checkpoints/v36b_ship_size_randomized_v35binit_lr1e6/best.pt --reset-progress --total-steps 200000 --learning-rate 1e-6 --run-name v37_visibility_waypoints_v36binit_lr1e6 --init-mode mixed_slot_approach --route-planner visibility --device cpu`
- 训练后用 128 回合诊断新 planner：
  `python3 diagnose_checkpoint.py --ckpt checkpoints/v37_visibility_waypoints_v36binit_lr1e6/best.pt --episodes 128 --seed 12345 --device cpu`
- 接受标准：相对 v36b 随机船型基线 `succ 86.7% / coll 13.3%`，碰撞率应下降且成功率不明显损失。
