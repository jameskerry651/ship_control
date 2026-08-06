# TensorBoard 指标说明

训练脚本：`scripts/train.py`  
默认日志目录：`runs/<run_name>/`  

```bash
tensorboard --logdir runs
```

横轴一般为 **环境步数** `global_step`（含所有并行 env × 拖轮）。  
`rollout/*` 多为最近约 100 个已完成 episode 的滑动均值；`eval/*` 仅在评估间隔写入。

任务阶段含义见 [task_spec.md](task_spec.md)（Approach → Capture → Track）。

---

## 1. 任务进度 `rollout/`

| 指标 | 含义 | 如何读 |
|------|------|--------|
| `ep_return_mean` | 单局回报（局内各步「全体拖轮奖励均值」再累加） | 上升通常表示学得更好；量级受奖励归一化与终端奖罚影响，宜与 success/collision 对照 |
| `success_rate` | 完整 Track 成功比例（Capture 后再连续 in-zone 满 `track_horizon_s`） | **主指标**；长期应上升 |
| `capture_rate` | 至少完成 Capture 的比例（短时全体到位，尚未要求长时跟随） | 先于 success 上升；高 capture、低 success 说明「进得去、跟不住」 |
| `track_in_zone_ratio` | 已 Capture 回合中，Track 阶段全体同时 in-zone 的步数占比 | 越接近 1 跟随越稳；未 Capture 的回合不参与该滑动窗口 |
| `collision_rate` | 因拖轮-大船或拖轮-拖轮碰撞结束的比例 | 应下降；与 success 权衡时优先看碰撞是否恶化 |
| `final_dist_mean` | 回合结束时各艇到目标 slot 距离的均值（米） | 早期下降表示在靠近；已 Capture 后仍很大则异常 |

控制台中的 `succ` / `cap` / `trk` / `coll` / `d` 与上表对应。

---

## 2. 周期性评估 `eval/`

确定性策略、固定评估 episode 数（`PPOConfig.eval_episodes`），比 `rollout/*` 更少噪声，用于存 `best.pt`。

| 指标 | 含义 |
|------|------|
| `return_mean` | 评估回报均值 |
| `success_rate` | 评估集完整 Track 成功率 |
| `capture_rate` | 评估集 Capture 率 |
| `track_in_zone_ratio` | 评估集中**已 Capture** 回合的 Track in-zone 占比均值；若无人 Capture 则为 0 |
| `collision_rate` | 评估碰撞率 |
| `final_dist_mean` | 评估终局平均距离（米） |

优先盯：`eval/success_rate` ↑、`eval/collision_rate` ↓、`eval/final_dist_mean` ↓。

---

## 3. PPO 损失与健康度 `loss/`

| 指标 | 含义 | 如何读 |
|------|------|--------|
| `policy` | Clipped surrogate 策略损失 | 可正可负；剧烈抖动且 KL 很大时警惕更新过猛 |
| `value` | Critic 价值损失（PopArt 归一化空间） | 应总体可控；长期爆炸说明回报尺度或 critic 异常 |
| `entropy` | 策略熵（探索强度代理） | 过早塌到很低 → 探索不足；一直很高且不收敛 → 学不动 |
| `approx_kl` | 新旧策略近似 KL | 常被 `target_kl` 约束；持续顶满说明步长偏大 |
| `explained_variance` | Critic 对 return 的解释度 | 越接近 **1** 越好；长期 ≤0 说明价值估计很差 |

---

## 4. 优化与探索 `opt/`

| 指标 | 含义 | 如何读 |
|------|------|--------|
| `learning_rate` | 当前 Adam 学习率 | 默认余弦退火，应缓慢下降 |
| `log_std_mean` | 动作高斯 `log_std` 均值 | 越大动作越噪；持续过低可能过早确定性收敛 |

---

## 5. 奖励分解 `reward/`

每个 update 内、对各 env 的 `reward_components` 做均值后再对步平均（未做运行归一化前的环境原始分量语义）。

| 指标 | 含义 | 如何读 |
|------|------|--------|
| `r_dist` | 距离/接近相关奖励（进度为主 + 弱绝对靠近，含 stall_scale） | Approach 阶段应有贡献；入位后 gate 切向 hold，此项变弱属正常 |
| `r_hold` | 站位保持分（位置/航向/速度综合，近距开启） | Capture/Track 阶段应上升或维持 |
| `p_collision` | 碰撞风险惩罚项（势垒 + CPA，船项可走廊软化；非负） | 越小越好；升高常伴随 `collision_rate` 变差 |
| `p_stall` | 停滞惩罚项（乘权重前；Hold 区为 0） | 长期打满却不靠近 → 假阳性/刷分；应随净接近下降 |

总步奖励还含 shaping、team、终端奖罚等，**未全部写入 TensorBoard**；上表只保留读训练最有用的项。

---

## 6. 建议阅读顺序

1. `eval/success_rate`、`eval/capture_rate`、`eval/collision_rate`、`eval/final_dist_mean`
2. 对照 `rollout/` 是否同向（噪声更大）
3. `loss/explained_variance`、`loss/approx_kl`、`loss/entropy` 看训练是否健康
4. `reward/r_dist`、`reward/r_hold`、`reward/p_collision` 看卡在接近、跟随还是避碰

---

## 7. 文本面板

| 面板 | 内容 |
|------|------|
| `hparams` | 本次运行的 `EnvConfig` / `PPOConfig` 全文（含 `reward_preset`） |

旧 run 的 event 文件可能仍含已删除的诊断曲线或历史标签；新训练只写入本文列出的指标。
