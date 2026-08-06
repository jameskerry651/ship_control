# 任务规格：Approach → Capture → Track

## 目标

拖轮智能体学会：

1. 自行接近分配 slot（反应式导航，不依赖外部航路）；
2. 接近大船时避免碰撞；
3. 入位后持续稳定跟随（站位保持）。

## 阶段

| 阶段 | 进入条件 | 行为 | 终止 |
|------|----------|------|------|
| Approach | episode 开始 | 靠近 slot + 避碰 | — |
| Capture | 全体 in-zone 连续 ≥ `hold_time_s` | 发一次 `reward_arrival_bonus`（仅此步），`phase=track` | **不结束** |
| Track | Capture 之后 | 继续相对大船保持 slot | 连续全体 in-zone ≥ `track_horizon_s` → **success**（不再发到位奖励） |

in-zone 定义：

- 位置误差 `< pos_tol_m`（默认 10 m）
- 航向误差 `< heading_tol_rad`（默认 30°）
- 相对大船速度误差 `< speed_tol_ms`（默认 3 m/s）

Track 期间若有艇离开 in-zone，`track_streak` 清零，但 `capture` 状态保留；需重新攒满连续跟随时长才算 success。

## 其它终止

- 碰撞（拖轮-大船 / 拖轮-拖轮）→ terminate，失败
- `max_episode_steps` → truncate（timeout）

## 指标

- `success_rate`：完成完整 Track
- `capture_rate`：至少完成 Capture
- `track_in_zone_ratio`：已 Capture 回合中，Track 阶段全体 in-zone 步数占比
- `collision_rate` / `final_dist_mean`

## 初始化

拖轮默认通过安全拒绝采样放在大船周围 120 m 圆环上：
`tug_init_mode=circle`，`tug_init_schema=safe_circle_v2`。初始化保证船体间隙和艇间距均超过硬碰撞阈值 5 m；显式使用较小半径时会对不可行方向做条件采样并发出警告。

位置生成后，默认以 `tug_slot_assignment_mode=minimax` 将匿名采样位置匹配到唯一 slot：先最小化最远单艇的初始直线距离，再以团队总距离和 slot 字典序消除平局。匹配后按 slot 顺序规范化为 canonical agent 角色，单个 episode 内保持不变；历史复现可使用 `--slot-assignment fixed`。

远距复现：

```bash
python scripts/train.py --arch transformer --run-name tf_r200 --init-radius 200
```

## 默认超参

见 `config.py`：`hold_time_s=2.0`，`track_horizon_s=30.0`，`tug_init_radius_m=120.0`。
