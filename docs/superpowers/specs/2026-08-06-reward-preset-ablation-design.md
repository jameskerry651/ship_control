# 奖励超参消融实验设计

日期：2026-08-06  
状态：待实现  
范围：不改 `env/reward.py` 公式结构；仅通过配置 preset 做权重/几何消融。

## 1. 背景与目标

`tf_r100`（`init_radius=100`，2M steps）显示：PPO 训练健康（EV≈0.9），但全程 `capture_rate=0` / `success_rate=0`。策略在「靠近」与「躲碰撞」之间震荡；中期曾把 `final_dist` 压到 ~159 m，但碰撞率升至 ~95%，随后退回 ~200 m。`r_hold` 几乎为 0。

本轮目标：用**可复现的短跑消融**比较现有奖励超参，选出更利于「碰撞可控下的 return」的 preset，再人工看曲线排除「躲远刷分」，优胜项加长训练。

不在本轮做：奖励公式大重构、新结构项、架构消融。

## 2. 固定条件

| 项 | 值 |
|----|-----|
| Actor | `--arch transformer` |
| 初始化半径 | `--init-radius 100` |
| 总步数（筛选） | `--total-steps 1000000` |
| Seed | `--seed 42` |
| Checkpoint | 不 resume；每 preset 独立 run |
| 终端奖罚 | 保持默认（`reward_arrival_bonus` / collision pen 不动） |

## 3. Preset 表

相对 `EnvConfig` 当前默认；未列出的字段保持默认。

| id | 假设 | 改动 |
|----|------|------|
| `rw_baseline` | 对照 | 无 |
| `rw_dist_up` | 靠近收益不够 | `reward_dist_w: 3 → 6` |
| `rw_ship_safe_dn` | 船软障过大挡进槽 | `reward_collision_ship_safe_m: 100 → 60` |
| `rw_coll_soft` | 稠密碰撞过重 | `reward_collision_w: 1 → 0.5`，`reward_collision_cpa_w: 2 → 1` |
| `rw_shape_up` | 需要更强势场引导 | `reward_shape_w: 0.3 → 0.8` |
| `rw_combo` | 靠近 + 缩软障 | `reward_dist_w=6` + `reward_collision_ship_safe_m=60` |

## 4. 实现约定

1. 在代码中集中定义 preset → 字段覆盖映射（建议 `config.py` 或 `env/reward_presets.py`）。
2. `scripts/train.py` 增加 `--reward-preset <id>`；启动时应用到 `EnvConfig`，并打日志 / 写入 checkpoint meta。
3. 非法 id 直接报错并列出合法 id。
4. 不修改 `FormationRewardComputer` 的计算公式。
5. 可选：单测断言每个 preset 的关键字段覆盖正确。

示例命令：

```bash
python scripts/train.py \
  --arch transformer \
  --init-radius 100 \
  --reward-preset rw_dist_up \
  --run-name rw_dist_up \
  --total-steps 1000000 \
  --seed 42
```

## 5. 筛选与晋级

**主指标**：末期 / 中后期 `eval/return_mean`（越高越好）。

**旁证（不设硬否决）**：`eval/collision_rate`（越低越好）。

**人工复核**：查看 `eval/final_dist_mean`、`reward/r_hold`、`eval/capture_rate`。若 return 高但明显「躲远」（距离相对对照显著变差且 `r_hold` 仍≈0），标为假阳性，不推荐加长训。

**晋级**：选 1–2 个优胜 preset，另开 run 训到 2M–5M（可用同 seed；可选再加 1 seed 验证）。

## 6. 交付物

- Preset 映射 + `--reward-preset` CLI
- 本设计文档；实现后可补一页实验协议 / 读结果说明（可并入 `docs/`）
- 可选：从 TensorBoard / 日志汇总最终指标表的小脚本

## 7. 后续（本轮不做）

若 6 跑后仍无 capture、且优胜项仍是「躲远刷 return」，再开一轮：增加 1 个可开关结构项（例如朝 slot 靠近时软化船软碰撞），与权重消融分开归因。
