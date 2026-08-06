# 奖励超参 Preset 消融

> 设计：`docs/superpowers/specs/2026-08-06-reward-preset-ablation-design.md`  
> 实现：`config.REWARD_PRESETS` / `apply_reward_preset`，CLI `--reward-preset`

## 固定条件

- `--arch transformer`
- `--init-radius 100`
- `--total-steps 1000000`
- `--seed 42`
- 不 resume；每 preset 独立 `--run-name`

## Preset

| id | 改动 |
|----|------|
| `rw_baseline` | 无（对照） |
| `rw_dist_up` | `reward_dist_w=6` |
| `rw_ship_safe_dn` | `reward_collision_ship_safe_m=60` |
| `rw_coll_soft` | `reward_collision_w=0.5`, `reward_collision_cpa_w=1` |
| `rw_shape_up` | `reward_shape_w=0.8` |
| `rw_combo` | `dist_w=6` + `ship_safe_m=60` |

## 命令

推荐用脚本串行跑（默认 `env-backend=sync`、`num-envs=2`，日志写到 `outputs/logs/reward_preset_ablation.log`）：

```bash
python scripts/run_reward_preset_ablation.py
```

常用选项：

```bash
# 只跑部分 preset
python scripts/run_reward_preset_ablation.py --presets rw_baseline rw_combo

# 预览命令不训练
python scripts/run_reward_preset_ablation.py --dry-run

# 覆盖设备 / 环境后端
python scripts/run_reward_preset_ablation.py --device mps --env-backend sync --num-envs 2
```

结束后会自动调用 `scripts/summarize_reward_presets.py`。

## 如何读结果

主看 `eval/return_mean`；旁证 `eval/collision_rate`。人工扫 `eval/final_dist_mean`、`reward/r_hold`、`eval/capture_rate`：return 高但明显躲远 → 假阳性，不加长训。优胜 1–2 个再训到 2M–5M。
