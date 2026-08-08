# 奖励尺度碰撞消融协议

设计规格：[superpowers/specs/2026-08-07-reward-scale-collision-ablation-design.md](superpowers/specs/2026-08-07-reward-scale-collision-ablation-design.md)

针对 `reward_no_orbit_smoke` 的「末距下降但捕获 0%、碰撞飙高」失败模式，扫描接近权重、碰撞稠密上限与走廊软化。

## 固定条件

| 项 | 值 |
|----|-----|
| `--arch` / `--tf-size` | `transformer` / `S` |
| `--init-radius` / slot | `120` / `minimax` |
| `--seed` | `42` |
| env / eval | `cuda` / `cuda` |
| `--num-envs` | `256` |
| `--rollout-steps` | `64` |
| `--minibatch-size` | `8192` |
| `--eval-workers` | `32` |
| 粗扫 / 复验步数 | `1e6` / `5e6` |

## Preset 表

见 `config.REWARD_PRESETS` 中 `rsc_*`；每项覆盖 `reward_dist_w`、`reward_collision_cap`、`reward_ship_soft_min_scale`。

## 推荐命令

```bash
# 1M 粗扫（可先 dry-run）
python scripts/run_reward_scale_ablation.py --dry-run
python scripts/run_reward_scale_ablation.py

# 汇总 + 晋级名单
python scripts/summarize_reward_scale.py --phase 1m --list-promote

# 对晋级者跑 5M
python scripts/run_reward_scale_ablation.py --promote

# 5M 汇总
python scripts/summarize_reward_scale.py --phase 5m
```

## 晋级 / 过关

- 1M：相对 `rsc_1m_baseline`，碰撞降 ≥15 pt 且末距不崩（或 capture>0）；最多 2 个。
- 5M：capture>0 且 final_dist<200 且 collision≤40% 才提议改默认（另任务，本协议不自动改）。
