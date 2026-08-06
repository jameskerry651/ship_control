# Reward Preset Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 `--reward-preset` 对 6 个奖励超参 preset 做可复现短跑消融，不改奖励公式。

**Architecture:** 在 `config.py` 集中定义 `REWARD_PRESETS` 映射与 `apply_reward_preset(env_cfg, preset_id)`；`scripts/train.py` 在构建 `EnvConfig` 后应用并打日志；单测锁定每个 preset 的字段覆盖；文档说明如何跑与如何读结果。

**Tech Stack:** Python 3、`dataclasses.EnvConfig`、pytest、现有 `scripts/train.py` / TensorBoard。

## Global Constraints

- 不修改 `env/reward.py` 中的计算公式。
- 固定筛选条件：`--arch transformer`、`--init-radius 100`、`--total-steps 1000000`、`--seed 42`。
- Preset id 必须与设计文档一致：`rw_baseline`、`rw_dist_up`、`rw_ship_safe_dn`、`rw_coll_soft`、`rw_shape_up`、`rw_combo`。
- 终端奖罚字段（`reward_arrival_bonus` / collision pen）本轮不动。
- 非法 preset id 必须报错并列出合法 id。

---

## File Structure

| 文件 | 职责 |
|------|------|
| `config.py` | `REWARD_PRESETS` 字典 + `apply_reward_preset` / `list_reward_presets` |
| `scripts/train.py` | CLI `--reward-preset`、应用、启动日志、写入 hparams |
| `tests/test_reward_presets.py` | preset 覆盖与错误处理 |
| `docs/reward_presets.md` | 实验协议与读结果说明 |
| `README.md` | 链到上述文档 + 示例命令 |
| `docs/superpowers/specs/2026-08-06-reward-preset-ablation-design.md` | 已存在的设计（只读参考） |

---

### Task 1: Preset 映射与应用函数（TDD）

**Files:**
- Create: `tests/test_reward_presets.py`
- Modify: `config.py`（在 `EnvConfig` 定义之后追加）

**Interfaces:**
- Consumes: `EnvConfig` dataclass fields listed in the design preset table
- Produces:
  - `REWARD_PRESETS: dict[str, dict[str, float]]`
  - `def list_reward_presets() -> list[str]`
  - `def apply_reward_preset(env_cfg: EnvConfig, preset_id: str | None) -> str | None`
    - `preset_id is None` or `""` → no-op，返回 `None`
    - 合法 id → 原地 `setattr` 覆盖，返回规范化 id 字符串
    - 非法 id → `ValueError`，message 含合法 id 列表

- [ ] **Step 1: Write the failing test**

Create `tests/test_reward_presets.py`:

```python
"""Reward preset overlays on EnvConfig."""

from __future__ import annotations

import pytest

from config import EnvConfig, REWARD_PRESETS, apply_reward_preset, list_reward_presets


EXPECTED = {
    "rw_baseline": {},
    "rw_dist_up": {"reward_dist_w": 6.0},
    "rw_ship_safe_dn": {"reward_collision_ship_safe_m": 60.0},
    "rw_coll_soft": {
        "reward_collision_w": 0.5,
        "reward_collision_cpa_w": 1.0,
    },
    "rw_shape_up": {"reward_shape_w": 0.8},
    "rw_combo": {
        "reward_dist_w": 6.0,
        "reward_collision_ship_safe_m": 60.0,
    },
}


def test_list_reward_presets_matches_design() -> None:
    assert list_reward_presets() == sorted(EXPECTED)
    assert set(REWARD_PRESETS) == set(EXPECTED)


@pytest.mark.parametrize("preset_id,overrides", sorted(EXPECTED.items()))
def test_apply_reward_preset_overrides(preset_id: str, overrides: dict[str, float]) -> None:
    cfg = EnvConfig()
    before = {k: getattr(cfg, k) for k in (
        "reward_dist_w",
        "reward_collision_ship_safe_m",
        "reward_collision_w",
        "reward_collision_cpa_w",
        "reward_shape_w",
        "reward_arrival_bonus",
    )}
    applied = apply_reward_preset(cfg, preset_id)
    assert applied == preset_id
    for key, value in overrides.items():
        assert getattr(cfg, key) == pytest.approx(value)
    # Unmentioned reward knobs stay at defaults (spot-check arrival bonus always untouched).
    assert cfg.reward_arrival_bonus == before["reward_arrival_bonus"]
    for key, value in before.items():
        if key not in overrides and key != "reward_arrival_bonus":
            # only assert keys that are in our watch list and not overridden
            if key in (
                "reward_dist_w",
                "reward_collision_ship_safe_m",
                "reward_collision_w",
                "reward_collision_cpa_w",
                "reward_shape_w",
            ) and key not in overrides:
                assert getattr(cfg, key) == value


def test_apply_reward_preset_none_is_noop() -> None:
    cfg = EnvConfig()
    dist = cfg.reward_dist_w
    assert apply_reward_preset(cfg, None) is None
    assert apply_reward_preset(cfg, "") is None
    assert cfg.reward_dist_w == dist


def test_apply_reward_preset_unknown_raises() -> None:
    cfg = EnvConfig()
    with pytest.raises(ValueError, match="rw_baseline"):
        apply_reward_preset(cfg, "not_a_preset")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_reward_presets.py -v`  
Expected: FAIL（`ImportError` 或 `NameError`：`REWARD_PRESETS` / `apply_reward_preset` 未定义）

- [ ] **Step 3: Write minimal implementation**

Append to `config.py` after `EnvConfig` (before `PPOConfig`):

```python
REWARD_PRESETS: dict[str, dict[str, float]] = {
    "rw_baseline": {},
    "rw_dist_up": {"reward_dist_w": 6.0},
    "rw_ship_safe_dn": {"reward_collision_ship_safe_m": 60.0},
    "rw_coll_soft": {
        "reward_collision_w": 0.5,
        "reward_collision_cpa_w": 1.0,
    },
    "rw_shape_up": {"reward_shape_w": 0.8},
    "rw_combo": {
        "reward_dist_w": 6.0,
        "reward_collision_ship_safe_m": 60.0,
    },
}


def list_reward_presets() -> list[str]:
    return sorted(REWARD_PRESETS)


def apply_reward_preset(env_cfg: EnvConfig, preset_id: str | None) -> str | None:
    if preset_id is None:
        return None
    key = str(preset_id).strip()
    if not key:
        return None
    if key not in REWARD_PRESETS:
        known = ", ".join(list_reward_presets())
        raise ValueError(f"Unknown reward preset {key!r}. Known: {known}")
    for field_name, value in REWARD_PRESETS[key].items():
        setattr(env_cfg, field_name, value)
    return key
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_reward_presets.py -v`  
Expected: PASS（全部用例）

- [ ] **Step 5: Commit**

```bash
git add config.py tests/test_reward_presets.py
git commit -m "$(cat <<'EOF'
feat: add reward preset overlays on EnvConfig

EOF
)"
```

---

### Task 2: CLI `--reward-preset` 接入 train.py

**Files:**
- Modify: `scripts/train.py`（argparse、EnvConfig 构建、启动日志、hparams dump）

**Interfaces:**
- Consumes: `apply_reward_preset`, `list_reward_presets`, `REWARD_PRESETS` from `config`
- Produces: training run that logs `[reward] preset=... overrides=...`

- [ ] **Step 1: Update import**

In `scripts/train.py` change:

```python
from config import EnvConfig, PPOConfig
```

to:

```python
from config import EnvConfig, PPOConfig, REWARD_PRESETS, apply_reward_preset, list_reward_presets
```

- [ ] **Step 2: Add argparse flag**

Near `--init-radius` (around line 756), add:

```python
    parser.add_argument(
        "--reward-preset",
        type=str,
        default=None,
        choices=list_reward_presets(),
        help=(
            "应用奖励超参 preset（见 config.REWARD_PRESETS / docs/reward_presets.md）；"
            f"可选: {', '.join(list_reward_presets())}"
        ),
    )
```

Note: `choices=` 会让 argparse 在非法 id 时直接失败；`apply_reward_preset` 仍保留 `ValueError` 供库/测试使用。若希望 unknown 走自定义文案，可去掉 `choices=` 并在应用处 `try/except SystemExit`。本计划采用 **去掉 choices、手动校验**，以便错误信息与设计一致：

```python
    parser.add_argument(
        "--reward-preset",
        type=str,
        default=None,
        help=(
            "应用奖励超参 preset（config.REWARD_PRESETS）；"
            f"可选: {', '.join(list_reward_presets())}"
        ),
    )
```

- [ ] **Step 3: Apply after other env overrides**

Immediately after the `args.speed_tol` block (before `total_steps` resolution）， insert:

```python
    reward_preset_id: str | None = None
    try:
        reward_preset_id = apply_reward_preset(env_cfg, args.reward_preset)
    except ValueError as exc:
        raise SystemExit(f"[reward] {exc}") from exc
```

Order note: apply **after** init-radius / tol overrides so CLI task knobs win for those fields; reward fields only come from preset (CLI does not expose individual reward weights in this task).

- [ ] **Step 4: Startup log + hparams**

After the existing `[init] actor_arch = ...` print, add:

```python
    if reward_preset_id is not None:
        overrides = dict(REWARD_PRESETS[reward_preset_id])
        print(f"[reward] preset = {reward_preset_id}, overrides = {overrides}")
        print(
            f"[reward] dist_w={env_cfg.reward_dist_w}, "
            f"ship_safe_m={env_cfg.reward_collision_ship_safe_m}, "
            f"coll_w={env_cfg.reward_collision_w}, "
            f"cpa_w={env_cfg.reward_collision_cpa_w}, "
            f"shape_w={env_cfg.reward_shape_w}"
        )
    else:
        print("[reward] preset = (none)")
```

Locate the existing hparams / text dump block (search `add_hparams` or writing `hparams` / `config.txt` near line 884). Add keys if a dict is being written:

```python
"reward_preset": reward_preset_id or "",
```

and, if dumping env fields individually, ensure the effective `reward_*` values (post-preset) are what get written — they already will if dump reads from `env_cfg` after apply.

If checkpoint save stores `env_cfg` (dataclass / asdict), no extra field is required; optional string `reward_preset` in the checkpoint dict is nice-to-have — add only if the save dict is constructed in the same file and a one-liner fits:

```python
# inside the checkpoint dict literal, if present:
"reward_preset": reward_preset_id,
```

- [ ] **Step 5: Smoke-check CLI help / bad id**

Run:

```bash
python scripts/train.py --help | grep -A2 reward-preset
python scripts/train.py --reward-preset not_a_preset --total-steps 1 2>&1 | head -20
```

Expected: help shows flag；bad id exits with message containing `rw_baseline`（或 Known presets 列表）。勿启动完整训练。

- [ ] **Step 6: Commit**

```bash
git add scripts/train.py
git commit -m "$(cat <<'EOF'
feat: wire --reward-preset into training CLI

EOF
)"
```

---

### Task 3: 文档与跑批说明

**Files:**
- Create: `docs/reward_presets.md`
- Modify: `README.md`（文档索引表 + 一小节示例）
- Modify: `docs/reward_function.md`（文末加一行链到 presets）

- [ ] **Step 1: Write `docs/reward_presets.md`**

```markdown
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

```bash
for p in rw_baseline rw_dist_up rw_ship_safe_dn rw_coll_soft rw_shape_up rw_combo; do
  python scripts/train.py \
    --arch transformer \
    --init-radius 100 \
    --reward-preset "$p" \
    --run-name "$p" \
    --total-steps 1000000 \
    --seed 42
done
```

## 如何读结果

主看 `eval/return_mean`；旁证 `eval/collision_rate`。人工扫 `eval/final_dist_mean`、`reward/r_hold`、`eval/capture_rate`：return 高但明显躲远 → 假阳性，不加长训。优胜 1–2 个再训到 2M–5M。
```

- [ ] **Step 2: Link from README and reward_function.md**

In `README.md` 文档索引表增加一行：

```markdown
| [docs/reward_presets.md](docs/reward_presets.md) | 奖励超参 preset 消融 |
```

In 对比实验一节下方或训练示例旁加：

```bash
python scripts/train.py --arch transformer --init-radius 100 \
  --reward-preset rw_combo --run-name rw_combo --total-steps 1000000 --seed 42
```

In `docs/reward_function.md` 文末增加：

```markdown
## Preset 消融

筛选用超参组合见 [reward_presets.md](reward_presets.md)（`--reward-preset`）。
```

- [ ] **Step 3: Commit**

```bash
git add docs/reward_presets.md README.md docs/reward_function.md
git commit -m "$(cat <<'EOF'
docs: add reward preset ablation protocol

EOF
)"
```

---

### Task 4:（可选）指标汇总小脚本

仅在需要从多次 run 快速出表时做；可跳过。

**Files:**
- Create: `scripts/summarize_reward_presets.py`

- [ ] **Step 1: Implement TB scalar reader**

```python
#!/usr/bin/env python3
"""Print last scalar values for reward-preset runs under runs/."""

from __future__ import annotations

import argparse
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAGS = [
    "eval/return_mean",
    "eval/collision_rate",
    "eval/final_dist_mean",
    "eval/capture_rate",
    "eval/success_rate",
]


def last_scalar(run_dir: Path, tag: str) -> float | None:
    ea = EventAccumulator(str(run_dir))
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return None
    events = ea.Scalars(tag)
    return float(events[-1].value) if events else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="runs")
    parser.add_argument(
        "--runs",
        nargs="+",
        default=[
            "rw_baseline",
            "rw_dist_up",
            "rw_ship_safe_dn",
            "rw_coll_soft",
            "rw_shape_up",
            "rw_combo",
        ],
    )
    args = parser.parse_args()
    root = Path(args.logdir)
    header = ["run", *TAGS]
    print("\t".join(header))
    for name in args.runs:
        row = [name]
        run_dir = root / name
        for tag in TAGS:
            val = last_scalar(run_dir, tag) if run_dir.is_dir() else None
            row.append(f"{val:.4g}" if val is not None else "NA")
        print("\t".join(row))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Manual smoke**（有 run 后）

```bash
python scripts/summarize_reward_presets.py --logdir runs
```

- [ ] **Step 3: Commit**

```bash
git add scripts/summarize_reward_presets.py
git commit -m "$(cat <<'EOF'
chore: add reward preset TensorBoard summary helper

EOF
)"
```

---

## Spec Coverage Checklist

| Spec 要求 | Task |
|-----------|------|
| Preset 映射表（6 ids） | Task 1 |
| `apply` + 非法 id 报错 | Task 1 |
| `--reward-preset` CLI | Task 2 |
| 日志 / checkpoint meta | Task 2 |
| 不改 `reward.py` 公式 | Global Constraints（无任务改该文件） |
| 实验协议文档 | Task 3 |
| 可选汇总脚本 | Task 4 |
| 固定筛选命令 | Task 3 文档中的 for-loop |

## Out of Scope（实现后由用户/后续会话执行）

- 实际跑满 6×1M 训练
- 人工选优胜并加长训
- 结构项奖励（第二轮）
