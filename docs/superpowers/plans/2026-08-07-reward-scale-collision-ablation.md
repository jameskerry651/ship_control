# Reward Scale Collision Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 注册 6 个 `rsc_*` 奖励尺度 preset，并提供 1M 串行粗扫 / 晋级 5M / TensorBoard 汇总工具，用于复现并缓解「冲撞式靠近」失败模式。

**Architecture:** 在已有 `config.REWARD_PRESETS` + `train.py --reward-preset` 上填表；纯函数模块负责 run 命名与 1M 晋级筛选；runner 串行调用 `train.py`；summarize 从 TB 抽 final/极值指标并可选打印晋级名单。本计划不改 `EnvConfig` 默认权重。

**Tech Stack:** Python 3、`EnvConfig` / `REWARD_PRESETS`、pytest、`scripts/train.py`、TensorBoard `EventAccumulator`。

## Global Constraints

- 规格：`docs/superpowers/specs/2026-08-07-reward-scale-collision-ablation-design.md`。
- Preset id 固定：`rsc_baseline`、`rsc_dist_soft`、`rsc_coll_mid`、`rsc_coll_hi`、`rsc_balanced`、`rsc_corridor_hard`。
- 每 preset 只覆盖三字段：`reward_dist_w`、`reward_collision_cap`、`reward_ship_soft_min_scale`（数值见 Task 1）。
- 固定训练栈（runner 必须显式传入）：`--arch transformer --tf-size S --init-radius 120 --slot-assignment minimax --seed 42 --device cuda --env-backend cuda --eval-backend cuda --num-envs 256 --rollout-steps 64 --minibatch-size 8192 --eval-workers 32`。
- 粗扫默认 `1_000_000` 步；复验默认 `5_000_000` 步。
- 不改硬碰撞阈值、终止逻辑、观测、网络、PPO、Capture/Track；不自动改 `EnvConfig` 默认奖励权重。
- 晋级最多 2 个；基线 run 为 `rsc_1m_baseline`。

---

## File Structure

| 文件 | 职责 |
|------|------|
| `config.py` | 填入 6 个 `REWARD_PRESETS` |
| `tests/test_reward_presets.py` | 锁定 preset 表与 apply 行为 |
| `rl/reward_scale_ablation.py` | run 命名、晋级纯函数（可单测） |
| `tests/test_reward_scale_ablation.py` | 命名与晋级规则 |
| `scripts/run_reward_scale_ablation.py` | 串行 1M / `--promote` 5M |
| `scripts/summarize_reward_scale.py` | TB 汇总 + 晋级名单 |
| `docs/reward_scale_ablation.md` | 实验协议 |
| `README.md` / `docs/architecture.md` | 链接 |

---

### Task 1: 注册 `rsc_*` reward presets（TDD）

**Files:**
- Modify: `config.py`（`REWARD_PRESETS` 空字典）
- Modify: `tests/test_reward_presets.py`

**Interfaces:**
- Consumes: `apply_reward_preset(env_cfg, preset_id) -> str | None`（已存在）
- Produces: `REWARD_PRESETS` 含下表 6 键；`list_reward_presets()` 返回排序后的 id 列表

- [ ] **Step 1: 改写失败测试**

将 `tests/test_reward_presets.py` 替换为：

```python
"""Reward preset CLI mapping on EnvConfig."""

from __future__ import annotations

import pytest

from config import EnvConfig, REWARD_PRESETS, apply_reward_preset, list_reward_presets

EXPECTED = {
    "rsc_baseline": {
        "reward_dist_w": 3.0,
        "reward_collision_cap": 2.0,
        "reward_ship_soft_min_scale": 0.15,
    },
    "rsc_dist_soft": {
        "reward_dist_w": 1.5,
        "reward_collision_cap": 2.0,
        "reward_ship_soft_min_scale": 0.15,
    },
    "rsc_coll_mid": {
        "reward_dist_w": 3.0,
        "reward_collision_cap": 4.0,
        "reward_ship_soft_min_scale": 0.15,
    },
    "rsc_coll_hi": {
        "reward_dist_w": 3.0,
        "reward_collision_cap": 6.0,
        "reward_ship_soft_min_scale": 0.15,
    },
    "rsc_balanced": {
        "reward_dist_w": 1.5,
        "reward_collision_cap": 4.0,
        "reward_ship_soft_min_scale": 0.15,
    },
    "rsc_corridor_hard": {
        "reward_dist_w": 3.0,
        "reward_collision_cap": 2.0,
        "reward_ship_soft_min_scale": 0.50,
    },
}


def test_list_reward_presets_matches_rsc_design() -> None:
    assert list_reward_presets() == sorted(EXPECTED)
    assert set(REWARD_PRESETS) == set(EXPECTED)


@pytest.mark.parametrize("preset_id,overrides", sorted(EXPECTED.items()))
def test_apply_reward_preset_overrides_rsc_fields(
    preset_id: str, overrides: dict[str, float]
) -> None:
    cfg = EnvConfig()
    applied = apply_reward_preset(cfg, preset_id)
    assert applied == preset_id
    for key, value in overrides.items():
        assert getattr(cfg, key) == pytest.approx(value)
    # Unlisted fields stay at EnvConfig defaults
    assert cfg.reward_collision_w == pytest.approx(1.0)
    assert cfg.reward_hold_w == pytest.approx(2.0)


def test_apply_reward_preset_none_is_noop() -> None:
    cfg = EnvConfig()
    dist = cfg.reward_dist_w
    assert apply_reward_preset(cfg, None) is None
    assert apply_reward_preset(cfg, "") is None
    assert cfg.reward_dist_w == dist


def test_apply_reward_preset_unknown_raises() -> None:
    cfg = EnvConfig()
    with pytest.raises(ValueError, match="rsc_baseline"):
        apply_reward_preset(cfg, "not_a_preset")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_reward_presets.py -v`  
Expected: FAIL（`REWARD_PRESETS == {}` 或 list 为空）

- [ ] **Step 3: 填入 presets**

在 `config.py` 将空字典改为：

```python
REWARD_PRESETS: dict[str, dict[str, float]] = {
    "rsc_baseline": {
        "reward_dist_w": 3.0,
        "reward_collision_cap": 2.0,
        "reward_ship_soft_min_scale": 0.15,
    },
    "rsc_dist_soft": {
        "reward_dist_w": 1.5,
        "reward_collision_cap": 2.0,
        "reward_ship_soft_min_scale": 0.15,
    },
    "rsc_coll_mid": {
        "reward_dist_w": 3.0,
        "reward_collision_cap": 4.0,
        "reward_ship_soft_min_scale": 0.15,
    },
    "rsc_coll_hi": {
        "reward_dist_w": 3.0,
        "reward_collision_cap": 6.0,
        "reward_ship_soft_min_scale": 0.15,
    },
    "rsc_balanced": {
        "reward_dist_w": 1.5,
        "reward_collision_cap": 4.0,
        "reward_ship_soft_min_scale": 0.15,
    },
    "rsc_corridor_hard": {
        "reward_dist_w": 3.0,
        "reward_collision_cap": 2.0,
        "reward_ship_soft_min_scale": 0.50,
    },
}
```

保留现有 `list_reward_presets` / `apply_reward_preset` 实现不变。

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_reward_presets.py -v`  
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add config.py tests/test_reward_presets.py
git commit -m "$(cat <<'EOF'
feat(reward): register rsc scale ablation presets

EOF
)"
```

---

### Task 2: run 命名与 1M 晋级纯函数（TDD）

**Files:**
- Create: `rl/reward_scale_ablation.py`
- Create: `tests/test_reward_scale_ablation.py`

**Interfaces:**
- Consumes: 无（纯函数）
- Produces:
  - `RSC_PRESET_ORDER: list[str]` — 设计表顺序的 6 个 preset id
  - `def rsc_short_name(preset_id: str) -> str` — `"rsc_baseline"` → `"baseline"`；非法（无 `rsc_` 前缀）→ `ValueError`
  - `def rsc_run_name(preset_id: str, *, phase: str) -> str` — `phase` 为 `"1m"` 或 `"5m"` → `"rsc_1m_baseline"` / `"rsc_5m_baseline"`
  - `def select_rsc_promotions(metrics_by_short: dict[str, dict[str, float]], *, max_promote: int = 2) -> list[str]`
    - `metrics_by_short` 键为 short name（`baseline`、`dist_soft`…）
    - 每个 value 至少含：`capture_rate`、`final_dist_mean`、`collision_rate`（final eval）
    - 必须存在 `baseline`；否则 `ValueError`
    - 规则（相对 baseline）：
      1. 若 `capture_rate > 0` → 自动合格
      2. 否则须：(`final_dist_mean < 200` **或** `final_dist_mean <= baseline_dist + 20`) **且** (`baseline_coll - collision_rate >= 0.15`)
      3. 排除 `baseline` 自身
      4. 排序：先按 `capture_rate > 0` 优先，再按碰撞降幅降序，再按 `final_dist_mean` 升序
      5. 返回最多 `max_promote` 个 **short name**

- [ ] **Step 1: 写失败测试**

Create `tests/test_reward_scale_ablation.py`:

```python
"""Reward scale ablation naming and promotion rules."""

from __future__ import annotations

import pytest

from rl.reward_scale_ablation import (
    RSC_PRESET_ORDER,
    rsc_run_name,
    rsc_short_name,
    select_rsc_promotions,
)


def test_rsc_preset_order_matches_design() -> None:
    assert RSC_PRESET_ORDER == [
        "rsc_baseline",
        "rsc_dist_soft",
        "rsc_coll_mid",
        "rsc_coll_hi",
        "rsc_balanced",
        "rsc_corridor_hard",
    ]


def test_rsc_short_and_run_name() -> None:
    assert rsc_short_name("rsc_coll_mid") == "coll_mid"
    assert rsc_run_name("rsc_coll_mid", phase="1m") == "rsc_1m_coll_mid"
    assert rsc_run_name("rsc_coll_mid", phase="5m") == "rsc_5m_coll_mid"


def test_rsc_short_name_rejects_bad_id() -> None:
    with pytest.raises(ValueError, match="rsc_"):
        rsc_short_name("baseline")


def test_select_promotions_picks_collision_improvers() -> None:
    metrics = {
        "baseline": {
            "capture_rate": 0.0,
            "final_dist_mean": 140.0,
            "collision_rate": 0.80,
        },
        "dist_soft": {
            "capture_rate": 0.0,
            "final_dist_mean": 150.0,
            "collision_rate": 0.70,  # only -10 pt → fail
        },
        "coll_mid": {
            "capture_rate": 0.0,
            "final_dist_mean": 155.0,
            "collision_rate": 0.60,  # -20 pt → pass
        },
        "coll_hi": {
            "capture_rate": 0.0,
            "final_dist_mean": 210.0,  # >200 and +70 vs baseline → fail dist
            "collision_rate": 0.50,
        },
        "balanced": {
            "capture_rate": 0.0,
            "final_dist_mean": 145.0,
            "collision_rate": 0.55,  # -25 pt → pass, best delta
        },
        "corridor_hard": {
            "capture_rate": 0.0,
            "final_dist_mean": 130.0,
            "collision_rate": 0.64,  # -16 pt → pass
        },
    }
    # top by delta: balanced (-25), coll_mid (-20); corridor_hard (-16) third
    assert select_rsc_promotions(metrics, max_promote=2) == ["balanced", "coll_mid"]


def test_select_promotions_capture_auto_promotes_first() -> None:
    metrics = {
        "baseline": {
            "capture_rate": 0.0,
            "final_dist_mean": 140.0,
            "collision_rate": 0.80,
        },
        "dist_soft": {
            "capture_rate": 0.1,
            "final_dist_mean": 250.0,  # would fail dist gate without capture
            "collision_rate": 0.79,
        },
        "coll_mid": {
            "capture_rate": 0.0,
            "final_dist_mean": 150.0,
            "collision_rate": 0.60,
        },
    }
    assert select_rsc_promotions(metrics, max_promote=2) == ["dist_soft", "coll_mid"]


def test_select_promotions_requires_baseline() -> None:
    with pytest.raises(ValueError, match="baseline"):
        select_rsc_promotions({"coll_mid": {
            "capture_rate": 0.0,
            "final_dist_mean": 100.0,
            "collision_rate": 0.1,
        }})
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_reward_scale_ablation.py -v`  
Expected: FAIL（模块不存在）

- [ ] **Step 3: 实现纯函数**

Create `rl/reward_scale_ablation.py`:

```python
"""Helpers for reward-scale collision ablation (naming + 1M promotion)."""

from __future__ import annotations

RSC_PRESET_ORDER: list[str] = [
    "rsc_baseline",
    "rsc_dist_soft",
    "rsc_coll_mid",
    "rsc_coll_hi",
    "rsc_balanced",
    "rsc_corridor_hard",
]


def rsc_short_name(preset_id: str) -> str:
    key = str(preset_id).strip()
    if not key.startswith("rsc_"):
        raise ValueError(f"preset id must start with 'rsc_', got {preset_id!r}")
    return key[len("rsc_") :]


def rsc_run_name(preset_id: str, *, phase: str) -> str:
    phase_norm = str(phase).strip().lower()
    if phase_norm not in {"1m", "5m"}:
        raise ValueError(f"phase must be '1m' or '5m', got {phase!r}")
    return f"rsc_{phase_norm}_{rsc_short_name(preset_id)}"


def select_rsc_promotions(
    metrics_by_short: dict[str, dict[str, float]],
    *,
    max_promote: int = 2,
) -> list[str]:
    if "baseline" not in metrics_by_short:
        raise ValueError("metrics_by_short must include 'baseline'")
    if max_promote < 0:
        raise ValueError("max_promote must be >= 0")

    base = metrics_by_short["baseline"]
    base_dist = float(base["final_dist_mean"])
    base_coll = float(base["collision_rate"])

    qualified: list[tuple[float, float, float, str]] = []
    for short, m in metrics_by_short.items():
        if short == "baseline":
            continue
        cap = float(m["capture_rate"])
        dist = float(m["final_dist_mean"])
        coll = float(m["collision_rate"])
        coll_delta = base_coll - coll
        dist_ok = dist < 200.0 or dist <= base_dist + 20.0
        coll_ok = coll_delta >= 0.15
        if cap > 0.0 or (dist_ok and coll_ok):
            # sort key: capture first (0 if capt, 1 else), -coll_delta, dist
            qualified.append(
                (0.0 if cap > 0.0 else 1.0, -coll_delta, dist, short)
            )

    qualified.sort()
    return [short for *_rest, short in qualified[:max_promote]]
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_reward_scale_ablation.py tests/test_reward_presets.py -v`  
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add rl/reward_scale_ablation.py tests/test_reward_scale_ablation.py
git commit -m "$(cat <<'EOF'
feat(reward): add rsc ablation promotion helpers

EOF
)"
```

---

### Task 3: 串行 runner

**Files:**
- Create: `scripts/run_reward_scale_ablation.py`
- Test: dry-run 手工 / 可选 subprocess 断言（本任务以 dry-run 验证为主）

**Interfaces:**
- Consumes: `RSC_PRESET_ORDER`、`rsc_run_name`、`select_rsc_promotions`；`--promote` 通过 `importlib` 加载 Task 4 的 `load_rsc_final_metrics`
- Produces: CLI 退出码；非 dry-run 时串行 `subprocess.run(train.py ...)`

**执行顺序：** 先完成 Task 4 的 `summarize_reward_scale.py`（至少提供 `load_rsc_final_metrics`），再提交本 runner；或本任务先落地不含 `--promote` 的 1M 路径，Task 4 后补 promote。下列代码为完整目标态。

- [ ] **Step 1: 创建 runner**

Create `scripts/run_reward_scale_ablation.py`:

```python
#!/usr/bin/env python3
"""串行跑奖励尺度碰撞消融（见 design spec）。"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rl.reward_scale_ablation import (  # noqa: E402
    RSC_PRESET_ORDER,
    rsc_run_name,
    select_rsc_promotions,
)


def _load_summarize_mod():
    spec = importlib.util.spec_from_file_location(
        "summarize_reward_scale",
        _ROOT / "scripts" / "summarize_reward_scale.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _train_cmd(
    *,
    preset_id: str,
    run_name: str,
    total_steps: int,
    seed: int,
    device: str,
    env_backend: str,
    eval_backend: str,
    num_envs: int,
    rollout_steps: int,
    minibatch_size: int,
    eval_workers: int,
) -> list[str]:
    return [
        sys.executable, "-u",
        str(_ROOT / "scripts" / "train.py"),
        "--arch", "transformer",
        "--tf-size", "S",
        "--init-radius", "120",
        "--slot-assignment", "minimax",
        "--reward-preset", preset_id,
        "--run-name", run_name,
        "--total-steps", str(total_steps),
        "--seed", str(seed),
        "--device", device,
        "--env-backend", env_backend,
        "--eval-backend", eval_backend,
        "--num-envs", str(num_envs),
        "--rollout-steps", str(rollout_steps),
        "--minibatch-size", str(minibatch_size),
        "--eval-workers", str(eval_workers),
    ]


def main() -> int:
    p = argparse.ArgumentParser(description="串行跑 reward scale collision 消融")
    p.add_argument(
        "--presets",
        nargs="+",
        default=list(RSC_PRESET_ORDER),
        help="rsc_* preset ids（默认设计表全序）",
    )
    p.add_argument("--total-steps", type=int, default=1_000_000)
    p.add_argument(
        "--phase",
        choices=("1m", "5m"),
        default="1m",
        help="决定 run-name 前缀；与 --total-steps 独立，调用方负责一致",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--env-backend", type=str, default="cuda")
    p.add_argument("--eval-backend", type=str, default="cuda")
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--rollout-steps", type=int, default=64)
    p.add_argument("--minibatch-size", type=int, default=8192)
    p.add_argument("--eval-workers", type=int, default=32)
    p.add_argument(
        "--promote",
        action="store_true",
        help="读取 1M runs，按规则最多选 2 个 short name 跑 5M（忽略 --presets）",
    )
    p.add_argument("--max-promote", type=int, default=2)
    p.add_argument("--promote-steps", type=int, default=5_000_000)
    p.add_argument("--logdir", type=str, default="runs")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--log-file",
        type=str,
        default="outputs/logs/reward_scale_ablation.log",
    )
    args = p.parse_args()

    jobs: list[tuple[str, str, int]] = []
    # (preset_id, run_name, steps)

    if args.promote:
        mod = _load_summarize_mod()
        metrics = mod.load_rsc_final_metrics(Path(args.logdir), phase="1m")
        try:
            shorts = select_rsc_promotions(metrics, max_promote=args.max_promote)
        except ValueError as exc:
            print(f"promote failed: {exc}", flush=True)
            return 1
        if not shorts:
            print("No presets promoted; stopping.", flush=True)
            return 0
        print(f"Promoting: {shorts}", flush=True)
        for short in shorts:
            preset_id = f"rsc_{short}"
            jobs.append((
                preset_id,
                rsc_run_name(preset_id, phase="5m"),
                args.promote_steps,
            ))
    else:
        for preset_id in args.presets:
            if not str(preset_id).startswith("rsc_"):
                raise SystemExit(f"expected rsc_* preset, got {preset_id!r}")
            jobs.append((
                preset_id,
                rsc_run_name(preset_id, phase=args.phase),
                args.total_steps,
            ))

    log_path = _ROOT / args.log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8") as f:
        f.write(
            f"=== reward scale ablation {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
        )
        for preset_id, run_name, steps in jobs:
            cmd = _train_cmd(
                preset_id=preset_id,
                run_name=run_name,
                total_steps=steps,
                seed=args.seed,
                device=args.device,
                env_backend=args.env_backend,
                eval_backend=args.eval_backend,
                num_envs=args.num_envs,
                rollout_steps=args.rollout_steps,
                minibatch_size=args.minibatch_size,
                eval_workers=args.eval_workers,
            )
            line = " ".join(cmd)
            print(line, flush=True)
            f.write(line + "\n")
            f.flush()
            if args.dry_run:
                continue
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            proc = subprocess.run(cmd, cwd=str(_ROOT), env=env)
            if proc.returncode != 0:
                return int(proc.returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: dry-run 验证命令**

Run:

```bash
python scripts/run_reward_scale_ablation.py --dry-run --presets rsc_baseline rsc_balanced
```

Expected stdout 两行，均含：
- `--reward-preset rsc_baseline` / `rsc_balanced`
- `--run-name rsc_1m_baseline` / `rsc_1m_balanced`
- `--total-steps 1000000`
- `--num-envs 256 --rollout-steps 64 --minibatch-size 8192 --eval-workers 32`
- `--eval-backend cuda`

- [ ] **Step 3: Commit**

```bash
git add scripts/run_reward_scale_ablation.py
git commit -m "$(cat <<'EOF'
feat(scripts): add reward scale ablation runner

EOF
)"
```

---

### Task 4: TensorBoard 汇总 + `load_rsc_final_metrics`

**Files:**
- Create: `scripts/summarize_reward_scale.py`
- Modify: `scripts/run_reward_scale_ablation.py`（若 Task 3 暂缺 promote，此处补齐联调）

**Interfaces:**
- Consumes: `runs/<run_name>/events.out.tfevents.*`
- Produces:
  - `def load_rsc_final_metrics(logdir: Path, *, phase: str = "1m") -> dict[str, dict[str, float]]`
    - 扫描 `logdir` 下匹配 `rsc_{phase}_*` 的子目录
    - short name = 去掉 `rsc_{phase}_` 前缀
    - 每个 short → `capture_rate` / `final_dist_mean` / `collision_rate` / `return_mean` 的 **最后一个** scalar
    - 缺目录或缺 tag：跳过该 short（不进 dict）；baseline 缺失时由 `select_rsc_promotions` 报错
  - CLI：打印 markdown 表；列包含 final 与极值（min dist、min coll、max capture）
  - CLI `--list-promote`：打印 `select_rsc_promotions` 结果

- [ ] **Step 1: 实现 summarize 脚本**

Create `scripts/summarize_reward_scale.py`:

```python
#!/usr/bin/env python3
"""汇总 reward scale 消融 run 的 eval 指标，并列出 1M 晋级候选。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from rl.reward_scale_ablation import select_rsc_promotions  # noqa: E402


def _scalars(ea: EventAccumulator, tag: str) -> list[float]:
    if tag not in ea.Tags().get("scalars", []):
        return []
    return [float(e.value) for e in ea.Scalars(tag)]


def load_run_eval_summary(run_dir: Path) -> dict[str, float] | None:
    if not run_dir.exists():
        return None
    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    capt = _scalars(ea, "eval/capture_rate")
    dist = _scalars(ea, "eval/final_dist_mean")
    coll = _scalars(ea, "eval/collision_rate")
    ret = _scalars(ea, "eval/return_mean")
    if not dist:
        return None
    return {
        "capture_rate": capt[-1] if capt else 0.0,
        "final_dist_mean": dist[-1],
        "collision_rate": coll[-1] if coll else float("nan"),
        "return_mean": ret[-1] if ret else float("nan"),
        "best_capture_rate": max(capt) if capt else 0.0,
        "best_final_dist_mean": min(dist),
        "best_collision_rate": min(coll) if coll else float("nan"),
        "best_return_mean": max(ret) if ret else float("nan"),
    }


def load_rsc_final_metrics(
    logdir: Path, *, phase: str = "1m"
) -> dict[str, dict[str, float]]:
    phase_norm = phase.strip().lower()
    prefix = f"rsc_{phase_norm}_"
    out: dict[str, dict[str, float]] = {}
    if not logdir.exists():
        return out
    for path in sorted(logdir.iterdir()):
        if not path.is_dir() or not path.name.startswith(prefix):
            continue
        short = path.name[len(prefix) :]
        summary = load_run_eval_summary(path)
        if summary is None:
            continue
        out[short] = {
            "capture_rate": summary["capture_rate"],
            "final_dist_mean": summary["final_dist_mean"],
            "collision_rate": summary["collision_rate"],
            "return_mean": summary["return_mean"],
        }
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", type=str, default="runs")
    p.add_argument("--phase", choices=("1m", "5m"), default="1m")
    p.add_argument("--runs", nargs="*", default=None,
                   help="显式 run 目录名；默认扫描 rsc_{phase}_*")
    p.add_argument("--list-promote", action="store_true")
    p.add_argument("--max-promote", type=int, default=2)
    args = p.parse_args()
    root = Path(args.logdir)

    if args.runs:
        names = list(args.runs)
    else:
        prefix = f"rsc_{args.phase}_"
        names = sorted(
            d.name for d in root.iterdir()
            if d.is_dir() and d.name.startswith(prefix)
        ) if root.exists() else []

    headers = [
        "run",
        "final_capture",
        "final_dist",
        "final_coll",
        "final_return",
        "best_capture",
        "best_dist",
        "best_coll",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")

    for name in names:
        summary = load_run_eval_summary(root / name)
        if summary is None:
            print(f"| {name} | " + " | ".join(["missing"] * (len(headers) - 1)) + " |")
            continue
        print(
            f"| {name} | "
            f"{summary['capture_rate']*100:.1f}% | "
            f"{summary['final_dist_mean']:.1f} | "
            f"{summary['collision_rate']*100:.1f}% | "
            f"{summary['return_mean']:.1f} | "
            f"{summary['best_capture_rate']*100:.1f}% | "
            f"{summary['best_final_dist_mean']:.1f} | "
            f"{summary['best_collision_rate']*100:.1f}% |"
        )

    if args.list_promote:
        metrics = load_rsc_final_metrics(root, phase=args.phase)
        try:
            promo = select_rsc_promotions(metrics, max_promote=args.max_promote)
        except ValueError as exc:
            print(f"promote: {exc}")
            return 1
        print("promote:", " ".join(promo) if promo else "(none)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: 语法检查**

Run: `python -m py_compile scripts/summarize_reward_scale.py scripts/run_reward_scale_ablation.py rl/reward_scale_ablation.py`  
Expected: 无输出，退出 0

- [ ] **Step 3: 无 runs 时 summarize 不崩**

Run: `python scripts/summarize_reward_scale.py --phase 1m`  
Expected: 只打印表头（或空表），退出 0

- [ ] **Step 4: dry-run promote 路径（无 1M 数据时应优雅退出）**

Run: `python scripts/run_reward_scale_ablation.py --dry-run --promote`  
Expected: 打印 `No presets promoted; stopping.` 或因缺 baseline 由 select 报错——**应在 runner 的 promote 分支捕获**：若 `load_rsc_final_metrics` 无 baseline，打印清晰错误并 `return 1`。在 runner 中包一层：

```python
try:
    shorts = select_rsc_promotions(metrics, max_promote=args.max_promote)
except ValueError as exc:
    print(f"promote failed: {exc}", flush=True)
    return 1
```

- [ ] **Step 5: 全量相关测试**

Run: `pytest tests/test_reward_presets.py tests/test_reward_scale_ablation.py -v`  
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/summarize_reward_scale.py scripts/run_reward_scale_ablation.py
git commit -m "$(cat <<'EOF'
feat(scripts): summarize and promote reward scale runs

EOF
)"
```

---

### Task 5: 协议文档与入口链接

**Files:**
- Create: `docs/reward_scale_ablation.md`
- Modify: `docs/architecture.md`（scripts 列表与 CLI 行）
- Modify: `README.md`（可选一行指向协议；若 README 过挤则只改 architecture）

- [ ] **Step 1: 写协议文档**

Create `docs/reward_scale_ablation.md`:

```markdown
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
```

- [ ] **Step 2: 更新 architecture 入口**

在 `docs/architecture.md` 的 `scripts/` 行加入 `run_reward_scale_ablation.py`、`summarize_reward_scale.py`；在提到 `REWARD_PRESETS` / 消融处增加指向 `[reward_scale_ablation.md](reward_scale_ablation.md)` 的链接。

- [ ] **Step 3: 跑测试不回归**

Run: `pytest tests/test_reward_presets.py tests/test_reward_scale_ablation.py -q`  
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add docs/reward_scale_ablation.md docs/architecture.md README.md
git commit -m "$(cat <<'EOF'
docs: add reward scale collision ablation protocol

EOF
)"
```

---

## Spec Coverage Checklist

| Spec 要求 | Task |
|-----------|------|
| 6 个 `rsc_*` presets | Task 1 |
| 固定训练栈显式传入 | Task 3 |
| run 命名 `rsc_1m_*` / `rsc_5m_*` | Task 2–3 |
| 1M 晋级规则（末距/碰撞/捕获、最多 2） | Task 2 + Task 3 `--promote` + Task 4 `--list-promote` |
| summarize final + 极值 | Task 4 |
| 协议文档 | Task 5 |
| 不改默认 EnvConfig 权重 | 全任务遵守（无改默认步骤） |
| 不改硬碰撞 / PPO / 网络 | 全任务遵守 |

## Out of scope（本计划不做）

- 实际启动 1M×6 / 5M 训练（实现验证用 dry-run；训练由用户或后续会话执行）
- 5M 过关后改写 `EnvConfig` 默认与 `docs/reward_function.md`
- 扫 `reward_hold_w` / `reward_collision_w`
