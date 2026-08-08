# Transformer Actor Scale Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 提供 `--tf-size {S,M,L}` 与串行消融脚本，在固定 r120+minimax 与 cuda 吞吐默认栈下用 50M steps 快筛 actor Transformer 容量对任务指标的影响。

**Architecture:** 在 `config.py` 集中 `TF_SIZE_PRESETS` 与 `apply_tf_size_preset`；`train.py` 在构造 `PPOConfig`/actor 前应用并打日志写入 hparams/ckpt；可选 runner + summarize 产出对照表；单测锁定 preset 与参数量量级。

**Tech Stack:** Python 3、`PPOConfig` dataclass、`build_actor` / `TransformerMAPPOActor`、pytest、现有 `scripts/train.py` / TensorBoard。

## Global Constraints

- 规格：`docs/superpowers/specs/2026-08-07-tf-scale-ablation-design.md`。
- Preset 键固定：`S` / `M` / `L`（CLI 输入规范化为大写）。
- 规模表必须与设计一致：S=`64/4/2/128`，M=`128/4/3/256`，L=`256/8/4/512`（字段顺序：d_model, nhead, layers, ffn）。
- 固定筛选条件：`--arch transformer`、`--init-radius 120`、`--slot-assignment minimax`、`--env-backend cuda`、`--num-envs 12288`、`--rollout-steps 128`、`--minibatch-size 65536`、`--total-steps 50000000`、`--seed 42`；无 reward-preset。Runner 必须显式传上述并行/预算参数，勿 silent 依赖全局默认漂移。
- 只改 actor 的 `tf_*`；不改 critic、不改奖励。
- 非法 `--tf-size` 必须报错并列出合法 id。

---

## File Structure

| 文件 | 职责 |
|------|------|
| `config.py` | `TF_SIZE_PRESETS` + `apply_tf_size_preset` / `list_tf_size_presets` |
| `scripts/train.py` | `--tf-size`、应用、日志、hparams |
| `tests/test_tf_size_presets.py` | preset 与参数量量级 |
| `scripts/run_tf_scale_ablation.py` | 串行三档 |
| `scripts/summarize_tf_scale.py` | 从 TB 抽末期指标表 |
| `docs/tf_scale_ablation.md` | 实验协议 |
| `README.md` / `docs/architecture.md` | 链接 |

---

### Task 1: TF size preset 映射（TDD）

**Files:**
- Create: `tests/test_tf_size_presets.py`
- Modify: `config.py`（`PPOConfig` 定义之后）

**Interfaces:**
- Consumes: `PPOConfig.tf_d_model` / `tf_nhead` / `tf_num_layers` / `tf_ffn_dim`
- Produces:
  - `TF_SIZE_PRESETS: dict[str, dict[str, int]]`
  - `def list_tf_size_presets() -> list[str]`
  - `def apply_tf_size_preset(ppo_cfg: PPOConfig, size_id: str | None) -> str | None`
    - `None` / `""` → no-op，返回 `None`
    - 合法（大小写不敏感）→ 原地 setattr，返回规范化大写 id
    - 非法 → `ValueError`，message 含合法 id

- [ ] **Step 1: Write the failing test**

Create `tests/test_tf_size_presets.py`:

```python
"""Transformer actor size presets on PPOConfig."""

from __future__ import annotations

import pytest

from config import PPOConfig, TF_SIZE_PRESETS, apply_tf_size_preset, list_tf_size_presets
from rl.actor import build_actor


EXPECTED = {
    "S": {
        "tf_d_model": 64,
        "tf_nhead": 4,
        "tf_num_layers": 2,
        "tf_ffn_dim": 128,
    },
    "M": {
        "tf_d_model": 128,
        "tf_nhead": 4,
        "tf_num_layers": 3,
        "tf_ffn_dim": 256,
    },
    "L": {
        "tf_d_model": 256,
        "tf_nhead": 8,
        "tf_num_layers": 4,
        "tf_ffn_dim": 512,
    },
}


def test_list_tf_size_presets_matches_design() -> None:
    assert list_tf_size_presets() == ["L", "M", "S"]
    assert set(TF_SIZE_PRESETS) == set(EXPECTED)


@pytest.mark.parametrize("size_id,overrides", sorted(EXPECTED.items()))
def test_apply_tf_size_preset_overrides(size_id: str, overrides: dict[str, int]) -> None:
    cfg = PPOConfig()
    applied = apply_tf_size_preset(cfg, size_id.lower())
    assert applied == size_id
    for key, value in overrides.items():
        assert getattr(cfg, key) == value


def test_apply_tf_size_preset_none_is_noop() -> None:
    cfg = PPOConfig()
    d = cfg.tf_d_model
    assert apply_tf_size_preset(cfg, None) is None
    assert apply_tf_size_preset(cfg, "") is None
    assert cfg.tf_d_model == d


def test_apply_tf_size_preset_unknown_raises() -> None:
    cfg = PPOConfig()
    with pytest.raises(ValueError, match="S"):
        apply_tf_size_preset(cfg, "XL")


@pytest.mark.parametrize(
    "size_id,min_params,max_params",
    [
        ("S", 180_000, 250_000),
        ("M", 450_000, 700_000),
        ("L", 1_800_000, 3_000_000),
    ],
)
def test_build_actor_param_count_in_band(size_id: str, min_params: int, max_params: int) -> None:
    cfg = PPOConfig(actor_arch="transformer")
    apply_tf_size_preset(cfg, size_id)
    actor = build_actor(
        arch="transformer",
        obs_dim=93,
        action_dim=4,
        hist_len=4,
        tf_d_model=cfg.tf_d_model,
        tf_nhead=cfg.tf_nhead,
        tf_num_layers=cfg.tf_num_layers,
        tf_ffn_dim=cfg.tf_ffn_dim,
        tf_dropout=cfg.tf_dropout,
    )
    n = sum(p.numel() for p in actor.parameters())
    assert min_params <= n <= max_params, n
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_tf_size_presets.py -v`  
Expected: FAIL（`TF_SIZE_PRESETS` / `apply_tf_size_preset` 未定义）

- [ ] **Step 3: Implement presets in config.py**

After `apply_reward_preset`（或 `PPOConfig` 定义之后），追加：

```python
TF_SIZE_PRESETS: dict[str, dict[str, int]] = {
    "S": {
        "tf_d_model": 64,
        "tf_nhead": 4,
        "tf_num_layers": 2,
        "tf_ffn_dim": 128,
    },
    "M": {
        "tf_d_model": 128,
        "tf_nhead": 4,
        "tf_num_layers": 3,
        "tf_ffn_dim": 256,
    },
    "L": {
        "tf_d_model": 256,
        "tf_nhead": 8,
        "tf_num_layers": 4,
        "tf_ffn_dim": 512,
    },
}


def list_tf_size_presets() -> list[str]:
    return sorted(TF_SIZE_PRESETS)


def apply_tf_size_preset(ppo_cfg: PPOConfig, size_id: str | None) -> str | None:
    if size_id is None:
        return None
    key = str(size_id).strip().upper()
    if not key:
        return None
    if key not in TF_SIZE_PRESETS:
        known = ", ".join(list_tf_size_presets())
        raise ValueError(f"Unknown tf size preset {size_id!r}. Known: {known}")
    for field_name, value in TF_SIZE_PRESETS[key].items():
        setattr(ppo_cfg, field_name, int(value))
    return key
```

注意：`apply_tf_size_preset` 必须定义在 `PPOConfig` **之后**（类型注解需要类已存在），或使用字符串注解/`from __future__ import annotations`（文件已有 future）。

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=. pytest tests/test_tf_size_presets.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add config.py tests/test_tf_size_presets.py
git commit -m "feat(config): add Transformer actor size presets S/M/L"
```

---

### Task 2: train.py `--tf-size` 接线

**Files:**
- Modify: `scripts/train.py`

**Interfaces:**
- Consumes: `apply_tf_size_preset` / `list_tf_size_presets` / `TF_SIZE_PRESETS`
- Produces: CLI `--tf-size`；应用后 `ppo_cfg.tf_*`；日志行 `[tf] size=M ... params=...`；hparams 含 `tf_size`

- [ ] **Step 1: Import helpers**

Near existing reward preset imports:

```python
from config import (
    EnvConfig,
    PPOConfig,
    REWARD_PRESETS,
    TF_SIZE_PRESETS,
    apply_reward_preset,
    apply_tf_size_preset,
    list_reward_presets,
    list_tf_size_presets,
)
```

（按文件现有 import 风格合并，勿重复导入。）

- [ ] **Step 2: Add argparse**

After `--arch`：

```python
parser.add_argument(
    "--tf-size",
    type=str,
    default=None,
    help=(
        "Transformer actor 规模 preset（config.TF_SIZE_PRESETS）；"
        f"可选: {', '.join(list_tf_size_presets())}"
    ),
)
```

- [ ] **Step 3: Apply after PPOConfig construction**

Immediately after `ppo_cfg = PPOConfig(...)`：

```python
tf_size_id: str | None = None
try:
    tf_size_id = apply_tf_size_preset(ppo_cfg, args.tf_size)
except ValueError as exc:
    raise SystemExit(f"[tf] {exc}") from exc
```

- [ ] **Step 4: Log size + param count after model build**

After `model = MAPPOActorCritic(**model_kwargs).to(device)`（或 actor 已存在处）：

```python
actor_params = sum(p.numel() for p in model.actor.parameters())
print(
    f"[tf] size={tf_size_id or '(default)'}, "
    f"d_model={ppo_cfg.tf_d_model}, nhead={ppo_cfg.tf_nhead}, "
    f"layers={ppo_cfg.tf_num_layers}, ffn={ppo_cfg.tf_ffn_dim}, "
    f"actor_params={actor_params}"
)
```

并把 `tf_size = {tf_size_id or ''}` 并入现有 hparams 文本列表（与 `reward_preset` 同行风格）。

确认 `model_kwargs` 使用的是应用 preset **之后** 的 `ppo_cfg.tf_*`（现有代码已读 `ppo_cfg.tf_*`，只要 apply 在 kwargs 构建之前即可）。

- [ ] **Step 5: Smoke help**

Run: `PYTHONPATH=. python scripts/train.py --help 2>&1 | rg -n "tf-size"`  
Expected: 显示 `--tf-size` 且 help 含 `S, M, L`（顺序随 `list_tf_size_presets`）。

Run: `PYTHONPATH=. python scripts/train.py --tf-size XL --arch transformer --total-steps 1 2>&1 | head -20`  
Expected: 非零退出，stderr/stdout 含 `Unknown tf size` 与 `S`。

- [ ] **Step 6: Commit**

```bash
git add scripts/train.py
git commit -m "feat(train): wire --tf-size preset into PPOConfig and logs"
```

---

### Task 3: 串行 runner + summarize

**Files:**
- Create: `scripts/run_tf_scale_ablation.py`
- Create: `scripts/summarize_tf_scale.py`

**Interfaces:**
- Consumes: `TF_SIZE_PRESETS`；调用 `scripts/train.py`
- Produces: 日志文件；Markdown/文本指标表（stdout）

- [ ] **Step 1: Write runner**

Create `scripts/run_tf_scale_ablation.py`（镜像已删的 reward ablation runner 风格）：

```python
#!/usr/bin/env python3
"""串行跑 Transformer 规模消融（见 docs/tf_scale_ablation.md）。"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import TF_SIZE_PRESETS, list_tf_size_presets  # noqa: E402

ORDER = ["S", "M", "L"]


def main() -> int:
    p = argparse.ArgumentParser(description="串行跑 TF 规模消融")
    p.add_argument("--sizes", nargs="+", default=ORDER, choices=ORDER)
    p.add_argument("--total-steps", type=int, default=50_000_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--env-backend", type=str, default="cuda")
    p.add_argument("--num-envs", type=int, default=12288)
    p.add_argument("--rollout-steps", type=int, default=128)
    p.add_argument("--minibatch-size", type=int, default=65536)
    p.add_argument("--eval-workers", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-summarize", action="store_true")
    p.add_argument("--log-file", type=str, default="outputs/logs/tf_scale_ablation.log")
    args = p.parse_args()

    log_path = _ROOT / args.log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmds: list[list[str]] = []
    for size in args.sizes:
        run_name = f"tf_scale_{size}_r120"
        cmds.append([
            sys.executable,
            str(_ROOT / "scripts" / "train.py"),
            "--arch", "transformer",
            "--tf-size", size,
            "--init-radius", "120",
            "--slot-assignment", "minimax",
            "--run-name", run_name,
            "--total-steps", str(args.total_steps),
            "--seed", str(args.seed),
            "--device", args.device,
            "--env-backend", args.env_backend,
            "--num-envs", str(args.num_envs),
            "--rollout-steps", str(args.rollout_steps),
            "--minibatch-size", str(args.minibatch_size),
            "--eval-workers", str(args.eval_workers),
        ])

    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"=== tf scale ablation {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        for cmd in cmds:
            line = " ".join(cmd)
            print(line)
            f.write(line + "\n")
            f.flush()
            if args.dry_run:
                continue
            proc = subprocess.run(cmd, cwd=str(_ROOT))
            if proc.returncode != 0:
                return int(proc.returncode)

    if not args.dry_run and not args.skip_summarize:
        summary = [
            sys.executable,
            str(_ROOT / "scripts" / "summarize_tf_scale.py"),
            "--runs",
            *[f"tf_scale_{s}_r120" for s in args.sizes],
        ]
        return int(subprocess.run(summary, cwd=str(_ROOT)).returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Write summarize script**

Create `scripts/summarize_tf_scale.py`：

```python
#!/usr/bin/env python3
"""汇总 TF 规模消融 run 的末期 eval 指标。"""

from __future__ import annotations

import argparse
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

KEYS = [
    "eval/final_dist_mean",
    "eval/collision_rate",
    "eval/capture_rate",
    "eval/success_rate",
    "eval/return_mean",
    "loss/explained_variance",
]


def last_scalar(ea: EventAccumulator, tag: str) -> float | None:
    if tag not in ea.Tags().get("scalars", []):
        return None
    ev = ea.Scalars(tag)
    return float(ev[-1].value) if ev else None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", type=str, default="runs")
    p.add_argument("--runs", nargs="+", required=True)
    args = p.parse_args()
    root = Path(args.logdir)

    rows = []
    for name in args.runs:
        path = root / name
        if not path.exists():
            rows.append((name, None))
            continue
        ea = EventAccumulator(str(path), size_guidance={"scalars": 0})
        ea.Reload()
        rows.append((name, {k: last_scalar(ea, k) for k in KEYS}))

    headers = ["run"] + KEYS
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for name, metrics in rows:
        if metrics is None:
            print(f"| {name} | " + " | ".join(["missing"] * len(KEYS)) + " |")
            continue
        cells = []
        for k in KEYS:
            v = metrics[k]
            if v is None:
                cells.append("n/a")
            elif "rate" in k:
                cells.append(f"{v*100:.1f}%")
            elif "dist" in k or "return" in k:
                cells.append(f"{v:.1f}")
            else:
                cells.append(f"{v:.3f}")
        print(f"| {name} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Dry-run smoke**

Run: `PYTHONPATH=. python scripts/run_tf_scale_ablation.py --dry-run`  
Expected: 打印三条含 `--tf-size S|M|L` 与 `tf_scale_*_r120` 的命令，退出 0。

- [ ] **Step 4: Commit**

```bash
git add scripts/run_tf_scale_ablation.py scripts/summarize_tf_scale.py
git commit -m "feat(scripts): add TF scale ablation runner and summarizer"
```

---

### Task 4: 文档对齐

**Files:**
- Create: `docs/tf_scale_ablation.md`
- Modify: `README.md`
- Modify: `docs/architecture.md`

- [ ] **Step 1: Write docs/tf_scale_ablation.md**

内容包含：设计链接、固定条件表、S/M/L 表、推荐命令（runner + 单条 train）、读结果（主看 final_dist/capture；假阳性说明）。

- [ ] **Step 2: Link from README**

在文档索引表增加一行：`docs/tf_scale_ablation.md` | Transformer 规模消融。  
在对比实验段落后加简短示例：

```bash
python scripts/train.py --arch transformer --tf-size M \
  --init-radius 120 --slot-assignment minimax \
  --env-backend cuda --num-envs 12288 --rollout-steps 128 \
  --minibatch-size 65536 --total-steps 50000000 \
  --run-name tf_scale_M_r120 --seed 42
```

- [ ] **Step 3: architecture.md**

CLI 列表增加 `--tf-size`；scripts 树可提 `run_tf_scale_ablation.py`；测试表增加 `tests/test_tf_size_presets.py`。

- [ ] **Step 4: Commit**

```bash
git add docs/tf_scale_ablation.md README.md docs/architecture.md
git commit -m "docs: add Transformer scale ablation protocol"
```

---

## Spec Coverage Checklist

| Spec 要求 | Task |
|-----------|------|
| TF_SIZE_PRESETS S/M/L | Task 1 |
| apply + 大小写规范化 | Task 1 |
| `--tf-size` + 日志/hparams | Task 2 |
| 参数量记录 | Task 2 |
| 串行 runner + summarize | Task 3 |
| 协议文档 + README/arch 链接 | Task 4 |
| 单测 | Task 1 |
| 不改奖励/critic | 全局约束 |

## Plan Self-Review Notes

- Preset 键统一为 `S|M|L`，与设计一致。
- `apply_tf_size_preset` 必须在 `PPOConfig` 构造之后、`model_kwargs` 之前调用。
- Runner 默认 `eval_workers=1` 以避免 CUDA fork/spawn 坑；与当前稳妥默认一致。
- 无 TBD；训练长跑本身不在本 plan 强制执行（实现后由用户或单独会话启动）。
