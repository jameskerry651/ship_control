# Training Throughput Defaults Implementation Plan

> **Historical note (2026-08-07):** `docs/reward_presets.md` 与 `scripts/run_reward_preset_ablation.py` 已删除；本计划中相关步骤忽略。只同步 `README` / `architecture` / `train.py` / `PPOConfig`。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 MAPPO 训练默认超参调到更高墙钟吞吐（`num_envs=32`、更大 minibatch、更轻更快的 eval），并同步文档默认。

**Architecture:** 只改 `PPOConfig` 字段默认值与引用这些默认值的文案/CLI；训练循环、网络、环境逻辑不动。用单测锁定默认值，避免文档与代码漂移。短跑验收对照设计文档中的诊断基线。

**Tech Stack:** Python 3、`dataclasses` `PPOConfig`、pytest、现有 `scripts/train.py` / CUDA。

## Global Constraints

- 不改训练算法、网络结构、奖励公式、`env-backend` 实现。
- 不做 AMP / `torch.compile` / 异步 pipeline / 性能埋点。
- 目标默认值必须与 `docs/superpowers/specs/2026-08-06-training-throughput-defaults-design.md` 一致：
  - `num_envs=32`
  - `minibatch_size=4096`
  - `eval_interval=5`
  - `eval_episodes=32`
  - `eval_workers=8`
- `rollout_steps=512`、`update_epochs=4` 保持不变。
- 弱机器路径保留：CLI 可下调 `--num-envs` / `--env-backend sync` / `--eval-workers 1`。
- 若工作区已有未提交的「16 env / cuda」WIP，最终落地状态必须是本 spec 的 32-env 默认，不要停在中间态。

---

## File Structure

| 文件 | 职责 |
|------|------|
| `config.py` | `PPOConfig` 默认值与注释 |
| `tests/test_ppo_throughput_defaults.py` | 锁定吞吐相关默认值 |
| `README.md` | 快速开始里的默认 env 说明 |
| `docs/architecture.md` | 训练入口默认描述 |
| `scripts/train.py` | 模块 docstring 示例中的 `--num-envs`（若仍写 16） |
| `docs/superpowers/specs/2026-08-06-training-throughput-defaults-design.md` | 只读参考 |

---

### Task 1: 锁定并更新 `PPOConfig` 默认值（TDD）

**Files:**
- Create: `tests/test_ppo_throughput_defaults.py`
- Modify: `config.py`（`PPOConfig` 字段与注释）

**Interfaces:**
- Consumes: `from config import PPOConfig`
- Produces: `PPOConfig` 实例字段满足设计表中的新默认值

- [ ] **Step 1: Write the failing test**

```python
"""锁定训练吞吐相关 PPOConfig 默认值（见 throughput defaults design）。"""

from config import PPOConfig


def test_throughput_related_ppo_defaults() -> None:
    cfg = PPOConfig()
    assert cfg.num_envs == 32
    assert cfg.minibatch_size == 4096
    assert cfg.eval_interval == 5
    assert cfg.eval_episodes == 32
    assert cfg.eval_workers == 8
    assert cfg.rollout_steps == 512
    assert cfg.update_epochs == 4
    assert cfg.device == "cuda"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ppo_throughput_defaults.py -v`

Expected: FAIL on `num_envs`（当前工作区多为 16，或 git HEAD 为更旧值）

- [ ] **Step 3: Update `PPOConfig` defaults**

In `config.py`, set:

```python
    rollout_steps: int = 512
    # 默认面向多核 CPU + CUDA GPU（如 RTX 3090）；弱机器可用 CLI 下调
    num_envs: int = 32
    minibatch_size: int = 4096
    update_epochs: int = 4
    # ...
    log_interval: int = 1
    # num_envs=32 后每 update 样本量更大；放宽 eval 间隔，并用并行 workers 控墙钟
    save_interval: int = 5
    eval_interval: int = 5
    eval_episodes: int = 32
    # CUDA 训练后 fork 子进程易挂死；并行 eval 用 spawn+CPU（见 train.py）
    eval_workers: int = 8
    device: str = "cuda"
```

Do not change `gamma` / `learning_rate` / `actor_arch` / `total_steps` / `save_interval` unless already required by unrelated WIP; if WIP only touched the fields above, fold it into these final values.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ppo_throughput_defaults.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_ppo_throughput_defaults.py config.py
git commit -m "$(cat <<'EOF'
feat: raise default training throughput knobs for multi-core CUDA

Increase num_envs/minibatch and lighten eval so wall-clock is less
dominated by sequential evaluation on 3090-class machines.
EOF
)"
```

---

### Task 2: 同步文档默认

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify: `scripts/train.py`（仅模块顶部用法示例行，若仍含 `--num-envs 16`）

**Interfaces:**
- Consumes: Task 1 落地后的 `PPOConfig.num_envs == 32` 等
- Produces: 文档文案与 `PPOConfig` 一致

- [ ] **Step 1: Update README quick start comment**

Change the training comment to mention 32 envs and new eval defaults, keep weak-machine hint:

```bash
# 训练（默认 cuda + subproc + 32 envs、eval_workers=8、init 半径 100 m、5M env-steps）
python scripts/train.py --arch transformer --run-name tf_r100
# 远距：python scripts/train.py --arch transformer --run-name tf_r200 --init-radius 200
# 弱机器：加 --device cpu --env-backend sync --num-envs 2 --eval-workers 1
```

- [ ] **Step 2: Update `docs/architecture.md` training section bullets**

```markdown
- 向量化：默认 `subproc` + `num_envs=32`（可用 `--env-backend sync` 覆盖）
- 设备：默认 `cuda`；评估默认 `eval_workers=8`、`eval_episodes=32`、`eval_interval=5`（CUDA 下并行 eval 用 spawn+CPU）
```

- [ ] **Step 3: Update `scripts/train.py` docstring example**

```text
    python scripts/train.py --total-steps 5000000 --num-envs 32 --device cuda
```

- [ ] **Step 4: Grep for stale defaults**

Run: `rg -n 'num_envs=16|num-envs 16|16 envs|eval_workers=1' README.md docs/architecture.md scripts/train.py config.py`

Expected: 无残留「默认 16」表述（设计文档/plans 里的基线叙述可保留）。`evaluate_policy` 形参默认 `eval_workers: int = 1` 是函数签名 fallback，可保留；真正训练路径必须走 `PPOConfig.eval_workers`。

- [ ] **Step 5: Commit**

```bash
git add README.md docs/architecture.md scripts/train.py
git commit -m "$(cat <<'EOF'
docs: sync training default env/eval counts with PPOConfig

Keep README and architecture text aligned with the higher-throughput CUDA defaults.
EOF
)"
```

---

### Task 3: 短跑验收

**Files:**
- 无代码改动（只跑命令、对照设计 §4）
- Optional note in commit message / leave local run artifacts untracked (`runs/`, `checkpoints/`)

**Interfaces:**
- Consumes: Task 1–2 默认值已生效（不传覆盖 CLI 时）
- Produces: 控制台日志可核对 `samples_per_update`、`sps`、eval 墙钟

- [ ] **Step 1: Confirm init banner uses new defaults**

Run:

```bash
python scripts/train.py --total-steps 65536 --run-name diag_throughput_a32_smoke --seed 0
```

Expected in log:
- `samples_per_update = 65536`（`512 × 32 × 4`）
- `workers=32` 或 `16 envs` 不再出现；应为 `32 envs`
- `eval_workers=8`

（该 smoke 只有 1 个 update，可能不触发 eval；主要用于确认 banner。）

- [ ] **Step 2: Acceptance run with eval**

Run:

```bash
python scripts/train.py --total-steps 262144 --run-name diag_throughput_a32 --seed 0
```

Expected:
- `total updates = 4`
- 训练段 `sps` 相对诊断基线（~5800–6700）明显上升（目标方向 ≥1.5×，不硬卡）
- 出现至少一次 `[eval]`；从 `elapsed` 跳变看，单次 eval 显著短于 ~110 s（目标方向约 15–30 s 量级）
- 进程正常退出 `[done]`，无 hang

If `eval_workers=8` hangs or crashes: stop, set default back to `2` or `1` in a follow-up fix, and document in the commit message. Do not ignore failures.

- [ ] **Step 3: Re-run unit test**

Run: `pytest tests/test_ppo_throughput_defaults.py -v`

Expected: PASS

- [ ] **Step 4: Mark design status（optional small edit）**

In `docs/superpowers/specs/2026-08-06-training-throughput-defaults-design.md`, change `状态：已批准，待实现` → `状态：已实现`.

- [ ] **Step 5: Commit status + any leftover default sync**

```bash
git add docs/superpowers/specs/2026-08-06-training-throughput-defaults-design.md
git commit -m "$(cat <<'EOF'
docs: mark throughput defaults design as implemented

Acceptance short run completed against the new PPOConfig defaults.
EOF
)"
```

Do **not** commit `runs/`、`checkpoints/`、`outputs/logs/` unless the user asks.

---

## Spec Coverage Checklist

| Spec 要求 | Task |
|-----------|------|
| `num_envs=32` | Task 1 |
| `minibatch_size=4096` | Task 1 |
| `eval_interval=5` | Task 1 |
| `eval_episodes=32` | Task 1 |
| `eval_workers=8` | Task 1 |
| 文档同步 README / architecture | Task 2 |
| 消融脚本/文档默认 32 | Task 2 |
| 不做 AMP/compile/埋点 | Global Constraints（全任务遵守） |
| 短跑验收 sps / eval 墙钟 | Task 3 |
| 弱机器 CLI 下调提示 | Task 2 README |

## Placeholder / Consistency Review

- 无 TBD/TODO 步骤。
- 数值与 design spec 字面一致。
- `evaluate_policy(..., eval_workers: int = 1)` 保留为函数默认；训练入口必须传 `PPOConfig.eval_workers`（现有 `train.py` 已如此）。
