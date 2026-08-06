# Docs–Code Sync Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align living docs with the current runnable training path and delete curriculum / route-planner dead code, tests, and CLI wiring.

**Architecture:** Surgical cleanup against truth sources (`config.py`, `env/obs_spec.py`, `env/reward.py`, `env/formation_env.py`, `scripts/train.py`). Delete unused modules/tests first, strip `--course` from train, then edit docs so they describe only what remains. Keep reward-preset superpowers design/plan; fix their outdated “course” ordering note.

**Tech Stack:** Python 3, pytest, existing MAPPO training scripts, Markdown docs under `docs/` and `README.md`.

## Global Constraints

- Scope = full audit of docs + legacy code/tests that contradict current main path (spec §2).
- Curriculum: delete completely — no CLI stub, no docs narration (spec §2 / §4.3).
- `route_planner`: delete module + related tests; living docs must not describe path planning (spec §2 / §4.2).
- `docs/superpowers/`: keep only artifacts that still map to code; fix outdated course ordering in reward-preset plan (spec §5.2).
- Truth sources win over old docs (spec §3).
- Out of scope: physics/reward formula rewrites, implementing GRU/LSTM, deleting simulator, rewriting TensorBoard event files, committing unrelated `outputs/` / `.playwright-cli/` (spec §6).
- Do not commit unless the user asks during execution (user rule); plan steps still list suggested commits for when requested.

## File Structure

| Path | Responsibility after this plan |
|------|--------------------------------|
| `env/route_planner.py` | **Deleted** |
| `tests/test_curricula.py` | **Deleted** |
| `tests/test_reward_route_progress.py` | **Deleted** |
| `tests/test_reward_precision.py` | **Deleted** |
| `docs/curriculum_training.md` | **Deleted** (already gone in working tree) |
| `scripts/evaluate_ready_counts.py` etc. | **Stay deleted** |
| `scripts/train.py` | No curricula import / `--course` / course checkpoint fields |
| `docs/architecture.md` | Current module map + tests; no route/course leftovers |
| `README.md` | Index + commands for current path only |
| `docs/task_spec.md` | Approach→Capture→Track; compact init-radius note |
| `docs/tensorboard_metrics.md` | Metrics actually logged; no live `course` panel |
| `docs/observation_space.md` | ObsSpec v2; no route/waypoint features |
| `docs/reward_function.md` / `reward_presets.md` / `arch_ablation.md` | Spot-check vs code; fix drift only |
| `docs/superpowers/plans/2026-08-06-reward-preset-ablation.md` | Order note without course |
| `config.py` | Comment cleanup only (no “课程” wording) |

---

### Task 1: Delete dead modules and orphan tests

**Files:**
- Delete: `env/route_planner.py`
- Delete: `tests/test_curricula.py`
- Delete: `tests/test_reward_route_progress.py`
- Delete: `tests/test_reward_precision.py`
- Confirm deleted (already absent): `docs/curriculum_training.md`, `scripts/evaluate_ready_counts.py`, `scripts/reproduce_c04_curriculum.py`, `scripts/train_strict_pos_curriculum.py`

**Interfaces:**
- Consumes: none
- Produces: those paths gone from the tree; no remaining Python imports of `RoutePlanner` / `curricula` except the still-present stubs in `scripts/train.py` (removed in Task 2)

- [ ] **Step 1: Confirm nothing imports route_planner except itself**

Run:

```bash
rg -n "route_planner|RoutePlanner" --glob '*.py' .
```

Expected: matches only under `env/route_planner.py` (and possibly comments in docs). If any other `.py` imports it, stop and update that file before deleting.

- [ ] **Step 2: Delete the four dead files**

```bash
rm -f env/route_planner.py \
  tests/test_curricula.py \
  tests/test_reward_route_progress.py \
  tests/test_reward_precision.py
```

Confirm curriculum doc/scripts stay gone:

```bash
test ! -e docs/curriculum_training.md
test ! -e scripts/evaluate_ready_counts.py
test ! -e scripts/reproduce_c04_curriculum.py
test ! -e scripts/train_strict_pos_curriculum.py
```

- [ ] **Step 3: Verify orphan tests are gone from collection**

Run:

```bash
pytest tests/ --collect-only -q 2>&1 | rg -n "curricula|route_progress|precision" || echo "OK: no orphan test names"
```

Expected: `OK: no orphan test names` (no collected tests with those names).

- [ ] **Step 4: Commit (only if user requested commits)**

```bash
git add -u env/route_planner.py tests/test_curricula.py \
  tests/test_reward_route_progress.py tests/test_reward_precision.py \
  docs/curriculum_training.md scripts/evaluate_ready_counts.py \
  scripts/reproduce_c04_curriculum.py scripts/train_strict_pos_curriculum.py
git commit -m "$(cat <<'EOF'
chore: remove curriculum and route-planner dead code

EOF
)"
```

---

### Task 2: Strip `--course` and curricula wiring from `train.py`

**Files:**
- Modify: `scripts/train.py` (import block ~47–54, argparse ~751–752, main course block ~808–818 / 838–839 / 875–881 / 915–917, `_save_ckpt` calls and signature ~1340 / 1363 / 1381 / 1404 / 1431–1432)
- Modify: `config.py` comment on `tug_init_radius_m` (~line 31) — remove “课程” wording

**Interfaces:**
- Consumes: Task 1 deletions (no curricula package)
- Produces: `python scripts/train.py --help` has no `--course`; checkpoint payload never writes `"course"`; `total_steps` comes only from `--total-steps` or `PPOConfig.total_steps`

- [ ] **Step 1: Remove curricula import block**

Delete this entire try/except at the top of `scripts/train.py`:

```python
try:
    from curricula.loader import CourseSpec, apply_course, load_course
    _CURRICULA_AVAILABLE = True
except ImportError:
    _CURRICULA_AVAILABLE = False
    CourseSpec = None  # type: ignore
    apply_course = None  # type: ignore
    load_course = None  # type: ignore
```

- [ ] **Step 2: Remove `--course` argparse and curriculum-flavored help text**

Delete:

```python
    parser.add_argument("--course", type=str, default=None,
                        help="课程 Python 文件路径；文件需导出 COURSE 字典")
```

Change `--hold-time` help from curriculum wording to neutral:

```python
    parser.add_argument("--hold-time", type=float, default=None,
                        help="覆盖 EnvConfig.hold_time_s")
```

- [ ] **Step 3: Remove course_spec / course_metadata from `main()`**

Replace the block that starts with `course_spec = None` through `course_metadata = course_spec.metadata()` with nothing — go straight to `if args.init_mode is not None:`.

Change total_steps selection to:

```python
    if args.total_steps is not None:
        total_steps = int(args.total_steps)
    else:
        total_steps = PPOConfig.total_steps
```

Delete the entire `if course_spec is not None: ... print(...)` startup block.

Delete TensorBoard course text:

```python
    if course_metadata is not None:
        course_text = "\n".join(f"{k} = {v}" for k, v in course_metadata.items())
        writer.add_text("course", course_text.replace("\n", "  \n"), 0)
```

- [ ] **Step 4: Simplify `_save_ckpt`**

Remove `course_metadata=` kwargs from all `_save_ckpt(...)` call sites.

Change signature and body to:

```python
def _save_ckpt(
    path: Path,
    model: MAPPOActorCritic,
    optimizer: torch.optim.Optimizer,
    env_cfg: EnvConfig,
    ppo_cfg: PPOConfig,
    *,
    update: int,
    global_step: int,
    metric: float,
    lr_scheduler: LRScheduler | None = None,
) -> None:
    payload = {
        "algo": "mappo",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_kwargs": {
            "obs_dim": model.obs_dim,
            "action_dim": model.action_dim,
            "n_agents": model.n_agents,
            "global_state_dim": model.global_state_dim,
            "actor_arch": getattr(model, "actor_arch", "mlp"),
            "hist_len": getattr(model, "hist_len", None),
            "observation_spec": model.observation_spec.to_dict(),
            "tf_d_model": getattr(model, "tf_d_model", ppo_cfg.tf_d_model),
            "tf_nhead": getattr(model, "tf_nhead", ppo_cfg.tf_nhead),
            "tf_num_layers": getattr(model, "tf_num_layers", ppo_cfg.tf_num_layers),
            "tf_ffn_dim": getattr(model, "tf_ffn_dim", ppo_cfg.tf_ffn_dim),
            "tf_dropout": getattr(model, "tf_dropout", ppo_cfg.tf_dropout),
        },
        "env_cfg": asdict(env_cfg),
        "ppo_cfg": asdict(ppo_cfg),
        "observation_spec": model.observation_spec.to_dict(),
        "update": update,
        "global_step": global_step,
        "metric": metric,
    }
    if lr_scheduler is not None:
        payload["lr_scheduler"] = lr_scheduler.state_dict()
        payload["lr_scheduler_type"] = type(lr_scheduler).__name__
    torch.save(payload, str(path))
```

- [ ] **Step 5: Neutralize config comment**

In `config.py`, change:

```python
    # 初始化（圆环半径；远距课程可调回 150/200，训练用 --init-radius）
```

to:

```python
    # 初始化（圆环半径；远距复现可用 --init-radius 200）
```

- [ ] **Step 6: Verify CLI and no course symbols**

Run:

```bash
python scripts/train.py --help 2>&1 | rg -n "course|reward-preset|init-radius|arch" 
rg -n "course|curricula|CourseSpec|course_metadata|_CURRICULA" scripts/train.py || echo "OK: train.py clean"
```

Expected:
- help shows `reward-preset`, `init-radius`, `arch`
- help does **not** show `course`
- second command prints `OK: train.py clean`

- [ ] **Step 7: Commit (only if user requested commits)**

```bash
git add scripts/train.py config.py
git commit -m "$(cat <<'EOF'
refactor: remove --course curriculum wiring from train

EOF
)"
```

---

### Task 3: Update architecture, README, task_spec, tensorboard docs

**Files:**
- Modify: `docs/architecture.md`
- Modify: `README.md` (spot-check only; already mostly current)
- Modify: `docs/task_spec.md`
- Modify: `docs/tensorboard_metrics.md`

**Interfaces:**
- Consumes: post–Task 1/2 tree (no route_planner, no --course)
- Produces: living docs that describe Approach→Capture→Track + current modules/tests

- [ ] **Step 1: Rewrite stale sections in `docs/architecture.md`**

Remove the paragraph starting with `**未接线遗留**：` entirely. After the ObservationSpec bullet, optionally add one short factual line (no route_planner filename):

```markdown
接近策略为反应式：slot 相对量 + 船体间隙 + 邻居风险（无外部航路模块）。
```

In §5, delete the entire sentence about `--course`. Optionally mention reward preset:

```markdown
- CLI：`--arch`、`--init-radius`、`--reward-preset`（见 [arch_ablation.md](arch_ablation.md) / [reward_presets.md](reward_presets.md)）
```

In §6 test table, add rows for current tests and **delete** the footnote about historical curriculum/route/precision tests:

```markdown
| 测试 | 覆盖 |
|------|------|
| `tests/test_actor_arch.py` | mlp / transformer 工厂与形状 |
| `tests/test_track_phase.py` | Capture / Track 终止 |
| `tests/test_observation_spec.py` | ObservationSpec 契约 |
| `tests/test_attention.py` | 邻居 Attention |
| `tests/test_init_radius.py` | init 半径配置 |
| `tests/test_reward_presets.py` | `REWARD_PRESETS` / `apply_reward_preset` |
| `tests/test_reward_cpa.py` | CPA 风险项 |
| `tests/test_maneuvers.py` | 动力学操纵性 |
```

Ensure `scripts/` line in the tree also lists `run_reward_preset_ablation.py` / `summarize_reward_presets.py` if you mention script inventory (keep brief).

- [ ] **Step 2: Tighten `docs/task_spec.md` init note**

Replace the init section paragraph that contrasts “旧默认 200 m” with:

```markdown
## 初始化

拖轮默认放在大船周围圆环上：`tug_init_mode=circle`，`tug_init_radius_m=100`。

远距复现：

```bash
python scripts/train.py --arch transformer --run-name tf_r200 --init-radius 200
```
```

Keep default hyperparams section pointing at `config.py`.

- [ ] **Step 3: Update `docs/tensorboard_metrics.md` §7**

Replace the `course` mention so living docs do not describe a current `course` panel:

```markdown
| 面板 | 内容 |
|------|------|
| `hparams` | 本次运行的 `EnvConfig` / `PPOConfig` 全文（含 `reward_preset`） |

旧 run 的 event 文件可能仍含已删除的诊断曲线或历史标签；新训练只写入本文列出的指标。
```

- [ ] **Step 4: Spot-check `README.md`**

Confirm the文档索引 table links only existing files (`task_spec`, `architecture`, `observation_space`, `reward_function`, `reward_presets`, `arch_ablation`, `tensorboard_metrics`) and that commands use `--arch` / `--init-radius` / `--reward-preset` without `--course`. Fix any drift found; do not invent new sections.

Run:

```bash
python - <<'PY'
from pathlib import Path
import re
text = Path("README.md").read_text()
links = re.findall(r"\((docs/[^)]+\.md)\)", text)
missing = [p for p in links if not Path(p).exists()]
print("links:", links)
print("missing:", missing or "none")
assert not missing
PY
```

Expected: `missing: none`.

- [ ] **Step 5: Commit (only if user requested commits)**

```bash
git add docs/architecture.md docs/task_spec.md docs/tensorboard_metrics.md README.md
git commit -m "$(cat <<'EOF'
docs: align architecture and task docs with current path

EOF
)"
```

---

### Task 4: Spot-sync remaining docs and superpowers plan note

**Files:**
- Modify: `docs/observation_space.md` (only if route/waypoint/curriculum drift exists)
- Modify: `docs/reward_function.md` / `docs/reward_presets.md` / `docs/arch_ablation.md` (spot-check)
- Modify: `docs/superpowers/plans/2026-08-06-reward-preset-ablation.md` (order note ~line 257)
- Keep: `docs/superpowers/specs/2026-08-06-reward-preset-ablation-design.md`

**Interfaces:**
- Consumes: `ObservationSpec`, `REWARD_PRESETS`, `FormationRewardComputer` defaults
- Produces: no living-doc references to removed APIs; preset plan order note matches Task 2 CLI

- [ ] **Step 1: Grep living docs for banned terms**

Run:

```bash
rg -n "curriculum|curricula|--course|RoutePlanner|route_planner|reward_precision|route_progress|未接线遗留" \
  README.md docs/*.md
```

Expected: no matches in living docs. (`docs/superpowers/specs/2026-08-06-docs-code-sync-design.md` and this plan may mention them as delete targets — that is OK. Living docs = `README.md` + `docs/*.md` top-level.)

If `observation_space.md` only says “无外部航路观测”, keep that (describes current absense, not a module). Do **not** reintroduce route_planner.

- [ ] **Step 2: Verify reward defaults in `docs/reward_function.md` against `EnvConfig`**

Check these defaults match `config.py` exactly; fix any mismatch:

| Field | Expected |
|-------|----------|
| `reward_dist_w` | 3.0 |
| `reward_hold_w` | 2.0 |
| `reward_velocity_w` | 0.0 |
| `reward_collision_w` | 1.0 |
| `reward_shape_w` | 0.3 |
| `reward_team_w` | 0.2 |
| `reward_arrival_bonus` | 80.0 |
| `reward_collision_pen_culprit` | 80.0 |
| `reward_collision_pen_bystander` | 15.0 |

Confirm doc has **no** `r_precision` / route progress components.

- [ ] **Step 3: Verify `docs/reward_presets.md` preset table matches `REWARD_PRESETS`**

Ids and overrides must match `config.REWARD_PRESETS` keys/values. Commands may keep `scripts/run_reward_preset_ablation.py`.

- [ ] **Step 4: Fix reward-preset plan order note**

In `docs/superpowers/plans/2026-08-06-reward-preset-ablation.md`, replace:

```markdown
Order note: apply **after** course / init-radius / tol overrides so CLI task knobs win for those fields; reward fields only come from preset (CLI does not expose individual reward weights in this task).
```

with:

```markdown
Order note: apply **after** init-radius / tol overrides so CLI task knobs win for those fields; reward fields only come from preset (CLI does not expose individual reward weights in this task).
```

- [ ] **Step 5: Spot-check `docs/arch_ablation.md`**

Confirm matrix still matches: `--arch mlp|transformer`, GRU/LSTM reserved, `obs_history_k=3` default, Transformer slicing narrative consistent with `rl/actor.py` / `rl/temporal.py`. Fix only factual drift.

- [ ] **Step 6: Commit (only if user requested commits)**

```bash
git add docs/observation_space.md docs/reward_function.md docs/reward_presets.md \
  docs/arch_ablation.md docs/superpowers/plans/2026-08-06-reward-preset-ablation.md
git commit -m "$(cat <<'EOF'
docs: remove leftover curriculum/route references

EOF
)"
```

---

### Task 5: Final verification sweep

**Files:**
- None required unless verification finds leftovers

**Interfaces:**
- Consumes: Tasks 1–4 complete tree
- Produces: evidence that success criteria in the design spec hold

- [ ] **Step 1: Banned-term sweep on code + living docs**

Run:

```bash
rg -n "curricula|CourseSpec|apply_course|load_course|--course|RoutePlanner|route_planner|reward_precision|route_progress" \
  --glob '!docs/superpowers/**' \
  --glob '!**/__pycache__/**' \
  .
```

Expected: no matches (or only this plan/spec if not excluded — prefer zero in code, `README.md`, and top-level `docs/*.md`).

Also:

```bash
rg -n "course|curricula|RoutePlanner|route_planner" README.md docs/architecture.md docs/task_spec.md docs/tensorboard_metrics.md docs/reward_function.md docs/reward_presets.md docs/arch_ablation.md || echo "OK: living docs clean"
```

Expected: `OK: living docs clean` (observation_space may still say “无外部航路观测” — that string is allowed; ensure it does not name `route_planner`).

- [ ] **Step 2: Run retained unit tests**

```bash
pytest tests/test_actor_arch.py tests/test_attention.py tests/test_init_radius.py \
  tests/test_observation_spec.py tests/test_reward_cpa.py tests/test_reward_presets.py \
  tests/test_track_phase.py -q
```

Expected: all passed.

- [ ] **Step 3: CLI help sanity**

```bash
python scripts/train.py --help 2>&1 | tee /tmp/train_help.txt
rg -n "^\s+--course\b" /tmp/train_help.txt && exit 1 || echo "OK: no --course"
rg -n "^\s+--arch\b|^\s+--reward-preset\b|^\s+--init-radius\b" /tmp/train_help.txt
```

Expected: `OK: no --course`, and the three flags present.

- [ ] **Step 4: README link existence**

```bash
python - <<'PY'
from pathlib import Path
import re
text = Path("README.md").read_text()
links = re.findall(r"\((docs/[^)]+\.md)\)", text)
missing = [p for p in links if not Path(p).exists()]
assert not missing, missing
print("OK:", len(links), "doc links")
PY
```

- [ ] **Step 5: Commit verification-only fixes if any (only if user requested commits)**

If Steps 1–4 forced small doc/code fixes, commit them:

```bash
git add -u
git commit -m "$(cat <<'EOF'
chore: finish docs-code sync verification fixes

EOF
)"
```

Otherwise no commit.

---

## Self-review (plan vs spec)

| Spec requirement | Task |
|------------------|------|
| Delete curriculum_training.md + curriculum scripts (stay deleted) | Task 1 |
| Delete route_planner + orphan tests | Task 1 |
| Remove --course / curricula from train + checkpoints/TB | Task 2 |
| Update architecture / README / task_spec / TB / other living docs | Tasks 3–4 |
| Keep reward-preset superpowers; fix course order note | Task 4 |
| Verification grep + pytest + --help | Task 5 |
| Out of scope items not scheduled | ✓ |

No TBD/TODO placeholders in task steps. Checkpoint/`_save_ckpt` signature in Task 2 matches call-site cleanup in the same task.
