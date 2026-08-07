# Reward Anti-Orbit-Farming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the farmable distance/stall reward with continuous approach progress, a bounded distance cost, target-local hold reward, and a shared laggard cost while preserving every hard-collision rule.

**Architecture:** The CPU `FormationRewardComputer` remains the reward oracle and the device-resident `FastBatchedStep` implements the tensor-equivalent formula. Per-agent reward depends only on current/previous slot distance plus existing safety geometry; the shared team term is a stable softmax-weighted maximum of per-agent distance cost. Obsolete stall and shaping history is removed after both paths have parity coverage.

**Tech Stack:** Python 3, NumPy, PyTorch, `EnvConfig`, `FormationEnv`, `CudaVecEnv`, pytest, TensorBoard.

## Global Constraints

- Approved spec: `docs/superpowers/specs/2026-08-07-reward-no-orbit-farming-design.md`.
- Do not modify observation, actor, PPO, initialization, Capture, or Track behavior.
- Preserve ship hard collision at `< 6 m` and tug-pair hard collision at `< 20 m`.
- Preserve collision termination and culprit/bystander terminal penalties exactly.
- Preserve existing ship-collision corridor softening and never soften tug-pair risk.
- Keep `reward_dist_w` as the public field name, with new pure-progress semantics.
- Keep `reward_shape_w=0.0` as a compatibility-only field; it must not affect reward computation.
- CPU and GPU component values must match within existing float tolerances.
- The worktree already contains unrelated user changes, including `scripts/train.py`; inspect and preserve them when editing overlapping files.

---

## File Structure

| File | Responsibility |
|---|---|
| `config.py` | New distance-cost weight and reward defaults; remove obsolete stall/shape-detail fields |
| `env/reward.py` | Authoritative NumPy reward formula and component diagnostics |
| `env/gpu/batched_step.py` | Tensor-equivalent reward formula and compact episode state |
| `env/state.py` | CPU episode state containing only history still consumed at runtime |
| `env/formation_env.py` | Initialize/update only the remaining one-step reward history |
| `tests/test_reward_no_orbit_farming.py` | Observable reward ordering, target transition, team laggard, and trajectory regression |
| `tests/test_reward_redesign_smoke.py` | New component contract and finite-value smoke test |
| `tests/parity/test_reward_terminate.py` | CPU/GPU reward-component parity |
| `tests/test_reward_corridor.py` | Existing safety-corridor regression with obsolete setup removed |
| `tests/test_reward_config_redesign.py` | Defaults still relevant to corridor safety |
| `scripts/train.py` | TensorBoard reward keys and reward startup summary |
| `docs/reward_function.md` | Implemented equations, defaults, and diagnostics |
| `docs/tensorboard_metrics.md` | New `p_distance` interpretation and reading order |

---

### Task 1: CPU reward behavior and default parameters

**Files:**
- Create: `tests/test_reward_no_orbit_farming.py`
- Modify: `config.py:53-94`
- Modify: `env/reward.py:1-346`

**Interfaces:**
- Consumes: `MutableEpisodeState.prev_dist`, slot distance, heading error, relative speed, existing collision geometry.
- Produces: `reward_components["r_dist"]`, `reward_components["p_distance"]`, `reward_components["r_hold"]`, and weighted `reward_components["r_team"]`.
- Formula defaults: `reward_dist_w=3.0`, `reward_distance_cost_w=0.2`, `reward_dist_progress_clip_m=1.0`, `reward_dist_scale_m=200.0`, `reward_hold_w=2.0`, `reward_team_w=0.2`.

- [ ] **Step 1: Write the failing CPU behavior tests**

Create `tests/test_reward_no_orbit_farming.py` with real `FormationEnv` state and hand-derived expectations:

```python
"""Reward relationships that prevent static or orbital farming."""

from __future__ import annotations

import math

import numpy as np
import pytest

from config import EnvConfig
from env.formation_env import FormationEnv
from physics.tugboat_dynamics_model import Vec3


def _isolated_env(*, team_w: float = 0.0) -> FormationEnv:
    cfg = EnvConfig()
    cfg.reward_collision_w = 0.0
    cfg.reward_velocity_w = 0.0
    cfg.reward_team_w = team_w
    env = FormationEnv(cfg=cfg, seed=17)
    env.reset()
    env.ship.x = 0.0
    env.ship.y = 0.0
    env.ship.psi = 0.0
    env.ship.u = 0.0
    env.ship.v = 0.0
    env.ship.r = 0.0
    return env


def _place_at_slot_distance(env: FormationEnv, tug_idx: int, distance_m: float) -> None:
    slots = env.ship.slot_positions_world()
    slot = slots[env.tug_to_slot[tug_idx]]
    angle = math.atan2(float(slot[1]) - env.ship.y, float(slot[0]) - env.ship.x)
    env.tugs[tug_idx].eta = Vec3(
        float(slot[0]) + distance_m * math.cos(angle),
        float(slot[1]) + distance_m * math.sin(angle),
        float(slot[2]),
    )
    env.tugs[tug_idx].nu = Vec3.zero()


def _components(env: FormationEnv, previous: list[float]) -> dict:
    env._episode.prev_dist[:] = np.asarray(previous, dtype=np.float32)
    actions = np.zeros((env.cfg.n_tugs, 4), dtype=np.float32)
    _, info = env._compute_rewards(actions)
    return info["reward_components"]


@pytest.mark.parametrize(
    ("distance_m", "expected_cost", "expected_total"),
    [(200.0, 1.0, -0.2), (100.0, 0.5, -0.1), (25.0, 0.125, -0.025)],
)
def test_static_outside_target_is_negative(
    distance_m: float, expected_cost: float, expected_total: float
) -> None:
    env = _isolated_env()
    for i, distance in enumerate((distance_m, 400.0, 450.0, 500.0)):
        _place_at_slot_distance(env, i, distance)
    comp = _components(env, [distance_m, 400.0, 450.0, 500.0])
    assert float(comp["r_dist"][0]) == pytest.approx(0.0)
    assert float(comp["p_distance"][0]) == pytest.approx(expected_cost)
    assert float(comp["r_total"][0]) == pytest.approx(expected_total, abs=1e-6)


def test_approach_beats_static_and_retreat() -> None:
    env = _isolated_env()
    for i, distance in enumerate((100.0, 400.0, 450.0, 500.0)):
        _place_at_slot_distance(env, i, distance)
    static = _components(env, [100.0, 400.0, 450.0, 500.0])
    approach = _components(env, [101.0, 400.0, 450.0, 500.0])
    retreat = _components(env, [99.0, 400.0, 450.0, 500.0])
    assert float(approach["r_dist"][0]) == pytest.approx(1.0)
    assert float(retreat["r_dist"][0]) == pytest.approx(-1.0)
    assert float(approach["r_total"][0]) > float(static["r_total"][0])
    assert float(static["r_total"][0]) > float(retreat["r_total"][0])


def test_progress_does_not_drop_at_old_150m_boundary() -> None:
    env = _isolated_env()
    values = []
    for distance_m in (149.0, 151.0):
        _place_at_slot_distance(env, 0, distance_m)
        for i, distance in enumerate((400.0, 450.0, 500.0), start=1):
            _place_at_slot_distance(env, i, distance)
        comp = _components(env, [distance_m + 1.0, 400.0, 450.0, 500.0])
        values.append(float(comp["r_dist"][0]))
    assert values == pytest.approx([1.0, 1.0])


def test_target_center_hold_is_positive() -> None:
    env = _isolated_env()
    for i, distance in enumerate((0.0, 400.0, 450.0, 500.0)):
        _place_at_slot_distance(env, i, distance)
    comp = _components(env, [0.0, 400.0, 450.0, 500.0])
    assert float(comp["p_distance"][0]) == pytest.approx(0.0)
    assert float(comp["r_hold"][0]) == pytest.approx(1.0)
    assert float(comp["r_total"][0]) == pytest.approx(2.0)


def test_team_cost_tracks_the_lagging_tug() -> None:
    env = _isolated_env(team_w=0.2)
    for i, distance in enumerate((20.0, 20.0, 20.0, 200.0)):
        _place_at_slot_distance(env, i, distance)
    lagging = _components(env, [20.0, 20.0, 20.0, 200.0])
    for i, distance in enumerate((20.0, 20.0, 20.0, 50.0)):
        _place_at_slot_distance(env, i, distance)
    recovered = _components(env, [20.0, 20.0, 20.0, 50.0])
    assert np.all(np.asarray(lagging["r_team"]) < 0.0)
    assert np.ptp(np.asarray(lagging["r_team"])) == pytest.approx(0.0)
    assert abs(float(recovered["r_team"][0])) < abs(float(lagging["r_team"][0]))


def test_approach_and_hold_trajectory_beats_longer_orbit() -> None:
    env = _isolated_env()
    parked = [400.0, 450.0, 500.0]

    def reward_at(distance_m: float, previous_m: float) -> float:
        _place_at_slot_distance(env, 0, distance_m)
        for i, distance in enumerate(parked, start=1):
            _place_at_slot_distance(env, i, distance)
        return float(_components(env, [previous_m, *parked])["r_total"][0])

    orbit_return = sum(reward_at(100.0, 100.0) for _ in range(80))
    distances = list(np.linspace(200.0, 0.0, 51))
    approach_return = sum(
        reward_at(distance, previous)
        for previous, distance in zip(distances, distances[1:])
    )
    approach_return += sum(reward_at(0.0, 0.0) for _ in range(10))
    assert orbit_return < 0.0
    assert approach_return > orbit_return
```

Mutation caught: reintroducing `dist_bonus`, gating progress at 150 m, omitting distance cost, making retreat unsigned, removing target hold, or reverting the team term makes at least one test fail.

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
pytest tests/test_reward_no_orbit_farming.py -v
```

Expected: FAIL because `p_distance` does not exist and current static `r_dist` is positive.

- [ ] **Step 3: Add the new defaults without deleting compatibility fields yet**

In `config.py`, change the reward block to these effective defaults while leaving obsolete fields temporarily available until Task 3 removes all consumers:

```python
# R = w_dist*R_progress - w_distance*C_distance + w_hold*R_hold
#     + w_vel*R_vel - w_coll*P_coll + R_team
reward_dist_w: float = 3.0
reward_distance_cost_w: float = 0.2
reward_hold_w: float = 2.0
reward_velocity_w: float = 0.0
reward_collision_w: float = 1.0
reward_collision_cap: float = 2.0

# 距离/目标切换
reward_dist_progress_clip_m: float = 1.0
reward_dist_scale_m: float = 200.0
reward_hold_start_m: float = 150.0  # collision-corridor range only

# Compatibility-only switch; legacy shaping is no longer computed.
reward_shape_w: float = 0.0
```

- [ ] **Step 4: Replace the CPU distance, hold, total, and team formula**

In `env/reward.py`, preserve all collision helper methods and the current collision block. Replace the old gate/distance/stall/shaping/team logic with:

```python
distance_cost_w = max(float(getattr(cfg, "reward_distance_cost_w", 0.2)), 0.0)
progress_clip = max(float(getattr(cfg, "reward_dist_progress_clip_m", 1.0)), 1e-6)
distance_scale = max(float(getattr(cfg, "reward_dist_scale_m", 200.0)), 1e-6)
target_tol = max(float(cfg.pos_tol_m), 1e-6)

# Per tug, after d/dpsi/speed_err are known:
target_x = float(np.clip(1.0 - d / target_tol, 0.0, 1.0))
target_gate = target_x * target_x * (3.0 - 2.0 * target_x)
progress = float(np.clip((float(episode.prev_dist[i]) - d) / progress_clip, -1.0, 1.0))
r_dist = (1.0 - target_gate) * progress
p_distance = (1.0 - target_gate) * float(np.clip(d / distance_scale, 0.0, 1.0))
head_score = max(0.0, 1.0 - abs(dpsi) / max(cfg.heading_tol_rad, 1e-6))
speed_score = max(0.0, 1.0 - speed_err / max(cfg.speed_tol_ms, 1e-6))
r_hold = target_gate * (0.5 + 0.25 * head_score + 0.25 * speed_score)
r_vel = -target_gate * (0.8 * speed_pen + 0.2 * yaw_pen)

r_total = (
    w_dist * r_dist
    - distance_cost_w * p_distance
    + w_hold * r_hold
    + w_vel * r_vel
    - w_coll * p_coll
)
```

Initialize and fill `comp["p_distance"]`; remove `r_shape`, `p_stall`, `stall_scale`, and reward `hold_gate` from the component dictionary. Store per-agent `p_distance` in a NumPy array for the shared team calculation.

After the per-agent loop, replace the old positive softmin with the stable weighted soft maximum:

```python
if w_team > 0.0:
    beta = max(float(getattr(cfg, "reward_team_softmin_beta", 4.0)), 1e-6)
    logits = beta * np.asarray(comp["p_distance"], dtype=np.float64)
    weights = np.exp(logits - float(np.max(logits)))
    team_cost = float(np.dot(weights, comp["p_distance"]) / max(float(weights.sum()), 1e-12))
    team_reward = np.float32(-w_team * team_cost)
    rewards += team_reward
    comp["r_total"] += team_reward
    comp["r_team"][:] = team_reward
```

- [ ] **Step 5: Run CPU reward and unchanged safety tests**

Run:

```bash
pytest tests/test_reward_no_orbit_farming.py tests/test_reward_corridor.py tests/test_reward_cpa.py -v
```

Expected: PASS. If a hand-derived total differs, inspect active collision/team configuration; do not weaken the ordering assertion.

- [ ] **Step 6: Commit the CPU behavior**

```bash
git add env/reward.py tests/test_reward_no_orbit_farming.py
git add -p config.py
git diff --cached --check
git diff --cached --name-only
git -c user.name='jameskerry651' -c user.email='jameskerry651@gmail.com' \
  commit -m "feat(reward): replace farmable distance reward"
```

At `git add -p config.py`, stage only the `EnvConfig` reward hunk; reject the
pre-existing PPO and Transformer-preset hunks.

---

### Task 2: GPU formula and CPU/GPU parity

**Files:**
- Modify: `tests/parity/test_reward_terminate.py:79-88`
- Modify: `env/gpu/batched_step.py:235-367`

**Interfaces:**
- Consumes: tensor `episode.prev_dist` and the same geometry/config values as the CPU path.
- Produces: tensor components with the same names, shapes, weighting conventions, and numerical semantics as `FormationRewardComputer`.

- [ ] **Step 1: Change parity coverage to the new component contract**

Update the component loop in `test_reward_and_phase_match_after_gpu_dynamics`:

```python
for key in (
    "r_total",
    "r_dist",
    "p_distance",
    "r_hold",
    "r_team",
    "p_collision",
    "corridor_gate",
    "ship_soft_scale",
):
    np.testing.assert_allclose(
        info_g["reward_components"][key],
        info_c["reward_components"][key],
        rtol=1e-5,
        atol=1e-5,
        err_msg=key,
    )
```

In `test_fast_batched_tug_cpa_matches_cpu`, compare the new fast-path components as well as the final total:

```python
for key in ("r_dist", "p_distance", "r_hold", "r_team"):
    np.testing.assert_allclose(
        info_fast[0]["reward_components"][key],
        info_cpu["reward_components"][key],
        rtol=1e-4,
        atol=1e-4,
        err_msg=f"{key}@{step}",
    )
```

- [ ] **Step 2: Run the parity tests and verify RED**

Run:

```bash
pytest tests/parity/test_reward_terminate.py -v
```

Expected: FAIL because the fast batched path still emits the old stall/shaping formula and lacks `p_distance`.

- [ ] **Step 3: Implement the tensor-equivalent per-agent formula**

In `FastBatchedStep.compute_rewards_batched`, replace the old hold gate, `dist_bonus`, stall, and shaping blocks with:

```python
target_tol = max(float(cfg.pos_tol_m), 1e-6)
target_x = (1.0 - dist / target_tol).clamp(0.0, 1.0)
target_gate = target_x.square() * (3.0 - 2.0 * target_x)
head_score = (1.0 - dpsi.abs() / max(float(cfg.heading_tol_rad), 1e-6)).clamp_min(0.0)
speed_score = (1.0 - speed_err / max(float(cfg.speed_tol_ms), 1e-6)).clamp_min(0.0)
r_hold = target_gate * (0.5 + 0.25 * head_score + 0.25 * speed_score)
in_zone = (dist < cfg.pos_tol_m) & (dpsi.abs() < cfg.heading_tol_rad) & (speed_err < cfg.speed_tol_ms)

progress = ((ep.prev_dist - dist) / max(float(cfg.reward_dist_progress_clip_m), 1e-6)).clamp(-1.0, 1.0)
r_dist = (1.0 - target_gate) * progress
p_distance = (1.0 - target_gate) * (
    dist / max(float(cfg.reward_dist_scale_m), 1e-6)
).clamp(0.0, 1.0)
```

Use `target_gate` for `r_vel`. Keep `hold_start = max(float(cfg.reward_hold_start_m), 1e-6)` only for the unchanged corridor condition.

Build the total without stall or shaping:

```python
rewards = (
    float(cfg.reward_dist_w) * r_dist
    - max(float(cfg.reward_distance_cost_w), 0.0) * p_distance
    + float(cfg.reward_hold_w) * r_hold
    + float(cfg.reward_velocity_w) * r_vel
    - float(cfg.reward_collision_w) * p_coll
)
rewards = rewards.to(torch.float32).to(self.dtype)
```

- [ ] **Step 4: Implement stable tensor team cost and diagnostics**

```python
r_team = torch.zeros_like(rewards)
if cfg.reward_team_w > 0.0:
    beta = max(float(cfg.reward_team_softmin_beta), 1e-6)
    logits = beta * p_distance
    weights = torch.softmax(logits, dim=1)
    team_cost = (weights * p_distance).sum(dim=1, keepdim=True)
    r_team = (-float(cfg.reward_team_w) * team_cost).to(torch.float32).to(self.dtype)
    rewards = rewards + r_team

components = {
    "r_total": rewards,
    "r_dist": r_dist,
    "p_distance": p_distance,
    "r_hold": r_hold,
    "r_velocity": r_vel,
    "r_team": r_team.expand(-1, self.n_tugs),
    "p_collision": p_coll,
    "p_ship_collision": p_ship,
    "p_tug_collision": p_tug,
    "dist_to_slot": dist,
    "heading_err_deg": dpsi.abs() * (180.0 / math.pi),
    "speed_err": speed_err,
    "hull_dist": d["hull_dist"],
    "in_zone": in_zone,
    "corridor_gate": corridor,
    "ship_soft_scale": soft,
}
```

Ensure the CPU `r_team` array and GPU expanded view both have shape `(n_tugs,)` per environment.

- [ ] **Step 5: Run parity and rollout tests**

Run:

```bash
pytest tests/parity/test_reward_terminate.py tests/parity/test_rollout_match.py -v
```

Expected: PASS on CPU float32/float64 paths; CUDA-marked cases skip only when CUDA is unavailable.

- [ ] **Step 6: Commit GPU parity**

```bash
git add env/gpu/batched_step.py tests/parity/test_reward_terminate.py
git -c user.name='jameskerry651' -c user.email='jameskerry651@gmail.com' \
  commit -m "feat(reward): match anti-farming reward on gpu"
```

---

### Task 3: Remove obsolete stall and shaping state

**Files:**
- Modify: `config.py:53-94`
- Modify: `env/state.py:270-295`
- Modify: `env/formation_env.py:122-166,229-247,284-295,362-375`
- Modify: `env/gpu/batched_step.py:40-125,430-455`
- Modify: `tests/test_reward_redesign_smoke.py`
- Modify: `tests/test_reward_corridor.py:15-21`
- Modify: `tests/test_reward_config_redesign.py`
- Delete: `tests/test_reward_stall.py`

**Interfaces:**
- Keeps: `prev_dist` as the only reward history required by the new formula.
- Removes: `prev_d_hull`, `prev_speed_err`, `prev_heading_err`, `dist_hist`, `dist_hist_head`, and `dist_hist_filled` from CPU and GPU episode state.

- [ ] **Step 1: Update the smoke contract while the suite is green**

Replace `REQUIRED` in `tests/test_reward_redesign_smoke.py` with:

```python
REQUIRED = {
    "r_total",
    "r_dist",
    "p_distance",
    "r_hold",
    "r_team",
    "p_collision",
    "corridor_gate",
    "ship_soft_scale",
}
```

Remove obsolete `reward_stall_w` and `reward_shape_w` setup from `tests/test_reward_corridor.py`. Replace `tests/test_reward_config_redesign.py` with the corridor/safety contract below; reward behavior tests, not source-field assertions, own the new distance defaults:

```python
"""Defaults that define the collision approach corridor."""

from config import EnvConfig


def test_reward_corridor_safety_defaults() -> None:
    cfg = EnvConfig()
    assert cfg.reward_corridor_half_width_m == 40.0
    assert cfg.reward_corridor_axial_slack_m == 30.0
    assert cfg.reward_ship_soft_min_scale == 0.15
    assert cfg.reward_collision_ship_safe_m == 80.0
```

- [ ] **Step 2: Run the focused suite before refactoring**

Run:

```bash
pytest tests/test_reward_no_orbit_farming.py tests/test_reward_redesign_smoke.py tests/test_reward_corridor.py tests/parity/test_reward_terminate.py -v
```

Expected: PASS. This is the green baseline for state cleanup.

- [ ] **Step 3: Remove CPU obsolete fields and ring-buffer methods**

Reduce `MutableEpisodeState` reward history to:

```python
in_zone_steps: np.ndarray
prev_dist: np.ndarray
```

Keep the existing Capture/Track fields below them. In `FormationEnv`:

- Construct `MutableEpisodeState` with only `in_zone_steps`, `prev_dist`, and Capture/Track arguments.
- Delete `_dist_hist_cap`, `_reset_dist_hist`, and `_push_dist_hist`.
- Delete reset-time history clearing.
- At initialization and after each step, update only `self._episode.prev_dist[i] = d_now` from the reward-history group.
- Delete `_tug_track_errors` only after `rg` confirms it has no observation/termination consumer.

- [ ] **Step 4: Remove GPU obsolete fields and updates**

Delete the six obsolete tensors from `BatchedEpisodeState`, its constructor, `reset_from_env`, and `update_histories`. Keep:

```python
ep.prev_dist.copy_(derived["dist"].to(torch.float32))
```

and remove the ring-buffer scatter block. Do not change motion/action observation histories.

- [ ] **Step 5: Remove obsolete configuration and test file**

Delete these effective configuration fields:

```text
reward_stall_w
reward_dist_progress_frac
reward_hold_full_m
reward_stall_window_s
reward_stall_min_progress_m
reward_stall_floor
reward_shape_gamma
reward_shape_d_ref_m
reward_shape_clip
```

Keep `reward_shape_w: float = 0.0` with a compatibility-only comment. Delete `tests/test_reward_stall.py`, whose behavior is superseded by direct static/orbit negative-reward tests.

- [ ] **Step 6: Verify cleanup did not change reward behavior**

Run:

```bash
pytest tests/test_reward_no_orbit_farming.py tests/test_reward_redesign_smoke.py tests/test_reward_corridor.py tests/test_reward_cpa.py tests/parity/test_reward_terminate.py tests/parity/test_rollout_match.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit state cleanup**

```bash
git add env/state.py env/formation_env.py env/gpu/batched_step.py tests/test_reward_redesign_smoke.py tests/test_reward_corridor.py tests/test_reward_config_redesign.py
git add -p config.py
git rm tests/test_reward_stall.py
git diff --cached --check
git -c user.name='jameskerry651' -c user.email='jameskerry651@gmail.com' \
  commit -m "refactor(reward): remove obsolete stall history"
```

Again stage only the reward-configuration hunk from `config.py`.

---

### Task 4: Training diagnostics and documentation

**Files:**
- Modify: `scripts/train.py:91-92,1193-1202`
- Modify: `docs/reward_function.md`
- Modify: `docs/tensorboard_metrics.md:70-90`

**Interfaces:**
- Produces TensorBoard scalars: `reward/r_dist`, `reward/p_distance`, `reward/r_hold`, `reward/p_collision`.
- Startup summary reports `dist_w`, `distance_cost_w`, `dist_scale_m`, and unchanged collision settings.

- [ ] **Step 1: Update training diagnostics without overwriting existing user edits**

Change only the reward-key tuple and reward print fields in `scripts/train.py`:

```python
_TB_REWARD_KEYS = ("r_dist", "p_distance", "r_hold", "p_collision")
```

```python
print(
    f"[reward] dist_w={env_cfg.reward_dist_w}, "
    f"distance_cost_w={env_cfg.reward_distance_cost_w}, "
    f"dist_scale_m={env_cfg.reward_dist_scale_m}, "
    f"ship_safe_m={env_cfg.reward_collision_ship_safe_m}, "
    f"coll_w={env_cfg.reward_collision_w}, "
    f"cpa_w={env_cfg.reward_collision_cpa_w}"
)
```

Because `scripts/train.py` is already modified in the worktree, inspect `git diff -- scripts/train.py` before and after applying this narrow patch.

- [ ] **Step 2: Rewrite the reward documentation to the implemented formula**

Replace the dense-reward sections of `docs/reward_function.md` with these exact formulas and retain the existing detailed collision/corridor and terminal tables unchanged:

```markdown
## 1. 稠密奖励

\[
R_i = 3R_{\mathrm{progress},i}
-0.2C_{\mathrm{distance},i}
+2R_{\mathrm{hold},i}
-P_{\mathrm{collision},i}
+w_vR_{\mathrm{velocity},i}
+R_{\mathrm{team}}.
\]

## 2. 目标门控、接近与距离代价

\[
x_i=\operatorname{clip}(1-d_i/\texttt{pos_tol_m},0,1),
\qquad g_i=x_i^2(3-2x_i)
\]

\[
R_{\mathrm{progress},i}=(1-g_i)\operatorname{clip}(
(d_{i,t-1}-d_{i,t})/1\mathrm m,-1,1)
\]

\[
C_{\mathrm{distance},i}=(1-g_i)\operatorname{clip}(
d_i/200\mathrm m,0,1)
\]

## 3. 目标保持

\[
R_{\mathrm{hold},i}=g_i(0.5+0.25s_{\mathrm{heading},i}
+0.25s_{\mathrm{speed},i})
\]

## 4. 团队最落后代价

\[
R_{\mathrm{team}}=-w_t
\frac{\sum_i c_i e^{\beta c_i}}{\sum_i e^{\beta c_i}},
\qquad c_i=C_{\mathrm{distance},i}
\]
```

Explicitly state below the formulas: progress stays active outside 10 m; static/orbital motion has zero progress but nonzero distance cost; `reward_hold_start_m=150` controls only corridor softening; no stall window or potential shaping participates in the total. Replace the diagnostics table with the exact component set from Task 3's `REQUIRED` contract plus the retained geometric fields.

- [ ] **Step 3: Update TensorBoard interpretation**

Replace the reward table rows with:

```markdown
| `r_dist` | 单步有符号接近进度（乘权重前） | 接近为正、等距为 0、远离为负；10 m 内平滑退出 |
| `p_distance` | 非负目标距离代价（乘权重前） | 外围应较高，接近目标持续下降；静止仍会产生代价 |
| `r_hold` | 10 m 目标区内的位置/航向/速度保持分 | Capture/Track 阶段应上升并维持 |
| `p_collision` | 势垒与 CPA 碰撞风险；船项可走廊软化 | 越小越好；结合 collision rate 判断安全性 |
```

Update the reading order to use `reward/p_distance` instead of `reward/p_stall`.

- [ ] **Step 4: Verify import, CLI, tests, and whitespace**

Run:

```bash
python -m py_compile config.py env/reward.py env/state.py env/formation_env.py env/gpu/batched_step.py scripts/train.py
python scripts/train.py --help >/dev/null
pytest tests/test_reward_no_orbit_farming.py tests/test_reward_redesign_smoke.py tests/parity/test_reward_terminate.py -v
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 5: Commit diagnostics and docs**

```bash
git add docs/reward_function.md docs/tensorboard_metrics.md
git add -p scripts/train.py
git diff --cached --check
git diff --cached -- scripts/train.py
git -c user.name='jameskerry651' -c user.email='jameskerry651@gmail.com' \
  commit -m "docs(reward): document anti-farming diagnostics"
```

Stage only `_TB_REWARD_KEYS` and the reward startup-summary hunk from
`scripts/train.py`; reject every pre-existing eval/backend/throughput hunk.

---

### Task 5: Full verification and behavioral smoke training

**Files:**
- No source files expected.
- Runtime outputs only: `runs/reward_no_orbit_smoke/`, `checkpoints/reward_no_orbit_smoke/`, and a captured training log under `training_logs/` if requested by the user.

**Interfaces:**
- Verifies all tests, CPU/GPU parity, and fixed-seed learning metrics against the approved behavioral thresholds.

- [ ] **Step 1: Run the complete test suite**

```bash
pytest -q
```

Expected: PASS, with only environment-dependent CUDA skips.

- [ ] **Step 2: Run static analysis and inspect the final diff**

```bash
python -m py_compile config.py env/*.py env/gpu/*.py scripts/train.py
git diff --check
git status --short
git diff --stat HEAD~4..HEAD
```

Expected: no syntax/whitespace errors; unrelated pre-existing worktree edits remain uncommitted and are not included in reward commits.

- [ ] **Step 3: Run a fixed-seed 5M-step smoke training**

Use a new run name and the existing fast CUDA/eval paths:

```bash
python -u scripts/train.py \
  --arch transformer \
  --tf-size S \
  --init-radius 120 \
  --slot-assignment minimax \
  --run-name reward_no_orbit_smoke \
  --total-steps 5000000 \
  --seed 42 \
  --device cuda \
  --env-backend cuda \
  --eval-backend cuda \
  --num-envs 256 \
  --rollout-steps 64 \
  --minibatch-size 8192 \
  --eval-workers 32
```

If CUDA is unavailable, report the environmental limitation; do not substitute incomparable CPU throughput results for the behavioral gate.

- [ ] **Step 4: Evaluate the behavioral gate**

From the fixed-seed evaluation lines, record:

- best and final `capture_rate`;
- best and final `final_dist_mean`;
- best and final `collision_rate`;
- whether `reward/p_distance` trends down as `reward/r_hold` rises.

Acceptance:

```text
capture_rate > 0
final_dist_mean < 200 m
collision_rate <= matching old-reward baseline + 10 percentage points
```

The existing logs establish the failure signature (`capture_rate=0`, typically 200–350 m), but compare collision rate only against a run with matching training/evaluation settings.

- [ ] **Step 5: Stop on failed behavior and return to diagnosis**

If any behavioral condition fails, do not stack weight changes. Inspect `r_dist`, `p_distance`, `r_hold`, `p_collision`, distance, and collision trajectories; state one root-cause hypothesis and add a failing test before any follow-up implementation.

- [ ] **Step 6: Record a clean final status**

Report test commands and results, smoke-training metrics, commit hashes, and any pre-existing unrelated worktree changes. Do not commit generated checkpoints, runs, logs, or TensorBoard events unless the user explicitly requests it.
