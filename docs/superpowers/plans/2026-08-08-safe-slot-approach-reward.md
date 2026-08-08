# Safe-Slot Approach Reward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现风险门控接近、走廊安全项 `R_safe`、加重碰撞/终端默认，并保持 CPU/GPU 一致，以消除冲撞式靠近。

**Architecture:** 在 `EnvConfig` 增加新字段并改默认；`FormationRewardComputer` 先算 `P_ship_raw` 再软化，对正 `R_progress` 乘 `(1-ρ)`，并加入 `R_safe`；`BatchedFormationKernels.compute_rewards_batched` 镜像同一公式；parity 与行为单测锁定；文档与 TB 诊断键同步。5M 验证为实现完成后的实验步骤，不阻塞代码合并。

**Tech Stack:** Python 3、NumPy、PyTorch CUDA batched env、pytest、现有 `FormationEnv` / parity 测试。

## Global Constraints

- 规格：`docs/superpowers/specs/2026-08-08-safe-slot-approach-reward-design.md`。
- 硬碰撞阈值保持：`ship_collision_dist_m=6`、`tug_collision_dist_m=20`。
- 不改 observation / actor / PPO / init / Capture-Track 几何与时间阈值。
- 仅当 `R_progress > 0` 时乘风险门控；`R_progress ≤ 0` 不乘。
- `P_ship_raw` 必须在乘 `ship_soft_scale` **之前**用于 `ρ`。
- `s_approach` 方向向量用**船心→拖轮的反方向**（拖轮指向船心）实现 YAGNI：`u = normalize(ship_xy - tug_xy)` 的单位向量，闭合速度为 `max(0, -v_rel · u)` 的对偶——与 spec「指向船体」一致时用 `u = normalize(tug - ship)` 则闭合为 `max(0, -v_rel · u)`；本计划固定：`u_hat = (ship_x - tug_x, ship_y - tug_y)` 归一化（从拖轮指向船心），`closing = max(0, v_rel · u_hat)`，`s_approach = clip(1 - closing/v_ref, 0, 1)`。
- CPU/GPU 分量键至少含：`r_safe`、`progress_risk`，且 `r_dist` 为门控后值。

---

## File Structure

| 文件 | 职责 |
|------|------|
| `config.py` | 新字段与默认值变更 |
| `env/reward.py` | CPU 奖励：风险门控、`R_safe`、诊断 |
| `env/gpu/batched_step.py` | GPU 镜像公式 |
| `scripts/train.py` | TB keys + 启动摘要 |
| `tests/test_safe_slot_reward.py` | 行为单测 |
| `tests/test_reward_redesign_smoke.py` | 更新默认断言与必需键 |
| `tests/test_reward_no_orbit_farming.py` | 隔离门控以免旧测误伤 |
| `tests/parity/test_reward_terminate.py`（或现有 parity） | CPU/GPU 对齐 |
| `docs/reward_function.md` | 公式文档 |

---

### Task 1: EnvConfig 默认与字段

**Files:**
- Modify: `config.py`（奖励段）
- Modify: `tests/test_reward_redesign_smoke.py`

**Interfaces:**
- Produces: `EnvConfig` 含  
  `reward_safe_w: float = 2.0`  
  `reward_progress_risk_gate: float = 0.5`  
  `reward_safe_closing_speed_mps: float = 1.0`  
  以及更新后的  
  `reward_collision_w=2.0`、`reward_collision_cap=4.0`、`reward_ship_soft_min_scale=0.70`、`reward_hold_w=3.0`、`reward_arrival_bonus=120.0`、`reward_collision_pen_culprit=160.0`、`reward_collision_pen_bystander=30.0`、`reward_collision_pen=160.0`（与 culprit 对齐）

- [ ] **Step 1: 更新 smoke 测试为失败态**

在 `tests/test_reward_redesign_smoke.py`：

```python
REQUIRED = {
    "r_total",
    "r_dist",
    "p_distance",
    "r_hold",
    "r_safe",
    "r_team",
    "p_collision",
    "progress_risk",
    "corridor_gate",
    "ship_soft_scale",
}

# ... existing loop ...

    assert env.cfg.ship_collision_dist_m == 6.0
    assert env.cfg.reward_arrival_bonus == 120.0
    assert env.cfg.reward_collision_w == 2.0
    assert env.cfg.reward_collision_cap == 4.0
    assert env.cfg.reward_ship_soft_min_scale == 0.70
    assert env.cfg.reward_safe_w == 2.0
    assert env.cfg.reward_progress_risk_gate == 0.5
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH=. pytest tests/test_reward_redesign_smoke.py -v`  
Expected: FAIL（缺字段或默认仍为旧值）

- [ ] **Step 3: 改 `config.py`**

将奖励段改为（保留注释风格）：

```python
    # ---------- 奖励 ----------
    # R = w_p*R_progress_gated - w_d*C_distance + w_s*R_safe + w_h*R_hold
    #     + w_vel*R_vel - w_c*P_coll + R_team
    reward_dist_w: float = 3.0
    reward_distance_cost_w: float = 0.2
    reward_safe_w: float = 2.0
    reward_hold_w: float = 3.0
    reward_velocity_w: float = 0.0
    reward_collision_w: float = 2.0
    reward_collision_cap: float = 4.0

    reward_dist_progress_clip_m: float = 1.0
    reward_dist_scale_m: float = 200.0
    reward_hold_start_m: float = 150.0
    reward_progress_risk_gate: float = 0.5
    reward_safe_closing_speed_mps: float = 1.0

    reward_collision_ship_safe_m: float = 80.0
    reward_collision_tug_safe_m: float = 120.0
    reward_cpa_horizon_s: float = 60.0
    reward_collision_cpa_w: float = 2.0

    reward_corridor_half_width_m: float = 40.0
    reward_corridor_axial_slack_m: float = 30.0
    reward_ship_soft_min_scale: float = 0.70

    reward_shape_w: float = 0.0

    reward_team_w: float = 0.2
    reward_team_softmin_beta: float = 4.0

    reward_arrival_bonus: float = 120.0
    reward_collision_pen: float = 160.0
    reward_collision_pen_culprit: float = 160.0
    reward_collision_pen_bystander: float = 30.0
```

- [ ] **Step 4: 跑 smoke（此时仍缺 `r_safe` 键 → 允许先只断言 cfg，或暂时从 REQUIRED 去掉 r_safe/progress_risk 直到 Task 2/3）**

推荐：本任务 REQUIRED **暂不**含 `r_safe`/`progress_risk`，只断言 cfg 默认；Task 3 再加回键。

Run: `PYTHONPATH=. pytest tests/test_reward_redesign_smoke.py -v`  
Expected: PASS（若只改 cfg 断言）

- [ ] **Step 5: Commit**

```bash
git add config.py tests/test_reward_redesign_smoke.py
git commit -m "$(cat <<'EOF'
feat(reward): update safe-slot EnvConfig defaults

EOF
)"
```

---

### Task 2: CPU 风险门控接近 + 诊断 `progress_risk`

**Files:**
- Modify: `env/reward.py`
- Create: `tests/test_safe_slot_reward.py`（本任务先写风险门控测例）
- Modify: `tests/test_reward_no_orbit_farming.py`（隔离门控）

**Interfaces:**
- Consumes: `reward_progress_risk_gate`
- Produces: `comp["r_dist"]` = 门控后 progress；`comp["progress_risk"]` = `ρ`

- [ ] **Step 1: 隔离旧测**

在 `tests/test_reward_no_orbit_farming.py` 的 `_isolated_env` 中增加：

```python
    cfg.reward_progress_risk_gate = 1e9  # ρ ≈ 0，保留无碰撞隔离语义
    cfg.reward_safe_w = 0.0
```

- [ ] **Step 2: 写失败测试**

Create `tests/test_safe_slot_reward.py`:

```python
"""Safe-slot approach reward: risk-gated progress and R_safe."""

from __future__ import annotations

import math

import numpy as np
import pytest

from config import EnvConfig
from env.formation_env import FormationEnv
from physics.tugboat_dynamics_model import Vec3


def _env(**overrides) -> FormationEnv:
    cfg = EnvConfig()
    cfg.reward_collision_w = 0.0
    cfg.reward_velocity_w = 0.0
    cfg.reward_team_w = 0.0
    cfg.reward_safe_w = 0.0
    for k, v in overrides.items():
        setattr(cfg, k, v)
    env = FormationEnv(cfg=cfg, seed=3)
    env.reset()
    env.ship.x = 0.0
    env.ship.y = 0.0
    env.ship.psi = 0.0
    env.ship.u = 0.0
    env.ship.v = 0.0
    env.ship.r = 0.0
    return env


def _place(env: FormationEnv, i: int, distance_m: float) -> None:
    slots = env.ship.slot_positions_world()
    slot = slots[env.tug_to_slot[i]]
    angle = math.atan2(float(slot[1]) - env.ship.y, float(slot[0]) - env.ship.x)
    env.tugs[i].eta = Vec3(
        float(slot[0]) + distance_m * math.cos(angle),
        float(slot[1]) + distance_m * math.sin(angle),
        float(slot[2]),
    )
    env.tugs[i].nu = Vec3.zero()


def _comp(env: FormationEnv, prev: list[float]) -> dict:
    env._episode.prev_dist[:] = np.asarray(prev, dtype=np.float32)
    actions = np.zeros((env.cfg.n_tugs, 4), dtype=np.float32)
    _, info = env._compute_rewards(actions)
    return info["reward_components"]


def test_positive_progress_gated_by_ship_risk() -> None:
    env = _env(reward_progress_risk_gate=0.5, reward_safe_w=0.0)
    # Place tug 0 very close to hull along slot axis (high P_ship_raw)
    for i, d in enumerate((8.0, 400.0, 450.0, 500.0)):
        _place(env, i, d)
    high = _comp(env, [9.0, 400.0, 450.0, 500.0])
    assert float(high["progress_risk"][0]) > 0.5
    assert float(high["r_dist"][0]) < 0.5  # ungated would be ~1.0

    # Far away: low risk, full progress
    for i, d in enumerate((120.0, 400.0, 450.0, 500.0)):
        _place(env, i, d)
    low = _comp(env, [121.0, 400.0, 450.0, 500.0])
    assert float(low["progress_risk"][0]) == pytest.approx(0.0, abs=1e-6)
    assert float(low["r_dist"][0]) == pytest.approx(1.0)


def test_negative_progress_not_risk_gated() -> None:
    env = _env(reward_progress_risk_gate=0.5, reward_safe_w=0.0)
    for i, d in enumerate((8.0, 400.0, 450.0, 500.0)):
        _place(env, i, d)
    retreat = _comp(env, [7.0, 400.0, 450.0, 500.0])
    assert float(retreat["r_dist"][0]) == pytest.approx(-1.0)
```

- [ ] **Step 3: 跑测试确认失败**

Run: `PYTHONPATH=. pytest tests/test_safe_slot_reward.py::test_positive_progress_gated_by_ship_risk -v`  
Expected: FAIL（无 `progress_risk` 或未门控）

- [ ] **Step 4: 实现 CPU 门控**

在 `env/reward.py` 的 `compute_rewards` 中，于计算 `p_ship`（prox+cpa）之后、乘 `soft` **之前**：

```python
            p_ship_raw = p_ship  # prox + cpa_w * cpa, before corridor soft
            gate = max(float(getattr(cfg, "reward_progress_risk_gate", 0.5)), 1e-6)
            rho = float(np.clip(p_ship_raw / gate, 0.0, 1.0))
            if r_dist > 0.0:
                r_dist = r_dist * (1.0 - rho)
            # then: soft = ...; p_ship = p_ship_raw * soft
```

确保 `comp` 初始化含 `"progress_risk": np.zeros(n, dtype=np.float32)`，并赋值 `comp["progress_risk"][i] = rho`。`comp["r_dist"][i] = r_dist` 写入门控后值。

将总奖励中的 `w_dist * r_dist` 使用门控后 `r_dist`。

- [ ] **Step 5: 跑测试**

Run: `PYTHONPATH=. pytest tests/test_safe_slot_reward.py tests/test_reward_no_orbit_farming.py -v`  
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add env/reward.py tests/test_safe_slot_reward.py tests/test_reward_no_orbit_farming.py
git commit -m "$(cat <<'EOF'
feat(reward): gate positive progress by ship collision risk

EOF
)"
```

---

### Task 3: CPU `R_safe` + 诊断键

**Files:**
- Modify: `env/reward.py`
- Modify: `tests/test_safe_slot_reward.py`
- Modify: `tests/test_reward_redesign_smoke.py`（REQUIRED 加回 `r_safe`/`progress_risk`）
- Modify: `scripts/train.py`（`_TB_REWARD_KEYS` + 启动摘要）

**Interfaces:**
- Produces: `comp["r_safe"]`；总奖励含 `+ reward_safe_w * r_safe`

- [ ] **Step 1: 追加失败测试到 `tests/test_safe_slot_reward.py`**

```python
def test_r_safe_zero_outside_corridor() -> None:
    env = _env(reward_safe_w=2.0, reward_progress_risk_gate=1e9)
    for i, d in enumerate((200.0, 400.0, 450.0, 500.0)):
        _place(env, i, d)
    comp = _comp(env, [201.0, 400.0, 450.0, 500.0])
    assert float(comp["corridor_gate"][0]) == pytest.approx(0.0)
    assert float(comp["r_safe"][0]) == pytest.approx(0.0)


def test_r_safe_higher_when_centered_and_approaching() -> None:
    env = _env(reward_safe_w=2.0, reward_progress_risk_gate=1e9, reward_dist_w=0.0)
    # On-axis near slot (inside hold_start)
    for i, d in enumerate((40.0, 400.0, 450.0, 500.0)):
        _place(env, i, d)
    on_axis = _comp(env, [41.0, 400.0, 450.0, 500.0])

    # Same distance but laterally offset ~ half_width
    slots = env.ship.slot_positions_world()
    slot = slots[env.tug_to_slot[0]]
    ax = float(slot[0]) - env.ship.x
    ay = float(slot[1]) - env.ship.y
    n = math.hypot(ax, ay)
    ex, ey = ax / n, ay / n
    # perpendicular
    px, py = -ey, ex
    env.tugs[0].eta = Vec3(
        float(slot[0]) + 40.0 * ex + 35.0 * px,
        float(slot[1]) + 40.0 * ey + 35.0 * py,
        float(slot[2]),
    )
    off = _comp(env, [41.0, 400.0, 450.0, 500.0])
    assert float(on_axis["r_safe"][0]) > float(off["r_safe"][0])


def test_r_safe_drops_when_closing_fast_on_ship() -> None:
    env = _env(reward_safe_w=2.0, reward_progress_risk_gate=1e9, reward_dist_w=0.0)
    for i, d in enumerate((40.0, 400.0, 450.0, 500.0)):
        _place(env, i, d)
    # Point body velocity toward ship center
    slots = env.ship.slot_positions_world()
    slot = slots[env.tug_to_slot[0]]
    tug = env.tugs[0]
    ux = env.ship.x - tug.x
    uy = env.ship.y - tug.y
    un = math.hypot(ux, uy) or 1.0
    # World velocity toward ship ≈ 3 m/s → expressed in body frame roughly via nu.u
    # Set world-aligned heading toward ship and surge
    tug.eta = Vec3(tug.x, tug.y, math.atan2(uy, ux))
    tug.nu = Vec3(3.0, 0.0, 0.0)
    fast = _comp(env, [40.0, 400.0, 450.0, 500.0])
    tug.nu = Vec3(0.0, 0.0, 0.0)
    still = _comp(env, [40.0, 400.0, 450.0, 500.0])
    assert float(still["r_safe"][0]) > float(fast["r_safe"][0])
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH=. pytest tests/test_safe_slot_reward.py::test_r_safe_zero_outside_corridor -v`  
Expected: FAIL

- [ ] **Step 3: 实现 `R_safe`**

在 `compute_rewards` 中，已有 `c_gate`、`progress`/`d`/`prev` 后计算：

```python
            clip_m = progress_clip
            s_axial = float(np.clip((float(episode.prev_dist[i]) - d) / clip_m, 0.0, 1.0))
            # lateral from same geometry as corridor (recompute lat_n or return from helper)
            # Prefer extending _corridor_gate to also return lateral_norm, or duplicate lat calc:
            # lat_n in [0,1] as used inside _corridor_gate
            s_lat = 1.0 - _smoothstep(lat_n)  # where lat_n = clip(lat/half_width,0,1)
            # closing speed toward ship center
            u_x = float(state.ship.x) - tug.x
            u_y = float(state.ship.y) - tug.y
            u_n = math.hypot(u_x, u_y)
            if u_n <= 1e-6:
                s_approach = 1.0
            else:
                u_x /= u_n
                u_y /= u_n
                closing = max(0.0, tug_vx_w * u_x + tug_vy_w * u_y)
                v_ref = max(float(getattr(cfg, "reward_safe_closing_speed_mps", 1.0)), 1e-6)
                s_approach = float(np.clip(1.0 - closing / v_ref, 0.0, 1.0))
            r_safe = c_gate * (1.0 - target_gate) * (0.5 * s_axial + 0.3 * s_lat + 0.2 * s_approach)
```

实现 `_smoothstep(x)`：`x` 已 clip 到 [0,1] 时 `x*x*(3-2*x)`。  
为得到 `lat_n`，将 `_corridor_gate` 改为返回 `(gate, lat_n)`，或新增 `_corridor_metrics` 返回二者——**推荐新增** `_corridor_metrics(...) -> tuple[float, float]`（gate, lat_n），`_corridor_gate` 转调它以保持兼容。

总奖励：

```python
            r_total = (
                w_dist * r_dist
                - distance_cost_w * p_distance
                + w_safe * r_safe
                + w_hold * r_hold
                + w_vel * r_vel
                - w_coll * p_coll
            )
```

其中 `w_safe = float(getattr(cfg, "reward_safe_w", 0.0))`。

- [ ] **Step 4: 更新 smoke REQUIRED 与 train TB**

`tests/test_reward_redesign_smoke.py` REQUIRED 加入 `r_safe`、`progress_risk`。

`scripts/train.py`：

```python
_TB_REWARD_KEYS = ("r_dist", "p_distance", "r_hold", "r_safe", "p_collision")
```

在奖励启动摘要 `print` 中加入 `safe_w`、`progress_risk_gate`、`coll_w`、`coll_cap`、`ship_soft_min`。

- [ ] **Step 5: 跑测试**

Run: `PYTHONPATH=. pytest tests/test_safe_slot_reward.py tests/test_reward_redesign_smoke.py tests/test_reward_no_orbit_farming.py -v`  
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add env/reward.py tests/test_safe_slot_reward.py tests/test_reward_redesign_smoke.py scripts/train.py
git commit -m "$(cat <<'EOF'
feat(reward): add corridor safe-approach term R_safe

EOF
)"
```

---

### Task 4: GPU 对齐 + parity

**Files:**
- Modify: `env/gpu/batched_step.py`（`compute_rewards_batched`）
- Test: 现有 `tests/parity/test_reward_terminate.py` 与/或 corridor/reward parity；若无专用文件则扩展 `tests/parity/` 中覆盖 reward 的测试

**Interfaces:**
- Produces: GPU `components` 含 `r_safe`、`progress_risk`；`r_dist` 门控后；奖励总和含 `reward_safe_w`

- [ ] **Step 1: 改 GPU 公式**

在 `compute_rewards_batched` 中，于 `p_ship = prox + cpa` 之后、`soft` 之前：

```python
        p_ship_raw = p_ship
        gate = max(float(getattr(cfg, "reward_progress_risk_gate", 0.5)), 1e-6)
        progress_risk = (p_ship_raw / gate).clamp(0.0, 1.0)
        r_dist = torch.where(r_dist > 0.0, r_dist * (1.0 - progress_risk), r_dist)
        # then soft / p_ship = p_ship_raw * soft as today
```

计算 `r_safe`（与 CPU 同系数），在已有 `corridor` 与 `lateral` 上：

```python
        lat_n = (lateral / max(float(cfg.reward_corridor_half_width_m), 1e-6)).clamp(0.0, 1.0)
        s_lat = 1.0 - lat_n.square() * (3.0 - 2.0 * lat_n)
        s_axial = progress.clamp(0.0, 1.0)
        u_x = s.x[:, None] - t.eta[..., 0]
        u_y = s.y[:, None] - t.eta[..., 1]
        u_n = torch.hypot(u_x, u_y).clamp_min(1e-6)
        closing = (d["tug_vx"] * (u_x / u_n) + d["tug_vy"] * (u_y / u_n)).clamp_min(0.0)
        v_ref = max(float(getattr(cfg, "reward_safe_closing_speed_mps", 1.0)), 1e-6)
        s_approach = (1.0 - closing / v_ref).clamp(0.0, 1.0)
        r_safe = corridor * (1.0 - target_gate) * (0.5 * s_axial + 0.3 * s_lat + 0.2 * s_approach)
```

奖励：

```python
        rewards = (
            float(cfg.reward_dist_w) * r_dist
            - max(float(cfg.reward_distance_cost_w), 0.0) * p_distance
            + float(getattr(cfg, "reward_safe_w", 0.0)) * r_safe
            + float(cfg.reward_hold_w) * r_hold
            + float(cfg.reward_velocity_w) * r_vel
            - float(cfg.reward_collision_w) * p_coll
        )
```

components 增加：

```python
            "r_safe": r_safe.to(self.dtype),
            "progress_risk": progress_risk.to(self.dtype),
```

- [ ] **Step 2: 跑 parity / 相关测试**

Run:

```bash
PYTHONPATH=. pytest tests/parity/test_reward_terminate.py tests/test_reward_corridor.py tests/test_safe_slot_reward.py -v
```

Expected: PASS（若某 parity 因默认权重变化失败，更新期望或在该测试内固定旧权重——优先固定测试 cfg，勿改回生产默认）

- [ ] **Step 3: 全量测试**

Run: `PYTHONPATH=. pytest -q`  
Expected: 全绿

- [ ] **Step 4: Commit**

```bash
git add env/gpu/batched_step.py tests/
git commit -m "$(cat <<'EOF'
feat(gpu): mirror safe-slot reward gating and R_safe

EOF
)"
```

---

### Task 5: 文档 + 验证命令

**Files:**
- Modify: `docs/reward_function.md`
- Optional link in `README.md` / `docs/architecture.md`（一行指向新 spec）

- [ ] **Step 1: 重写 `docs/reward_function.md` 稠密公式段**

更新为含 \(R_{\mathrm{safe}}\)、风险门控、新默认权重表；终端 culprit/bystander/arrival 新值；链接  
`docs/superpowers/specs/2026-08-08-safe-slot-approach-reward-design.md`。

- [ ] **Step 2: 记录验证命令（实现者或用户执行）**

```bash
python -u scripts/train.py \
  --arch transformer --tf-size S \
  --init-radius 120 --slot-assignment minimax \
  --run-name safe_slot_v1_r120 \
  --total-steps 5000000 --seed 42 \
  --device cuda --env-backend cuda --eval-backend cuda \
  --num-envs 256 --rollout-steps 64 --minibatch-size 8192 \
  --eval-workers 32
```

过关：capture>0，final_dist<200，collision≤40%。对照 `runs/rsc_5m_baseline`。

- [ ] **Step 3: Commit docs**

```bash
git add docs/reward_function.md docs/architecture.md README.md
git commit -m "$(cat <<'EOF'
docs(reward): document safe-slot approach reward

EOF
)"
```

---

## Spec Coverage

| Spec | Task |
|------|------|
| 新字段与默认表 §7 | Task 1 |
| 风险门控 progress §5 | Task 2 |
| `R_safe` §6 | Task 3 |
| 诊断键 / TB / 启动摘要 §8 | Task 3 |
| GPU 一致 §2.1 | Task 4 |
| 文档 §10.5 | Task 5 |
| 5M 验证 §9 | Task 5 命令（实验执行，非代码门禁） |

## Out of scope

- 自动改 `rsc_*` presets 语义（可继续覆盖字段；默认 EnvConfig 已变）
- 在本计划内强制跑完 5M（可作为执行会话后续步骤）
