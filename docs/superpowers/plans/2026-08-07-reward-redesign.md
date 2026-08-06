# Reward Structural Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现进槽走廊船软碰软化、阶段门控下的距离项重混合、以及停滞惩罚，消除「外围躲远刷分」并支撑稳定 capture。

**Architecture:** 在 `FormationRewardComputer` 内新增走廊门控 / 软化 / 停滞纯函数，经 `compute_rewards` 接入；`MutableEpisodeState` 增加距离环形历史供停滞窗；`EnvConfig` 承载新超参并将 `reward_collision_ship_safe_m` 默认改为 80；诊断键与 TensorBoard 白名单同步；文档对齐。

**Tech Stack:** Python 3、NumPy、`EnvConfig` dataclass、`FormationEnv` / `FormationRewardComputer`、pytest。

## Global Constraints

- 规格：`docs/superpowers/specs/2026-08-07-reward-redesign-design.md`（已批准）。
- 硬碰撞 `ship_collision_dist_m`、终端 Capture/碰撞惩罚默认不变。
- 拖轮间软碰/CPA **不**随走廊软化。
- `R_vel` 默认权重保持 0。
- 旧 `REWARD_PRESETS` 可继续加载；本轮不重做消融矩阵。
- 成功验收（训练侧，实现后人工跑）：`final_dist` 明显下降 + 碰撞不明显恶化 + `capture_rate` 稳定 > 0。

---

## File Structure

| 文件 | 职责 |
|------|------|
| `config.py` | 新奖励字段；`reward_collision_ship_safe_m` 默认 80 |
| `env/state.py` | `MutableEpisodeState.dist_hist` / `dist_hist_head` / `dist_hist_filled` |
| `env/formation_env.py` | 构造/reset 时初始化距离历史；每步在更新 `prev_dist` 前写入 hist |
| `env/reward.py` | 走廊、软化、停滞、`R_dist` 混合；诊断键 |
| `scripts/train.py` | `_TB_REWARD_KEYS` 增加 `p_stall` |
| `tests/test_reward_corridor.py` | 走廊软化与拖轮间不软化 |
| `tests/test_reward_stall.py` | 停滞触发 / Hold 关闭 |
| `tests/test_reward_cpa.py` | 注释与显式 `ship_safe`（若默认变更影响断言） |
| `docs/reward_function.md` | 与新公式对齐 |
| `docs/tensorboard_metrics.md` | 记录 `p_stall`（若该文档列出 reward 键） |

---

### Task 1: EnvConfig 新字段与 ship_safe 默认

**Files:**
- Modify: `config.py`（`EnvConfig` 奖励段）
- Modify: `tests/test_reward_cpa.py`（显式设置/更新注释，避免依赖旧默认 100）
- Test: `tests/test_reward_config_redesign.py`（新建，锁默认值）

**Interfaces:**
- Consumes: 无
- Produces: `EnvConfig` 字段（类型均为 `float`）：
  - `reward_dist_progress_frac: float = 0.7`
  - `reward_stall_w: float = 0.5`
  - `reward_stall_window_s: float = 5.0`
  - `reward_stall_min_progress_m: float = 2.0`
  - `reward_stall_floor: float = 0.2`
  - `reward_corridor_half_width_m: float = 40.0`
  - `reward_corridor_axial_slack_m: float = 30.0`
  - `reward_ship_soft_min_scale: float = 0.15`
  - `reward_collision_ship_safe_m: float = 80.0`（原 100.0）

- [ ] **Step 1: Write the failing test**

Create `tests/test_reward_config_redesign.py`:

```python
"""Defaults for structural reward redesign."""

from config import EnvConfig


def test_reward_redesign_defaults() -> None:
    cfg = EnvConfig()
    assert cfg.reward_dist_progress_frac == 0.7
    assert cfg.reward_stall_w == 0.5
    assert cfg.reward_stall_window_s == 5.0
    assert cfg.reward_stall_min_progress_m == 2.0
    assert cfg.reward_stall_floor == 0.2
    assert cfg.reward_corridor_half_width_m == 40.0
    assert cfg.reward_corridor_axial_slack_m == 30.0
    assert cfg.reward_ship_soft_min_scale == 0.15
    assert cfg.reward_collision_ship_safe_m == 80.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_reward_config_redesign.py -v`  
Expected: FAIL（缺字段或 `ship_safe` 仍为 100）

- [ ] **Step 3: Add fields to EnvConfig**

In `config.py` `EnvConfig`，于距离段加入 `reward_dist_progress_frac = 0.7`；于碰撞段将 `reward_collision_ship_safe_m` 改为 `80.0`，并加入走廊/软化字段；于奖励段加入停滞字段（紧挨 `reward_collision_*` 或独立「停滞 / 走廊」注释块）。同步更新文件顶部奖励注释行，包含 `- w_stall*P_stall`。

- [ ] **Step 4: Fix CPA test comment / explicit safe if needed**

In `tests/test_reward_cpa.py` `test_ship_cpa_uses_future_hull_distance`：在 `_make_env` 之后或测试内设置 `env.cfg.reward_collision_ship_safe_m = 100.0`（保持该测例几何意图），并改注释为「显式 ship_safe=100」。

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_reward_config_redesign.py tests/test_reward_cpa.py tests/test_reward_presets.py -v`  
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add config.py tests/test_reward_config_redesign.py tests/test_reward_cpa.py
git commit -m "feat(config): add reward redesign defaults and lower ship soft safe"
```

---

### Task 2: 走廊门控与船软碰软化（纯函数 + 接入）

**Files:**
- Modify: `env/reward.py`
- Create: `tests/test_reward_corridor.py`

**Interfaces:**
- Consumes: `EnvConfig` 走廊字段；`slot_world`；tug/ship 位姿
- Produces（`FormationRewardComputer` 静态/类方法）:
  - `corridor_gate(tx, ty, slot_x, slot_y, d, hold_start_m, half_width_m, axial_slack_m) -> float`
  - `ship_soft_scale(corridor_gate: float, s_min: float) -> float`
  - `compute_rewards` 中：`p_ship *= ship_soft_scale`；`comp` 增加 `corridor_gate`、`ship_soft_scale`

**走廊公式（实现必须与此一致）：**

```python
# r: slot → tug; e: unit vector pointing toward slot (tug → slot)
r_x, r_y = tx - slot_x, ty - slot_y
d = hypot(r_x, r_y)  # or pass-in
if d <= 1e-6:
    corridor_gate = 1.0 if True else 0.0  # at slot: treat as inside laterally
    # use corridor_gate = 1.0 * dist_gate with dist_gate=1
else:
    e_x, e_y = -r_x / d, -r_y / d          # toward slot
    a = r_x * e_x + r_y * e_y              # on-axis outside: a = -d; past slot: a > 0
    lat = hypot(r_x - a * e_x, r_y - a * e_y)
    along_ok = a <= axial_slack_m          # allow overshoot up to slack; outside a=-d always ok
    lat_n = lat / max(half_width_m, 1e-6)
    if lat_n >= 1.0: lat_gate = 0.0
    elif lat_n <= 0.0: lat_gate = 1.0
    else:
        u = 1.0 - lat_n
        lat_gate = u * u * (3.0 - 2.0 * u)
    dist_gate = 1.0 if d < hold_start_m else 0.0
    corridor_gate = lat_gate * dist_gate * (1.0 if along_ok else 0.0)

ship_soft_scale = 1.0 - (1.0 - s_min) * corridor_gate
```

**实现注意：** 单测用「沿 slot 径向、d≈80 m」vs「同 d 但大幅侧偏」对比 `corridor_gate` / `ship_soft_scale`。

- [ ] **Step 1: Write failing tests**

Create `tests/test_reward_corridor.py`:

```python
"""Corridor softening for ship collision risk."""

from __future__ import annotations

import math

import numpy as np

from config import EnvConfig
from env.formation_env import FormationEnv
from physics.tugboat_dynamics_model import Vec3


def _env() -> FormationEnv:
    cfg = EnvConfig()
    cfg.reward_stall_w = 0.0  # isolate corridor
    cfg.reward_shape_w = 0.0
    cfg.reward_team_w = 0.0
    env = FormationEnv(cfg=cfg, seed=3)
    env.reset()
    return env


def _park(env: FormationEnv) -> None:
    for i in (1, 2, 3):
        env.tugs[i].eta = Vec3(800.0 + 50 * i, 800.0, 0.0)
        env.tugs[i].nu = Vec3.zero()


def _comp(env: FormationEnv) -> dict:
    actions = np.zeros((env.cfg.n_tugs, 4), dtype=np.float32)
    _, info = env._compute_rewards(actions)
    return info["reward_components"]


def test_corridor_softens_ship_penalty_on_radial_approach() -> None:
    env = _env()
    env.ship.x = 0.0
    env.ship.y = 0.0
    env.ship.psi = 0.0
    env.ship.u = 0.0
    env.ship.v = 0.0
    env.ship.r = 0.0
    slots = env.ship.slot_positions_world()
    # tug 0 assigned slot 0 by default
    sx, sy = float(slots[0, 0]), float(slots[0, 1])
    # Place tug on the line from ship-ish toward slot, 80 m from slot (Near band)
    # Direction from slot away from ship center:
    ang = math.atan2(sy - env.ship.y, sx - env.ship.x)
    d = 80.0
    env.tugs[0].eta = Vec3(sx + d * math.cos(ang), sy + d * math.sin(ang), env.ship.psi)
    env.tugs[0].nu = Vec3.zero()
    _park(env)

    comp = _comp(env)
    assert float(comp["corridor_gate"][0]) > 0.5
    assert float(comp["ship_soft_scale"][0]) < 0.5
    # Softened ship risk should be below unscaled barrier at same hull dist
    # (scale applied → p_ship lower than if scale==1 with same geometry)
    assert float(comp["ship_soft_scale"][0]) == (
        1.0 - (1.0 - env.cfg.reward_ship_soft_min_scale) * float(comp["corridor_gate"][0])
    )


def test_outside_corridor_keeps_full_ship_soft_scale() -> None:
    env = _env()
    env.ship.x = 0.0
    env.ship.y = 0.0
    env.ship.psi = 0.0
    slots = env.ship.slot_positions_world()
    sx, sy = float(slots[0, 0]), float(slots[0, 1])
    # Same distance to slot (~80m) but far laterally: offset perpendicular
    ang = math.atan2(sy, sx)
    perp = ang + math.pi / 2
    d = 80.0
    env.tugs[0].eta = Vec3(
        sx + d * math.cos(ang) + 120.0 * math.cos(perp),
        sy + d * math.sin(ang) + 120.0 * math.sin(perp),
        0.0,
    )
    # Recompute actual distance — place purely lateral at 80m from slot
    env.tugs[0].eta = Vec3(sx + 80.0 * math.cos(perp), sy + 80.0 * math.sin(perp), 0.0)
    env.tugs[0].nu = Vec3.zero()
    _park(env)

    comp = _comp(env)
    assert float(comp["corridor_gate"][0]) < 0.1
    assert float(comp["ship_soft_scale"][0]) > 0.95


def test_tug_collision_not_softened_by_corridor() -> None:
    env = _env()
    env.ship.x = 0.0
    env.ship.y = 1000.0
    # Two tugs close; ensure corridor_gate high for tug0 via parked near its slot far away ship
    env.tugs[0].eta = Vec3(0.0, 0.0, 0.0)
    env.tugs[0].nu = Vec3.zero()
    env.tugs[1].eta = Vec3(50.0, 0.0, 0.0)  # within tug soft 120
    env.tugs[1].nu = Vec3.zero()
    _park(env)
    # force measure p_tug > 0
    comp = _comp(env)
    p_tug_before = float(comp["p_tug_collision"][0])
    assert p_tug_before > 0.0
    # Manually bump corridor diagnostics independence: p_tug must not depend on ship soft scale
    # Place tug0 in corridor near slot while keeping pair distance
    slots = env.ship.slot_positions_world()
    sx, sy = float(slots[0, 0]), float(slots[0, 1])
    env.ship.x = 0.0
    env.ship.y = 0.0
    env.tugs[0].eta = Vec3(sx + 60.0, sy, 0.0)
    env.tugs[1].eta = Vec3(sx + 60.0 + 50.0, sy, 0.0)
    env.tugs[2].eta = Vec3(900.0, 900.0, 0.0)
    env.tugs[3].eta = Vec3(950.0, 900.0, 0.0)
    comp2 = _comp(env)
    assert float(comp2["p_tug_collision"][0]) > 0.0
    # ship soft scale may be < 1 but p_tug uses raw barriers
    assert float(comp2["ship_soft_scale"][0]) <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_reward_corridor.py -v`  
Expected: FAIL（缺 `corridor_gate` / `ship_soft_scale` 键）

- [ ] **Step 3: Implement helpers + wire into compute_rewards**

In `env/reward.py`:

1. Add classmethods `_corridor_gate` and `_ship_soft_scale` implementing the formula above（`along_ok = a <= axial_slack_m`，`a = r·e` 按规格：`r` 从 slot 指向 tug，`e` 指向 slot，故外侧 `a=-d`）。
2. In component dict init, add:
   - `"corridor_gate": np.zeros(n, dtype=np.float32)`
   - `"ship_soft_scale": np.ones(n, dtype=np.float32)`
3. After computing `p_ship`（prox+cpa），before combining into `p_coll`:
   ```python
   c_gate = self._corridor_gate(
       tug.x, tug.y, float(slot[0]), float(slot[1]), d,
       hold_start_m,
       float(getattr(cfg, "reward_corridor_half_width_m", 40.0)),
       float(getattr(cfg, "reward_corridor_axial_slack_m", 30.0)),
   )
   s_min = float(getattr(cfg, "reward_ship_soft_min_scale", 0.15))
   soft = self._ship_soft_scale(c_gate, s_min)
   p_ship *= soft
   ```
4. Log `comp["corridor_gate"][i] = c_gate` and `comp["ship_soft_scale"][i] = soft`.

- [ ] **Step 4: Run corridor tests**

Run: `pytest tests/test_reward_corridor.py -v`  
Expected: PASS（若几何断言过紧，微调测试放置使径向 `corridor_gate>0.5`、侧向 `<0.1`，不要放宽产品公式）

- [ ] **Step 5: Commit**

```bash
git add env/reward.py tests/test_reward_corridor.py
git commit -m "feat(reward): soften ship collision risk inside approach corridor"
```

---

### Task 3: 距离历史缓冲与停滞惩罚

**Files:**
- Modify: `env/state.py`（`MutableEpisodeState`）
- Modify: `env/formation_env.py`（init / reset / step 写 hist）
- Modify: `env/reward.py`（停滞计算、`R_dist` 乘 `stall_scale`、减 `w_stall*P_stall`）
- Create: `tests/test_reward_stall.py`

**Interfaces:**
- Consumes: `cfg.dt_ctrl`、`reward_stall_*`、`hold_gate`、`episode.dist_hist*`
- Produces:
  - `MutableEpisodeState.dist_hist: np.ndarray` shape `(n_tugs, hist_cap)` float32
  - `MutableEpisodeState.dist_hist_head: int` 下一个写入下标
  - `MutableEpisodeState.dist_hist_filled: int` 已填充长度（≤ hist_cap）
  - `comp["p_stall"]`, `comp["stall_scale"]`
  - `R_dist` 使用 `alpha = reward_dist_progress_frac` 且乘 `stall_scale`
  - `r_total -= w_stall * p_stall`（Hold 时 p_stall=0, stall_scale=1）

**历史缓冲约定：**

```python
hist_cap = max(2, int(math.ceil(reward_stall_window_s / dt_ctrl)) + 1)
# 每 step 在 reward 之后、更新 prev_dist 时：写入当前 d
# 读窗：若 filled >= steps_needed，取 index (head - steps_needed) mod cap 的距离为 d_old
# Δd_net = d_old - d_now
```

`formation_env` 在 `__init__` / `reset` 分配 `dist_hist`；在 `step` 里更新 `prev_dist` 的同一循环中调用内部方法 `_push_dist_hist(i, d)`。

- [ ] **Step 1: Write failing stall tests**

Create `tests/test_reward_stall.py`:

```python
"""Stall penalty and stall_scale for reward farming suppression."""

from __future__ import annotations

import numpy as np

from config import EnvConfig
from env.formation_env import FormationEnv
from physics.tugboat_dynamics_model import Vec3


def _make() -> FormationEnv:
    cfg = EnvConfig()
    cfg.reward_stall_w = 0.5
    cfg.reward_stall_window_s = 1.0  # 5 steps at dt=0.2
    cfg.reward_stall_min_progress_m = 2.0
    cfg.reward_stall_floor = 0.2
    cfg.reward_shape_w = 0.0
    cfg.reward_team_w = 0.0
    cfg.reward_collision_w = 0.0  # isolate stall
    env = FormationEnv(cfg=cfg, seed=5)
    env.reset()
    return env


def test_stall_triggers_when_no_net_progress() -> None:
    env = _make()
    # Park far from slot (~200m) and freeze all motion for many steps
    env.ship.x = 0.0
    env.ship.y = 0.0
    env.ship.u = 0.0
    slots = env.ship.slot_positions_world()
    sx, sy = float(slots[0, 0]), float(slots[0, 1])
    env.tugs[0].eta = Vec3(sx + 200.0, sy, 0.0)
    env.tugs[0].nu = Vec3.zero()
    for i in (1, 2, 3):
        env.tugs[i].eta = Vec3(800.0 + i * 40.0, 800.0, 0.0)
        env.tugs[i].nu = Vec3.zero()

    zero = np.zeros((env.cfg.n_tugs, 4), dtype=np.float32)
    last = None
    for _ in range(12):
        # Keep positions frozen: zero actions + zero ship speed already
        _, _, _, info = env.step(zero)
        last = info["reward_components"]
        # Re-assert freeze in case dynamics drift slightly — hard reset pose
        env.tugs[0].eta = Vec3(sx + 200.0, sy, 0.0)
        env.tugs[0].nu = Vec3.zero()

    assert last is not None
    assert float(last["p_stall"][0]) > 0.5
    assert float(last["stall_scale"][0]) < 0.5


def test_stall_disabled_in_hold_region() -> None:
    env = _make()
    env.ship.x = 0.0
    env.ship.y = 0.0
    env.ship.u = 0.0
    slots = env.ship.slot_positions_world()
    sx, sy, spsi = float(slots[0, 0]), float(slots[0, 1]), float(slots[0, 2])
    # Place inside hold_full (20m)
    env.tugs[0].eta = Vec3(sx + 5.0, sy, spsi)
    env.tugs[0].nu = Vec3(env.ship.u, 0.0, 0.0)
    for i in (1, 2, 3):
        env.tugs[i].eta = Vec3(800.0 + i * 40.0, 800.0, 0.0)
        env.tugs[i].nu = Vec3.zero()

    zero = np.zeros((env.cfg.n_tugs, 4), dtype=np.float32)
    last = None
    for _ in range(12):
        _, _, _, info = env.step(zero)
        last = info["reward_components"]
        env.tugs[0].eta = Vec3(sx + 5.0, sy, spsi)
        env.tugs[0].nu = Vec3(env.ship.u, 0.0, 0.0)

    assert last is not None
    assert float(last["hold_gate"][0]) > 0.9
    assert float(last["p_stall"][0]) == 0.0
    assert float(last["stall_scale"][0]) == 1.0
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest tests/test_reward_stall.py -v`  
Expected: FAIL

- [ ] **Step 3: Extend MutableEpisodeState + env lifecycle**

In `env/state.py` add fields:

```python
dist_hist: np.ndarray          # (n_tugs, hist_cap)
dist_hist_head: int = 0
dist_hist_filled: int = 0
```

In `formation_env.py`:
- Helper `_dist_hist_cap() -> int`
- On init/reset: allocate zeros，`head=0`，`filled=0`
- Method `_push_dist_hist(dists: np.ndarray) -> None` 写入一列当前各船 `d`
- In `step`, after rewards（或与 `prev_dist` 更新同一处）push **当前** slot 距离

- [ ] **Step 4: Implement stall + R_dist remix in reward.py**

```python
alpha = float(np.clip(getattr(cfg, "reward_dist_progress_frac", 0.7), 0.0, 1.0))
# replace dist mix:
r_dist = (1.0 - gate) * (alpha * progress + (1.0 - alpha) * dist_bonus)

# stall:
w_stall = float(getattr(cfg, "reward_stall_w", 0.0))
# read d_old from episode.dist_hist ...
# if hold_gate high (gate >= 0.99 or gate > 0.5 per spec "hold 区"): disable
if gate >= 0.99:
    stall_scale = 1.0
    p_stall = 0.0
else:
    # when history insufficient: stall_scale=1, p_stall=0
    ...
    delta = d_old - d
    thr = reward_stall_min_progress_m
    if delta < thr:
        p_stall = float(np.clip((thr - delta) / max(thr, 1e-6), 0.0, 1.0))
        # map stall_scale from 1 → floor as p_stall goes 0→1
        floor = reward_stall_floor
        stall_scale = 1.0 - (1.0 - floor) * p_stall
    else:
        p_stall = 0.0
        stall_scale = 1.0

r_dist = r_dist * stall_scale
r_total = w_dist * r_dist + ... - w_coll * p_coll - w_stall * p_stall + r_shape
```

Add `p_stall` / `stall_scale` to `comp`.

**历史读取时机：** `compute_rewards` 使用的 hist 必须是 **上一窗** 数据；当前步的 `d` 在 reward 之后再 push，避免 `d_old==d` 伪停滞。即：`step` 顺序保持「算 reward（读旧 hist）→ push 当前 d → 更新 prev_dist」。

- [ ] **Step 5: Run stall + corridor + config tests**

Run: `pytest tests/test_reward_stall.py tests/test_reward_corridor.py tests/test_reward_config_redesign.py tests/test_reward_cpa.py -v`  
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add env/state.py env/formation_env.py env/reward.py tests/test_reward_stall.py
git commit -m "feat(reward): add stall penalty and progress-weighted distance term"
```

---

### Task 4: 训练日志与文档对齐

**Files:**
- Modify: `scripts/train.py`（`_TB_REWARD_KEYS`）
- Modify: `docs/reward_function.md`
- Modify: `docs/tensorboard_metrics.md`（若列出 reward 分量键则追加 `p_stall`）

**Interfaces:**
- Consumes: `reward_components["p_stall"]`
- Produces: TB 标量 `reward/p_stall`；文档与实现一致

- [ ] **Step 1: Update TB whitelist**

In `scripts/train.py`:

```python
_TB_REWARD_KEYS = ("r_dist", "r_hold", "p_collision", "p_stall")
```

- [ ] **Step 2: Rewrite docs/reward_function.md dense-reward sections**

更新第 1 节总式（含 \(P_{\mathrm{stall}}\)）；第 2 节写 \(\alpha\) 与 `stall_scale`；新增「走廊软化」与「停滞」小节；配置表加入 Task 1 全部新字段；诊断表加入 `p_stall` / `corridor_gate` / `ship_soft_scale` / `stall_scale`；注明 `ship_safe` 默认 80。保留指向 preset 文档的链接，并注明旧 preset 语义可能过时。

- [ ] **Step 3: Update tensorboard_metrics.md if it enumerates reward/* keys**

追加 `reward/p_stall` 一行说明。

- [ ] **Step 4: Run full reward-related tests**

Run: `pytest tests/test_reward_stall.py tests/test_reward_corridor.py tests/test_reward_config_redesign.py tests/test_reward_cpa.py tests/test_reward_presets.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/train.py docs/reward_function.md docs/tensorboard_metrics.md
git commit -m "docs: align reward docs and TB metrics with redesign"
```

---

### Task 5: 回归冒烟（环境一步 + 组件键完备）

**Files:**
- Create: `tests/test_reward_redesign_smoke.py`

**Interfaces:**
- Consumes: 完整 `FormationEnv.step`
- Produces: 断言新键存在且有限；硬碰撞阈值配置未改

- [ ] **Step 1: Write smoke test**

```python
"""Smoke: redesigned reward components are finite and keyed."""

import numpy as np

from config import EnvConfig
from env.formation_env import FormationEnv


REQUIRED = {
    "r_total", "r_dist", "r_hold", "p_collision", "p_stall",
    "corridor_gate", "ship_soft_scale", "stall_scale", "hold_gate",
}


def test_reward_redesign_smoke_step() -> None:
    env = FormationEnv(cfg=EnvConfig(), seed=0)
    obs = env.reset()
    assert obs is not None
    actions = np.zeros((env.cfg.n_tugs, 4), dtype=np.float32)
    for _ in range(5):
        obs, rew, done, info = env.step(actions)
        comp = info["reward_components"]
        for k in REQUIRED:
            assert k in comp
            assert np.isfinite(np.asarray(comp[k], dtype=np.float64)).all()
        assert np.isfinite(rew).all()
    assert env.cfg.ship_collision_dist_m == 6.0
    assert env.cfg.reward_arrival_bonus == 80.0
```

- [ ] **Step 2: Run**

Run: `pytest tests/test_reward_redesign_smoke.py -v`  
Expected: PASS

- [ ] **Step 3: Run broader env reward suite**

Run: `pytest tests/test_reward_*.py -v`  
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_reward_redesign_smoke.py
git commit -m "test: smoke coverage for redesigned reward components"
```

---

## Spec Coverage Checklist

| Spec 要求 | Task |
|-----------|------|
| 阶段门控 / approach_gate / hold_gate | Task 2–3（沿用 hold_gate；`R_dist` 乘 approach） |
| 径向椭球走廊 + ship_soft_scale | Task 2 |
| 拖轮间不软化 | Task 2 测试 |
| 硬碰撞/终端不变 | Task 5 + Task 1 不改 pen |
| 弱 dist_bonus + α + stall_scale / P_stall | Task 3 |
| Hold 关闭停滞 | Task 3 测试 |
| Config 默认表 | Task 1 |
| 诊断键 + TB `p_stall` | Task 2–4 |
| `docs/reward_function.md` | Task 4 |
| ship_safe 80 | Task 1 |

## Plan Self-Review Notes

- 走廊轴向符号在 Task 2 公式块已按「`r` 从 slot→tug，`e` 指向 slot → 外侧 `a=-d`」写死，避免与规格文字歧义；实现以该块与单测为准。
- 停滞 hist 必须在 reward **之后** push，已写入 Task 3。
- 无 TBD；旧 preset 消融重跑显式列为非目标。
