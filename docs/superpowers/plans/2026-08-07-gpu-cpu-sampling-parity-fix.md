# GPU/CPU Sampling Parity Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the real batched CUDA environment preserve CPU reward semantics and per-environment RNG/reset sequences while retaining float32 training dynamics.

**Architecture:** Keep `FormationEnv` as the oracle and reset sampler. Correct the batched tug CPA vector convention, use each shadow environment's NumPy RNG only at rare ship-resample boundaries, and keep the resample countdown in float64 so both backends trigger on the same step.

**Tech Stack:** Python 3, NumPy, PyTorch/CUDA, pytest.

## Global Constraints

- Do not change reward weights, CPA definitions, observation schema, or the default training backend.
- CUDA dynamics and rewards remain float32; only `time_to_resample` is float64.
- Discrete events must match exactly.
- Non-TCPA observations and global state use `rtol=1e-4, atol=1e-4`; neighbor TCPA uses `rtol=1e-4, atol=5e-2`.
- Preserve all unrelated dirty-worktree changes.

---

### Task 1: Correct batched tug CPA semantics

**Files:**
- Modify: `tests/parity/test_reward_terminate.py`
- Modify: `env/gpu/batched_step.py:311-323`

**Interfaces:**
- Consumes: `FormationEnv.step(actions)` and `CudaVecEnv.step(actions)` reward component dictionaries.
- Produces: `FastBatchedStep.compute_rewards_batched()` with `other - self` conventions for both relative position and velocity.

- [ ] **Step 1: Write the failing fast-path regression test**

Add a test that uses the real float32 fast path on CPU, replays the known seed/action sequence, and compares tug CPA components:

```python
def test_fast_batched_tug_cpa_matches_cpu():
    cfg = EnvConfig()
    cpu = FormationEnv(cfg=cfg, seed=123)
    fast = CudaVecEnv(cfg, n_envs=1, base_seed=123, device="cpu", dtype=torch.float32)
    cpu.reset()
    fast.reset()
    rng = np.random.default_rng(2026)

    for step in range(20):
        actions = rng.uniform(-1, 1, size=(8, cfg.n_tugs, 4)).astype(np.float32)[0]
        _, reward_cpu, _, info_cpu = cpu.step(actions)
        _, reward_fast, _, info_fast, *_ = fast.step(actions[None])
        np.testing.assert_allclose(
            info_fast[0]["reward_components"]["p_tug_collision"],
            info_cpu["reward_components"]["p_tug_collision"],
            rtol=1e-4,
            atol=1e-4,
            err_msg=f"p_tug_collision@{step}",
        )
        np.testing.assert_allclose(reward_fast[0], reward_cpu, rtol=1e-4, atol=1e-4)
```

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m pytest -q tests/parity/test_reward_terminate.py::test_fast_batched_tug_cpa_matches_cpu`

Expected: FAIL near step 9/10 with `p_tug_collision` differing by approximately `0.089`.

- [ ] **Step 3: Apply the minimal CPA fix**

Change only the batched call site:

```python
_, _, pair_risk = self._cpa_risk(
    pair_dx, pair_dy, rvx, rvy,
    float(cfg.tug_collision_dist_m), tug_safe, horizon,
)
```

- [ ] **Step 4: Run the focused test and parity suite**

Run: `python -m pytest -q tests/parity/test_reward_terminate.py::test_fast_batched_tug_cpa_matches_cpu tests/parity/test_reward_terminate.py`

Expected: PASS.

### Task 2: Synchronize ship-resample RNG and trigger timing

**Files:**
- Modify: `tests/parity/test_ship_step.py`
- Modify: `tests/parity/test_rollout_match.py`
- Modify: `physics/batched/ship.py:28-48,69-87`
- Modify: `env/gpu/batched_step.py:138-169`
- Modify: `env/gpu/vec_env.py:111-113`

**Interfaces:**
- Consumes: `dynamics_step(actions, rng_envs)` where `rng_envs[i].ship.rng` is the CPU oracle stream.
- Produces: `BatchedShipState.time_to_resample: torch.Tensor` with dtype `torch.float64`, and identical per-environment draw order across CPU/CUDA.

- [ ] **Step 1: Write a failing countdown-dtype test**

```python
def test_ship_resample_countdown_stays_float64_for_float32_dynamics():
    state = BatchedShipState.zeros(2, dtype=torch.float32)
    assert state.x.dtype == torch.float32
    assert state.time_to_resample.dtype == torch.float64
```

- [ ] **Step 2: Write a failing CUDA reset-RNG regression**

Add a CUDA-conditional test using `dataclasses.replace`:

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_resample_preserves_cpu_rng_through_auto_reset():
    cfg = replace(EnvConfig(), ship_speed_min=0.6, ship_speed_max=1.4, max_episode_steps=1)
    cpu = FormationEnv(cfg=cfg, seed=123)
    cuda = CudaVecEnv(cfg, n_envs=1, base_seed=123, device="cuda", dtype=torch.float32)
    cpu.reset()
    cuda.reset()
    cpu.ship._time_to_resample = 0.1
    cuda.batch.ships.time_to_resample[0] = 0.1
    actions = np.zeros((1, cfg.n_tugs, 4), dtype=np.float32)

    _, _, done_cpu, _ = cpu.step(actions[0])
    obs_cpu_reset = cpu.reset()
    obs_cuda_reset, _, done_cuda, *_ = cuda.step(actions)

    assert bool(done_cpu.any()) and bool(done_cuda[0])
    np.testing.assert_array_equal(obs_cuda_reset[0], obs_cpu_reset)
```

- [ ] **Step 3: Run both tests and verify RED**

Run: `python -m pytest -q tests/parity/test_ship_step.py::test_ship_resample_countdown_stays_float64_for_float32_dynamics tests/parity/test_rollout_match.py::test_cuda_resample_preserves_cpu_rng_through_auto_reset`

Expected: dtype assertion fails; on CUDA, reset observation also differs.

- [ ] **Step 4: Keep only the countdown tensor in float64**

In `BatchedShipState.zeros`, construct it independently:

```python
time_to_resample=torch.full((n,), 25.0, device=device, dtype=torch.float64),
```

When generating an interval without injected samples, use `state.time_to_resample.dtype`.

- [ ] **Step 5: Pre-sample from shadow NumPy RNGs for every fast backend**

In `dynamics_step`, when `need.any()` and `rng_envs` is supplied, collect needed row indices once, draw speed then interval in index order, and upload two tensors once:

```python
needed = torch.nonzero(need, as_tuple=False).flatten().cpu().tolist()
u_values = np.zeros(self.n_envs, dtype=np.float64)
interval_values = np.zeros(self.n_envs, dtype=np.float64)
for i in needed:
    ship = rng_envs[i].ship
    u_values[i] = ship.rng.uniform(ship.speed_min, ship.speed_max)
    interval_values[i] = ship.rng.uniform(
        ship.target_resample_min_s, ship.target_resample_max_s
    )
u_samples = torch.as_tensor(u_values, device=self.device, dtype=self.dtype)
interval_samples = torch.as_tensor(
    interval_values,
    device=self.device,
    dtype=self.batch.ships.time_to_resample.dtype,
)
```

Pass `self.envs` from `CudaVecEnv.step` on both CPU and CUDA fast paths.

- [ ] **Step 6: Run the focused tests and verify GREEN**

Run the command from Step 3.

Expected: PASS, with no CUDA skip on the target RTX 3090 machine.

### Task 3: Cover the real CUDA rollout contract

**Files:**
- Modify: `tests/parity/test_rollout_match.py`

**Interfaces:**
- Consumes: public `FormationEnv` and `CudaVecEnv` reset/step/global-state APIs plus `ObservationSpec.neighbor_item_slice()`.
- Produces: parameterized true-CUDA parity coverage for N in `{1, 4, 8}`.

- [ ] **Step 1: Add observation comparison helpers**

Build the neighbor TCPA column list as `neighbor_item_slice(i).start + 8`, compare all other columns with `1e-4`, and compare only TCPA columns with `atol=5e-2`.

- [ ] **Step 2: Add the CUDA rollout test**

For each `n_envs` in `(1, 4, 8)`, reset CPU and CUDA environments with identical seeds, replay 40 deterministic float32 action steps, and on every step assert:

```python
np.testing.assert_allclose(reward_cuda, reward_cpu, rtol=1e-4, atol=1e-4)
assert np.array_equal(done_cuda, done_cpu)
np.testing.assert_allclose(global_cuda, global_cpu, rtol=1e-4, atol=1e-4)
```

Also compare `collision`, `success`, `phase`, `terminated`, and `truncated` exactly and use the field-specific observation helper.

- [ ] **Step 3: Run the new CUDA rollout test**

Run: `python -m pytest -q tests/parity/test_rollout_match.py::test_cuda_fast_rollout_matches_cpu`

Expected: PASS after Tasks 1–2. If a non-TCPA field fails, diagnose it before widening any tolerance.

- [ ] **Step 4: Run the entire parity suite**

Run: `python -m pytest -q tests/parity`

Expected: all tests pass.

### Task 4: Full regression and throughput verification

**Files:**
- Modify only if a test exposes an in-scope parity defect.

**Interfaces:**
- Consumes: the completed parity fixes.
- Produces: verified repository test status and a before/after-compatible CUDA hot path.

- [ ] **Step 1: Run the complete test suite**

Run: `python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 2: Run a short CUDA throughput smoke test**

Run: `python scripts/bench_env_throughput.py --backend cuda --num-envs 64 --rollout-steps 64`

Expected: command succeeds, finite SPS is reported, and no CPU per-environment loop runs on ordinary non-resample steps.

- [ ] **Step 3: Inspect the final diff**

Run: `git diff --check && git diff --stat && git status --short`

Expected: no whitespace errors; only the planned source/tests plus pre-existing user changes are present.

- [ ] **Step 4: Commit the implementation if explicitly requested**

Do not include unrelated dirty-worktree files. Suggested message: `fix: align cuda environment sampling with cpu`.
