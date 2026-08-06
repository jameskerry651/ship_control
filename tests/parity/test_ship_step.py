"""L0: batched ship kinematics vs CPU ``LargeShipModel``."""

from __future__ import annotations

import math

import numpy as np
import torch

from physics.batched.geometry import slot_positions_world
from physics.batched.ship import BatchedShipState, step_ships
from physics.large_ship_model import LargeShipModel

RTOL = 1e-10
ATOL = 1e-12


def test_ship_resample_countdown_stays_float64_for_float32_dynamics():
    state = BatchedShipState.zeros(2, dtype=torch.float32)

    assert state.x.dtype == torch.float32
    assert state.time_to_resample.dtype == torch.float64


def test_ship_step_no_resample_matches_cpu():
    rng = np.random.default_rng(0)
    cpu = LargeShipModel(rng=rng)
    cpu.reset(rng)
    # Freeze resample so paths stay deterministic without RNG sync.
    cpu._time_to_resample = 1e9
    cpu._u_target = cpu.u
    cpu._r_target = 0.0

    st = BatchedShipState.zeros(1, dtype=torch.float64)
    st.x[0] = cpu.x
    st.y[0] = cpu.y
    st.psi[0] = cpu.psi
    st.u[0] = cpu.u
    st.v[0] = cpu.v
    st.r[0] = cpu.r
    st.u_target[0] = cpu._u_target
    st.r_target[0] = cpu._r_target
    st.time_to_resample[0] = cpu._time_to_resample

    dt = 0.2
    for _ in range(40):
        step_ships(
            st,
            dt,
            speed_min=cpu.speed_min,
            speed_max=cpu.speed_max,
            speed_tau=cpu.speed_tau,
            target_resample_min_s=cpu.target_resample_min_s,
            target_resample_max_s=cpu.target_resample_max_s,
        )
        cpu.step(dt)
        assert math.isclose(float(st.x[0]), cpu.x, rel_tol=RTOL, abs_tol=ATOL)
        assert math.isclose(float(st.y[0]), cpu.y, rel_tol=RTOL, abs_tol=ATOL)
        assert math.isclose(float(st.u[0]), cpu.u, rel_tol=RTOL, abs_tol=ATOL)
        assert math.isclose(float(st.psi[0]), cpu.psi, rel_tol=RTOL, abs_tol=ATOL)


def test_slot_positions_world_match():
    rng = np.random.default_rng(1)
    cpu = LargeShipModel(rng=rng)
    cpu.reset(rng)
    st_x = torch.tensor([cpu.x], dtype=torch.float64)
    st_y = torch.tensor([cpu.y], dtype=torch.float64)
    st_psi = torch.tensor([cpu.psi], dtype=torch.float64)
    slots = slot_positions_world(
        st_x,
        st_y,
        st_psi,
        cpu.length_m,
        cpu.beam_m,
        cpu.slot_lon_offset_m,
        cpu.slot_lat_offset_m,
    )
    cpu_slots = cpu.slot_positions_world()
    torch.testing.assert_close(
        slots[0],
        torch.as_tensor(cpu_slots, dtype=torch.float64),
        rtol=RTOL,
        atol=ATOL,
    )


def test_ship_resample_with_injected_samples():
    rng = np.random.default_rng(2)
    cpu = LargeShipModel(rng=rng)
    cpu.x = 0.0
    cpu.y = 0.0
    cpu.psi = 0.1
    cpu.u = 1.0
    cpu.v = 0.0
    cpu.r = 0.0
    cpu._u_target = 1.0
    cpu._r_target = 0.0
    cpu._time_to_resample = 0.1  # will trigger on first step with dt=0.2

    # Record what CPU will draw
    # We can't easily intercept; instead drive both with known samples by
    # patching: run CPU once and mirror values into batched after detecting trigger.
    # Simpler: set CPU rng to known sequence by constructing Generator with seed
    # and replaying the same draws into batched.
    cpu.rng = np.random.default_rng(99)
    # Peek next two uniforms the CPU will use for u_target and interval
    peek = np.random.default_rng(99)
    u_sample = float(peek.uniform(cpu.speed_min, cpu.speed_max))
    t_sample = float(peek.uniform(cpu.target_resample_min_s, cpu.target_resample_max_s))

    st = BatchedShipState.zeros(1, dtype=torch.float64)
    st.x[0] = cpu.x
    st.y[0] = cpu.y
    st.psi[0] = cpu.psi
    st.u[0] = cpu.u
    st.u_target[0] = cpu._u_target
    st.time_to_resample[0] = cpu._time_to_resample

    dt = 0.2
    step_ships(
        st,
        dt,
        speed_min=cpu.speed_min,
        speed_max=cpu.speed_max,
        speed_tau=cpu.speed_tau,
        target_resample_min_s=cpu.target_resample_min_s,
        target_resample_max_s=cpu.target_resample_max_s,
        u_target_samples=torch.tensor([u_sample], dtype=torch.float64),
        resample_interval_samples=torch.tensor([t_sample], dtype=torch.float64),
    )
    cpu.step(dt)
    assert math.isclose(float(st.u_target[0]), cpu._u_target, rel_tol=RTOL, abs_tol=ATOL)
    assert math.isclose(float(st.time_to_resample[0]), cpu._time_to_resample, rel_tol=RTOL, abs_tol=ATOL)
    assert math.isclose(float(st.u[0]), cpu.u, rel_tol=RTOL, abs_tol=ATOL)
    assert math.isclose(float(st.x[0]), cpu.x, rel_tol=RTOL, abs_tol=ATOL)
