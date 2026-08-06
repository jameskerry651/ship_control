"""L0: batched tug dynamics vs CPU ``TugboatDynamicsModel``."""

from __future__ import annotations

import math

import torch

from physics.batched.tugboat import BatchedTugParams, BatchedTugState, set_control_commands, step_tugs
from physics.tugboat_dynamics_model import TugboatDynamicsModel, Vec3

RTOL = 1e-10
ATOL = 1e-12


def _sync_cpu_to_batched(cpu: TugboatDynamicsModel, st: BatchedTugState, idx: tuple[int, ...]) -> None:
    st.eta[idx] = torch.tensor([cpu.eta.x, cpu.eta.y, cpu.eta.z], dtype=st.eta.dtype)
    st.nu[idx] = torch.tensor([cpu.nu.x, cpu.nu.y, cpu.nu.z], dtype=st.nu.dtype)
    st.rpm_cmd[idx] = torch.tensor([cpu._port_rpm_cmd, cpu._starboard_rpm_cmd], dtype=st.eta.dtype)
    st.az_cmd_deg[idx] = torch.tensor(
        [cpu._port_azimuth_cmd_deg, cpu._starboard_azimuth_cmd_deg], dtype=st.eta.dtype
    )
    st.rpm_actual[idx] = torch.tensor(
        [cpu._port_rpm_actual, cpu._starboard_rpm_actual], dtype=st.eta.dtype
    )
    st.az_actual_deg[idx] = torch.tensor(
        [cpu._port_azimuth_actual_deg, cpu._starboard_azimuth_actual_deg], dtype=st.eta.dtype
    )


def _assert_close(cpu: TugboatDynamicsModel, st: BatchedTugState, idx: tuple[int, ...]) -> None:
    eta = st.eta[idx].detach().cpu().numpy()
    nu = st.nu[idx].detach().cpu().numpy()
    assert math.isclose(eta[0], cpu.eta.x, rel_tol=RTOL, abs_tol=ATOL)
    assert math.isclose(eta[1], cpu.eta.y, rel_tol=RTOL, abs_tol=ATOL)
    assert math.isclose(eta[2], cpu.eta.z, rel_tol=RTOL, abs_tol=ATOL)
    assert math.isclose(nu[0], cpu.nu.x, rel_tol=RTOL, abs_tol=ATOL)
    assert math.isclose(nu[1], cpu.nu.y, rel_tol=RTOL, abs_tol=ATOL)
    assert math.isclose(nu[2], cpu.nu.z, rel_tol=RTOL, abs_tol=ATOL)
    torch.testing.assert_close(
        st.rpm_actual[idx],
        torch.tensor([cpu._port_rpm_actual, cpu._starboard_rpm_actual], dtype=st.eta.dtype),
        rtol=RTOL,
        atol=ATOL,
    )
    torch.testing.assert_close(
        st.az_actual_deg[idx],
        torch.tensor(
            [cpu._port_azimuth_actual_deg, cpu._starboard_azimuth_actual_deg],
            dtype=st.eta.dtype,
        ),
        rtol=RTOL,
        atol=ATOL,
    )


def test_tug_step_matches_cpu_single():
    cpu = TugboatDynamicsModel()
    cpu.set_state(Vec3(10.0, -5.0, 0.3), Vec3(1.5, 0.2, -0.05))
    cpu.set_control_commands(80.0, -40.0, 20.0, -10.0)
    cpu.snap_actuators_to_commands()

    p = BatchedTugParams.from_model(cpu)
    st = BatchedTugState.zeros(1, 1, dtype=torch.float64)
    _sync_cpu_to_batched(cpu, st, (0, 0))

    actions = torch.tensor([[[80.0 / p.rpm_limit, -40.0 / p.rpm_limit, 20.0 / p.azimuth_limit_deg, -10.0 / p.azimuth_limit_deg]]], dtype=torch.float64)
    dt = 0.2
    for _ in range(25):
        set_control_commands(st, actions, p)
        step_tugs(st, p, dt)
        cpu.set_control_commands(80.0, -40.0, 20.0, -10.0)
        cpu.step(dt)
        _assert_close(cpu, st, (0, 0))


def test_tug_step_batched_matches_cpu_grid():
    """N=2,K=2 independent tugs each match their CPU twin."""
    cfgs = [
        (Vec3(0, 0, 0), Vec3(0, 0, 0), (0.0, 0.0, 0.0, 0.0)),
        (Vec3(5, 2, 1.0), Vec3(2.0, -0.5, 0.1), (100.0, 100.0, 45.0, -45.0)),
        (Vec3(-3, 8, -0.5), Vec3(0.5, 0.5, -0.2), (-120.0, 60.0, -30.0, 15.0)),
        (Vec3(20, -10, 2.0), Vec3(3.0, 0.0, 0.0), (200.0, -200.0, 90.0, 90.0)),
    ]
    cpus = []
    for eta, nu, cmd in cfgs:
        m = TugboatDynamicsModel()
        m.set_state(eta, nu)
        m.set_control_commands(*cmd)
        m.snap_actuators_to_commands()
        cpus.append((m, cmd))

    p = BatchedTugParams.from_model()
    st = BatchedTugState.zeros(2, 2, dtype=torch.float64)
    actions = torch.zeros(2, 2, 4, dtype=torch.float64)
    for i, (m, cmd) in enumerate(cpus):
        n, k = divmod(i, 2)
        _sync_cpu_to_batched(m, st, (n, k))
        actions[n, k] = torch.tensor(
            [
                cmd[0] / p.rpm_limit,
                cmd[1] / p.rpm_limit,
                cmd[2] / p.azimuth_limit_deg,
                cmd[3] / p.azimuth_limit_deg,
            ],
            dtype=torch.float64,
        )

    dt = 0.2
    for _ in range(15):
        set_control_commands(st, actions, p)
        step_tugs(st, p, dt)
        for i, (m, cmd) in enumerate(cpus):
            n, k = divmod(i, 2)
            m.set_control_commands(*cmd)
            m.step(dt)
            _assert_close(m, st, (n, k))
