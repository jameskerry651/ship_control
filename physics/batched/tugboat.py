"""Batched 3DOF tug dynamics matching ``TugboatDynamicsModel``."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from physics.batched.geometry import wrap_pi
from physics.tugboat_dynamics_model import TugboatDynamicsModel


@dataclass
class BatchedTugParams:
    """Scalar hydrodynamic / actuator parameters (broadcast over N,K)."""

    mass_kg: float = 699000.0
    length_m: float = 36.0
    beam_m: float = 11.0
    draft_m: float = 2.5
    radius_of_gyration_ratio: float = 0.25
    added_mass_surge_ratio: float = 0.08
    added_mass_sway_ratio: float = 0.30
    added_inertia_yaw_ratio: float = 0.55
    rho_water: float = 1025.0
    cd_surge: float = 0.70
    cd_sway: float = 4.00
    linear_damping_ratio: float = 0.20
    linear_damping_ref_speed: float = 1.0
    yaw_damping_gain: float = 15.0
    cd_cross_coupling: float = 2.0
    skeg_area_m2: float = 6.0
    skeg_lift_slope: float = 3.0
    skeg_x_m: float = -15.0
    prop_diameter_m: float = 2.4
    kt_forward: float = 0.40
    reverse_kt_scale: float = 0.60
    thruster_x_m: float = -12.0
    thruster_half_span_m: float = 2.5
    rpm_limit: float = 240.0
    azimuth_limit_deg: float = 90.0
    rpm_rate_limit: float = 120.0
    azimuth_rate_limit_deg: float = 30.0
    use_rigid_body_coriolis_only: bool = False
    max_integration_step_s: float = 0.02

    @classmethod
    def from_model(cls, model: TugboatDynamicsModel | None = None) -> "BatchedTugParams":
        m = model or TugboatDynamicsModel()
        return cls(
            mass_kg=m.mass_kg,
            length_m=m.length_m,
            beam_m=m.beam_m,
            draft_m=m.draft_m,
            radius_of_gyration_ratio=m.radius_of_gyration_ratio,
            added_mass_surge_ratio=m.added_mass_surge_ratio,
            added_mass_sway_ratio=m.added_mass_sway_ratio,
            added_inertia_yaw_ratio=m.added_inertia_yaw_ratio,
            rho_water=m.rho_water,
            cd_surge=m.cd_surge,
            cd_sway=m.cd_sway,
            linear_damping_ratio=m.linear_damping_ratio,
            linear_damping_ref_speed=m.linear_damping_ref_speed,
            yaw_damping_gain=m.yaw_damping_gain,
            cd_cross_coupling=m.cd_cross_coupling,
            skeg_area_m2=m.skeg_area_m2,
            skeg_lift_slope=m.skeg_lift_slope,
            skeg_x_m=m.skeg_x_m,
            prop_diameter_m=m.prop_diameter_m,
            kt_forward=m.kt_forward,
            reverse_kt_scale=m.reverse_kt_scale,
            thruster_x_m=m.thruster_x_m,
            thruster_half_span_m=m.thruster_half_span_m,
            rpm_limit=m.rpm_limit,
            azimuth_limit_deg=m.azimuth_limit_deg,
            rpm_rate_limit=m.rpm_rate_limit,
            azimuth_rate_limit_deg=m.azimuth_rate_limit_deg,
            use_rigid_body_coriolis_only=m.use_rigid_body_coriolis_only,
            max_integration_step_s=m.max_integration_step_s,
        )


@dataclass
class BatchedTugState:
    """SoA tug state. Shapes: eta/nu/tau/nu_dot (..., 3); actuators (..., 4)."""

    eta: torch.Tensor  # (..., 3) x,y,psi
    nu: torch.Tensor  # (..., 3) u,v,r
    rpm_cmd: torch.Tensor  # (..., 2) port, stbd
    az_cmd_deg: torch.Tensor  # (..., 2)
    rpm_actual: torch.Tensor  # (..., 2)
    az_actual_deg: torch.Tensor  # (..., 2)
    last_tau: torch.Tensor  # (..., 3)
    last_nu_dot: torch.Tensor  # (..., 3)

    @classmethod
    def zeros(
        cls,
        *batch_shape: int,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> "BatchedTugState":
        device = device or torch.device("cpu")
        z3 = torch.zeros(*batch_shape, 3, device=device, dtype=dtype)
        z2 = torch.zeros(*batch_shape, 2, device=device, dtype=dtype)
        return cls(
            eta=z3.clone(),
            nu=z3.clone(),
            rpm_cmd=z2.clone(),
            az_cmd_deg=z2.clone(),
            rpm_actual=z2.clone(),
            az_actual_deg=z2.clone(),
            last_tau=z3.clone(),
            last_nu_dot=z3.clone(),
        )


def _move_toward(current: torch.Tensor, target: torch.Tensor, max_delta: torch.Tensor | float) -> torch.Tensor:
    delta = target - current
    return torch.where(delta.abs() <= max_delta, target, current + torch.sign(delta) * max_delta)


def _prop_thrust(n_rpm: torch.Tensor, p: BatchedTugParams) -> torch.Tensor:
    n_rps = n_rpm / 60.0
    kt = torch.where(
        n_rpm >= 0.0,
        torch.full_like(n_rpm, p.kt_forward),
        torch.full_like(n_rpm, p.kt_forward * p.reverse_kt_scale),
    )
    return kt * p.rho_water * (p.prop_diameter_m ** 4) * n_rps * n_rps.abs()


def _update_actuators(state: BatchedTugState, p: BatchedTugParams, dt: float) -> None:
    rpm_lim = p.rpm_rate_limit * dt
    az_lim = p.azimuth_rate_limit_deg * dt
    state.rpm_actual = _move_toward(state.rpm_actual, state.rpm_cmd, rpm_lim)
    state.az_actual_deg = _move_toward(state.az_actual_deg, state.az_cmd_deg, az_lim)
    state.rpm_actual = torch.where(state.rpm_actual.abs() < 0.05, torch.zeros_like(state.rpm_actual), state.rpm_actual)
    state.az_actual_deg = torch.where(
        state.az_actual_deg.abs() < 0.01, torch.zeros_like(state.az_actual_deg), state.az_actual_deg
    )


def _mass_terms(p: BatchedTugParams, ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    safe_mass = max(p.mass_kg, 1.0)
    rg = max(p.radius_of_gyration_ratio * p.length_m, 0.001)
    i_z = safe_mass * rg * rg
    m11 = safe_mass * max(1.0 + p.added_mass_surge_ratio, 0.001)
    m22 = safe_mass * max(1.0 + p.added_mass_sway_ratio, 0.001)
    m33 = i_z * max(1.0 + p.added_inertia_yaw_ratio, 0.001)
    return (
        ref.new_full(ref.shape, m11),
        ref.new_full(ref.shape, m22),
        ref.new_full(ref.shape, m33),
    )


def _step_dynamics_once(state: BatchedTugState, p: BatchedTugParams, dt: float) -> None:
    u = state.nu[..., 0]
    v = state.nu[..., 1]
    r = state.nu[..., 2]

    t_port = _prop_thrust(state.rpm_actual[..., 0], p)
    t_stbd = _prop_thrust(state.rpm_actual[..., 1], p)
    a_port = torch.deg2rad(state.az_actual_deg[..., 0])
    a_stbd = torch.deg2rad(state.az_actual_deg[..., 1])
    f_port_x = torch.cos(a_port) * t_port
    f_port_y = torch.sin(a_port) * t_port
    f_stbd_x = torch.cos(a_stbd) * t_stbd
    f_stbd_y = torch.sin(a_stbd) * t_stbd
    p_port_x = p.thruster_x_m
    p_port_y = p.thruster_half_span_m
    p_stbd_x = p.thruster_x_m
    p_stbd_y = -p.thruster_half_span_m
    n_port = p_port_x * f_port_y - p_port_y * f_port_x
    n_stbd = p_stbd_x * f_stbd_y - p_stbd_y * f_stbd_x
    tau_x = f_port_x + f_stbd_x
    tau_y = f_port_y + f_stbd_y
    tau_z = n_port + n_stbd

    m11, m22, m33 = _mass_terms(p, u)
    if p.use_rigid_body_coriolis_only:
        m_cx = u.new_full(u.shape, p.mass_kg)
        m_cy = u.new_full(u.shape, p.mass_kg)
        yaw_coupling = torch.zeros_like(u)
    else:
        m_cx, m_cy = m11, m22
        yaw_coupling = (m22 - m11) * u * v
    cnu_x = -m_cy * v * r
    cnu_y = m_cx * u * r
    cnu_z = yaw_coupling

    a_front = p.beam_m * p.draft_m
    a_side = p.length_m * p.draft_m
    x_uu = 0.5 * p.rho_water * p.cd_surge * a_front
    y_vv = 0.5 * p.rho_water * p.cd_sway * a_side
    n_rr = y_vv * p.length_m * p.length_m / 12.0
    u_ref = max(p.linear_damping_ref_speed, 1e-6)
    r_ref = u_ref / max(p.length_m, 1e-6)
    x_u = x_uu * u_ref * p.linear_damping_ratio
    y_v = y_vv * u_ref * p.linear_damping_ratio
    n_r = n_rr * r_ref * p.linear_damping_ratio * p.yaw_damping_gain
    n_rr = n_rr * p.yaw_damping_gain
    x_cross = 0.5 * p.rho_water * p.cd_cross_coupling * a_front * v.abs() * u
    damp_x = x_u * u + x_uu * u.abs() * u + x_cross
    damp_y = y_v * v + y_vv * v.abs() * v
    damp_z = n_r * r + n_rr * r.abs() * r

    v_local = v + r * p.skeg_x_m
    fy = -0.5 * p.rho_water * p.skeg_area_m2 * p.skeg_lift_slope * u.abs() * v_local
    skeg_x = torch.zeros_like(u)
    skeg_y = fy
    skeg_z = p.skeg_x_m * fy

    nu_dot_x = (tau_x + skeg_x - cnu_x - damp_x) / m11
    nu_dot_y = (tau_y + skeg_y - cnu_y - damp_y) / m22
    nu_dot_z = (tau_z + skeg_z - cnu_z - damp_z) / m33
    state.last_nu_dot = torch.stack([nu_dot_x, nu_dot_y, nu_dot_z], dim=-1)
    state.nu = state.nu + state.last_nu_dot * dt

    psi = state.eta[..., 2]
    eta_dot_x = torch.cos(psi) * state.nu[..., 0] - torch.sin(psi) * state.nu[..., 1]
    eta_dot_y = torch.sin(psi) * state.nu[..., 0] + torch.cos(psi) * state.nu[..., 1]
    eta_dot_z = state.nu[..., 2]
    new_eta = state.eta.clone()
    new_eta[..., 0] = state.eta[..., 0] + eta_dot_x * dt
    new_eta[..., 1] = state.eta[..., 1] + eta_dot_y * dt
    new_eta[..., 2] = wrap_pi(state.eta[..., 2] + eta_dot_z * dt)
    state.eta = new_eta
    state.last_tau = torch.stack([tau_x, tau_y, tau_z], dim=-1)


def set_control_commands(
    state: BatchedTugState,
    actions_norm: torch.Tensor,
    p: BatchedTugParams,
) -> None:
    """actions_norm (..., 4) in [-1,1] → rpm/azimuth commands."""
    actions = actions_norm.clamp(-1.0, 1.0)
    state.rpm_cmd = torch.stack(
        [actions[..., 0] * p.rpm_limit, actions[..., 1] * p.rpm_limit], dim=-1
    )
    state.az_cmd_deg = torch.stack(
        [actions[..., 2] * p.azimuth_limit_deg, actions[..., 3] * p.azimuth_limit_deg],
        dim=-1,
    )


def step_tugs(state: BatchedTugState, p: BatchedTugParams, dt: float) -> None:
    """Advance tug state by ``dt`` with the same substepping as CPU model."""
    step_limit = max(p.max_integration_step_s, 0.001)
    num_substeps = max(1, int(torch.ceil(torch.tensor(dt / step_limit)).item()))
    step_dt = dt / float(num_substeps)
    for _ in range(num_substeps):
        _update_actuators(state, p, step_dt)
        _step_dynamics_once(state, p, step_dt)
