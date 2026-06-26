"""
纯动力学模型层：只负责参数、状态、执行器动态和 3DOF 积分。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _move_toward(current: float, target: float, max_delta: float) -> float:
    if abs(target - current) <= max_delta:
        return target
    return current + math.copysign(max_delta, target - current)


def _wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    @classmethod
    def zero(cls) -> "Vec3":
        return cls(0.0, 0.0, 0.0)


@dataclass
class Vec2:
    x: float = 0.0
    y: float = 0.0

    def length(self) -> float:
        return math.hypot(self.x, self.y)


@dataclass
class TugboatDynamicsModel:
    # 船体主尺度与质量参数
    mass_kg: float = 699000.0
    length_m: float = 36.0
    beam_m: float = 11.0
    draft_m: float = 2.5
    radius_of_gyration_ratio: float = 0.25

    # 附加质量与附加转动惯量比例
    added_mass_surge_ratio: float = 0.08
    added_mass_sway_ratio: float = 0.30
    added_inertia_yaw_ratio: float = 0.55

    # 水动力阻尼参数
    rho_water: float = 1025.0
    cd_surge: float = 0.70
    cd_sway: float = 4.00
    linear_damping_ratio: float = 0.20
    # 线性阻尼参考速度尺度：把二次阻尼系数（kg/m）换算成线性阻尼系数（kg/s）。
    linear_damping_ref_speed: float = 1.0
    yaw_damping_gain: float = 15.0

    # 尾鳍（skeg）升力参数：提供航向恢复力矩，对抗 Munk 力矩造成的航向不稳定。
    # 鳍处横向来流 v_local = v + r * skeg_x_m 产生攻角，升力 ≈ 0.5*rho*A*Cl_a*|u|*v_local，
    # 方向抵抗攻角，作用于艉部 → 恢复性偏航力矩。
    skeg_area_m2: float = 6.0
    skeg_lift_slope: float = 3.0      # 升力线斜率 Cl_alpha（每弧度，含有效展弦比修正）
    skeg_x_m: float = -15.0           # 鳍纵向位置（艉部，负值）

    # 螺旋桨推力模型参数
    prop_diameter_m: float = 2.4
    kt_forward: float = 0.40
    reverse_kt_scale: float = 0.60

    # 推进器安装位置
    thruster_x_m: float = -12.0
    thruster_half_span_m: float = 2.5

    # 控制上限、执行器速率限制与积分选项
    rpm_limit: float = 240.0
    azimuth_limit_deg: float = 90.0
    rpm_rate_limit: float = 120.0
    azimuth_rate_limit_deg: float = 30.0
    use_rigid_body_coriolis_only: bool = False
    max_integration_step_s: float = 0.02

    # eta = [x, y, psi]：地理坐标系下的位置与航向
    # nu  = [u, v, r]：船体系下的纵向、横向和偏航角速度
    eta: Vec3 = field(default_factory=Vec3.zero)
    nu: Vec3 = field(default_factory=Vec3.zero)

    # 控制指令值
    _port_rpm_cmd: float = 0.0
    _starboard_rpm_cmd: float = 0.0
    _port_azimuth_cmd_deg: float = 0.0
    _starboard_azimuth_cmd_deg: float = 0.0

    # 执行器实际值（考虑速率限制后的真实生效值）
    _port_rpm_actual: float = 0.0
    _starboard_rpm_actual: float = 0.0
    _port_azimuth_actual_deg: float = 0.0
    _starboard_azimuth_actual_deg: float = 0.0

    # 缓存最近一步的总控制力/力矩 tau=[Fx, Fy, N]
    _last_tau: Vec3 = field(default_factory=Vec3.zero)
    # 缓存最近一个内部积分子步的体系加速度 nu_dot=[u_dot, v_dot, r_dot]
    _last_nu_dot: Vec3 = field(default_factory=Vec3.zero)

    # 将外部配置字典同步到动力学模型内部。
    def set_config(self, config: dict[str, Any]) -> None:
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)

    # 设置控制指令，并在这里统一做限幅。
    def set_control_commands(
        self,
        port_rpm_cmd: float,
        starboard_rpm_cmd: float,
        port_azimuth_cmd_deg: float,
        starboard_azimuth_cmd_deg: float,
    ) -> None:
        self._port_rpm_cmd = _clamp(port_rpm_cmd, -self.rpm_limit, self.rpm_limit)
        self._starboard_rpm_cmd = _clamp(starboard_rpm_cmd, -self.rpm_limit, self.rpm_limit)
        self._port_azimuth_cmd_deg = _clamp(
            port_azimuth_cmd_deg, -self.azimuth_limit_deg, self.azimuth_limit_deg
        )
        self._starboard_azimuth_cmd_deg = _clamp(
            starboard_azimuth_cmd_deg, -self.azimuth_limit_deg, self.azimuth_limit_deg
        )

    # 启动或重置时，让执行器实际值与指令值直接对齐。
    def snap_actuators_to_commands(self) -> None:
        self._port_rpm_actual = self._port_rpm_cmd
        self._starboard_rpm_actual = self._starboard_rpm_cmd
        self._port_azimuth_actual_deg = self._port_azimuth_cmd_deg
        self._starboard_azimuth_actual_deg = self._starboard_azimuth_cmd_deg

    # 重置动力学状态与执行器状态。
    def reset(self) -> None:
        self.eta = Vec3.zero()
        self.nu = Vec3.zero()
        self.set_control_commands(0.0, 0.0, 0.0, 0.0)
        self._port_rpm_actual = 0.0
        self._starboard_rpm_actual = 0.0
        self._port_azimuth_actual_deg = 0.0
        self._starboard_azimuth_actual_deg = 0.0
        self._last_tau = Vec3.zero()
        self._last_nu_dot = Vec3.zero()

    # 允许场景适配层在不依赖内部实现的情况下同步位姿/速度。
    def set_state(self, new_eta: Vec3, new_nu: Vec3 | None = None) -> None:
        self.eta = Vec3(new_eta.x, new_eta.y, new_eta.z)
        self.nu = Vec3(new_nu.x, new_nu.y, new_nu.z) if new_nu is not None else Vec3.zero()
        self.eta.z = _wrap_pi(self.eta.z)
        self._last_nu_dot = Vec3.zero()

    # 动力学主入口：按最大内部步长切成多个子步。
    def step(self, delta: float) -> None:
        step_limit = max(self.max_integration_step_s, 0.001)
        num_substeps = max(1, int(math.ceil(delta / step_limit)))
        step_dt = delta / float(num_substeps)
        for _ in range(num_substeps):
            self._update_actuators(step_dt)
            self._last_tau = self._step_dynamics(step_dt)

    # 返回状态快照。
    def get_state_snapshot(self) -> dict[str, Any]:
        return {
            "eta": self.eta,
            "nu": self.nu,
            "position_m": Vec2(self.eta.x, self.eta.y),
            "heading_rad": self.eta.z,
            "heading_deg": math.degrees(self.eta.z),
            "u_ms": self.nu.x,
            "v_ms": self.nu.y,
            "speed_ms": Vec2(self.nu.x, self.nu.y).length(),
            "yaw_rate_rads": self.nu.z,
            "yaw_rate_degs": math.degrees(self.nu.z),
            "u_dot_mss": self._last_nu_dot.x,
            "v_dot_mss": self._last_nu_dot.y,
            "yaw_accel_radss": self._last_nu_dot.z,
        }

    # 返回控制快照。
    def get_control_snapshot(self) -> dict[str, Any]:
        return {
            "port_rpm_cmd": self._port_rpm_cmd,
            "starboard_rpm_cmd": self._starboard_rpm_cmd,
            "port_rpm_actual": self._port_rpm_actual,
            "starboard_rpm_actual": self._starboard_rpm_actual,
            "port_azimuth_cmd_deg": self._port_azimuth_cmd_deg,
            "starboard_azimuth_cmd_deg": self._starboard_azimuth_cmd_deg,
            "port_azimuth_actual_deg": self._port_azimuth_actual_deg,
            "starboard_azimuth_actual_deg": self._starboard_azimuth_actual_deg,
        }

    # 返回推进器在船体系下的位置和推力向量。
    def get_thruster_snapshot(self) -> dict[str, Vec2]:
        t_port = self._prop_thrust_from_rpm(self._port_rpm_actual)
        t_stbd = self._prop_thrust_from_rpm(self._starboard_rpm_actual)
        a_port = math.radians(self._port_azimuth_actual_deg)
        a_stbd = math.radians(self._starboard_azimuth_actual_deg)
        return {
            "port_position_body_m": Vec2(self.thruster_x_m, self.thruster_half_span_m),
            "starboard_position_body_m": Vec2(self.thruster_x_m, -self.thruster_half_span_m),
            "port_force_body_n": Vec2(math.cos(a_port) * t_port, math.sin(a_port) * t_port),
            "starboard_force_body_n": Vec2(math.cos(a_stbd) * t_stbd, math.sin(a_stbd) * t_stbd),
        }

    def get_last_tau(self) -> Vec3:
        return self._last_tau

    def get_last_nu_dot(self) -> Vec3:
        return self._last_nu_dot

    # 计算等效质量矩阵 M 的对角项。
    def _mass_terms(self) -> Vec3:
        safe_mass = max(self.mass_kg, 1.0)
        rg = max(self.radius_of_gyration_ratio * self.length_m, 0.001)
        i_z = safe_mass * rg * rg
        m_11 = safe_mass * max(1.0 + self.added_mass_surge_ratio, 0.001)
        m_22 = safe_mass * max(1.0 + self.added_mass_sway_ratio, 0.001)
        m_33 = i_z * max(1.0 + self.added_inertia_yaw_ratio, 0.001)
        return Vec3(m_11, m_22, m_33)

    # 计算 C(nu)*nu 项。
    def _coriolis_times_nu(self, m_terms: Vec3) -> Vec3:
        u, v, r = self.nu.x, self.nu.y, self.nu.z
        if self.use_rigid_body_coriolis_only:
            m_cx = self.mass_kg
            m_cy = self.mass_kg
            yaw_coupling = 0.0
        else:
            m_cx = m_terms.x
            m_cy = m_terms.y
            yaw_coupling = (m_terms.y - m_terms.x) * u * v
        return Vec3(-m_cy * v * r, m_cx * u * r, yaw_coupling)

    # 计算阻尼项 D(nu)。
    def _damping_forces(self) -> Vec3:
        a_front = self.beam_m * self.draft_m
        a_side = self.length_m * self.draft_m

        # 二次阻尼系数（量纲 kg/m）：力 = 系数 * |速度| * 速度
        x_uu = 0.5 * self.rho_water * self.cd_surge * a_front
        y_vv = 0.5 * self.rho_water * self.cd_sway * a_side
        n_rr = y_vv * self.length_m * self.length_m / 12.0

        # 线性阻尼系数：用参考速度尺度把 kg/m 换算成 kg/s，保证线性项也是力。
        # 平动用参考速度 u_ref，偏航用对应的参考角速度 u_ref / length。
        u_ref = max(self.linear_damping_ref_speed, 1e-6)
        r_ref = u_ref / max(self.length_m, 1e-6)
        x_u = x_uu * u_ref * self.linear_damping_ratio
        y_v = y_vv * u_ref * self.linear_damping_ratio
        n_r = n_rr * r_ref * self.linear_damping_ratio * self.yaw_damping_gain
        n_rr *= self.yaw_damping_gain

        u, v, r = self.nu.x, self.nu.y, self.nu.z
        return Vec3(
            x_u * u + x_uu * abs(u) * u,
            y_v * v + y_vv * abs(v) * v,
            n_r * r + n_rr * abs(r) * r,
        )

    # 由推进器位置和推力向量合成总控制输入 tau=[Fx, Fy, N]。
    def _compute_tau(self) -> Vec3:
        thrusters = self.get_thruster_snapshot()
        f_port: Vec2 = thrusters["port_force_body_n"]
        f_stbd: Vec2 = thrusters["starboard_force_body_n"]
        p_port: Vec2 = thrusters["port_position_body_m"]
        p_stbd: Vec2 = thrusters["starboard_position_body_m"]

        n_port = p_port.x * f_port.y - p_port.y * f_port.x
        n_stbd = p_stbd.x * f_stbd.y - p_stbd.y * f_stbd.x

        return Vec3(f_port.x + f_stbd.x, f_port.y + f_stbd.y, n_port + n_stbd)

    # 尾鳍升力：鳍处横向来流产生攻角，升力抵抗攻角并在艉部形成恢复偏航力矩。
    # 返回作用力/力矩 [Fx, Fy, N]（Fx≈0，鳍主要产生横向力与偏航力矩）。
    def _skeg_forces(self) -> Vec3:
        u, v, r = self.nu.x, self.nu.y, self.nu.z
        # 鳍处横向局部来流速度（船体系 y 向）。
        v_local = v + r * self.skeg_x_m
        # 升力 ∝ 0.5*rho*A*Cl_alpha*|u|*v_local，方向抵抗局部横向来流（取负号）。
        # 用 |u| 而非 u² 保证升力随纵向流速线性、随 v_local 线性（小攻角线化）。
        fy = -0.5 * self.rho_water * self.skeg_area_m2 * self.skeg_lift_slope * abs(u) * v_local
        # 力作用在 skeg_x_m 处：N = x * Fy（Fx 为 0）。
        n = self.skeg_x_m * fy
        return Vec3(0.0, fy, n)

    # 单个内部积分步。
    def _step_dynamics(self, delta: float) -> Vec3:
        tau = self._compute_tau()
        m_terms = self._mass_terms()
        cnu = self._coriolis_times_nu(m_terms)
        damping = self._damping_forces()
        skeg = self._skeg_forces()

        # skeg 是外加水动力（恢复力），与控制 tau 同号进入分子、与阻尼/科氏反号。
        nu_dot_x = (tau.x + skeg.x - cnu.x - damping.x) / m_terms.x
        nu_dot_y = (tau.y + skeg.y - cnu.y - damping.y) / m_terms.y
        nu_dot_z = (tau.z + skeg.z - cnu.z - damping.z) / m_terms.z
        self._last_nu_dot = Vec3(nu_dot_x, nu_dot_y, nu_dot_z)

        self.nu = Vec3(
            self.nu.x + nu_dot_x * delta,
            self.nu.y + nu_dot_y * delta,
            self.nu.z + nu_dot_z * delta,
        )

        psi = self.eta.z
        eta_dot_x = math.cos(psi) * self.nu.x - math.sin(psi) * self.nu.y
        eta_dot_y = math.sin(psi) * self.nu.x + math.cos(psi) * self.nu.y
        eta_dot_z = self.nu.z

        self.eta = Vec3(
            self.eta.x + eta_dot_x * delta,
            self.eta.y + eta_dot_y * delta,
            _wrap_pi(self.eta.z + eta_dot_z * delta),
        )
        return tau

    # 单推进器推力模型：T = Kt * rho * D^4 * n * |n|
    def _prop_thrust_from_rpm(self, n_rpm: float) -> float:
        n_rps = n_rpm / 60.0
        kt = self.kt_forward if n_rpm >= 0.0 else self.kt_forward * self.reverse_kt_scale
        return kt * self.rho_water * (self.prop_diameter_m ** 4) * n_rps * abs(n_rps)

    # 执行器动态：让实际值以有限速率逼近指令值。
    def _update_actuators(self, delta: float) -> None:
        rpm_delta_limit = self.rpm_rate_limit * delta
        az_delta_limit = self.azimuth_rate_limit_deg * delta

        self._port_rpm_actual = _move_toward(
            self._port_rpm_actual, self._port_rpm_cmd, rpm_delta_limit
        )
        self._starboard_rpm_actual = _move_toward(
            self._starboard_rpm_actual, self._starboard_rpm_cmd, rpm_delta_limit
        )
        self._port_azimuth_actual_deg = _move_toward(
            self._port_azimuth_actual_deg, self._port_azimuth_cmd_deg, az_delta_limit
        )
        self._starboard_azimuth_actual_deg = _move_toward(
            self._starboard_azimuth_actual_deg, self._starboard_azimuth_cmd_deg, az_delta_limit
        )

        if abs(self._port_rpm_actual) < 0.05:
            self._port_rpm_actual = 0.0
        if abs(self._starboard_rpm_actual) < 0.05:
            self._starboard_rpm_actual = 0.0
        if abs(self._port_azimuth_actual_deg) < 0.01:
            self._port_azimuth_actual_deg = 0.0
        if abs(self._starboard_azimuth_actual_deg) < 0.01:
            self._starboard_azimuth_actual_deg = 0.0
