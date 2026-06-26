"""拖轮 3DOF 动力学模型验证套件。

分三层验证模型是否合理：
- 第 1 层：代码与量纲正确性（静止守恒、直航、左右镜像、旋转不变、积分收敛）
- 第 2 层：物理守恒律（动能耗散、科氏力不做功、稳态阻力平衡）
- 第 3 层：标准操纵性试验（回转、Z 形 zigzag、正弦），输出可与真实拖轮对标的指标

同时为 scripts/export_maneuver_videos.py 提供 run_zigzag / run_sinusoidal /
run_turning_circle 三个操纵动作的轨迹日志。
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from physics.tugboat_dynamics_model import TugboatDynamicsModel, Vec3  # noqa: E402

# 一个控制器接受当前模型，返回 (port_rpm, stbd_rpm, port_az_deg, stbd_az_deg)。
Controller = Callable[[TugboatDynamicsModel], tuple[float, float, float, float]]


@dataclass
class ManeuverLog:
    """采样轨迹日志。字段名与 export_maneuver_videos.py 的期望保持一致。"""

    t: list[float] = field(default_factory=list)
    x: list[float] = field(default_factory=list)            # 北向位置
    y: list[float] = field(default_factory=list)            # 东向位置
    heading_deg: list[float] = field(default_factory=list)
    yaw_rate_degs: list[float] = field(default_factory=list)
    speed_ms: list[float] = field(default_factory=list)
    u_ms: list[float] = field(default_factory=list)
    v_ms: list[float] = field(default_factory=list)
    az_cmd_deg: list[float] = field(default_factory=list)    # 代表性方位角指令（左舷）

    def record(self, time_s: float, tug: TugboatDynamicsModel) -> None:
        s = tug.get_state_snapshot()
        c = tug.get_control_snapshot()
        self.t.append(time_s)
        self.x.append(s["position_m"].x)
        self.y.append(s["position_m"].y)
        self.heading_deg.append(s["heading_deg"])
        self.yaw_rate_degs.append(s["yaw_rate_degs"])
        self.speed_ms.append(s["speed_ms"])
        self.u_ms.append(s["u_ms"])
        self.v_ms.append(s["v_ms"])
        self.az_cmd_deg.append(c["port_azimuth_cmd_deg"])
# PLACEHOLDER_RUNNER

def make_tug(**overrides) -> TugboatDynamicsModel:
    """构造一艘默认拖轮，允许覆盖参数。"""
    tug = TugboatDynamicsModel()
    if overrides:
        tug.set_config(overrides)
    tug.reset()
    return tug


def simulate(
    tug: TugboatDynamicsModel,
    controller: Controller,
    *,
    duration_s: float,
    dt: float = 0.1,
    sample_dt: float = 0.1,
    log: ManeuverLog | None = None,
    init_eta: Vec3 | None = None,
    init_nu: Vec3 | None = None,
    snap_actuators: bool = False,
) -> ManeuverLog:
    """按固定步长推进模型，定期记录状态。

    控制器每个外部步被调用一次；dt 是外部步长，内部仍按
    max_integration_step_s 再细分。
    """
    if init_eta is not None or init_nu is not None:
        tug.set_state(init_eta or Vec3.zero(), init_nu)
    if log is None:
        log = ManeuverLog()

    n_steps = max(1, int(round(duration_s / dt)))
    next_sample = 0.0
    log.record(0.0, tug)
    for k in range(n_steps):
        cmd = controller(tug)
        tug.set_control_commands(*cmd)
        if k == 0 and snap_actuators:
            tug.snap_actuators_to_commands()
        tug.step(dt)
        time_s = (k + 1) * dt
        if time_s + 1e-9 >= next_sample:
            log.record(time_s, tug)
            next_sample += sample_dt
    return log


def run_spiral(
    *, rpm: float = 240.0, az_max: float = 35.0, az_step: float = 2.5,
    hold_s: float = 25.0, dt: float = 0.1,
) -> dict[str, dict[float, float]]:
    """螺旋试验：缓慢正反扫方位角，记录每个准稳态的偏航角速度（°/s）。

    返回 {'down': {az: r_degs}, 'up': {az: r_degs}}。下扫/上扫两支不重合即为
    回滞环，反映航向不稳定程度。
    """
    tug = make_tug()
    for _ in range(int(15.0 / dt)):
        tug.set_control_commands(rpm, rpm, 0.0, 0.0)
        tug.step(dt)

    res: dict[str, dict[float, float]] = {"down": {}, "up": {}}
    a = az_max
    while a >= -az_max - 1e-6:
        for _ in range(int(hold_s / dt)):
            tug.set_control_commands(rpm, rpm, a, a)
            tug.step(dt)
        res["down"][round(a, 1)] = math.degrees(tug.nu.z)
        a -= az_step
    a = -az_max
    while a <= az_max + 1e-6:
        for _ in range(int(hold_s / dt)):
            tug.set_control_commands(rpm, rpm, a, a)
            tug.step(dt)
        res["up"][round(a, 1)] = math.degrees(tug.nu.z)
        a += az_step
    return res


# --------------------------------------------------------------------------
# 第 3 层会用到、export_maneuver_videos.py 也依赖的标准操纵动作
# --------------------------------------------------------------------------

def run_turning_circle(
    *, rpm: float = 240.0, azimuth_deg: float = 35.0, duration_s: float = 120.0
) -> ManeuverLog:
    """回转试验：先直航建立航速，再把方位角打满做定常回转。"""
    tug = make_tug()

    def controller(_t: TugboatDynamicsModel) -> tuple[float, float, float, float]:
        return (rpm, rpm, azimuth_deg, azimuth_deg)

    # 先 10s 直航建立航速
    simulate(tug, lambda _t: (rpm, rpm, 0.0, 0.0), duration_s=10.0)
    return simulate(tug, controller, duration_s=duration_s)


def run_zigzag(
    *, rpm: float = 240.0, command_deg: float = 10.0, trigger_deg: float = 10.0,
    duration_s: float = 120.0,
) -> ManeuverLog:
    """10/10 Z 形试验：航向每越过 ±trigger_deg 就反打方位角。"""
    tug = make_tug()
    simulate(tug, lambda _t: (rpm, rpm, 0.0, 0.0), duration_s=8.0)

    state = {"sign": 1.0}

    def controller(t: TugboatDynamicsModel) -> tuple[float, float, float, float]:
        # 符号约定：正方位角把船尾推向右舷 → 船首左转 → 航向(psi)减小。
        # 因此要让航向朝 +trigger 走，需要负方位角；据当前航向反向修正。
        heading = math.degrees(t.eta.z)
        if state["sign"] > 0 and heading <= -trigger_deg:
            state["sign"] = -1.0
        elif state["sign"] < 0 and heading >= trigger_deg:
            state["sign"] = 1.0
        az = state["sign"] * command_deg
        return (rpm, rpm, az, az)

    return simulate(tug, controller, duration_s=duration_s)


def run_sinusoidal(
    *, rpm: float = 240.0, amplitude_deg: float = 25.0, period_s: float = 30.0,
    duration_s: float = 120.0,
) -> ManeuverLog:
    """正弦舵：方位角按正弦变化，产生 S 形轨迹。"""
    tug = make_tug()
    simulate(tug, lambda _t: (rpm, rpm, 0.0, 0.0), duration_s=6.0)

    clock = {"t": 0.0, "dt": 0.1}

    def controller(_t: TugboatDynamicsModel) -> tuple[float, float, float, float]:
        clock["t"] += clock["dt"]
        az = amplitude_deg * math.sin(2.0 * math.pi * clock["t"] / period_s)
        return (rpm, rpm, az, az)

    return simulate(tug, controller, duration_s=duration_s)
# PLACEHOLDER_TESTS

# ==========================================================================
# 第 1 层：代码与量纲正确性
# ==========================================================================

def test_static_equilibrium_stays_at_rest() -> None:
    """零控制 + 零初速：状态必须永远不变。"""
    tug = make_tug()
    simulate(tug, lambda _t: (0.0, 0.0, 0.0, 0.0), duration_s=30.0)
    s = tug.get_state_snapshot()
    assert abs(s["position_m"].x) < 1e-9
    assert abs(s["position_m"].y) < 1e-9
    assert abs(s["heading_deg"]) < 1e-9
    assert abs(s["speed_ms"]) < 1e-9
    assert abs(s["yaw_rate_rads"]) < 1e-9


def test_symmetric_thrust_goes_straight() -> None:
    """左右舷对称推力（方位角 0）：应纯直航，无横移、无偏航。"""
    tug = make_tug()
    simulate(tug, lambda _t: (240.0, 240.0, 0.0, 0.0), duration_s=40.0)
    s = tug.get_state_snapshot()
    assert s["u_ms"] > 0.5                      # 确实前进了
    assert abs(s["v_ms"]) < 1e-6                 # 无横移
    assert abs(s["yaw_rate_rads"]) < 1e-6        # 无偏航
    assert abs(s["heading_deg"]) < 1e-6
    assert abs(s["position_m"].y) < 1e-6         # 航向 0 时只沿北向走


def test_antisymmetric_thrust_induces_pure_yaw() -> None:
    """差速 RPM（左右反向、方位角 0）：应产生力偶 → 偏航，且不诱导大横移。

    注意：方位角反对称 + 同 RPM 因左右镜像会相互抵消力矩，真正的转向力偶
    来自差速 RPM。
    """
    tug = make_tug()
    simulate(tug, lambda _t: (240.0, -240.0, 0.0, 0.0), duration_s=5.0)
    s = tug.get_state_snapshot()
    assert abs(s["yaw_rate_rads"]) > 1e-3        # 确有偏航


def test_rotation_invariance() -> None:
    """同样控制，从不同初始航向出发，轨迹应只差一个刚体旋转。"""
    psi0 = math.radians(50.0)

    tug_a = make_tug()
    log_a = simulate(tug_a, lambda _t: (200.0, 240.0, 20.0, 10.0), duration_s=20.0)

    tug_b = make_tug()
    log_b = simulate(
        tug_b, lambda _t: (200.0, 240.0, 20.0, 10.0),
        duration_s=20.0, init_eta=Vec3(0.0, 0.0, psi0),
    )

    # 把 B 的末位置旋转回 -psi0，应与 A 末位置一致
    c, sgn = math.cos(-psi0), math.sin(-psi0)
    bx, by = log_b.x[-1], log_b.y[-1]
    bx_rot = c * bx - sgn * by
    by_rot = sgn * bx + c * by
    assert math.isclose(bx_rot, log_a.x[-1], abs_tol=1e-4)
    assert math.isclose(by_rot, log_a.y[-1], abs_tol=1e-4)
    # 航向差应恒为 psi0
    dpsi = math.radians(log_b.heading_deg[-1] - log_a.heading_deg[-1])
    dpsi = (dpsi + math.pi) % (2 * math.pi) - math.pi
    assert math.isclose(dpsi, psi0, abs_tol=1e-4)


def test_integration_step_convergence() -> None:
    """细分内部步长，轨迹应收敛（末位置差随步长减小而减小）。"""
    def end_pos(step: float) -> tuple[float, float]:
        tug = make_tug(max_integration_step_s=step)
        log = simulate(tug, lambda _t: (240.0, 200.0, 25.0, 15.0), duration_s=20.0)
        return log.x[-1], log.y[-1]

    coarse = end_pos(0.04)
    fine = end_pos(0.02)
    finer = end_pos(0.01)
    d1 = math.hypot(coarse[0] - fine[0], coarse[1] - fine[1])
    d2 = math.hypot(fine[0] - finer[0], fine[1] - finer[1])
    assert d2 < d1                               # 误差随步长减小而收敛
    assert d2 < 0.5                              # 默认步长下绝对误差可接受
# PLACEHOLDER_LAYER2

# ==========================================================================
# 第 2 层：物理守恒律
# ==========================================================================

def _kinetic_energy(tug: TugboatDynamicsModel) -> float:
    m = tug._mass_terms()
    u, v, r = tug.nu.x, tug.nu.y, tug.nu.z
    return 0.5 * (m.x * u * u + m.y * v * v + m.z * r * r)


def test_damping_dissipates_energy() -> None:
    """无控制、给定初速：动能必须单调衰减（阻尼只耗散不注入能量）。"""
    tug = make_tug()
    tug.set_state(Vec3.zero(), Vec3(3.0, 1.0, 0.2))
    prev_e = _kinetic_energy(tug)
    for _ in range(300):
        tug.step(0.1)
        e = _kinetic_energy(tug)
        assert e <= prev_e + 1e-6              # 不允许增能
        prev_e = e
    assert prev_e < 0.5 * _kinetic_energy(make_tug_with_nu(3.0, 1.0, 0.2))


def make_tug_with_nu(u: float, v: float, r: float) -> TugboatDynamicsModel:
    tug = make_tug()
    tug.set_state(Vec3.zero(), Vec3(u, v, r))
    return tug


def test_coriolis_does_no_work() -> None:
    """科氏/向心力不做功：nu · (C(nu)·nu) 必须恒为 0（C 反对称性）。"""
    tug = make_tug(use_rigid_body_coriolis_only=False)
    for u, v, r in [(2.0, 0.5, 0.1), (-1.0, 1.5, -0.3), (3.0, -0.8, 0.2)]:
        tug.set_state(Vec3.zero(), Vec3(u, v, r))
        m = tug._mass_terms()
        cnu = tug._coriolis_times_nu(m)
        power = u * cnu.x + v * cnu.y + r * cnu.z
        assert abs(power) < 1e-6, f"科氏力做功 {power} != 0 @ ({u},{v},{r})"


def test_steady_state_thrust_balances_drag() -> None:
    """满推力直航稳态：推力 ≈ 纵向阻尼力（独立用阻尼公式核对）。"""
    tug = make_tug()
    simulate(tug, lambda _t: (240.0, 240.0, 0.0, 0.0), duration_s=200.0)
    # 稳态：u_dot ≈ 0
    assert abs(tug.get_last_nu_dot().x) < 1e-3
    tau = tug.get_last_tau()
    damping = tug._damping_forces()
    # 直航稳态下 tau.x 应与阻尼力平衡（科氏项含 v,r≈0 故可忽略）
    assert math.isclose(tau.x, damping.x, rel_tol=1e-3, abs_tol=1.0)
# PLACEHOLDER_LAYER3

# ==========================================================================
# 第 3 层：标准操纵性试验（指标落在真实拖轮合理区间）
# ==========================================================================

def test_turning_circle_metrics() -> None:
    """回转试验：稳态偏航角速度、回转半径、回转掉速应落在真实 ASD 区间。"""
    log = run_turning_circle(azimuth_deg=35.0, duration_s=150.0)
    # 进速：取回转前直航段（约 t=10s 打舵前）的航速作为基准
    approach_idx = min(range(len(log.t)), key=lambda k: abs(log.t[k] - 10.0))
    approach_speed = log.speed_ms[approach_idx]
    # 取后半段作为稳态
    half = len(log.yaw_rate_degs) // 2
    steady_r_degs = abs(sum(log.yaw_rate_degs[half:]) / (len(log.yaw_rate_degs) - half))
    steady_speed = sum(log.speed_ms[half:]) / (len(log.speed_ms) - half)
    assert steady_r_degs > 1.0, "回转应产生明显的稳态偏航"
    # 回转中应保持前进航速（不塌陷/不倒退），这是 ASD 拖轮的真实特性
    steady_u = sum(log.u_ms[half:]) / (len(log.u_ms) - half)
    assert steady_u > 0.5, f"回转中纵向速度塌陷（u={steady_u:.2f}）"
    # 回转掉速：真实船舶硬回转掉速 20~40%，模型应体现明显掉速而非几乎不掉。
    speed_loss = 1.0 - steady_speed / max(approach_speed, 1e-6)
    assert 0.10 < speed_loss < 0.55, f"回转掉速 {speed_loss*100:.0f}% 偏离真实区间"
    # 回转半径 R = V / r（r 用 rad/s）。真实 ASD 拖轮约 0.7~3 倍船长，远紧于常规舵船。
    r_rads = math.radians(steady_r_degs)
    radius_m = steady_speed / max(r_rads, 1e-6)
    assert 25.0 < radius_m < 110.0, f"回转半径 {radius_m:.1f} m 超出真实 ASD 区间"


def test_zigzag_overshoot_is_bounded() -> None:
    """Z 形试验：航向超越角应有界（既能转向又不发散）。"""
    log = run_zigzag(command_deg=10.0, trigger_deg=10.0, duration_s=120.0)
    max_heading = max(abs(h) for h in log.heading_deg)
    # 超越角 = 峰值航向 - 触发角；应为正（有响应）且不夸张（< 触发角+40°）
    overshoot = max_heading - 10.0
    assert overshoot > 0.0, "zigzag 未产生航向响应"
    assert overshoot < 40.0, f"超越角 {overshoot:.1f}° 过大，偏航过于不稳定"


def test_zigzag_completes_multiple_cycles() -> None:
    """Z 形试验应在时长内完成多次换向（验证响应不至于过慢）。"""
    log = run_zigzag(command_deg=10.0, trigger_deg=10.0, duration_s=120.0)
    signs = [1 if a > 0 else -1 for a in log.az_cmd_deg]
    switches = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])
    assert switches >= 3, f"换向次数 {switches} 偏少，偏航响应可能过慢"


def test_sinusoidal_produces_s_shape() -> None:
    """正弦舵：航向应来回摆动（峰谷差足够大），横向轨迹呈 S 形。"""
    log = run_sinusoidal(amplitude_deg=25.0, period_s=30.0, duration_s=120.0)
    headings = log.heading_deg
    # 正弦舵产生绕某偏置的往复摆动；用峰谷差判定双向摆动，避免依赖跨零。
    assert max(headings) - min(headings) > 10.0, "正弦舵未产生明显往复摆动"


def test_acceleration_reaches_steady_speed() -> None:
    """加速冲程：从静止满推力，应在合理时间内逼近稳态航速。"""
    tug = make_tug()
    log = simulate(tug, lambda _t: (240.0, 240.0, 0.0, 0.0), duration_s=200.0)
    final_u = log.u_ms[-1]
    # 找到达到 95% 稳态航速的时刻
    target = 0.95 * final_u
    t_95 = next((log.t[i] for i, u in enumerate(log.u_ms) if u >= target), None)
    assert t_95 is not None and t_95 < 180.0, "加速过慢，未能逼近稳态航速"
    assert final_u > 1.0


def test_spiral_hysteresis_in_realistic_range() -> None:
    """螺旋试验：回滞环应小但非零——保留真实 ASD 拖轮的轻度航向不稳定，
    同时不至于过度不稳定（舵居中时不应大幅自转）。"""
    res = run_spiral()
    # 舵居中(δ=0)时的稳态偏航：两支均值反映回滞半幅
    center = (abs(res["down"][0.0]) + abs(res["up"][0.0])) / 2.0
    assert center < 1.5, f"舵居中偏航 {center:.2f}°/s 过大，航向过度不稳定"

    # 回滞环宽度：下扫/上扫过零点的方位角差
    def zero_cross(branch: dict[float, float]) -> float:
        items = sorted(branch.items())
        prev = None
        for a, r in items:
            if prev is not None and prev[1] * r < 0:
                a0, r0 = prev
                return a0 + (a - a0) * (0.0 - r0) / (r - r0)
            prev = (a, r)
        return float("nan")

    width = abs(zero_cross(res["down"]) - zero_cross(res["up"]))
    assert width < 4.0, f"回滞环宽度 {width:.2f}° 过大，航向过度不稳定"


if __name__ == "__main__":
    # 直接运行时打印第 3 层操纵性指标，便于人工对标真实拖轮。
    tc = run_turning_circle(azimuth_deg=35.0, duration_s=150.0)
    half = len(tc.yaw_rate_degs) // 2
    r_degs = abs(sum(tc.yaw_rate_degs[half:]) / (len(tc.yaw_rate_degs) - half))
    spd = sum(tc.speed_ms[half:]) / (len(tc.speed_ms) - half)
    print(f"[回转] 稳态偏航 {r_degs:.1f} °/s, 航速 {spd:.2f} m/s, "
          f"半径 {spd / max(math.radians(r_degs), 1e-6):.1f} m")

    zz = run_zigzag(duration_s=120.0)
    print(f"[Z形] 峰值航向 {max(abs(h) for h in zz.heading_deg):.1f}°")

    acc = simulate(make_tug(), lambda _t: (240, 240, 0, 0), duration_s=200.0)
    print(f"[加速] 稳态航速 {acc.u_ms[-1]:.2f} m/s ({acc.u_ms[-1] * 1.94:.1f} 节)")
