"""Z字试验与S形试验。

Z字试验（Zigzag maneuver）：经典 IMO 操纵性试验。两侧推进器以固定正转 RPM 前进，
方位角在 +delta / -delta 之间切换，当航向偏离到 +psi_switch 时翻向负方位角，
偏离到 -psi_switch 时翻向正方位角，循环若干次。

S形试验（Sinusoidal maneuver）：方位角指令为 sin 波形，船舶在水面上画出 S 形轨迹。

在项目根目录运行: python tests/test_maneuvers.py
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from utils.mpl_fonts import configure_matplotlib_fonts
from physics.tugboat_dynamics_model import TugboatDynamicsModel


OUTPUT_DIR = Path(__file__).resolve().parent


@dataclass
class TestLog:
    t: list[float]
    x: list[float]
    y: list[float]
    heading_deg: list[float]
    yaw_rate_degs: list[float]
    speed_ms: list[float]
    u_ms: list[float]
    v_ms: list[float]
    az_cmd_deg: list[float]
    az_actual_deg: list[float]
    rpm_actual: list[float]


def _new_log() -> TestLog:
    return TestLog([], [], [], [], [], [], [], [], [], [], [])


def _record(log: TestLog, t: float, model: TugboatDynamicsModel) -> None:
    state = model.get_state_snapshot()
    ctrl = model.get_control_snapshot()
    log.t.append(t)
    log.x.append(state["position_m"].x)
    log.y.append(state["position_m"].y)
    log.heading_deg.append(state["heading_deg"])
    log.yaw_rate_degs.append(state["yaw_rate_degs"])
    log.speed_ms.append(state["speed_ms"])
    log.u_ms.append(state["u_ms"])
    log.v_ms.append(state["v_ms"])
    log.az_cmd_deg.append(0.5 * (ctrl["port_azimuth_cmd_deg"] + ctrl["starboard_azimuth_cmd_deg"]))
    log.az_actual_deg.append(
        0.5 * (ctrl["port_azimuth_actual_deg"] + ctrl["starboard_azimuth_actual_deg"])
    )
    log.rpm_actual.append(0.5 * (ctrl["port_rpm_actual"] + ctrl["starboard_rpm_actual"]))


def run_zigzag(
    delta_deg: float = 15.0,
    psi_switch_deg: float = 10.0,
    rpm_cruise: float = 150.0,
    duration_s: float = 300.0,
    dt: float = 0.05,
    warmup_s: float = 20.0,
) -> TestLog:
    """Z字试验：先直航预热到稳态前进，再开始 ±delta 翻舵试验。

    在本动力学约定下，方位角与偏航力矩反号（正方位角 → 负偏航），
    因此 target_side 与 azimuth_cmd 取反号：target_side=+1 表示想把航向推向 +psi_switch，
    此时应施加 -delta_deg 的方位角。
    """
    model = TugboatDynamicsModel()
    model.reset()

    log = _new_log()
    n_steps = int(duration_s / dt)
    target_side = +1  # 起始想把航向推向哪一边

    for i in range(n_steps):
        t = i * dt

        if t < warmup_s:
            az_cmd = 0.0
        else:
            heading_deg = model.get_state_snapshot()["heading_deg"]
            # 越过阈值就翻向相反目标航向
            if target_side > 0 and heading_deg >= psi_switch_deg:
                target_side = -1
            elif target_side < 0 and heading_deg <= -psi_switch_deg:
                target_side = +1
            az_cmd = -target_side * delta_deg

        model.set_control_commands(rpm_cruise, rpm_cruise, az_cmd, az_cmd)
        model.step(dt)
        _record(log, t, model)

    return log


def run_sinusoidal(
    amplitude_deg: float = 25.0,
    period_s: float = 60.0,
    rpm_cruise: float = 180.0,
    duration_s: float = 240.0,
    dt: float = 0.05,
    warmup_s: float = 15.0,
) -> TestLog:
    """S形试验：方位角指令为正弦波。"""
    model = TugboatDynamicsModel()
    model.reset()

    log = _new_log()
    n_steps = int(duration_s / dt)
    omega = 2.0 * math.pi / period_s

    for i in range(n_steps):
        t = i * dt

        if t < warmup_s:
            az_cmd = 0.0
        else:
            az_cmd = amplitude_deg * math.sin(omega * (t - warmup_s))

        model.set_control_commands(rpm_cruise, rpm_cruise, az_cmd, az_cmd)
        model.step(dt)
        _record(log, t, model)

    return log


def plot_test(log: TestLog, title: str, save_path: str) -> None:
    configure_matplotlib_fonts()
    t = np.array(log.t)

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)

    # 轨迹图
    ax_traj = fig.add_subplot(gs[:, 0])
    ax_traj.plot(log.y, log.x, color="tab:blue", linewidth=1.5)
    ax_traj.scatter([log.y[0]], [log.x[0]], color="green", s=60, zorder=5, label="start")
    ax_traj.scatter([log.y[-1]], [log.x[-1]], color="red", s=60, zorder=5, label="end")
    ax_traj.set_xlabel("East  y [m]")
    ax_traj.set_ylabel("North  x [m]")
    ax_traj.set_title(f"{title} — trajectory (NED)")
    ax_traj.axis("equal")
    ax_traj.grid(True, alpha=0.3)
    ax_traj.legend(loc="best")

    # 航向角
    ax_psi = fig.add_subplot(gs[0, 1])
    ax_psi.plot(t, log.heading_deg, color="tab:blue", label="heading psi")
    ax_psi.set_ylabel("heading [deg]")
    ax_psi.set_title("Heading & azimuth command")
    ax_psi.grid(True, alpha=0.3)
    ax_psi.legend(loc="upper left")
    ax_az = ax_psi.twinx()
    ax_az.plot(t, log.az_cmd_deg, color="tab:orange", linestyle="--", label="azimuth cmd")
    ax_az.plot(t, log.az_actual_deg, color="tab:red", linestyle=":", label="azimuth actual")
    ax_az.set_ylabel("azimuth [deg]")
    ax_az.legend(loc="upper right")

    # 偏航角速度
    ax_r = fig.add_subplot(gs[1, 1])
    ax_r.plot(t, log.yaw_rate_degs, color="tab:purple")
    ax_r.set_ylabel("yaw rate [deg/s]")
    ax_r.set_title("Yaw rate r")
    ax_r.grid(True, alpha=0.3)

    # 速度
    ax_v = fig.add_subplot(gs[2, 1])
    ax_v.plot(t, log.speed_ms, color="tab:green", label="speed")
    ax_v.plot(t, log.u_ms, color="tab:blue", linestyle="--", label="u (surge)")
    ax_v.plot(t, log.v_ms, color="tab:red", linestyle=":", label="v (sway)")
    ax_v.set_xlabel("t [s]")
    ax_v.set_ylabel("velocity [m/s]")
    ax_v.set_title("Body-frame velocities")
    ax_v.grid(True, alpha=0.3)
    ax_v.legend(loc="best")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {save_path}")


def main() -> None:
    print("running zigzag test ...")
    zigzag_log = run_zigzag()
    plot_test(
        zigzag_log,
        "Z-shape (Zigzag) test — 15° azimuth, ±10° heading switch, 150 RPM",
        os.path.join(OUTPUT_DIR, "zigzag_test.png"),
    )

    print("running sinusoidal (S-shape) test ...")
    sine_log = run_sinusoidal()
    plot_test(
        sine_log,
        "S-shape (Sinusoidal) test — ±25° azimuth, T=60 s, both thrusters at 180 RPM",
        os.path.join(OUTPUT_DIR, "s_shape_test.png"),
    )


if __name__ == "__main__":
    main()
