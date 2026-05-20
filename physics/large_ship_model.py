"""大船的简化运动学模型与 4 个 slot 的几何定义。

大船被视为 3DOF 刚体（位置 x/y、航向 psi、体系速度 u/v/r），
但不再用拖轮那套水动力学，而是用一阶低通跟随随机目标，
让大船以平滑变化的速度与航向沿水面前进。

坐标约定（与 tugboat_dynamics_model 保持一致）：
- 世界系：x 朝北、y 朝东（NED），psi 顺时针为正
- 船体系：x 朝船首、y 朝右舷
- 旋转矩阵：world_pos = R(psi) @ body_pos
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def _wrap_pi(angle: float) -> float:
    """把任意角度规整到 (-pi, pi]。"""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class LargeShipModel:
    # 主尺度
    length_m: float = 200.0
    beam_m: float = 30.0

    # slot 在船体系下的偏移
    slot_lon_offset_m: float = 30.0
    slot_lat_offset_m: float = 25.0

    # 速度/航向变化范围
    speed_min: float = 0.5
    speed_max: float = 2.0
    yaw_rate_max: float = 0.02

    # 一阶低通时间常数
    speed_tau: float = 15.0
    yaw_tau: float = 20.0

    # 周期性重新采样目标速度/角速度
    target_resample_min_s: float = 20.0
    target_resample_max_s: float = 40.0

    # 当前状态
    x: float = 0.0
    y: float = 0.0
    psi: float = 0.0
    u: float = 1.0          # 体系纵向速度
    v: float = 0.0          # 体系横向速度（基本固定为 0）
    r: float = 0.0          # 偏航角速度
    u_dot: float = 0.0      # 体系纵向加速度
    v_dot: float = 0.0      # 体系横向加速度
    r_dot: float = 0.0      # 偏航角加速度

    # 内部目标值与计时
    _u_target: float = 1.0
    _r_target: float = 0.0
    _time_to_resample: float = 25.0

    # 随机数生成器（由环境注入）
    rng: np.random.Generator = field(default=None)

    def __post_init__(self) -> None:
        if self.rng is None:
            self.rng = np.random.default_rng()

    # 重置到初始状态：原点出发，航向随机，速度随机
    def reset(self, rng: np.random.Generator | None = None) -> None:
        if rng is not None:
            self.rng = rng
        self.x = 0.0
        self.y = 0.0
        self.psi = float(self.rng.uniform(-math.pi, math.pi))
        self.u = float(self.rng.uniform(self.speed_min, self.speed_max))
        self.v = 0.0
        self.r = 0.0
        self.u_dot = 0.0
        self.v_dot = 0.0
        self.r_dot = 0.0
        self._u_target = self.u
        self._r_target = 0.0
        self._time_to_resample = float(
            self.rng.uniform(self.target_resample_min_s, self.target_resample_max_s)
        )

    # 推进一步：一阶低通逼近目标速度，再积分位姿。
    # 当前训练阶段：大船仅做直行（不转向），航向在 reset 时随机一次后保持不变。
    def step(self, dt: float) -> None:
        self._time_to_resample -= dt
        if self._time_to_resample <= 0.0:
            self._u_target = float(self.rng.uniform(self.speed_min, self.speed_max))
            # 大船不转向：目标偏航角速度强制为 0
            self._r_target = 0.0
            self._time_to_resample = float(
                self.rng.uniform(self.target_resample_min_s, self.target_resample_max_s)
            )

        # 一阶低通跟随目标速度
        self.u_dot = (self._u_target - self.u) / max(self.speed_tau, 1e-3)
        self.u += self.u_dot * dt
        self.v_dot = 0.0
        # 偏航强制为 0（不转向）
        self.r_dot = 0.0
        self.r = 0.0

        # 位姿积分（与拖轮动力学一致的旋转约定）
        cos_p = math.cos(self.psi)
        sin_p = math.sin(self.psi)
        self.x += (cos_p * self.u - sin_p * self.v) * dt
        self.y += (sin_p * self.u + cos_p * self.v) * dt
        # psi 保持不变（直行）

    # 4 个 slot 在船体系下的固定坐标（顺序：船首左、船首右、船尾左、船尾右）
    def slot_positions_body(self) -> np.ndarray:
        L = self.length_m / 2.0
        lon = self.slot_lon_offset_m
        lat = self.slot_lat_offset_m + self.beam_m / 2.0
        return np.array(
            [
                [+L + lon, -lat],   # 船首左舷（右舷为 +y_body，左舷为 -y_body）
                [+L + lon, +lat],   # 船首右舷
                [-L - lon, -lat],   # 船尾左舷
                [-L - lon, +lat],   # 船尾右舷
            ],
            dtype=np.float64,
        )

    # 4 个 slot 在世界系下的位置和期望航向，shape=(4, 3)
    def slot_positions_world(self) -> np.ndarray:
        slots_body = self.slot_positions_body()
        cos_p = math.cos(self.psi)
        sin_p = math.sin(self.psi)
        # body → world：world = R @ body
        rot = np.array([[cos_p, -sin_p], [sin_p, cos_p]])
        slots_world_xy = slots_body @ rot.T
        slots_world_xy[:, 0] += self.x
        slots_world_xy[:, 1] += self.y
        # 所有 slot 期望航向 = 大船当前航向（拖轮船首与大船同向）
        psis = np.full((4, 1), self.psi)
        return np.concatenate([slots_world_xy, psis], axis=1)

    # 大船船体多边形（世界系），用于碰撞距离计算与可视化
    def hull_polygon_world(self) -> np.ndarray:
        L = self.length_m / 2.0
        B = self.beam_m / 2.0
        # 船首做尖头，船尾平直
        verts_body = np.array(
            [
                [+L,        0.0],
                [+L * 0.78, -B],
                [-L,        -B],
                [-L,        +B],
                [+L * 0.78, +B],
            ]
        )
        cos_p = math.cos(self.psi)
        sin_p = math.sin(self.psi)
        rot = np.array([[cos_p, -sin_p], [sin_p, cos_p]])
        verts_world = verts_body @ rot.T
        verts_world[:, 0] += self.x
        verts_world[:, 1] += self.y
        return verts_world

    # 任意点到大船船体外表面的最短距离（点在船体内时返回 0）
    def distance_from_hull(self, x_world: float, y_world: float) -> float:
        # 把世界点旋转到船体系，再用 box-outside-distance 公式
        dx = x_world - self.x
        dy = y_world - self.y
        cos_p = math.cos(self.psi)
        sin_p = math.sin(self.psi)
        x_b = cos_p * dx + sin_p * dy
        y_b = -sin_p * dx + cos_p * dy
        L = self.length_m / 2.0
        B = self.beam_m / 2.0
        ex = max(abs(x_b) - L, 0.0)
        ey = max(abs(y_b) - B, 0.0)
        return math.hypot(ex, ey)
