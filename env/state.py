"""仿真状态快照与可变状态容器。

用于解耦 ``env/`` 子模块与 ``FormationEnv`` 之间的紧耦合：
子模块不再持有 ``FormationEnv`` 引用，而是通过 ``SimState``（只读快照）
和 ``MutableEpisodeState``（可变追踪状态）接收所需数据。

架构::

    FormationEnv (owns all state)
      │
      ├── SimState (frozen snapshot, rebuilt each step)
      │     ├── cfg, n_tugs, dt_ctrl
      │     ├── ship snapshot
      │     ├── tug snapshots (tuple)
      │     ├── slot positions
      │     ├── route waypoints (pre-computed)
      │     └── derived helpers (coordinate transforms, hull queries)
      │
      └── MutableEpisodeState (mutated by submodules, returned as updates)
            ├── in_zone_steps
            ├── route_stage
            ├── route_waypoints_body_cache
            └── prev_* tracking arrays
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from config import EnvConfig
from env.obs_spec import _GLOBAL_ACCEL_PER_TUG_DIM, _GLOBAL_PER_TUG_DIM, _GLOBAL_SHIP_DIM, _SHIP_LINEAR_ACCEL_SCALE, _TUG_LINEAR_ACCEL_SCALE, _TUG_YAW_ACCEL_SCALE
from physics.large_ship_model import _wrap_pi


def _closest_hull_point_body(
    x_body: float, y_body: float, length_m: float, beam_m: float
) -> tuple[float, float]:
    """Closest point on the rectangular collision hull in ship body coordinates."""
    l_half = length_m / 2.0
    b_half = beam_m / 2.0
    clamped_x = float(np.clip(x_body, -l_half, l_half))
    clamped_y = float(np.clip(y_body, -b_half, b_half))

    if abs(x_body) <= l_half and abs(y_body) <= b_half:
        dx_edge = l_half - abs(x_body)
        dy_edge = b_half - abs(y_body)
        if dx_edge < dy_edge:
            clamped_x = math.copysign(l_half, x_body if x_body != 0.0 else 1.0)
        else:
            clamped_y = math.copysign(b_half, y_body if y_body != 0.0 else 1.0)

    return clamped_x, clamped_y


def world_to_local(dx: float, dy: float, psi_local: float) -> tuple[float, float]:
    """世界系向量旋转到本地坐标系。"""
    c = math.cos(psi_local)
    s = math.sin(psi_local)
    return c * dx + s * dy, -s * dx + c * dy


def local_to_world(dx_local: float, dy_local: float, psi_local: float) -> tuple[float, float]:
    """局部坐标系向量旋转到世界系。"""
    c = math.cos(psi_local)
    s = math.sin(psi_local)
    return c * dx_local - s * dy_local, s * dx_local + c * dy_local


@dataclass
class ShipSnapshot:
    """不可变的大船状态快照。"""
    x: float
    y: float
    psi: float
    u: float       # 船体系纵荡速度
    v: float       # 船体系横荡速度
    r: float       # 船体系艏摇角速度
    u_dot: float   # 船体系纵荡加速度
    length_m: float
    beam_m: float
    slot_lon_offset_m: float = 30.0
    slot_lat_offset_m: float = 10.0

    def body_to_world(self, x_body: float, y_body: float) -> tuple[float, float]:
        """船体坐标 → 世界坐标。"""
        dx_w, dy_w = local_to_world(x_body, y_body, self.psi)
        return self.x + dx_w, self.y + dy_w

    def world_to_body(self, x_world: float, y_world: float) -> tuple[float, float]:
        """世界坐标 → 船体坐标。"""
        return world_to_local(x_world - self.x, y_world - self.y, self.psi)

    def world_velocity(self) -> tuple[float, float]:
        """大船在世界系下的速度分量。"""
        c = math.cos(self.psi)
        s = math.sin(self.psi)
        return c * self.u - s * self.v, s * self.u + c * self.v

    def distance_from_hull(self, x_world: float, y_world: float) -> float:
        """世界坐标点到船体外廓的最近距离（内部为负值）。"""
        dx = x_world - self.x
        dy = y_world - self.y
        cos_p = math.cos(self.psi)
        sin_p = math.sin(self.psi)
        x_b = cos_p * dx + sin_p * dy
        y_b = -sin_p * dx + cos_p * dy
        l_half = self.length_m / 2.0
        b_half = self.beam_m / 2.0
        ex = max(abs(x_b) - l_half, 0.0)
        ey = max(abs(y_b) - b_half, 0.0)
        return math.hypot(ex, ey)

    def distance_from_hull_pose(
        self,
        x_world: float,
        y_world: float,
        ship_x: float,
        ship_y: float,
        ship_psi: float,
    ) -> float:
        """给定大船位姿下，世界坐标点到船体外廓的最近距离。"""
        dx = x_world - ship_x
        dy = y_world - ship_y
        cos_p = math.cos(ship_psi)
        sin_p = math.sin(ship_psi)
        x_b = cos_p * dx + sin_p * dy
        y_b = -sin_p * dx + cos_p * dy
        l_half = self.length_m / 2.0
        b_half = self.beam_m / 2.0
        ex = max(abs(x_b) - l_half, 0.0)
        ey = max(abs(y_b) - b_half, 0.0)
        return math.hypot(ex, ey)

    def slot_positions_body(self) -> np.ndarray:
        """返回 4 个 slot 在船体坐标系下的位置 (4, 2)。"""
        L = self.length_m / 2.0
        lon = self.slot_lon_offset_m
        lat = self.slot_lat_offset_m + self.beam_m / 2.0
        return np.array(
            [
                [+L + lon, -lat],
                [+L + lon, +lat],
                [-L - lon, -lat],
                [-L - lon, +lat],
            ],
            dtype=np.float64,
        )

    def slot_positions_world(self) -> np.ndarray:
        """返回 4 个 slot 在世界坐标系下的位置和朝向 (4, 3)。"""
        body = self.slot_positions_body()
        world = np.zeros((4, 3), dtype=np.float64)
        for k in range(4):
            wx, wy = self.body_to_world(float(body[k, 0]), float(body[k, 1]))
            world[k, 0] = wx
            world[k, 1] = wy
            # slot 期望航向 = 大船航向（与 LargeShipModel 一致）
            world[k, 2] = self.psi
        return world


@dataclass
class TugSnapshot:
    """不可变的单艇状态快照。"""
    x: float
    y: float
    psi: float
    u: float       # 船体系纵荡速度
    v: float       # 船体系横荡速度
    r: float       # 船体系艏摇角速度
    u_dot: float
    v_dot: float
    r_dot: float
    port_rpm_actual: float
    starboard_rpm_actual: float
    port_azimuth_actual_deg: float
    starboard_azimuth_actual_deg: float
    rpm_limit: float
    azimuth_limit_deg: float

    def world_velocity(self) -> tuple[float, float]:
        """世界系速度分量。"""
        c = math.cos(self.psi)
        s = math.sin(self.psi)
        return c * self.u - s * self.v, s * self.u + c * self.v


@dataclass
class SimState:
    """一次 step 的不可变仿真快照。

    由 ``FormationEnv`` 在 reset/step 后构建，传递给 Observer、RewardComputer、
    RoutePlanner 等子模块，替代原来直接传递 ``FormationEnv`` 引用的紧耦合设计。

    子模块通过 ``SimState`` 读取所需数据，通过 ``MutableEpisodeState`` 读写可变追踪状态。
    """
    cfg: EnvConfig
    n_tugs: int
    dt_ctrl: float
    ship: ShipSnapshot
    tugs: tuple[TugSnapshot, ...]
    slot_positions_world: np.ndarray   # (n_tugs, 3) world frame
    tug_to_slot: np.ndarray            # (n_tugs,) int
    route_stage: np.ndarray            # (n_tugs,) int
    route_waypoints_body: dict[int, np.ndarray]   # tug_idx → (M, 2) body frame
    route_waypoints_world: dict[int, np.ndarray]  # tug_idx → (M, 2) world frame
    last_actions: np.ndarray           # (n_tugs, 4)
    init_mode: str

    @property
    def uses_route_mode(self) -> bool:
        return self.init_mode == "mixed_slot_approach"

    # -- route accessors (delegated from RoutePlanner) --

    def route_waypoints_body_for_tug(self, tug_idx: int) -> np.ndarray:
        return self.route_waypoints_body.get(tug_idx, np.zeros((0, 2), dtype=np.float64))

    def route_waypoints_world_for_tug(self, tug_idx: int) -> np.ndarray:
        return self.route_waypoints_world.get(tug_idx, np.zeros((0, 2), dtype=np.float64))

    def current_route_target_world(self, tug_idx: int) -> np.ndarray:
        waypoints = self.route_waypoints_world_for_tug(tug_idx)
        stage = int(np.clip(self.route_stage[tug_idx], 0, len(waypoints) - 1))
        return waypoints[stage]

    def route_remaining_distance(self, tug_idx: int) -> float:
        waypoints = self.route_waypoints_world_for_tug(tug_idx)
        stage = int(np.clip(self.route_stage[tug_idx], 0, len(waypoints) - 1))
        tug = self.tugs[tug_idx]
        rem = math.hypot(tug.x - waypoints[stage, 0], tug.y - waypoints[stage, 1])
        for k in range(stage, len(waypoints) - 1):
            rem += float(np.linalg.norm(waypoints[k + 1] - waypoints[k]))
        return float(rem)

    # -- hull clearance helper --

    def closest_hull_point_world(self, tug_idx: int) -> tuple[float, float]:
        """拖轮在世界系下最近船体边界点的世界坐标。"""
        tug = self.tugs[tug_idx]
        tug_x_body, tug_y_body = self.ship.world_to_body(tug.x, tug.y)
        hull_x_body, hull_y_body = _closest_hull_point_body(
            float(tug_x_body), float(tug_y_body),
            self.ship.length_m, self.ship.beam_m,
        )
        return self.ship.body_to_world(hull_x_body, hull_y_body)

    # -- global state construction helper (used by Observer.get_global_state) --

    def build_global_state_array(self, in_zone_steps: np.ndarray) -> np.ndarray:
        """构建 90 维全局状态数组（供中心化 Critic 使用）。"""
        n = self.n_tugs
        total_dim = _GLOBAL_SHIP_DIM + _GLOBAL_PER_TUG_DIM * n + _GLOBAL_ACCEL_PER_TUG_DIM * n
        state = np.zeros(total_dim, dtype=np.float32)

        state[0] = float(self.ship.u) / 5.0
        state[1] = float(self.ship.u_dot) / _SHIP_LINEAR_ACCEL_SCALE

        cs_s = math.cos(self.ship.psi)
        sn_s = math.sin(self.ship.psi)
        hold_steps = max(1, int(round(self.cfg.hold_time_s / self.dt_ctrl)))

        for i, tug in enumerate(self.tugs):
            base = _GLOBAL_SHIP_DIM + i * _GLOBAL_PER_TUG_DIM

            x_b, y_b = self.ship.world_to_body(tug.x, tug.y)
            state[base + 0] = float(x_b) / 100.0
            state[base + 1] = float(y_b) / 100.0

            ci = math.cos(tug.psi)
            si = math.sin(tug.psi)
            vx_w = ci * tug.u - si * tug.v
            vy_w = si * tug.u + ci * tug.v
            u_b = cs_s * vx_w + sn_s * vy_w
            v_b = -sn_s * vx_w + cs_s * vy_w
            state[base + 2] = float(u_b) / 5.0
            state[base + 3] = float(v_b) / 5.0

            dpsi_ship = _wrap_pi(tug.psi - self.ship.psi)
            state[base + 4] = math.sin(dpsi_ship)
            state[base + 5] = math.cos(dpsi_ship)
            state[base + 6] = float(tug.r) / 0.5

            state[base + 7] = tug.port_rpm_actual / tug.rpm_limit
            state[base + 8] = tug.starboard_rpm_actual / tug.rpm_limit
            state[base + 9] = tug.port_azimuth_actual_deg / tug.azimuth_limit_deg
            state[base + 10] = tug.starboard_azimuth_actual_deg / tug.azimuth_limit_deg

            state[base + 11:base + 15] = self.last_actions[i]

            route_len = len(self.route_waypoints_body_for_tug(i))
            stage_norm = float(self.route_stage[i]) / max(route_len - 1, 1)
            state[base + 15] = float(np.clip(stage_norm, 0.0, 1.0))
            state[base + 16] = float(self.route_remaining_distance(i)) / 500.0

            state[base + 17] = float(in_zone_steps[i]) / float(hold_steps)

            d_hull = self.ship.distance_from_hull(tug.x, tug.y)
            state[base + 18] = float(d_hull) / 50.0

            acc_base = _GLOBAL_SHIP_DIM + _GLOBAL_PER_TUG_DIM * n + i * _GLOBAL_ACCEL_PER_TUG_DIM
            ax_w = ci * tug.u_dot - si * tug.v_dot
            ay_w = si * tug.u_dot + ci * tug.v_dot
            ax_b = cs_s * ax_w + sn_s * ay_w
            ay_b = -sn_s * ax_w + cs_s * ay_w
            state[acc_base + 0] = float(ax_b) / _TUG_LINEAR_ACCEL_SCALE
            state[acc_base + 1] = float(ay_b) / _TUG_LINEAR_ACCEL_SCALE
            state[acc_base + 2] = float(tug.r_dot) / _TUG_YAW_ACCEL_SCALE

        np.clip(state, -10.0, 10.0, out=state)
        return state


def _make_tug_snapshot(tug) -> TugSnapshot:
    """从 ``TugboatDynamicsModel`` 构建 ``TugSnapshot``。"""
    ctrl = tug.get_control_snapshot()
    acc = tug.get_last_nu_dot()
    return TugSnapshot(
        x=tug.eta.x,
        y=tug.eta.y,
        psi=tug.eta.z,
        u=tug.nu.x,
        v=tug.nu.y,
        r=tug.nu.z,
        u_dot=acc.x,
        v_dot=acc.y,
        r_dot=acc.z,
        port_rpm_actual=ctrl["port_rpm_actual"],
        starboard_rpm_actual=ctrl["starboard_rpm_actual"],
        port_azimuth_actual_deg=ctrl["port_azimuth_actual_deg"],
        starboard_azimuth_actual_deg=ctrl["starboard_azimuth_actual_deg"],
        rpm_limit=tug.rpm_limit,
        azimuth_limit_deg=tug.azimuth_limit_deg,
    )


def _make_ship_snapshot(ship) -> ShipSnapshot:
    """从 ``LargeShipModel`` 构建 ``ShipSnapshot``。"""
    return ShipSnapshot(
        x=ship.x,
        y=ship.y,
        psi=ship.psi,
        u=ship.u,
        v=ship.v,
        r=ship.r,
        u_dot=ship.u_dot,
        length_m=ship.length_m,
        beam_m=ship.beam_m,
        slot_lon_offset_m=getattr(ship, "slot_lon_offset_m", 30.0),
        slot_lat_offset_m=getattr(ship, "slot_lat_offset_m", 10.0),
    )


@dataclass
class MutableEpisodeState:
    """可变容器：episode 级别的追踪状态，由子模块读写。

    ``FormationEnv`` 拥有此对象，子模块通过引用来读写其中的字段。
    这种设计避免了子模块需要通过 ``self._env`` 来修改 env 的私有状态。
    """
    in_zone_steps: np.ndarray           # (n_tugs,) int
    route_stage: np.ndarray             # (n_tugs,) int
    route_waypoints_body_cache: dict[int, np.ndarray]
    mixed_ready_tugs: set[int]
    prev_dist: np.ndarray               # (n_tugs,)
    prev_route_remaining: np.ndarray    # (n_tugs,)
    prev_d_hull: np.ndarray             # (n_tugs,)
    prev_speed_err: np.ndarray          # (n_tugs,)
    prev_heading_err: np.ndarray        # (n_tugs,)
