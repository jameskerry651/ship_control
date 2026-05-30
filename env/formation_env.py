"""多智能体拖轮编队环境。

任务：4 艘拖轮 + 1 艘移动的大船。每艘拖轮被分配到大船周围一个固定的 slot
（船首左/右、船尾左/右），需要平滑驶入 slot 并在大船前进过程中保持就位。

设计要点：
- 单环境一次接受 (n_tugs, action_dim) 的动作，返回 (n_tugs, obs_dim) 的观察。
- 4 个智能体共享同一份策略网络（参数共享），观察都用"以自身为参考系"的相对量。
- 奖励是逐 agent 计算，但碰撞/成功这种全局事件所有 agent 同步收到。
- 初始化时 slot 角色固定为 tug i → slot i。
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from scipy.interpolate import splprep, splev

from config import EnvConfig
from env.reward import FormationRewardComputer
from physics.large_ship_model import LargeShipModel, _wrap_pi
from physics.tugboat_dynamics_model import TugboatDynamicsModel, Vec3


# ---------- 动作/观察维度 ----------
ACTION_DIM = 4

# 单个智能体的 actor 观察默认 71 维，按以下顺序拼接：
#  [0:28]  自身历史 4 帧 × (u, v, sinψ, cosψ, Δu, Δv, Δr)
#  [28:44] 动作历史 4 帧 × (n_L, n_R, δ_L, δ_R)
#  [44:50] 大船相对位置与状态 (dx, dy, u, v, sinΔψ, cosΔψ)
#  [50:56] 大船中心轨迹前瞻 3 点 × (dx, dy)
#  [56:71] 3 个邻居的 attention 输入，每个 (dx, dy, du, dv, distance)
_EGO_MOTION_OBS_DIM = 7
_ACTION_HISTORY_OBS_DIM = ACTION_DIM
_SHIP_REL_OBS_DIM = 6
_SHIP_PREVIEW_POINT_DIM = 2
_NEIGHBOR_COUNT = 3
_NEIGHBOR_OBS_DIM = 5

# Critic 用的全局状态：所有量都在大船船体系下表达，跨 agent 共享。
# - ship 段 (5)：(u/5, v/5, r/0.05, length_bias, beam_bias)
# - per-tug 段 (23)：
#     [0:2]   tug 在船体系下的位置 (x_b, y_b) / 100
#     [2:4]   tug 在船体系下的速度 (u_b, v_b) / 5
#     [4:6]   tug 朝向相对船体系 (sin, cos)
#     [6]     tug 体系角速度 r / 0.5
#     [7:11]  执行器实际值（已经归一化）
#     [11:15] 上一步动作
#     [15:19] slot one-hot
#     [19]    route stage 进度 [0, 1]
#     [20]    route remaining / 500
#     [21]    in_zone 标志
#     [22]    距船体的最近距离 / 50
# - acceleration tail：每条 tug 6 维
#     [0:3]   tug 线加速度投影到船体系 + yaw acceleration，按 1 / 1 / 0.1 缩放
#     [3:6]   大船体系加速度，按 0.2 / 0.2 / 0.01 缩放
_GLOBAL_SHIP_DIM = 5
_GLOBAL_PER_TUG_DIM = 23
_GLOBAL_ACCEL_PER_TUG_DIM = 6

_TUG_LINEAR_ACCEL_SCALE = 1.0
_TUG_YAW_ACCEL_SCALE = 0.1
_SHIP_LINEAR_ACCEL_SCALE = 0.2
_SHIP_YAW_ACCEL_SCALE = 0.01


def _world_to_local(dx: float, dy: float, psi_local: float) -> tuple[float, float]:
    """把世界系下的相对向量旋转到本地坐标系（朝向 psi_local 的物体的体系）。"""
    c = math.cos(psi_local)
    s = math.sin(psi_local)
    # 逆旋转 R(psi)^T = R(-psi)
    x_local = c * dx + s * dy
    y_local = -s * dx + c * dy
    return x_local, y_local


def _local_to_world(dx_local: float, dy_local: float, psi_local: float) -> tuple[float, float]:
    """把局部坐标系向量旋转到世界系。"""
    c = math.cos(psi_local)
    s = math.sin(psi_local)
    x_world = c * dx_local - s * dy_local
    y_world = s * dx_local + c * dy_local
    return x_world, y_world


@dataclass
class FormationEnv:
    """多智能体拖轮编队环境，遵循类 Gymnasium 的接口。"""

    cfg: EnvConfig = field(default_factory=EnvConfig)
    seed: int | None = None

    # 内部状态（运行时填充）
    rng: np.random.Generator = field(init=False)
    tugs: list[TugboatDynamicsModel] = field(init=False)
    ship: LargeShipModel = field(init=False)
    n_tugs: int = field(init=False)
    step_count: int = field(init=False)
    last_actions: np.ndarray = field(init=False)
    last_action_changes: np.ndarray = field(init=False)   # da = a_t - a_{t-1}, for jerk
    motion_history: np.ndarray = field(init=False)
    action_history: np.ndarray = field(init=False)
    last_reward_components: dict = field(init=False)
    _route_waypoints_body_cache: dict[int, np.ndarray] = field(init=False)  # tug_idx -> waypoints
    _mixed_ready_tugs: set[int] = field(init=False)
    prev_dist: np.ndarray = field(init=False)
    in_zone_steps: np.ndarray = field(init=False)
    route_stage: np.ndarray = field(init=False)
    tug_to_slot: np.ndarray = field(init=False)  # 长度 n_tugs，元素是 slot 索引

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.n_tugs = self.cfg.n_tugs
        self.tugs = [TugboatDynamicsModel() for _ in range(self.n_tugs)]
        self.ship = LargeShipModel(
            length_m=self.cfg.ship_length_m,
            beam_m=self.cfg.ship_beam_m,
            slot_lon_offset_m=self.cfg.slot_lon_offset_m,
            slot_lat_offset_m=self.cfg.slot_lat_offset_m,
            speed_min=self.cfg.ship_speed_min,
            speed_max=self.cfg.ship_speed_max,
            yaw_rate_max=self.cfg.ship_yaw_rate_max,
            speed_tau=self.cfg.ship_speed_tau_s,
            yaw_tau=self.cfg.ship_yaw_tau_s,
            target_resample_min_s=self.cfg.ship_target_resample_min_s,
            target_resample_max_s=self.cfg.ship_target_resample_max_s,
            rng=self.rng,
        )
        self.step_count = 0
        self.last_actions = np.zeros((self.n_tugs, ACTION_DIM), dtype=np.float32)
        self.last_action_changes = np.zeros((self.n_tugs, ACTION_DIM), dtype=np.float32)
        hist_len = int(getattr(self.cfg, "obs_history_k", 3)) + 1
        self.motion_history = np.zeros(
            (self.n_tugs, hist_len, _EGO_MOTION_OBS_DIM), dtype=np.float32
        )
        self.action_history = np.zeros(
            (self.n_tugs, hist_len, ACTION_DIM), dtype=np.float32
        )
        self.last_reward_components = {}
        self._route_waypoints_body_cache = {}
        self._mixed_ready_tugs = set()
        self.prev_dist = np.zeros(self.n_tugs, dtype=np.float32)
        self.in_zone_steps = np.zeros(self.n_tugs, dtype=np.int32)
        self.route_stage = np.zeros(self.n_tugs, dtype=np.int32)
        self.tug_to_slot = np.arange(self.n_tugs, dtype=np.int32)
        self._reward = FormationRewardComputer(self)

    @property
    def obs_dim(self) -> int:
        hist_len = int(getattr(self.cfg, "obs_history_k", 3)) + 1
        preview_times = tuple(getattr(self.cfg, "obs_ship_preview_times_s", (5.0, 10.0, 15.0)))
        return (
            hist_len * _EGO_MOTION_OBS_DIM
            + hist_len * _ACTION_HISTORY_OBS_DIM
            + _SHIP_REL_OBS_DIM
            + len(preview_times) * _SHIP_PREVIEW_POINT_DIM
            + (self.n_tugs - 1) * _NEIGHBOR_OBS_DIM
        )

    @property
    def global_state_dim(self) -> int:
        """Critic 用的全局状态维度。所有 agent 在同一 env 内共享同一个向量。"""
        return (
            _GLOBAL_SHIP_DIM
            + _GLOBAL_PER_TUG_DIM * self.n_tugs
            + _GLOBAL_ACCEL_PER_TUG_DIM * self.n_tugs
        )

    @property
    def action_dim(self) -> int:
        return ACTION_DIM

    def _ensure_history_buffers(self) -> None:
        hist_len = int(getattr(self.cfg, "obs_history_k", 3)) + 1
        if (
            getattr(self, "motion_history", None) is None
            or self.motion_history.shape != (self.n_tugs, hist_len, _EGO_MOTION_OBS_DIM)
        ):
            self.motion_history = np.zeros(
                (self.n_tugs, hist_len, _EGO_MOTION_OBS_DIM), dtype=np.float32
            )
        if (
            getattr(self, "action_history", None) is None
            or self.action_history.shape != (self.n_tugs, hist_len, ACTION_DIM)
        ):
            self.action_history = np.zeros(
                (self.n_tugs, hist_len, ACTION_DIM), dtype=np.float32
            )

    def _motion_frame(self, tug: TugboatDynamicsModel, prev_nu: Vec3 | None) -> np.ndarray:
        if prev_nu is None:
            du = dv = dr = 0.0
        else:
            du = float(tug.nu.x - prev_nu.x)
            dv = float(tug.nu.y - prev_nu.y)
            dr = float(tug.nu.z - prev_nu.z)
        return np.asarray(
            [
                float(tug.nu.x) / 5.0,
                float(tug.nu.y) / 5.0,
                math.sin(float(tug.eta.z)),
                math.cos(float(tug.eta.z)),
                du / 5.0,
                dv / 5.0,
                dr / 0.5,
            ],
            dtype=np.float32,
        )

    def _fill_obs_history(self, actions: np.ndarray) -> None:
        self._ensure_history_buffers()
        actions = np.clip(actions, -1.0, 1.0).astype(np.float32, copy=False)
        for i, tug in enumerate(self.tugs):
            self.motion_history[i, :, :] = self._motion_frame(tug, prev_nu=None)
            self.action_history[i, :, :] = actions[i]

    def _append_obs_history(self, actions: np.ndarray, prev_nu: np.ndarray) -> None:
        self._ensure_history_buffers()
        self.motion_history[:, 1:, :] = self.motion_history[:, :-1, :].copy()
        self.action_history[:, 1:, :] = self.action_history[:, :-1, :].copy()
        for i, tug in enumerate(self.tugs):
            prev = Vec3(float(prev_nu[i, 0]), float(prev_nu[i, 1]), float(prev_nu[i, 2]))
            self.motion_history[i, 0, :] = self._motion_frame(tug, prev_nu=prev)
        self.action_history[:, 0, :] = np.clip(actions, -1.0, 1.0).astype(
            np.float32, copy=False
        )

    def _init_mode(self) -> str:
        return str(getattr(self.cfg, "tug_init_mode", "mixed_slot_approach"))

    def _uses_route_mode(self, mode: str | None = None) -> bool:
        """哪些初始化模式需要按 route 计算观察与奖励。"""
        init_mode = self._init_mode() if mode is None else str(mode)
        return init_mode == "mixed_slot_approach"

    def _sample_ship_size(self) -> None:
        """每个 episode 采样大船尺度；关闭随机化时回到基准尺度。"""
        cfg = self.cfg
        base_length = float(getattr(cfg, "ship_length_m", 200.0))
        base_beam = float(getattr(cfg, "ship_beam_m", 30.0))
        if not bool(getattr(cfg, "ship_size_randomize", False)):
            self.ship.length_m = base_length
            self.ship.beam_m = base_beam
            return

        length_min = float(getattr(cfg, "ship_length_min_m", base_length))
        length_max = float(getattr(cfg, "ship_length_max_m", base_length))
        beam_min = float(getattr(cfg, "ship_beam_min_m", base_beam))
        beam_max = float(getattr(cfg, "ship_beam_max_m", base_beam))
        length_lo, length_hi = sorted((max(1.0, length_min), max(1.0, length_max)))
        beam_lo, beam_hi = sorted((max(1.0, beam_min), max(1.0, beam_max)))
        self.ship.length_m = float(self.rng.uniform(length_lo, length_hi))
        self.ship.beam_m = float(self.rng.uniform(beam_lo, beam_hi))
        self._route_waypoints_body_cache.clear()       # 船体尺寸变化，waypoint 缓存失效

    def _slot_side_sign(self, slot_idx: int) -> float:
        # slot 顺序：船首左、船首右、船尾左、船尾右；船体系 y>0 为右舷。
        return -1.0 if int(slot_idx) in (0, 2) else 1.0

    def _slot_is_bow(self, slot_idx: int) -> bool:
        return int(slot_idx) in (0, 1)

    def _slot_lane_lat_abs(self, slot_idx: int) -> float:
        cfg = self.cfg
        if self._slot_is_bow(slot_idx):
            return float(getattr(cfg, "route_bow_lane_lat_m", 90.0))
        return float(getattr(cfg, "route_stern_lane_lat_m", 55.0))

    def _slot_rear_start_dist_range(self, slot_idx: int) -> tuple[float, float]:
        cfg = self.cfg
        if self._slot_is_bow(slot_idx):
            lo = float(getattr(cfg, "tug_init_rear_bow_slot_dist_min_m", 150.0))
            hi = float(getattr(cfg, "tug_init_rear_bow_slot_dist_max_m", 230.0))
        else:
            lo = float(getattr(cfg, "tug_init_rear_stern_slot_dist_min_m", 60.0))
            hi = float(getattr(cfg, "tug_init_rear_stern_slot_dist_max_m", 100.0))
        return lo, max(lo, hi)

    def _ship_body_xy(self, x_world: float, y_world: float) -> tuple[float, float]:
        return _world_to_local(x_world - self.ship.x, y_world - self.ship.y, self.ship.psi)

    def _ship_body_to_world_xy(self, x_body: float, y_body: float) -> tuple[float, float]:
        dx_w, dy_w = _local_to_world(x_body, y_body, self.ship.psi)
        return self.ship.x + dx_w, self.ship.y + dy_w

    def _distance_from_ship_hull_pose(
        self,
        x_world: float,
        y_world: float,
        ship_x: float,
        ship_y: float,
        ship_psi: float,
    ) -> float:
        """点到指定船位姿下的大船船体外表面的最短距离。"""
        dx = x_world - ship_x
        dy = y_world - ship_y
        cos_p = math.cos(ship_psi)
        sin_p = math.sin(ship_psi)
        x_b = cos_p * dx + sin_p * dy
        y_b = -sin_p * dx + cos_p * dy
        l_half = self.ship.length_m / 2.0
        b_half = self.ship.beam_m / 2.0
        ex = max(abs(x_b) - l_half, 0.0)
        ey = max(abs(y_b) - b_half, 0.0)
        return math.hypot(ex, ey)

    def _hull_rect_body(self) -> tuple[float, float, float, float]:
        """大船船体在船体系下的轴对齐矩形边界。"""
        l_half = self.ship.length_m / 2.0
        b_half = self.ship.beam_m / 2.0
        return (-l_half, l_half, -b_half, b_half)

    @staticmethod
    def _point_in_rect_interior(
        point: tuple[float, float],
        rect: tuple[float, float, float, float],
        eps: float = 1e-6,
    ) -> bool:
        x_min, x_max, y_min, y_max = rect
        x, y = point
        return x_min + eps < x < x_max - eps and y_min + eps < y < y_max - eps

    @staticmethod
    def _segment_intersects_closed_rect(
        p0: tuple[float, float],
        p1: tuple[float, float],
        rect: tuple[float, float, float, float],
        eps: float = 1e-9,
    ) -> bool:
        """Liang-Barsky segment/axis-aligned-rectangle intersection."""
        x_min, x_max, y_min, y_max = rect
        x0, y0 = p0
        x1, y1 = p1
        dx = x1 - x0
        dy = y1 - y0
        t0, t1 = 0.0, 1.0

        for p, q in (
            (-dx, x0 - x_min),
            (dx, x_max - x0),
            (-dy, y0 - y_min),
            (dy, y_max - y0),
        ):
            if abs(p) < eps:
                if q < 0.0:
                    return False
                continue
            r = q / p
            if p < 0.0:
                if r > t1:
                    return False
                t0 = max(t0, r)
            else:
                if r < t0:
                    return False
                t1 = min(t1, r)
        return t1 >= t0 and t1 > eps and t0 < 1.0 - eps

    def _body_segment_visible(
        self,
        p0: tuple[float, float],
        p1: tuple[float, float],
        rect: tuple[float, float, float, float],
    ) -> bool:
        if self._point_in_rect_interior(p0, rect) or self._point_in_rect_interior(p1, rect):
            return False
        x_min, x_max, y_min, y_max = rect
        # 检测是否穿过船体开内部；贴着边界走允许通过。
        shrunk = (x_min + 1e-6, x_max - 1e-6, y_min + 1e-6, y_max - 1e-6)
        if shrunk[0] >= shrunk[1] or shrunk[2] >= shrunk[3]:
            return True
        return not self._segment_intersects_closed_rect(p0, p1, shrunk)

    def _simplify_path_los(
        self,
        points: list[tuple[float, float]],
        rect: tuple[float, float, float, float],
    ) -> list[tuple[float, float]]:
        """用视线可见性贪心简化 A* 折线，减少冗余拐点。"""
        if len(points) <= 2:
            return list(points)
        out = [points[0]]
        i = 0
        while i < len(points) - 1:
            j = len(points) - 1
            while j > i + 1 and not self._body_segment_visible(points[i], points[j], rect):
                j -= 1
            out.append(points[j])
            i = j
        return out

    def _astar_path_body(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
        rect: tuple[float, float, float, float],
        side: float,
        *,
        allow_los_shortcut: bool = True,
    ) -> list[tuple[float, float]]:
        """在船体外用网格 A* 连接两个船体系路径点。"""
        if allow_los_shortcut and self._body_segment_visible(start, goal, rect):
            return [start, goal]

        cfg = self.cfg
        margin = float(
            getattr(
                cfg,
                "route_astar_margin_m",
                getattr(cfg, "route_visibility_node_margin_m", 10.0),
            )
        )
        cell = float(getattr(cfg, "route_astar_cell_m", 4.0))
        lane_min = float(getattr(cfg, "route_lane_min_lat_m", 32.0))
        lane_penalty = float(getattr(cfg, "route_astar_lane_penalty", 10_000.0))

        x_lo = min(start[0], goal[0], rect[0]) - margin
        x_hi = max(start[0], goal[0], rect[1]) + margin
        y_lo = min(start[1], goal[1], rect[2]) - margin
        y_hi = max(start[1], goal[1], rect[3]) + margin

        nx = max(2, int(math.ceil((x_hi - x_lo) / cell)) + 1)
        ny = max(2, int(math.ceil((y_hi - y_lo) / cell)) + 1)

        def to_idx(x: float, y: float) -> tuple[int, int]:
            ix = int(round((x - x_lo) / cell))
            iy = int(round((y - y_lo) / cell))
            return max(0, min(nx - 1, ix)), max(0, min(ny - 1, iy))

        def to_xy(ix: int, iy: int) -> tuple[float, float]:
            return x_lo + ix * cell, y_lo + iy * cell

        def blocked(ix: int, iy: int) -> bool:
            return self._point_in_rect_interior(to_xy(ix, iy), rect)

        start_idx = to_idx(start[0], start[1])
        goal_idx = to_idx(goal[0], goal[1])
        if blocked(*start_idx) or blocked(*goal_idx):
            return [start, goal]

        neighbors = (
            (1, 0, 1.0),
            (-1, 0, 1.0),
            (0, 1, 1.0),
            (0, -1, 1.0),
            (1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (-1, -1, math.sqrt(2.0)),
        )

        def heuristic(ix: int, iy: int) -> float:
            return math.hypot(ix - goal_idx[0], iy - goal_idx[1]) * cell

        def edge_cost(ix0: int, iy0: int, ix1: int, iy1: int, step_scale: float) -> float:
            if blocked(ix1, iy1):
                return float("inf")
            _, y0 = to_xy(ix0, iy0)
            _, y1 = to_xy(ix1, iy1)
            cost = step_scale * cell
            if side * (0.5 * (y0 + y1)) < lane_min:
                cost += lane_penalty
            return cost

        open_heap: list[tuple[float, float, tuple[int, int]]] = [
            (heuristic(*start_idx), 0.0, start_idx)
        ]
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score: dict[tuple[int, int], float] = {start_idx: 0.0}
        closed: set[tuple[int, int]] = set()
        goal_found = False

        while open_heap:
            _, g, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            closed.add(current)
            if current == goal_idx:
                goal_found = True
                break

            cx, cy = current
            for dx, dy, step_scale in neighbors:
                nxt = (cx + dx, cy + dy)
                nx_i, ny_i = nxt
                if nx_i < 0 or nx_i >= nx or ny_i < 0 or ny_i >= ny:
                    continue
                if nxt in closed:
                    continue
                tentative = g + edge_cost(cx, cy, nx_i, ny_i, step_scale)
                if not math.isfinite(tentative):
                    continue
                if tentative < g_score.get(nxt, float("inf")):
                    g_score[nxt] = tentative
                    came_from[nxt] = current
                    heapq.heappush(
                        open_heap,
                        (tentative + heuristic(nx_i, ny_i), tentative, nxt),
                    )

        if not goal_found:
            return [start, goal]

        path_idx: list[tuple[int, int]] = []
        cur: tuple[int, int] | None = goal_idx
        while cur is not None:
            path_idx.append(cur)
            if cur == start_idx:
                break
            cur = came_from.get(cur)
        path_idx.reverse()
        if not path_idx or path_idx[0] != start_idx:
            return [start, goal]

        path_xy = [to_xy(ix, iy) for ix, iy in path_idx]
        simplified = self._simplify_path_los(path_xy, rect)
        if not simplified:
            return [start, goal]
        simplified[0] = start
        simplified[-1] = goal
        return simplified

    def _dedupe_route_points(self, points: list[tuple[float, float]]) -> np.ndarray:
        if not points:
            return np.zeros((0, 2), dtype=np.float64)
        min_spacing = float(getattr(self.cfg, "route_min_waypoint_spacing_m", 2.0))
        out: list[tuple[float, float]] = [points[0]]
        for point in points[1:-1]:
            if math.hypot(point[0] - out[-1][0], point[1] - out[-1][1]) >= min_spacing:
                out.append(point)
        if math.hypot(points[-1][0] - out[-1][0], points[-1][1] - out[-1][1]) >= 1e-6:
            out.append(points[-1])
        else:
            out[-1] = points[-1]
        return np.asarray(out, dtype=np.float64)

    @staticmethod
    def _resample_route_fixed_count(
        points: np.ndarray,
        num_points: int,
        start: tuple[float, float],
        goal: tuple[float, float],
    ) -> np.ndarray:
        """沿折线弧长等距重采样为固定点数；首尾强制为拖轮位置与 slot。"""
        n = max(2, int(num_points))
        if len(points) < 2:
            return np.array([start, goal], dtype=np.float64)

        seg_lens = [
            float(np.linalg.norm(points[i + 1] - points[i]))
            for i in range(len(points) - 1)
        ]
        total = float(sum(seg_lens))
        if total < 1e-6:
            return np.array([start, goal], dtype=np.float64)

        cum = [0.0]
        for seg_len in seg_lens:
            cum.append(cum[-1] + seg_len)

        targets = np.linspace(0.0, total, n)
        sampled: list[tuple[float, float]] = []
        seg_idx = 0
        for target in targets:
            while seg_idx < len(seg_lens) - 1 and cum[seg_idx + 1] < target:
                seg_idx += 1
            seg_len = cum[seg_idx + 1] - cum[seg_idx]
            alpha = 0.0 if seg_len < 1e-9 else (target - cum[seg_idx]) / seg_len
            p = points[seg_idx] * (1.0 - alpha) + points[seg_idx + 1] * alpha
            sampled.append((float(p[0]), float(p[1])))

        result = np.asarray(sampled, dtype=np.float64)
        result[0] = start
        result[-1] = goal
        return result

    @staticmethod
    def _smooth_waypoints(points: np.ndarray, num_points: int | None = None,
                          s: float = 5.0) -> np.ndarray:
        """B-spline 平滑路径并等距重采样。

        对 Dijkstra 输出的折线段拟合三次 B-Spline，再按指定点数均匀采样，
        消除折角，产生更贴近真实拖轮操控轨迹的平滑路径。
        """
        if len(points) <= 2:
            return points
        k = min(3, len(points) - 1)
        tck, u = splprep([points[:, 0], points[:, 1]], s=s, k=k)
        if num_points is None:
            num_points = max(len(points) * 3, 12)
        u_new = np.linspace(0, 1, num_points)
        smoothed = np.column_stack(splev(u_new, tck))
        return smoothed

    def _route_at_slot_skip_tol_m(self) -> float:
        cfg = self.cfg
        return float(
            getattr(
                cfg,
                "route_at_slot_skip_tol_m",
                max(15.0, 0.5 * float(getattr(cfg, "pos_tol_m", 60.0))),
            )
        )

    def _route_body_distance(
        self,
        a: tuple[float, float],
        b: tuple[float, float],
    ) -> float:
        return float(math.hypot(a[0] - b[0], a[1] - b[1]))

    def _route_already_at_slot(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
    ) -> bool:
        """拖轮已在目标 slot 附近：无需绕路 A*，避免生成回到 slot 的环形路径。"""
        return self._route_body_distance(start, goal) <= self._route_at_slot_skip_tol_m()

    def _plan_route_segments_body(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
        slot_idx: int,
    ) -> list[tuple[float, float]]:
        """从拖轮位置到 slot 的分段 A*：锚点由起终点几何推导，非固定走廊。"""
        if self._route_already_at_slot(start, goal):
            return [start, goal]

        cfg = self.cfg
        side = self._slot_side_sign(slot_idx)
        rect = self._hull_rect_body()
        l_half = self.ship.length_m / 2.0
        lane_y = side * self._slot_lane_lat_abs(slot_idx)
        stern_back = float(getattr(cfg, "route_stern_gate_dist_m", 60.0))

        anchors: list[tuple[float, float]] = [start]
        if start[0] < -l_half - 25.0:
            anchors.append((-l_half - stern_back, lane_y))
        # 仅当起点仍在船尾侧、需绕到船首 slot 时才插入船首外侧锚点
        if self._slot_is_bow(slot_idx) and start[0] < l_half * 0.5:
            anchors.append((0.0, lane_y))
            anchors.append(
                (l_half + max(20.0, float(getattr(cfg, "route_astar_margin_m", 10.0))), lane_y)
            )
        anchors.append(goal)

        unique: list[tuple[float, float]] = [anchors[0]]
        for point in anchors[1:]:
            if math.hypot(point[0] - unique[-1][0], point[1] - unique[-1][1]) > 12.0:
                unique.append(point)
            else:
                unique[-1] = point
        if unique[-1] != goal:
            unique.append(goal)

        planned: list[tuple[float, float]] = []
        for seg_start, seg_goal in zip(unique[:-1], unique[1:]):
            segment = self._astar_path_body(
                seg_start, seg_goal, rect, side, allow_los_shortcut=False
            )
            if not planned:
                planned.extend(segment)
            else:
                planned.extend(segment[1:])
        return planned

    def _route_waypoints_body_for_tug(self, tug_idx: int) -> np.ndarray:
        """从当前拖轮船体系位置规划到其目标 slot 的 waypoint 序列。

        首点始终是拖轮当前位置，末点是 slot；中间用船体障碍网格 A* 绕开大船。
        结果按 tug_idx 缓存在 _route_waypoints_body_cache，reset 后首次访问时计算。
        """
        cached = self._route_waypoints_body_cache.get(tug_idx)
        if cached is not None:
            return cached

        cfg = self.cfg
        tug = self.tugs[tug_idx]
        start_xy = self._ship_body_xy(tug.eta.x, tug.eta.y)
        start = (float(start_xy[0]), float(start_xy[1]))
        slot_idx = int(self.tug_to_slot[tug_idx])
        slot_arr = self.ship.slot_positions_body()[slot_idx, :2]
        goal = (float(slot_arr[0]), float(slot_arr[1]))

        at_slot = self._route_already_at_slot(start, goal)
        if at_slot:
            planned: list[tuple[float, float]] = [start, goal]
        else:
            planned = self._plan_route_segments_body(start, goal, slot_idx)

        points = np.asarray(planned, dtype=np.float64)
        if len(points) < 2:
            points = np.array([start, goal], dtype=np.float64)

        path_len = self._route_body_distance(start, goal)
        use_smooth = (
            not at_slot
            and path_len > self._route_at_slot_skip_tol_m()
            and len(points) >= 4
            and bool(getattr(cfg, "route_spline_smooth", True))
        )
        if use_smooth:
            smoothed = self._smooth_waypoints(points)
            result = self._dedupe_route_points(
                [(float(p[0]), float(p[1])) for p in smoothed]
            )
        else:
            result = self._dedupe_route_points(
                [(float(p[0]), float(p[1])) for p in points]
            )

        if len(result) < 2:
            result = np.array([start, goal], dtype=np.float64)
        else:
            result = np.asarray(result, dtype=np.float64).copy()
            result[0] = start
            result[-1] = goal

        num_fixed = int(getattr(cfg, "route_num_waypoints", 0))
        if num_fixed >= 2:
            result = self._resample_route_fixed_count(result, num_fixed, start, goal)

        self._route_waypoints_body_cache[tug_idx] = result
        return result

    def _route_waypoints_world_for_tug(self, tug_idx: int) -> np.ndarray:
        points_body = self._route_waypoints_body_for_tug(tug_idx)
        points_world = np.zeros_like(points_body)
        for k, (x_b, y_b) in enumerate(points_body):
            points_world[k] = self._ship_body_to_world_xy(float(x_b), float(y_b))
        return points_world

    def _advance_route_stage(self, tug_idx: int) -> None:
        waypoints = self._route_waypoints_world_for_tug(tug_idx)
        tol = float(getattr(self.cfg, "route_waypoint_tol_m", 35.0))
        tug = self.tugs[tug_idx]
        while int(self.route_stage[tug_idx]) < len(waypoints) - 1:
            target = waypoints[int(self.route_stage[tug_idx])]
            if math.hypot(tug.eta.x - target[0], tug.eta.y - target[1]) > tol:
                break
            self.route_stage[tug_idx] += 1

    def _route_remaining_distance(self, tug_idx: int) -> float:
        waypoints = self._route_waypoints_world_for_tug(tug_idx)
        stage = int(np.clip(self.route_stage[tug_idx], 0, len(waypoints) - 1))
        tug = self.tugs[tug_idx]
        rem = math.hypot(tug.eta.x - waypoints[stage, 0], tug.eta.y - waypoints[stage, 1])
        for k in range(stage, len(waypoints) - 1):
            rem += float(np.linalg.norm(waypoints[k + 1] - waypoints[k]))
        return float(rem)

    def _current_route_target_world(self, tug_idx: int) -> np.ndarray:
        waypoints = self._route_waypoints_world_for_tug(tug_idx)
        stage = int(np.clip(self.route_stage[tug_idx], 0, len(waypoints) - 1))
        return waypoints[stage]

    # ---------- 环境主接口 ----------
    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.ship.rng = self.rng

        self.step_count = 0
        self.last_actions[:] = 0.0
        self.last_action_changes[:] = 0.0
        self._ensure_history_buffers()
        self.motion_history[:] = 0.0
        self.action_history[:] = 0.0
        self.in_zone_steps[:] = 0
        self.route_stage[:] = 0
        self._route_waypoints_body_cache.clear()       # 新 episode，waypoint 缓存失效

        # 大船初始化（每个 episode 先采样尺度，再重置位置/航向/速度）
        self._sample_ship_size()
        self.ship.reset(self.rng)

        slot_world = self.ship.slot_positions_world()
        init_mode = self._init_mode()
        tug_psi = np.zeros(self.n_tugs, dtype=np.float64)
        tug_nu = np.zeros((self.n_tugs, 3), dtype=np.float64)
        init_actions = np.zeros((self.n_tugs, ACTION_DIM), dtype=np.float32)

        self._mixed_ready_tugs = set()
        if init_mode == "mixed_slot_approach":
            # 固定角色，并随机选择一部分 slot 已经被拖轮合理占据。
            self.tug_to_slot = np.arange(self.n_tugs, dtype=np.int32)
            tug_xy, tug_psi, tug_nu, init_actions = self._sample_mixed_slot_approach_states()
        else:
            raise ValueError(
                f"未知 tug_init_mode: {init_mode!r}；"
                "当前仅支持 mixed_slot_approach"
            )

        for i, tug in enumerate(self.tugs):
            tug.reset()
            tug.set_state(
                Vec3(float(tug_xy[i, 0]), float(tug_xy[i, 1]), float(tug_psi[i])),
                Vec3(float(tug_nu[i, 0]), float(tug_nu[i, 1]), float(tug_nu[i, 2])),
            )
            if self._uses_route_mode(init_mode):
                tug.set_control_commands(
                    float(init_actions[i, 0]) * tug.rpm_limit,
                    float(init_actions[i, 1]) * tug.rpm_limit,
                    float(init_actions[i, 2]) * tug.azimuth_limit_deg,
                    float(init_actions[i, 3]) * tug.azimuth_limit_deg,
                )
                tug.snap_actuators_to_commands()

        self.last_actions = init_actions.copy()
        self.last_action_changes[:] = 0.0
        self._fill_obs_history(init_actions)

        # 拖轮就位后按当前位置生成各 tug 的 route（首点=拖轮，末点=slot）。
        for i in range(self.n_tugs):
            self._route_waypoints_body_for_tug(i)

        for i in range(self.n_tugs):
            route_len = len(self._route_waypoints_body_for_tug(i))
            if not self._uses_route_mode(init_mode):
                self.route_stage[i] = max(route_len - 1, 0)
            elif init_mode == "mixed_slot_approach" and i in self._mixed_ready_tugs:
                self.route_stage[i] = max(route_len - 1, 0)
            else:
                self.route_stage[i] = 0

        # 缓存初始距离（用于进度奖励）
        for i, tug in enumerate(self.tugs):
            slot = slot_world[self.tug_to_slot[i]]
            self.prev_dist[i] = float(math.hypot(tug.eta.x - slot[0], tug.eta.y - slot[1]))

        return self._build_obs()

    def _sample_ship_tracking_motion(
        self,
        *,
        speed_offset_min: float,
        speed_offset_max: float,
        heading_noise_rad: float,
        sway_noise_ms: float,
        yaw_rate_noise_rads: float,
        forward_action: float,
        position_body: tuple[float, float] | None = None,
        target_body: tuple[float, float] | None = None,
        approach_speed_min: float | None = None,
        approach_speed_max: float | None = None,
        forward_action_jitter: float = 0.0,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """生成接近大船运动、必要时朝目标缓慢靠近的初始运动状态。"""
        speed_offset = float(self.rng.uniform(speed_offset_min, speed_offset_max))
        sway = float(self.rng.uniform(-sway_noise_ms, sway_noise_ms))
        ship_body_vx = self.ship.u + speed_offset
        ship_body_vy = self.ship.v + sway

        if position_body is not None and target_body is not None:
            dx = float(target_body[0] - position_body[0])
            dy = float(target_body[1] - position_body[1])
            dist = math.hypot(dx, dy)
            if dist > 1e-6:
                lo = speed_offset_min if approach_speed_min is None else float(approach_speed_min)
                hi = speed_offset_max if approach_speed_max is None else float(approach_speed_max)
                lo, hi = sorted((max(0.0, lo), max(0.0, hi)))
                approach_speed = float(self.rng.uniform(lo, hi))
                ship_body_vx = self.ship.u + approach_speed * dx / dist
                ship_body_vy = self.ship.v + approach_speed * dy / dist + sway

        vx_w, vy_w = _local_to_world(ship_body_vx, ship_body_vy, self.ship.psi)
        if math.hypot(vx_w, vy_w) > 1e-4:
            heading_base = math.atan2(vy_w, vx_w)
        else:
            heading_base = self.ship.psi
        psi = _wrap_pi(
            heading_base + float(self.rng.uniform(-heading_noise_rad, heading_noise_rad))
        )
        u_tug, v_tug = _world_to_local(vx_w, vy_w, psi)
        r_tug = float(self.rng.uniform(-yaw_rate_noise_rads, yaw_rate_noise_rads))

        action = np.zeros(ACTION_DIM, dtype=np.float32)
        forward = float(
            np.clip(
                forward_action + self.rng.uniform(-forward_action_jitter, forward_action_jitter),
                -1.0,
                1.0,
            )
        )
        action[:] = (forward, forward, 0.0, 0.0)
        return psi, np.asarray((u_tug, v_tug, r_tug), dtype=np.float64), action

    def _init_position_is_safe(
        self,
        x_w: float,
        y_w: float,
        placed: list[tuple[float, float]],
        min_pair_dist: float,
        min_hull_dist: float,
    ) -> bool:
        if self.ship.distance_from_hull(x_w, y_w) < min_hull_dist:
            return False
        for px, py in placed:
            if math.hypot(x_w - px, y_w - py) < min_pair_dist:
                return False
        return True

    def _mixed_init_safety_margins(self) -> tuple[float, float]:
        cfg = self.cfg
        min_pair_dist = max(
            2.0 * cfg.tug_collision_dist_m,
            float(getattr(cfg, "tug_init_mixed_pair_min_dist_m", 120.0)),
        )
        min_hull_dist = max(
            2.0 * cfg.ship_collision_dist_m,
            float(getattr(cfg, "ship_safety_dist_m", cfg.ship_collision_dist_m * 3.0)),
        )
        return min_pair_dist, min_hull_dist

    def _sample_ready_slot_state(
        self,
        slot_idx: int,
        placed: list[tuple[float, float]] | None = None,
    ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        """已就位拖轮：在目标 slot 外侧小扰动，速度/航向接近大船。"""
        cfg = self.cfg
        slot_body = self.ship.slot_positions_body()[int(slot_idx), :2]
        side = self._slot_side_sign(slot_idx)
        outward = float(getattr(cfg, "tug_init_ready_outward_offset_m", 0.0))
        jitter = float(getattr(cfg, "tug_init_ready_pos_jitter_m", 0.0))
        placed_safe = [] if placed is None else placed
        min_pair_dist, min_hull_dist = self._mixed_init_safety_margins()

        chosen_xy: tuple[float, float] | None = None
        for _ in range(120):
            x_b = float(slot_body[0] + self.rng.uniform(-jitter, jitter))
            y_b = float(slot_body[1] + side * outward + self.rng.uniform(-jitter, jitter))
            x_w, y_w = self._ship_body_to_world_xy(x_b, y_b)
            if self._init_position_is_safe(x_w, y_w, placed_safe, min_pair_dist, min_hull_dist):
                chosen_xy = (x_w, y_w)
                break

        if chosen_xy is None:
            for k in range(16):
                x_b = float(slot_body[0])
                y_b = float(slot_body[1] + side * (outward + 10.0 * (k + 1)))
                x_w, y_w = self._ship_body_to_world_xy(x_b, y_b)
                if self._init_position_is_safe(x_w, y_w, placed_safe, min_pair_dist, min_hull_dist):
                    chosen_xy = (x_w, y_w)
                    break
        if chosen_xy is None:
            x_b = float(slot_body[0])
            y_b = float(slot_body[1] + side * (outward + 180.0))
            chosen_xy = self._ship_body_to_world_xy(x_b, y_b)

        speed_noise = float(getattr(cfg, "tug_init_ready_speed_noise_ms", 0.1))
        psi, nu, action = self._sample_ship_tracking_motion(
            speed_offset_min=-speed_noise,
            speed_offset_max=speed_noise,
            heading_noise_rad=float(getattr(cfg, "tug_init_ready_heading_noise_rad", math.radians(5.0))),
            sway_noise_ms=float(getattr(cfg, "tug_init_ready_sway_noise_ms", 0.03)),
            yaw_rate_noise_rads=float(getattr(cfg, "tug_init_ready_yaw_rate_noise_rads", 0.004)),
            forward_action=float(getattr(cfg, "tug_init_ready_forward_action", 0.22)),
            forward_action_jitter=float(getattr(cfg, "tug_init_action_jitter", 0.04)) * 0.5,
        )
        return np.asarray(chosen_xy, dtype=np.float64), psi, nu, action

    def _sample_mixed_route_body_candidate(
        self,
        slot_idx: int,
        zone: str,
    ) -> tuple[float, float, tuple[float, float]]:
        """采样未就位拖轮的船体系候选点和它应缓慢接近的目标点。"""
        cfg = self.cfg
        side = self._slot_side_sign(slot_idx)
        slot_body_arr = self.ship.slot_positions_body()[int(slot_idx), :2]
        slot_body = (float(slot_body_arr[0]), float(slot_body_arr[1]))
        l_half = self.ship.length_m / 2.0
        lane_min = float(getattr(cfg, "route_lane_min_lat_m", 32.0))
        lane_abs = max(self._slot_lane_lat_abs(slot_idx), lane_min + 5.0)
        gate_dist = float(getattr(cfg, "route_stern_gate_dist_m", 60.0))
        lon_jitter = float(getattr(cfg, "tug_init_mixed_route_longitudinal_jitter_m", 18.0))
        lat_jitter = float(getattr(cfg, "tug_init_mixed_route_lateral_jitter_m", 20.0))

        if zone == "rear_lane":
            dist_min, dist_max = self._slot_rear_start_dist_range(slot_idx)
            x_b = float(
                -l_half
                - self.rng.uniform(dist_min, dist_max)
                + self.rng.uniform(-lon_jitter, lon_jitter)
            )
            y_b = float(side * lane_abs + self.rng.uniform(-lat_jitter, lat_jitter))
            target = (-l_half - gate_dist, side * lane_abs)
        elif zone == "stern_gate":
            gate_lo = max(20.0, gate_dist * 0.4)
            gate_hi = max(gate_lo, gate_dist * 1.6)
            x_b = float(
                -l_half
                - self.rng.uniform(gate_lo, gate_hi)
                + self.rng.uniform(-0.5 * lon_jitter, 0.5 * lon_jitter)
            )
            y_b = float(side * lane_abs + self.rng.uniform(-0.6 * lat_jitter, 0.6 * lat_jitter))
            target = slot_body
        elif zone == "side_lane":
            side_lat = max(
                lane_abs,
                abs(slot_body[1]) + float(getattr(cfg, "route_outer_holding_extra_m", 18.0)),
            )
            if self._slot_is_bow(slot_idx):
                x_lo = -l_half - 0.5 * gate_dist
                x_hi = min(slot_body[0] - 20.0, l_half + 0.5 * float(getattr(cfg, "slot_lon_offset_m", 30.0)))
            else:
                x_lo = slot_body[0] - gate_dist
                x_hi = -l_half - 8.0
            x_lo, x_hi = sorted((x_lo, x_hi))
            x_b = float(
                self.rng.uniform(x_lo, x_hi)
                + self.rng.uniform(-0.3 * lon_jitter, 0.3 * lon_jitter)
            )
            y_b = float(side * side_lat + self.rng.uniform(-0.6 * lat_jitter, 0.6 * lat_jitter))
            target = slot_body
        elif zone == "outer_slot":
            outward_min = max(25.0, float(getattr(cfg, "route_outer_holding_extra_m", 18.0)) + 15.0)
            outward_max = outward_min + 70.0
            x_span = max(20.0, lon_jitter)
            x_b = float(slot_body[0] + self.rng.uniform(-x_span, x_span))
            y_b = float(
                slot_body[1]
                + side * self.rng.uniform(outward_min, outward_max)
                + self.rng.uniform(-0.5 * lat_jitter, 0.5 * lat_jitter)
            )
            target = slot_body
        elif zone == "opposite_stern":
            dist_min = float(getattr(cfg, "tug_init_mixed_opposite_stern_dist_min_m", 220.0))
            dist_max = float(getattr(cfg, "tug_init_mixed_opposite_stern_dist_max_m", 420.0))
            dist_lo, dist_hi = sorted((max(80.0, dist_min), max(80.0, dist_max)))
            lateral_extra = float(getattr(cfg, "tug_init_mixed_opposite_lateral_extra_m", 35.0))
            opposite_lane_abs = max(lane_abs + lateral_extra, lane_min + lateral_extra)
            x_b = float(
                -l_half
                - self.rng.uniform(dist_lo, dist_hi)
                + self.rng.uniform(-lon_jitter, lon_jitter)
            )
            y_b = float(-side * opposite_lane_abs + self.rng.uniform(-lat_jitter, lat_jitter))
            target = (-l_half - gate_dist, side * lane_abs)
        else:
            raise ValueError(f"unknown mixed init zone: {zone!r}")

        return x_b, y_b, target

    def _mixed_route_fallback_candidates(
        self,
        slot_idx: int,
        *,
        force_opposite_side: bool,
    ) -> list[tuple[float, float, tuple[float, float]]]:
        cfg = self.cfg
        side = self._slot_side_sign(slot_idx)
        l_half = self.ship.length_m / 2.0
        lane_min = float(getattr(cfg, "route_lane_min_lat_m", 32.0))
        lane_abs = max(self._slot_lane_lat_abs(slot_idx), lane_min + 5.0)
        gate_dist = float(getattr(cfg, "route_stern_gate_dist_m", 60.0))
        _, dist_max = self._slot_rear_start_dist_range(slot_idx)
        opposite_dist_max = float(getattr(cfg, "tug_init_mixed_opposite_stern_dist_max_m", 420.0))
        base_dist = max(dist_max, opposite_dist_max if force_opposite_side else dist_max)
        lateral_extra = float(getattr(cfg, "tug_init_mixed_opposite_lateral_extra_m", 35.0))

        fallback_sides = [-side, side] if force_opposite_side else [side, -side]
        candidates: list[tuple[float, float, tuple[float, float]]] = []
        for fallback_side in fallback_sides:
            fallback_lane_abs = lane_abs + (lateral_extra if fallback_side != side else 0.0)
            target = (
                (-l_half - gate_dist, side * lane_abs)
                if fallback_side != side
                else tuple(float(v) for v in self.ship.slot_positions_body()[int(slot_idx), :2])
            )
            for k in range(48):
                x_b = float(-l_half - base_dist - 80.0 - 70.0 * k)
                y_b = float(fallback_side * fallback_lane_abs)
                candidates.append((x_b, y_b, target))
        return candidates

    def _sample_random_route_state(
        self,
        slot_idx: int,
        placed: list[tuple[float, float]],
        *,
        force_opposite_side: bool = False,
    ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        """未就位拖轮：从多个安全区域采样初始位置，route_stage 始终从 0 开始。"""
        cfg = self.cfg
        side = self._slot_side_sign(slot_idx)
        min_pair_dist, min_hull_dist = self._mixed_init_safety_margins()
        lane_min = float(getattr(cfg, "route_lane_min_lat_m", 32.0))
        zones = (
            ["opposite_stern"]
            if force_opposite_side
            else ["rear_lane", "stern_gate", "side_lane", "outer_slot", "opposite_stern"]
        )
        zones = list(self.rng.permutation(np.asarray(zones, dtype=object)))

        chosen_xy: tuple[float, float] | None = None
        chosen_body: tuple[float, float] | None = None
        target_body: tuple[float, float] | None = None
        for zone in zones:
            for _ in range(80):
                x_b, y_b, target = self._sample_mixed_route_body_candidate(slot_idx, str(zone))
                if str(zone) == "opposite_stern":
                    if side * y_b > -lane_min:
                        continue
                elif side * y_b < lane_min:
                    continue

                x_w, y_w = self._ship_body_to_world_xy(x_b, y_b)
                if not self._init_position_is_safe(x_w, y_w, placed, min_pair_dist, min_hull_dist):
                    continue
                chosen_xy = (x_w, y_w)
                chosen_body = (x_b, y_b)
                target_body = target
                break
            if chosen_xy is not None:
                break

        if chosen_xy is None:
            for x_b, y_b, target in self._mixed_route_fallback_candidates(
                slot_idx,
                force_opposite_side=force_opposite_side,
            ):
                x_w, y_w = self._ship_body_to_world_xy(x_b, y_b)
                if self._init_position_is_safe(x_w, y_w, placed, min_pair_dist, min_hull_dist):
                    chosen_xy = (x_w, y_w)
                    chosen_body = (x_b, y_b)
                    target_body = target
                    break

        if chosen_xy is None or chosen_body is None or target_body is None:
            raise RuntimeError("failed to sample a safe mixed tug initialization position")

        psi, nu, action = self._sample_ship_tracking_motion(
            speed_offset_min=float(getattr(cfg, "tug_init_speed_boost_min_ms", 0.2)),
            speed_offset_max=float(getattr(cfg, "tug_init_speed_boost_max_ms", 0.8)),
            heading_noise_rad=float(getattr(cfg, "tug_init_heading_noise_rad", math.radians(12.0))),
            sway_noise_ms=float(getattr(cfg, "tug_init_sway_noise_ms", 0.08)),
            yaw_rate_noise_rads=float(getattr(cfg, "tug_init_yaw_rate_noise_rads", 0.01)),
            forward_action=float(getattr(cfg, "tug_init_forward_action", 0.35)),
            position_body=chosen_body,
            target_body=target_body,
            approach_speed_min=float(getattr(cfg, "tug_init_mixed_approach_speed_min_ms", 0.20)),
            approach_speed_max=float(getattr(cfg, "tug_init_mixed_approach_speed_max_ms", 0.80)),
            forward_action_jitter=float(getattr(cfg, "tug_init_action_jitter", 0.04)),
        )
        return (
            np.asarray(chosen_xy, dtype=np.float64),
            psi,
            nu,
            action,
        )

    def _sample_mixed_slot_approach_states(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """随机 0/1/2/3 艘拖轮已就位，其余拖轮从多种路线区域起步。"""
        cfg = self.cfg
        n = self.n_tugs
        ready_counts_raw = getattr(cfg, "tug_init_mixed_ready_counts", (2, 3))
        ready_counts = [int(v) for v in ready_counts_raw if 0 <= int(v) < n]
        if not ready_counts:
            ready_counts = [max(0, n - 1)]
        ready_count = int(self.rng.choice(np.asarray(ready_counts, dtype=np.int32)))
        ready_slots = set(int(v) for v in self.rng.choice(n, size=ready_count, replace=False))
        self._mixed_ready_tugs = ready_slots

        positions = np.zeros((n, 2), dtype=np.float64)
        psis = np.zeros(n, dtype=np.float64)
        nus = np.zeros((n, 3), dtype=np.float64)
        actions = np.zeros((n, ACTION_DIM), dtype=np.float32)

        placed: list[tuple[float, float]] = []
        for i in range(n):
            if i not in ready_slots:
                continue
            pos, psi, nu, action = self._sample_ready_slot_state(i, placed)
            positions[i] = pos
            psis[i] = psi
            nus[i] = nu
            actions[i] = action
            placed.append((float(pos[0]), float(pos[1])))

        free_slots = [i for i in range(n) if i not in ready_slots]
        force_single_opposite = ready_count == n - 1 and len(free_slots) == 1
        for i in self.rng.permutation(free_slots):
            pos, psi, nu, action = self._sample_random_route_state(
                int(i),
                placed,
                force_opposite_side=force_single_opposite,
            )
            positions[i] = pos
            psis[i] = psi
            nus[i] = nu
            actions[i] = action
            placed.append((float(pos[0]), float(pos[1])))

        return positions, psis, nus, actions

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """推进一个控制周期。

        actions: shape (n_tugs, 4)，连续动作，已经在 [-1, 1] 范围内（外部裁剪）。
                 4 维含义：[port_rpm_norm, stbd_rpm_norm, port_az_norm, stbd_az_norm]。
        返回：(obs, rewards, dones, info)
              obs:     (n_tugs, obs_dim)
              rewards: (n_tugs,)
              dones:   (n_tugs,)，环境是合作型，要么全 True 要么全 False
              info:    dict，含 reward 分量、是否成功、是否碰撞等
        """
        actions = np.clip(actions, -1.0, 1.0).astype(np.float32)
        actions = self._apply_route_speed_governor(actions)
        prev_nu = np.asarray(
            [[tug.nu.x, tug.nu.y, tug.nu.z] for tug in self.tugs],
            dtype=np.float32,
        )

        # 把归一化动作映射到拖轮的物理控制指令
        for i, tug in enumerate(self.tugs):
            port_rpm = float(actions[i, 0]) * tug.rpm_limit
            stbd_rpm = float(actions[i, 1]) * tug.rpm_limit
            port_az_deg = float(actions[i, 2]) * tug.azimuth_limit_deg
            stbd_az_deg = float(actions[i, 3]) * tug.azimuth_limit_deg
            tug.set_control_commands(port_rpm, stbd_rpm, port_az_deg, stbd_az_deg)
            tug.step(self.cfg.dt_ctrl)

        # 大船同步推进
        self.ship.step(self.cfg.dt_ctrl)
        self.step_count += 1

        if self._uses_route_mode():
            for i in range(self.n_tugs):
                self._advance_route_stage(i)

        # 计算奖励与基础信息
        slot_world = self.ship.slot_positions_world()
        rewards, info = self._reward.compute_rewards(actions, slot_world)
        self.last_reward_components = info.get("reward_components", {})

        # 终止判定
        dones, term_info = self._check_termination(slot_world)
        info.update(term_info)

        # 终端奖励单独存入 info，不混入稠密奖励。
        # 训练脚本只对稠密奖励做归一化，终端奖励在归一化之后直接叠加，
        # 保证碰撞惩罚/成功奖励的信号强度不被稠密奖励的方差压缩。
        # v56 (P1-#4): collision 只惩罚肇事 tug，其他 agent 不吃 -20。
        # v56 (P1-#11): success 给所有 agent 一次性正向 bonus，advantage 更尖锐。
        terminal_reward = np.zeros(self.n_tugs, dtype=np.float32)
        if term_info.get("collision"):
            kind = term_info.get("collision_kind")
            pen = float(self.cfg.reward_collision_pen)
            if kind == "tug_vs_ship":
                idx = int(term_info.get("collision_tug", 0))
                terminal_reward[idx] -= pen
            elif kind == "tug_vs_tug":
                i, j = term_info.get("collision_pair", (0, 1))
                terminal_reward[int(i)] -= pen
                terminal_reward[int(j)] -= pen
            else:
                terminal_reward -= pen
        if term_info.get("success") and self.cfg.reward_arrival_bonus > 0.0:
            terminal_reward += float(self.cfg.reward_arrival_bonus)
        info["terminal_reward"] = terminal_reward

        # 缓存上一步动作与距离
        self.last_action_changes = actions - self.last_actions
        self.last_actions = actions.copy()
        self._append_obs_history(actions, prev_nu)
        for i, tug in enumerate(self.tugs):
            slot = slot_world[self.tug_to_slot[i]]
            self.prev_dist[i] = float(math.hypot(tug.eta.x - slot[0], tug.eta.y - slot[1]))

        obs = self._build_obs()
        # 最终奖励 = 稠密奖励 + 终端奖励（训练脚本会分别处理归一化）
        return obs, rewards + terminal_reward, dones, info

    def _apply_route_speed_governor(self, actions: np.ndarray) -> np.ndarray:
        """非 final 路线阶段的速度安全层。

        策略仍输出原始动作，但在追赶/绕行阶段限制正向油门，避免远距离为了
        route progress 奖励持续满油门。进入 final slot 阶段后不做限幅。
        """
        if not bool(getattr(self.cfg, "route_speed_governor", False)):
            return actions
        if not self._uses_route_mode():
            return actions

        governed = actions.copy()
        max_chase = float(getattr(self.cfg, "route_chase_speed_max_ms", 0.9))
        speed_limit = float(getattr(self.cfg, "route_tug_speed_soft_limit_ms", 3.0))
        base_cap = float(getattr(self.cfg, "route_nonfinal_forward_action_cap", 0.45))
        min_cap = float(getattr(self.cfg, "route_speed_governor_min_forward_action", 0.05))
        slope = float(getattr(self.cfg, "route_speed_governor_cap_slope", 0.30))

        cs = math.cos(self.ship.psi)
        sn = math.sin(self.ship.psi)
        ship_vx_w = cs * self.ship.u - sn * self.ship.v
        ship_vy_w = sn * self.ship.u + cs * self.ship.v

        for i, tug in enumerate(self.tugs):
            route_len = len(self._route_waypoints_body_for_tug(i))
            if int(self.route_stage[i]) >= route_len - 1:
                continue

            ci = math.cos(tug.eta.z)
            si = math.sin(tug.eta.z)
            tug_vx_w = ci * tug.nu.x - si * tug.nu.y
            tug_vy_w = si * tug.nu.x + ci * tug.nu.y
            dvx = tug_vx_w - ship_vx_w
            dvy = tug_vy_w - ship_vy_w
            rel_u_ship, _ = _world_to_local(dvx, dvy, self.ship.psi)
            tug_speed_world = math.hypot(tug_vx_w, tug_vy_w)

            excess_chase = max(0.0, rel_u_ship - max_chase)
            excess_speed = max(0.0, tug_speed_world - speed_limit)
            cap = base_cap - slope * excess_chase - 0.5 * slope * excess_speed
            cap = float(np.clip(cap, min_cap, base_cap))
            governed[i, 0] = min(float(governed[i, 0]), cap)
            governed[i, 1] = min(float(governed[i, 1]), cap)

        return governed

    # ---------- 观察构造 ----------
    def _build_obs(self) -> np.ndarray:
        self._ensure_history_buffers()
        obs = np.zeros((self.n_tugs, self.obs_dim), dtype=np.float32)
        ship_u, ship_v, ship_r = self.ship.u, self.ship.v, self.ship.r
        preview_times = tuple(getattr(self.cfg, "obs_ship_preview_times_s", (5.0, 10.0, 15.0)))

        tug_world_vx = np.zeros(self.n_tugs, dtype=np.float32)
        tug_world_vy = np.zeros(self.n_tugs, dtype=np.float32)
        for i, tug in enumerate(self.tugs):
            ci = math.cos(tug.eta.z)
            si = math.sin(tug.eta.z)
            tug_world_vx[i] = ci * tug.nu.x - si * tug.nu.y
            tug_world_vy[i] = si * tug.nu.x + ci * tug.nu.y

        for i, tug in enumerate(self.tugs):
            idx = 0
            motion_len = self.motion_history.shape[1] * _EGO_MOTION_OBS_DIM
            obs[i, idx:idx + motion_len] = self.motion_history[i].reshape(-1)
            idx += motion_len

            action_len = self.action_history.shape[1] * ACTION_DIM
            obs[i, idx:idx + action_len] = self.action_history[i].reshape(-1)
            idx += action_len

            ship_dx_w = self.ship.x - tug.eta.x
            ship_dy_w = self.ship.y - tug.eta.y
            ship_dx_local, ship_dy_local = _world_to_local(ship_dx_w, ship_dy_w, tug.eta.z)
            dpsi_ship = _wrap_pi(self.ship.psi - tug.eta.z)
            obs[i, idx + 0] = ship_dx_local / 100.0
            obs[i, idx + 1] = ship_dy_local / 100.0
            obs[i, idx + 2] = ship_u / 3.0
            obs[i, idx + 3] = ship_v / 3.0
            obs[i, idx + 4] = math.sin(dpsi_ship)
            obs[i, idx + 5] = math.cos(dpsi_ship)
            idx += _SHIP_REL_OBS_DIM

            cs = math.cos(self.ship.psi)
            sn = math.sin(self.ship.psi)
            ship_vx_w = cs * ship_u - sn * ship_v
            ship_vy_w = sn * ship_u + cs * ship_v
            for tau in preview_times:
                t = float(tau)
                if abs(ship_r) < 1e-6:
                    ship_x_f = self.ship.x + ship_vx_w * t
                    ship_y_f = self.ship.y + ship_vy_w * t
                else:
                    dx_body = (ship_u * math.sin(ship_r * t) + ship_v * (math.cos(ship_r * t) - 1.0)) / ship_r
                    dy_body = (ship_u * (1.0 - math.cos(ship_r * t)) + ship_v * math.sin(ship_r * t)) / ship_r
                    dx_world, dy_world = _local_to_world(dx_body, dy_body, self.ship.psi)
                    ship_x_f = self.ship.x + dx_world
                    ship_y_f = self.ship.y + dy_world
                dx_local, dy_local = _world_to_local(
                    ship_x_f - tug.eta.x,
                    ship_y_f - tug.eta.y,
                    tug.eta.z,
                )
                obs[i, idx + 0] = dx_local / 100.0
                obs[i, idx + 1] = dy_local / 100.0
                idx += _SHIP_PREVIEW_POINT_DIM

            other_idx = [j for j in range(self.n_tugs) if j != i]
            for j in other_idx:
                other = self.tugs[j]
                dx_w = other.eta.x - tug.eta.x
                dy_w = other.eta.y - tug.eta.y
                dx_local, dy_local = _world_to_local(dx_w, dy_w, tug.eta.z)
                du_local, dv_local = _world_to_local(
                    float(tug_world_vx[j] - tug_world_vx[i]),
                    float(tug_world_vy[j] - tug_world_vy[i]),
                    tug.eta.z,
                )
                obs[i, idx + 0] = dx_local / 100.0
                obs[i, idx + 1] = dy_local / 100.0
                obs[i, idx + 2] = du_local / 5.0
                obs[i, idx + 3] = dv_local / 5.0
                obs[i, idx + 4] = min(math.hypot(dx_w, dy_w) / 100.0, 10.0)
                idx += _NEIGHBOR_OBS_DIM

        # 数值兜底：把 NaN/Inf 截断为 0/clip，防止极端步导致网络爆炸
        np.clip(obs, -10.0, 10.0, out=obs)
        return obs

    # ---------- Critic 用的全局状态 ----------
    def get_global_state(self) -> np.ndarray:
        """构造跨 agent 共享的 canonical global state（大船船体系）。

        所有几何量都在船体系下表达；跨 4 个 agent 完全一致。这样 critic 不需要
        学一个把"四套不同自身参考系拼接"还原到全局状态的非线性函数，容量得以
        真正用于学 V。
        """
        cfg = self.cfg
        n = self.n_tugs
        state = np.zeros(self.global_state_dim, dtype=np.float32)

        # 大船段：船体系速度 + 尺度 bias
        state[0] = float(self.ship.u) / 5.0
        state[1] = float(self.ship.v) / 5.0
        state[2] = float(self.ship.r) / 0.05
        base_length = max(float(getattr(cfg, "ship_length_m", 200.0)), 1e-6)
        base_beam = max(float(getattr(cfg, "ship_beam_m", 30.0)), 1e-6)
        state[3] = float(self.ship.length_m) / base_length - 1.0
        state[4] = float(self.ship.beam_m) / base_beam - 1.0

        # 预计算每条 tug 的世界系速度，再投影到船体系
        cs_s = math.cos(self.ship.psi)
        sn_s = math.sin(self.ship.psi)
        acc_tail_start = _GLOBAL_SHIP_DIM + _GLOBAL_PER_TUG_DIM * n

        hold_steps = max(1, int(round(cfg.hold_time_s / cfg.dt_ctrl)))

        for i, tug in enumerate(self.tugs):
            base = _GLOBAL_SHIP_DIM + i * _GLOBAL_PER_TUG_DIM

            # tug 位置（船体系）
            x_b, y_b = self._ship_body_xy(tug.eta.x, tug.eta.y)
            state[base + 0] = float(x_b) / 100.0
            state[base + 1] = float(y_b) / 100.0

            # tug 速度：先投到世界系，再投到船体系
            ci = math.cos(tug.eta.z)
            si = math.sin(tug.eta.z)
            vx_w = ci * tug.nu.x - si * tug.nu.y
            vy_w = si * tug.nu.x + ci * tug.nu.y
            u_b = cs_s * vx_w + sn_s * vy_w
            v_b = -sn_s * vx_w + cs_s * vy_w
            state[base + 2] = float(u_b) / 5.0
            state[base + 3] = float(v_b) / 5.0

            # 朝向（相对船体系）
            dpsi_ship = _wrap_pi(tug.eta.z - self.ship.psi)
            state[base + 4] = math.sin(dpsi_ship)
            state[base + 5] = math.cos(dpsi_ship)
            state[base + 6] = float(tug.nu.z) / 0.5

            # 执行器
            ctrl = tug.get_control_snapshot()
            state[base + 7] = ctrl["port_rpm_actual"] / tug.rpm_limit
            state[base + 8] = ctrl["starboard_rpm_actual"] / tug.rpm_limit
            state[base + 9] = ctrl["port_azimuth_actual_deg"] / tug.azimuth_limit_deg
            state[base + 10] = ctrl["starboard_azimuth_actual_deg"] / tug.azimuth_limit_deg

            # 上一步动作
            state[base + 11:base + 15] = self.last_actions[i]

            # slot one-hot（agent-id 等价）
            slot_idx = int(self.tug_to_slot[i])
            state[base + 15 + slot_idx] = 1.0

            # route stage 进度
            route_len = len(self._route_waypoints_body_for_tug(i))
            stage_norm = float(self.route_stage[i]) / max(route_len - 1, 1)
            state[base + 19] = float(np.clip(stage_norm, 0.0, 1.0))
            state[base + 20] = float(self._route_remaining_distance(i)) / 500.0

            # in_zone 进度（不仅是当前是否 in_zone，还包含已 hold 多久）
            state[base + 21] = float(self.in_zone_steps[i]) / float(hold_steps)

            # 距船体的最近距离（让 critic 看到全员的接近风险）
            d_hull = self.ship.distance_from_hull(tug.eta.x, tug.eta.y)
            state[base + 22] = float(d_hull) / 50.0

            # acceleration tail：每条 tug 的自身 3D 加速度 + 大船 3D 加速度。
            acc = tug.get_last_nu_dot()
            ax_w = ci * acc.x - si * acc.y
            ay_w = si * acc.x + ci * acc.y
            ax_b = cs_s * ax_w + sn_s * ay_w
            ay_b = -sn_s * ax_w + cs_s * ay_w
            acc_base = acc_tail_start + i * _GLOBAL_ACCEL_PER_TUG_DIM
            state[acc_base + 0] = float(ax_b) / _TUG_LINEAR_ACCEL_SCALE
            state[acc_base + 1] = float(ay_b) / _TUG_LINEAR_ACCEL_SCALE
            state[acc_base + 2] = float(acc.z) / _TUG_YAW_ACCEL_SCALE
            state[acc_base + 3] = float(self.ship.u_dot) / _SHIP_LINEAR_ACCEL_SCALE
            state[acc_base + 4] = float(self.ship.v_dot) / _SHIP_LINEAR_ACCEL_SCALE
            state[acc_base + 5] = float(self.ship.r_dot) / _SHIP_YAW_ACCEL_SCALE

        np.clip(state, -10.0, 10.0, out=state)
        return state

    # ---------- 终止判定 ----------
    def _check_termination(self, slot_world: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """返回 (dones, info)。

        info 同时给出 `terminated`（碰撞/成功，真正的吸收态）与 `truncated`
        （超时，应继续 bootstrap）；外层 GAE 用 `terminated` 决定是否截断 V。
        """
        n = self.n_tugs
        dones = np.zeros(n, dtype=bool)
        info: dict[str, Any] = {
            "success": False,
            "collision": False,
            "timeout": False,
            "terminated": False,
            "truncated": False,
        }
        cfg = self.cfg

        # 1) 拖轮与大船船体的碰撞
        for i, tug in enumerate(self.tugs):
            d_to_hull = self.ship.distance_from_hull(tug.eta.x, tug.eta.y)
            if d_to_hull < cfg.ship_collision_dist_m:
                dones[:] = True
                info["collision"] = True
                info["collision_kind"] = "tug_vs_ship"
                info["collision_tug"] = int(i)
                info["terminated"] = True
                return dones, info

        # 2) 拖轮之间的相互碰撞
        for i in range(n):
            for j in range(i + 1, n):
                dij = math.hypot(
                    self.tugs[i].eta.x - self.tugs[j].eta.x,
                    self.tugs[i].eta.y - self.tugs[j].eta.y,
                )
                if dij < cfg.tug_collision_dist_m:
                    dones[:] = True
                    info["collision"] = True
                    info["collision_kind"] = "tug_vs_tug"
                    info["collision_pair"] = (i, j)
                    info["terminated"] = True
                    return dones, info

        # 3) 成功：所有 agent 都连续保持 hold_time
        hold_steps = int(round(cfg.hold_time_s / cfg.dt_ctrl))
        if all(int(self.in_zone_steps[i]) >= hold_steps for i in range(n)):
            dones[:] = True
            info["success"] = True
            info["terminated"] = True
            return dones, info

        # 4) 超时——episode 强制结束以触发 reset，但不是吸收态：
        # GAE 应继续 bootstrap next_value，避免长 horizon 任务系统性低估 V。
        if self.step_count >= cfg.max_episode_steps:
            dones[:] = True
            info["timeout"] = True
            info["truncated"] = True

        return dones, info

    # ---------- 给可视化用的快照 ----------
    def render_snapshot(self) -> dict[str, Any]:
        """收集当前帧绘图所需的数据。可视化模块只读这一份字典。"""
        slot_world = self.ship.slot_positions_world()
        tugs_data = []
        tug_world_vx = []
        tug_world_vy = []
        for i, tug in enumerate(self.tugs):
            ctrl = tug.get_control_snapshot()
            thr = tug.get_thruster_snapshot()
            ci = math.cos(tug.eta.z)
            si = math.sin(tug.eta.z)
            vx_w = ci * tug.nu.x - si * tug.nu.y
            vy_w = si * tug.nu.x + ci * tug.nu.y
            acc = tug.get_last_nu_dot()
            tug_world_vx.append(vx_w)
            tug_world_vy.append(vy_w)
            tugs_data.append({
                "x": tug.eta.x,
                "y": tug.eta.y,
                "psi": tug.eta.z,
                "u": tug.nu.x,
                "v": tug.nu.y,
                "r": tug.nu.z,
                "u_dot": acc.x,
                "v_dot": acc.y,
                "r_dot": acc.z,
                "vx_w": vx_w,
                "vy_w": vy_w,
                "length": tug.length_m,
                "beam": tug.beam_m,
                "ctrl": ctrl,
                "thruster": thr,
                "slot_idx": int(self.tug_to_slot[i]),
                "route_stage": int(self.route_stage[i]),
                "route_remaining": float(self._route_remaining_distance(i)),
                "route_target": self._current_route_target_world(i).copy(),
                "in_zone_steps": int(self.in_zone_steps[i]),
                "r_target": float(self.last_reward_components.get("r_target", np.zeros(self.n_tugs))[i]),
                "r_velocity": float(self.last_reward_components.get("r_velocity", np.zeros(self.n_tugs))[i]),
                "dist_to_slot": float(self.last_reward_components.get("dist_to_slot", np.zeros(self.n_tugs))[i]),
            })

        # 计算所有拖轮对之间的 CPA
        cpa_pairs = []
        n = self.n_tugs
        for i in range(n):
            for j in range(i + 1, n):
                dx = self.tugs[j].eta.x - self.tugs[i].eta.x
                dy = self.tugs[j].eta.y - self.tugs[i].eta.y
                vrx = tug_world_vx[j] - tug_world_vx[i]
                vry = tug_world_vy[j] - tug_world_vy[i]
                vr_sq = vrx * vrx + vry * vry
                if vr_sq < 1e-6:
                    dcpa = math.hypot(dx, dy)
                    tcpa = 0.0
                    cpa_x = self.tugs[i].eta.x + dx / 2.0
                    cpa_y = self.tugs[i].eta.y + dy / 2.0
                else:
                    tcpa = -(dx * vrx + dy * vry) / vr_sq
                    if tcpa < 0:
                        dcpa = math.hypot(dx, dy)
                        tcpa = 0.0
                        cpa_x = self.tugs[i].eta.x + dx * 0.5
                        cpa_y = self.tugs[i].eta.y + dy * 0.5
                    else:
                        cpa_dx = dx + vrx * tcpa
                        cpa_dy = dy + vry * tcpa
                        dcpa = math.hypot(cpa_dx, cpa_dy)
                        cpa_x = self.tugs[i].eta.x + cpa_dx * 0.5
                        cpa_y = self.tugs[i].eta.y + cpa_dy * 0.5
                cpa_pairs.append({
                    "i": i, "j": j,
                    "dcpa": dcpa,
                    "tcpa": tcpa,
                    "cpa_x": cpa_x,
                    "cpa_y": cpa_y,
                })

        # 计算拖轮→大船的 CPA（v34）
        ship_vx_w = 0.0
        ship_vy_w = 0.0
        cs_s = math.cos(self.ship.psi)
        sn_s = math.sin(self.ship.psi)
        ship_vx_w = cs_s * self.ship.u - sn_s * self.ship.v
        ship_vy_w = sn_s * self.ship.u + cs_s * self.ship.v
        cpa_ship = []
        for i in range(n):
            ti = self.tugs[i]
            dx = self.ship.x - ti.eta.x
            dy = self.ship.y - ti.eta.y
            vrx = ship_vx_w - tug_world_vx[i]
            vry = ship_vy_w - tug_world_vy[i]
            vr_sq = vrx * vrx + vry * vry
            if vr_sq < 1e-6:
                dcpa = math.hypot(dx, dy)
                tcpa = 0.0
                cpa_x = ti.eta.x + dx * 0.5
                cpa_y = ti.eta.y + dy * 0.5
            else:
                tcpa = -(dx * vrx + dy * vry) / vr_sq
                if tcpa < 0:
                    dcpa = math.hypot(dx, dy)
                    tcpa = 0.0
                    cpa_x = ti.eta.x + dx * 0.5
                    cpa_y = ti.eta.y + dy * 0.5
                else:
                    cpa_dx = dx + vrx * tcpa
                    cpa_dy = dy + vry * tcpa
                    dcpa = math.hypot(cpa_dx, cpa_dy)
                    cpa_x = ti.eta.x + cpa_dx * 0.5
                    cpa_y = ti.eta.y + cpa_dy * 0.5
            cpa_ship.append({
                "tug_idx": i,
                "dcpa": dcpa,
                "tcpa": tcpa,
                "cpa_x": cpa_x,
                "cpa_y": cpa_y,
            })

        return {
            "step": self.step_count,
            "ship": {
                "x": self.ship.x,
                "y": self.ship.y,
                "psi": self.ship.psi,
                "u": self.ship.u,
                "v": self.ship.v,
                "r": self.ship.r,
                "u_dot": self.ship.u_dot,
                "v_dot": self.ship.v_dot,
                "r_dot": self.ship.r_dot,
                "length_m": self.ship.length_m,
                "beam_m": self.ship.beam_m,
                "hull": self.ship.hull_polygon_world(),
            },
            "slots": slot_world,
            "tugs": tugs_data,
            "cpa_pairs": cpa_pairs,
            "cpa_ship": cpa_ship,
        }
