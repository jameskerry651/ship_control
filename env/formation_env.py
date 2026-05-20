"""多智能体拖轮编队环境。

任务：4 艘拖轮 + 1 艘移动的大船。每艘拖轮被分配到大船周围一个固定的 slot
（船首左/右、船尾左/右），需要平滑驶入 slot 并在大船前进过程中保持就位。

设计要点：
- 单环境一次接受 (n_tugs, action_dim) 的动作，返回 (n_tugs, obs_dim) 的观察。
- 4 个智能体共享同一份策略网络（参数共享），观察都用"以自身为参考系"的相对量。
- 奖励是逐 agent 计算，但碰撞/成功这种全局事件所有 agent 同步收到。
- astern/mixed 模式下 slot 角色在 reset 时固定为 tug i → slot i。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from scipy.interpolate import splprep, splev

from config import EnvConfig
from physics.large_ship_model import LargeShipModel, _wrap_pi
from physics.tugboat_dynamics_model import TugboatDynamicsModel, Vec3


# ---------- 动作/观察维度 ----------
ACTION_DIM = 4

# 单个智能体的观察分块，按以下顺序拼接：
#  [0:3]   自身体系速度 (u, v, r)，按 5 / 5 / 0.5 缩放
#  [3:5]   自身位置在 slot 局部系下的偏移 (dx, dy)/50
#  [5:8]   自身→slot 极坐标 (log1p(d)/5, sinθ, cosθ)
#  [8:10]  航向误差 (sin dpsi, cos dpsi)
#  [10:12] 大船相对自身位置（自身体系下） (dx, dy)/100
#  [12:14] 大船航向相对自身 (sin dpsi_ship, cos dpsi_ship)
#  [14:17] 大船体系速度 (u, v, r)，按 3 / 3 / 0.05 缩放
#  [17:21] 自身执行器实际值（归一化到 [-1, 1]）
#  [21:25] 上一步动作（归一化）
#  [25:31] 其他 3 个拖轮在自身体系下的相对位置 (dx, dy)/100
#  [31:35] slot one-hot（船首左/船首右/船尾左/船尾右）
#  [35:43] 路线特征：next waypoint 相对位置/极坐标、stage、左/右舷、剩余路线
#  [43:55] CPA 特征：对其他 3 条拖轮的 (dcpa/100, tanh(tcpa/60), sin_bearing, cos_bearing)
#  [55:59] CPA 特征：对大船的 (dcpa/100, tanh(tcpa/60), sin_bearing, cos_bearing)
#  [59:61] 大船尺度特征：(length/base_length-1, beam/base_beam-1)
#  [61:64] 自身体系加速度 (u_dot, v_dot, r_dot)，按 1 / 1 / 0.1 缩放
# v28：加 slot one-hot，让参数共享网络能区分"我是哪个 slot"。
# 不加这个，v27 P5（差异化奖励）那种改动在共享策略下没法收效——
# 网络前向传播时连"我是谁"都看不到，梯度只能学出 4 个 slot 的平均策略。
# v29：加 astern route，让策略从船尾后方追赶并沿舷侧 waypoint 绕行，而不是直奔 final slot。
# v32：加 CPA 特征（DCPA/TCPA/会遇方位），让智能体显式感知与其他拖轮的碰撞风险。
# v34：加 拖轮-大船 CPA 特征，补全碰撞风险感知。
# v36：加大船尺度 domain randomization，并显式观测 length/beam。
# v58：加自身体系加速度，让 actor 看到动力学瞬态响应。
_BASE_OBS_DIM = 31
_ONEHOT_OBS_DIM = 4
_ROUTE_OBS_DIM = 8
_CPA_OBS_DIM = 12   # 3 条其他拖轮 × (dcpa/100, tanh(tcpa/60), sin_cpa_bearing, cos_cpa_bearing)
_CPA_SHIP_OBS_DIM = 4  # 拖轮→大船 CPA: (dcpa/100, tanh(tcpa/60), sin_bearing, cos_bearing)
_SHIP_SIZE_OBS_DIM = 2  # 大船 length/beam 相对 EnvConfig 基准尺度的偏差
_EGO_ACCEL_OBS_DIM = 3  # 自身体系加速度 (u_dot, v_dot, r_dot)

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
    last_reward_components: dict = field(init=False)
    _route_waypoints_body_cache: dict[int, np.ndarray] = field(init=False)
    prev_dist: np.ndarray = field(init=False)
    prev_route_remaining: np.ndarray = field(init=False)
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
        self.last_reward_components = {}
        self._route_waypoints_body_cache = {}
        self.prev_dist = np.zeros(self.n_tugs, dtype=np.float32)
        self.prev_route_remaining = np.zeros(self.n_tugs, dtype=np.float32)
        self.in_zone_steps = np.zeros(self.n_tugs, dtype=np.int32)
        self.route_stage = np.zeros(self.n_tugs, dtype=np.int32)
        self.tug_to_slot = np.arange(self.n_tugs, dtype=np.int32)

    @property
    def obs_dim(self) -> int:
        dim = _BASE_OBS_DIM
        if self.cfg.obs_include_slot_onehot:
            dim += _ONEHOT_OBS_DIM
        if getattr(self.cfg, "obs_include_route", False):
            dim += _ROUTE_OBS_DIM
        if getattr(self.cfg, "obs_include_cpa", False):
            dim += _CPA_OBS_DIM
        if getattr(self.cfg, "obs_include_cpa_ship", False):
            dim += _CPA_SHIP_OBS_DIM
        if getattr(self.cfg, "obs_include_ship_size", False):
            dim += _SHIP_SIZE_OBS_DIM
        if getattr(self.cfg, "obs_include_ego_accel", False):
            dim += _EGO_ACCEL_OBS_DIM
        return dim

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

    def _init_mode(self) -> str:
        return str(getattr(self.cfg, "tug_init_mode", "astern_approach"))

    def _uses_route_mode(self, mode: str | None = None) -> bool:
        """哪些初始化模式需要沿 astern route 计算路线观察与奖励。"""
        init_mode = self._init_mode() if mode is None else str(mode)
        return init_mode in ("astern_approach", "mixed_slot_approach")

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

    def _slot_astern_dist_range(self, slot_idx: int) -> tuple[float, float]:
        cfg = self.cfg
        if self._slot_is_bow(slot_idx):
            lo = float(getattr(cfg, "tug_init_astern_bow_dist_min_m", 150.0))
            hi = float(getattr(cfg, "tug_init_astern_bow_dist_max_m", 230.0))
        else:
            lo = float(getattr(cfg, "tug_init_astern_stern_dist_min_m", 60.0))
            hi = float(getattr(cfg, "tug_init_astern_stern_dist_max_m", 100.0))
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

    def _manual_route_waypoints_body(self, slot_idx: int) -> np.ndarray:
        """旧版手写 waypoint 模板，用于复现实验或 ablation。"""
        cfg = self.cfg
        side = self._slot_side_sign(slot_idx)
        l_half = self.ship.length_m / 2.0
        lane_y = side * self._slot_lane_lat_abs(slot_idx)
        stern_gate = float(getattr(cfg, "route_stern_gate_dist_m", 60.0))
        slot_body = self.ship.slot_positions_body()[int(slot_idx), :2]

        if self._slot_is_bow(slot_idx):
            points = [
                (-l_half - stern_gate, lane_y),
                (-l_half - 20.0, lane_y),
                (0.0, lane_y),
                (l_half + 20.0, lane_y),
                (float(slot_body[0]), float(slot_body[1])),
            ]
        else:
            points = [
                (-l_half - stern_gate * 0.75, lane_y),
                (-l_half - 25.0, lane_y),
                (float(slot_body[0]), float(slot_body[1])),
            ]
        return np.asarray(points, dtype=np.float64)

    def _route_hull_clearance_m(self) -> float:
        """visibility planner 的硬避障膨胀距离。

        final slot 本身在船舷外 `slot_lat_offset_m` 处；如果把障碍膨胀得比
        slot 还远，planner 会把目标点也视作障碍内部。因此这里把硬避障距离
        限制在 slot 外侧净距以内，最终贴近阶段再由 reward 的 soft safety 约束。
        """
        cfg = self.cfg
        collision_clear = float(getattr(cfg, "ship_collision_dist_m", 6.0)) + 2.0
        slot_clear = max(collision_clear, float(getattr(cfg, "slot_lat_offset_m", 25.0)) - 2.0)
        default_clear = min(
            float(getattr(cfg, "ship_safety_dist_m", collision_clear)),
            slot_clear,
        )
        requested = float(getattr(cfg, "route_hull_clearance_m", default_clear))
        return max(collision_clear, min(requested, slot_clear))

    def _inflated_hull_rect_body(self, clearance: float) -> tuple[float, float, float, float]:
        l_half = self.ship.length_m / 2.0
        b_half = self.ship.beam_m / 2.0
        return (
            -l_half - clearance,
            l_half + clearance,
            -b_half - clearance,
            b_half + clearance,
        )

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
        # 检测是否穿过膨胀船体的开内部；贴着边界走允许通过。
        shrunk = (x_min + 1e-6, x_max - 1e-6, y_min + 1e-6, y_max - 1e-6)
        if shrunk[0] >= shrunk[1] or shrunk[2] >= shrunk[3]:
            return True
        return not self._segment_intersects_closed_rect(p0, p1, shrunk)

    @staticmethod
    def _append_unique_body_point(
        points: list[tuple[float, float]],
        point: tuple[float, float],
        tol: float = 1e-6,
    ) -> None:
        for existing in points:
            if math.hypot(existing[0] - point[0], existing[1] - point[1]) < tol:
                return
        points.append(point)

    def _visibility_path_body(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
        extra_nodes: list[tuple[float, float]],
        rect: tuple[float, float, float, float],
        side: float,
    ) -> list[tuple[float, float]]:
        """在膨胀船体外用 visibility graph + Dijkstra 连接两个语义 waypoint。"""
        if self._body_segment_visible(start, goal, rect):
            return [start, goal]

        nodes: list[tuple[float, float]] = []
        for point in [start, goal, *extra_nodes]:
            if self._point_in_rect_interior(point, rect):
                continue
            self._append_unique_body_point(nodes, point)

        n = len(nodes)
        if n < 2:
            return [start, goal]

        lane_min = float(getattr(self.cfg, "route_lane_min_lat_m", 32.0))
        dist = [float("inf")] * n
        prev = [-1] * n
        used = [False] * n
        dist[0] = 0.0

        for _ in range(n):
            u = -1
            best = float("inf")
            for idx in range(n):
                if not used[idx] and dist[idx] < best:
                    best = dist[idx]
                    u = idx
            if u < 0 or u == 1:
                break
            used[u] = True
            for v in range(n):
                if used[v] or u == v:
                    continue
                if not self._body_segment_visible(nodes[u], nodes[v], rect):
                    continue
                d = math.hypot(nodes[u][0] - nodes[v][0], nodes[u][1] - nodes[v][1])
                mid_y = 0.5 * (nodes[u][1] + nodes[v][1])
                if side * mid_y < lane_min:
                    d += 10_000.0
                nd = dist[u] + d
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u

        if not math.isfinite(dist[1]):
            return [start, goal]

        rev: list[tuple[float, float]] = []
        idx = 1
        while idx >= 0:
            rev.append(nodes[idx])
            idx = prev[idx]
        return list(reversed(rev))

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

    def _route_jitter_scale(self) -> float:
        """v39: 每 episode 对锚点施加随机扰动的幅度 (0=关闭)。

        船体尺寸越大，抖动幅度成比例放大。"""
        cfg = self.cfg
        if not bool(getattr(cfg, "route_anchor_jitter", True)):
            return 0.0
        base_jitter = float(getattr(cfg, "route_anchor_jitter_m", 10.0))
        scale = max(self.ship.length_m / 200.0, self.ship.beam_m / 30.0)
        return float(self.rng.uniform(0.3, 1.0)) * base_jitter * scale

    def _route_stagger_offset(self, slot_idx: int) -> float:
        """v39: 同舷推错峰 — 船尾 slot stern_gate 更靠后，船首 slot 更靠前。

        避免同侧两艘拖轮在 stern gate 处追尾或拥堵。
        """
        cfg = self.cfg
        if not bool(getattr(cfg, "route_stagger", True)):
            return 0.0
        stagger = float(getattr(cfg, "route_stagger_dist_m", 50.0))
        base_length = float(getattr(cfg, "ship_length_m", 200.0))
        scale = self.ship.length_m / base_length
        if self._slot_is_bow(slot_idx):
            return stagger * scale      # 船首拖轮起点更靠近大船
        else:
            return -stagger * scale     # 船尾拖轮起点更靠后

    def _visibility_route_waypoints_body(self, slot_idx: int) -> np.ndarray:
        """按大船尺寸、slot 和膨胀船体自动生成同舷 waypoint。

        v39: 引入三个改进
          P0: 锚点随机抖动（每 episode 独立），提升路线泛化；
          P1: B-spline 路径平滑（消除 Dijkstra 折线段）；
          P2: 同舷错峰（船尾 slot stern_gate 更靠后，船首更靠前）。
        路线由少量语义 anchor 构成；每两个 anchor 之间先尝试直连，若直线
        穿过膨胀船体，则用同舷 visibility graph 绕开。最后经 B-spline 平滑、
        等距重采样、去重得到最终 waypoint 序列。
        """
        cfg = self.cfg
        side = self._slot_side_sign(slot_idx)
        l_half = self.ship.length_m / 2.0
        b_half = self.ship.beam_m / 2.0
        slot_body_arr = self.ship.slot_positions_body()[int(slot_idx), :2]
        slot_body = (float(slot_body_arr[0]), float(slot_body_arr[1]))

        clearance = self._route_hull_clearance_m()
        rect = self._inflated_hull_rect_body(clearance)
        stern_gate = float(getattr(cfg, "route_stern_gate_dist_m", 60.0))
        margin = float(getattr(cfg, "route_visibility_node_margin_m", 10.0))
        stagger = self._route_stagger_offset(slot_idx)
        jitter = self._route_jitter_scale()

        lane_abs = max(
            self._slot_lane_lat_abs(slot_idx),
            b_half + clearance + margin,
            float(getattr(cfg, "route_lane_min_lat_m", 32.0)) + margin,
        )
        lane_y = side * lane_abs
        side_rect_y = rect[3] if side > 0.0 else rect[2]

        final_lat_extra = float(getattr(cfg, "route_final_entry_lat_extra_m", 22.0))
        holding_extra = float(getattr(cfg, "route_outer_holding_extra_m", 18.0))
        final_lon_offset = float(getattr(cfg, "route_final_entry_lon_offset_m", 12.0))
        slot_y_abs = abs(slot_body[1])
        final_entry_y_abs = max(
            slot_y_abs + final_lat_extra,
            b_half + clearance + margin,
        )
        holding_y_abs = max(lane_abs, slot_y_abs + holding_extra, final_entry_y_abs)

        # --- P0 + P2: 锚点随机抖动 + 同舷错峰 ---
        def _jx(mag: float) -> float:
            return self.rng.uniform(-mag, mag) if jitter > 0 else 0.0

        stern_gate_x = -l_half - stern_gate + stagger + _jx(jitter * 0.5)
        stern_gate_y = lane_y + _jx(jitter * 0.8)
        stern_side_x = -l_half - max(clearance + margin, 20.0) + _jx(jitter * 0.3)
        bow_side_x = l_half + max(clearance + margin, 20.0) + _jx(jitter * 0.3)
        lane_mid_x = 0.0 + _jx(jitter * 0.5)

        if self._slot_is_bow(slot_idx):
            final_entry = (
                slot_body[0] + _jx(jitter * 0.3),
                side * max(final_entry_y_abs, slot_y_abs + final_lat_extra) + _jx(jitter * 0.3),
            )
            required = [
                (stern_gate_x, stern_gate_y),
                (stern_side_x, lane_y),
                (lane_mid_x, lane_y),
                (bow_side_x, lane_y),
                final_entry,
                slot_body,
            ]
        else:
            entry_x = slot_body[0] - final_lon_offset + _jx(jitter * 0.3)
            holding_back = max(2.0 * final_lon_offset, 0.35 * stern_gate) + _jx(jitter * 0.3)
            outer_holding = (
                slot_body[0] - holding_back,
                side * holding_y_abs + _jx(jitter * 0.5),
            )
            final_entry = (
                entry_x,
                side * max(final_entry_y_abs, holding_y_abs) + _jx(jitter * 0.3),
            )
            required = [
                (stern_gate_x, stern_gate_y),
                outer_holding,
                final_entry,
                slot_body,
            ]

        extra_nodes: list[tuple[float, float]] = []
        for point in (
            (rect[0], side_rect_y),
            (rect[1], side_rect_y),
            (rect[0] - margin, side_rect_y + side * margin),
            (rect[1] + margin, side_rect_y + side * margin),
            (-l_half - stern_gate, side_rect_y + side * margin),
            (l_half + stern_gate, side_rect_y + side * margin),
        ):
            self._append_unique_body_point(extra_nodes, point)

        planned: list[tuple[float, float]] = []
        for start, goal in zip(required[:-1], required[1:]):
            segment = self._visibility_path_body(start, goal, extra_nodes, rect, side)
            if not planned:
                planned.extend(segment)
            else:
                planned.extend(segment[1:])

        # --- P1: B-spline 平滑 + 等距重采样 ---
        points = np.asarray(planned, dtype=np.float64)
        if len(points) >= 4:
            smoothed = self._smooth_waypoints(points)
            return self._dedupe_route_points(
                [(float(p[0]), float(p[1])) for p in smoothed]
            )
        return self._dedupe_route_points(planned)

    def _route_waypoints_body(self, slot_idx: int) -> np.ndarray:
        """返回指定 slot 的船体系 waypoint。

        默认使用 visibility planner：在大船船体系下对船体做安全膨胀，并在同舷
        走廊内生成 stern gate、holding/final-entry、final slot 等 waypoint。
        最后一个 waypoint 始终是 final slot。
        结果缓存在 _route_waypoints_body_cache 中，每 episode 仅计算一次；
        reset() / _sample_ship_size() 会清空缓存。
        """
        cached = self._route_waypoints_body_cache.get(slot_idx)
        if cached is not None:
            return cached
        planner = str(getattr(self.cfg, "route_planner", "visibility")).lower()
        if planner == "manual":
            result = self._manual_route_waypoints_body(slot_idx)
        else:
            result = self._visibility_route_waypoints_body(slot_idx)
        self._route_waypoints_body_cache[slot_idx] = result
        return result

    def _route_waypoints_world(self, slot_idx: int) -> np.ndarray:
        points_body = self._route_waypoints_body(slot_idx)
        points_world = np.zeros_like(points_body)
        for k, (x_b, y_b) in enumerate(points_body):
            points_world[k] = self._ship_body_to_world_xy(float(x_b), float(y_b))
        return points_world

    def _advance_route_stage(self, tug_idx: int) -> None:
        slot_idx = int(self.tug_to_slot[tug_idx])
        waypoints = self._route_waypoints_world(slot_idx)
        tol = float(getattr(self.cfg, "route_waypoint_tol_m", 35.0))
        tug = self.tugs[tug_idx]
        while int(self.route_stage[tug_idx]) < len(waypoints) - 1:
            target = waypoints[int(self.route_stage[tug_idx])]
            if math.hypot(tug.eta.x - target[0], tug.eta.y - target[1]) > tol:
                break
            self.route_stage[tug_idx] += 1

    def _route_remaining_distance(self, tug_idx: int) -> float:
        slot_idx = int(self.tug_to_slot[tug_idx])
        waypoints = self._route_waypoints_world(slot_idx)
        stage = int(np.clip(self.route_stage[tug_idx], 0, len(waypoints) - 1))
        tug = self.tugs[tug_idx]
        rem = math.hypot(tug.eta.x - waypoints[stage, 0], tug.eta.y - waypoints[stage, 1])
        for k in range(stage, len(waypoints) - 1):
            rem += float(np.linalg.norm(waypoints[k + 1] - waypoints[k]))
        return float(rem)

    def _current_route_target_world(self, tug_idx: int) -> np.ndarray:
        slot_idx = int(self.tug_to_slot[tug_idx])
        waypoints = self._route_waypoints_world(slot_idx)
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
        init_route_stage = np.zeros(self.n_tugs, dtype=np.int32)

        if init_mode == "astern_approach":
            # 固定角色：左舷拖轮走左侧航道，右舷拖轮走右侧航道。
            self.tug_to_slot = np.arange(self.n_tugs, dtype=np.int32)
            tug_xy, tug_psi, tug_nu, init_actions = self._sample_astern_approach_states()
        elif init_mode == "mixed_slot_approach":
            # 固定角色，并随机选择一部分 slot 已经被拖轮合理占据。
            self.tug_to_slot = np.arange(self.n_tugs, dtype=np.int32)
            tug_xy, tug_psi, tug_nu, init_actions, init_route_stage = (
                self._sample_mixed_slot_approach_states()
            )
        else:
            raise ValueError(
                f"未知 tug_init_mode: {init_mode!r}；"
                "支持 astern_approach / mixed_slot_approach"
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

        # route 模式使用初始化给出的路线阶段。
        for i in range(self.n_tugs):
            route_len = len(self._route_waypoints_body(int(self.tug_to_slot[i])))
            if self._uses_route_mode(init_mode):
                self.route_stage[i] = int(np.clip(init_route_stage[i], 0, max(route_len - 1, 0)))
            else:
                self.route_stage[i] = max(route_len - 1, 0)

        # 缓存初始距离（用于进度奖励）
        for i, tug in enumerate(self.tugs):
            slot = slot_world[self.tug_to_slot[i]]
            self.prev_dist[i] = float(math.hypot(tug.eta.x - slot[0], tug.eta.y - slot[1]))
            self.prev_route_remaining[i] = float(self._route_remaining_distance(i))

        return self._build_obs()

    def _sample_astern_approach_states(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """从大船船尾后方生成拖轮位姿、初始速度和初始推进器动作。

        真实作业中拖轮通常跟在大船后方，以略高于大船的速度追上目标舷侧；
        因此这里固定 slot 角色，并把初始速度设置为大船速度 + 小幅追赶速度。
        """
        cfg = self.cfg
        n = self.n_tugs
        min_pair_dist = max(2.0 * cfg.tug_collision_dist_m,
                            float(getattr(cfg, "tug_init_pair_min_dist_m", 60.0)))
        min_hull_dist = 2.0 * cfg.ship_collision_dist_m
        l_half = self.ship.length_m / 2.0

        positions = np.zeros((n, 2), dtype=np.float64)
        psis = np.zeros(n, dtype=np.float64)
        nus = np.zeros((n, 3), dtype=np.float64)
        actions = np.zeros((n, ACTION_DIM), dtype=np.float32)

        for i in range(n):
            slot_idx = int(self.tug_to_slot[i])
            side = self._slot_side_sign(slot_idx)
            dist_min, dist_max = self._slot_astern_dist_range(slot_idx)
            lane_abs = self._slot_lane_lat_abs(slot_idx)
            lateral_jitter = float(getattr(cfg, "tug_init_astern_lateral_jitter_m", 8.0))
            for _ in range(200):
                stern_dist = float(self.rng.uniform(dist_min, dist_max))
                lateral = lane_abs + float(self.rng.uniform(-lateral_jitter, lateral_jitter))
                x_b = -l_half - stern_dist
                y_b = side * lateral
                x_w, y_w = self._ship_body_to_world_xy(x_b, y_b)

                if self.ship.distance_from_hull(x_w, y_w) < min_hull_dist:
                    continue
                ok = True
                for j in range(i):
                    if math.hypot(x_w - positions[j, 0], y_w - positions[j, 1]) < min_pair_dist:
                        ok = False
                        break
                if ok:
                    positions[i] = (x_w, y_w)
                    break
            else:
                # 兜底：保持对应舷侧和内外航道，按 slot 类型错开纵向距离。
                x_b = -l_half - (dist_min + dist_max) * 0.5
                y_b = side * lane_abs
                positions[i] = self._ship_body_to_world_xy(x_b, y_b)

            psi = _wrap_pi(
                self.ship.psi + float(self.rng.uniform(-cfg.tug_init_heading_noise_rad,
                                                        cfg.tug_init_heading_noise_rad))
            )
            boost = float(self.rng.uniform(cfg.tug_init_speed_boost_min_ms,
                                           cfg.tug_init_speed_boost_max_ms))
            sway = side * float(self.rng.uniform(-cfg.tug_init_sway_noise_ms,
                                                 cfg.tug_init_sway_noise_ms))
            ship_body_vx = self.ship.u + boost
            ship_body_vy = sway
            vx_w, vy_w = _local_to_world(ship_body_vx, ship_body_vy, self.ship.psi)
            u_tug, v_tug = _world_to_local(vx_w, vy_w, psi)
            r_tug = float(self.rng.uniform(-cfg.tug_init_yaw_rate_noise_rads,
                                           cfg.tug_init_yaw_rate_noise_rads))

            psis[i] = psi
            nus[i] = (u_tug, v_tug, r_tug)
            forward = float(np.clip(cfg.tug_init_forward_action, -1.0, 1.0))
            actions[i] = (forward, forward, 0.0, 0.0)

        return positions, psis, nus, actions

    def _sample_ship_tracking_motion(
        self,
        *,
        speed_offset_min: float,
        speed_offset_max: float,
        heading_noise_rad: float,
        sway_noise_ms: float,
        yaw_rate_noise_rads: float,
        forward_action: float,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """生成接近大船航向/速度的拖轮初始运动状态和推进器动作。"""
        psi = _wrap_pi(
            self.ship.psi + float(self.rng.uniform(-heading_noise_rad, heading_noise_rad))
        )
        speed_offset = float(self.rng.uniform(speed_offset_min, speed_offset_max))
        sway = float(self.rng.uniform(-sway_noise_ms, sway_noise_ms))
        ship_body_vx = self.ship.u + speed_offset
        ship_body_vy = sway
        vx_w, vy_w = _local_to_world(ship_body_vx, ship_body_vy, self.ship.psi)
        u_tug, v_tug = _world_to_local(vx_w, vy_w, psi)
        r_tug = float(self.rng.uniform(-yaw_rate_noise_rads, yaw_rate_noise_rads))

        action = np.zeros(ACTION_DIM, dtype=np.float32)
        forward = float(np.clip(forward_action, -1.0, 1.0))
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
            float(getattr(
                cfg,
                "tug_init_mixed_pair_min_dist_m",
                getattr(cfg, "tug_init_pair_min_dist_m", 60.0),
            )),
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
        )
        return np.asarray(chosen_xy, dtype=np.float64), psi, nu, action

    def _sample_random_route_state(
        self,
        slot_idx: int,
        placed: list[tuple[float, float]],
        *,
        force_opposite_side: bool = False,
    ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, int]:
        """未就位拖轮：从目标 slot 对应路线的船尾/舷侧随机阶段起步。

        force_opposite_side 用于 ready_count=3 的鲁棒性场景：最后一艘拖轮
        故意从目标 slot 的对侧船尾外侧起步，使其必须绕过船尾再进入自身舷侧路线。
        """
        cfg = self.cfg
        side = self._slot_side_sign(slot_idx)
        waypoints = self._route_waypoints_body(slot_idx)
        max_stage = max(0, len(waypoints) - 2)  # 不直接从 final slot 起步
        min_pair_dist, min_hull_dist = self._mixed_init_safety_margins()
        lon_jitter = float(getattr(cfg, "tug_init_mixed_route_longitudinal_jitter_m", 18.0))
        lat_jitter = float(getattr(cfg, "tug_init_mixed_route_lateral_jitter_m", 20.0))
        lane_min = float(getattr(cfg, "route_lane_min_lat_m", 32.0))

        chosen_stage = 0
        chosen_xy: tuple[float, float] | None = None
        for _ in range(300):
            if force_opposite_side:
                stage = 0
                l_half = self.ship.length_m / 2.0
                dist_min = float(getattr(cfg, "tug_init_mixed_opposite_stern_dist_min_m", 220.0))
                dist_max = float(getattr(cfg, "tug_init_mixed_opposite_stern_dist_max_m", 420.0))
                dist_lo, dist_hi = sorted((max(80.0, dist_min), max(80.0, dist_max)))
                lateral_extra = float(getattr(cfg, "tug_init_mixed_opposite_lateral_extra_m", 35.0))
                lane_abs = max(
                    self._slot_lane_lat_abs(slot_idx) + lateral_extra,
                    abs(float(waypoints[0, 1])) + lateral_extra,
                    lane_min + lateral_extra,
                )
                x_b = float(-l_half - self.rng.uniform(dist_lo, dist_hi)
                            + self.rng.uniform(-lon_jitter, lon_jitter))
                y_b = float(-side * lane_abs + self.rng.uniform(-lat_jitter, lat_jitter))
            else:
                stage = int(self.rng.integers(0, max_stage + 1))
                if stage == 0:
                    dist_min, dist_max = self._slot_astern_dist_range(slot_idx)
                    back_min = max(45.0, 0.45 * dist_min)
                    back_max = max(back_min + 20.0, 0.80 * dist_max)
                    back = float(self.rng.uniform(back_min, back_max))
                    x_b = float(waypoints[0, 0] - back + self.rng.uniform(-lon_jitter, lon_jitter))
                    y_b = float(waypoints[0, 1] + self.rng.uniform(-lat_jitter, lat_jitter))
                else:
                    prev = waypoints[stage - 1]
                    target = waypoints[stage]
                    seg = target - prev
                    seg_len = float(np.linalg.norm(seg))
                    if seg_len < 1e-6:
                        ux, uy = 1.0, 0.0
                    else:
                        ux, uy = float(seg[0] / seg_len), float(seg[1] / seg_len)
                    nx, ny = -uy, ux
                    frac = float(self.rng.uniform(0.15, 0.85))
                    base = prev + frac * seg
                    x_b = float(base[0] + ux * self.rng.uniform(-lon_jitter, lon_jitter)
                                + nx * self.rng.uniform(-lat_jitter, lat_jitter))
                    y_b = float(base[1] + uy * self.rng.uniform(-lon_jitter, lon_jitter)
                                + ny * self.rng.uniform(-lat_jitter, lat_jitter))

            if force_opposite_side:
                if side * y_b > -lane_min:
                    continue
            elif side * y_b < lane_min:
                continue

            x_w, y_w = self._ship_body_to_world_xy(x_b, y_b)
            if not self._init_position_is_safe(x_w, y_w, placed, min_pair_dist, min_hull_dist):
                continue
            chosen_stage = stage
            chosen_xy = (x_w, y_w)
            break

        if chosen_xy is None and force_opposite_side:
            l_half = self.ship.length_m / 2.0
            dist_max = float(getattr(cfg, "tug_init_mixed_opposite_stern_dist_max_m", 420.0))
            lateral_extra = float(getattr(cfg, "tug_init_mixed_opposite_lateral_extra_m", 35.0))
            lane_abs = max(
                self._slot_lane_lat_abs(slot_idx) + lateral_extra,
                abs(float(waypoints[0, 1])) + lateral_extra,
                lane_min + lateral_extra,
            )
            for k in range(16):
                x_b = float(-l_half - dist_max - 60.0 - 55.0 * k)
                y_b = float(-side * lane_abs)
                x_w, y_w = self._ship_body_to_world_xy(x_b, y_b)
                if self._init_position_is_safe(x_w, y_w, placed, min_pair_dist, min_hull_dist):
                    chosen_xy = (x_w, y_w)
                    chosen_stage = 0
                    break

        if chosen_xy is None:
            # 兜底：沿该 slot 的第一路线点继续向船尾方向拉远，直到满足间距。
            dist_min, dist_max = self._slot_astern_dist_range(slot_idx)
            for k in range(12):
                x_b = float(waypoints[0, 0] - dist_max - 60.0 - 45.0 * k)
                y_b = float(waypoints[0, 1])
                x_w, y_w = self._ship_body_to_world_xy(x_b, y_b)
                if self._init_position_is_safe(x_w, y_w, placed, min_pair_dist, min_hull_dist):
                    chosen_xy = (x_w, y_w)
                    chosen_stage = 0
                    break
            if chosen_xy is None:
                x_w, y_w = self._ship_body_to_world_xy(
                    float(waypoints[0, 0] - dist_max - 200.0),
                    float(waypoints[0, 1]),
                )
                chosen_xy = (x_w, y_w)
                chosen_stage = 0

        psi, nu, action = self._sample_ship_tracking_motion(
            speed_offset_min=float(getattr(cfg, "tug_init_speed_boost_min_ms", 0.2)),
            speed_offset_max=float(getattr(cfg, "tug_init_speed_boost_max_ms", 0.8)),
            heading_noise_rad=float(getattr(cfg, "tug_init_heading_noise_rad", math.radians(12.0))),
            sway_noise_ms=float(getattr(cfg, "tug_init_sway_noise_ms", 0.08)),
            yaw_rate_noise_rads=float(getattr(cfg, "tug_init_yaw_rate_noise_rads", 0.01)),
            forward_action=float(getattr(cfg, "tug_init_forward_action", 0.35)),
        )
        return (
            np.asarray(chosen_xy, dtype=np.float64),
            psi,
            nu,
            action,
            int(chosen_stage),
        )

    def _sample_mixed_slot_approach_states(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """随机 1/2/3 艘拖轮已就位，其余拖轮从路线周围起步。"""
        cfg = self.cfg
        n = self.n_tugs
        ready_counts_raw = getattr(cfg, "tug_init_mixed_ready_counts", (2, 3))
        ready_counts = [int(v) for v in ready_counts_raw if 0 <= int(v) < n]
        if not ready_counts:
            ready_counts = [max(0, n - 1)]
        ready_count = int(self.rng.choice(np.asarray(ready_counts, dtype=np.int32)))
        ready_slots = set(int(v) for v in self.rng.choice(n, size=ready_count, replace=False))

        positions = np.zeros((n, 2), dtype=np.float64)
        psis = np.zeros(n, dtype=np.float64)
        nus = np.zeros((n, 3), dtype=np.float64)
        actions = np.zeros((n, ACTION_DIM), dtype=np.float32)
        route_stages = np.zeros(n, dtype=np.int32)

        placed: list[tuple[float, float]] = []
        for i in range(n):
            if i not in ready_slots:
                continue
            pos, psi, nu, action = self._sample_ready_slot_state(i, placed)
            positions[i] = pos
            psis[i] = psi
            nus[i] = nu
            actions[i] = action
            route_stages[i] = max(len(self._route_waypoints_body(i)) - 1, 0)
            placed.append((float(pos[0]), float(pos[1])))

        free_slots = [i for i in range(n) if i not in ready_slots]
        force_single_opposite = ready_count == n - 1 and len(free_slots) == 1
        for i in self.rng.permutation(free_slots):
            pos, psi, nu, action, stage = self._sample_random_route_state(
                int(i),
                placed,
                force_opposite_side=force_single_opposite,
            )
            positions[i] = pos
            psis[i] = psi
            nus[i] = nu
            actions[i] = action
            route_stages[i] = stage
            placed.append((float(pos[0]), float(pos[1])))

        return positions, psis, nus, actions, route_stages

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

        # 计算奖励与基础信息
        slot_world = self.ship.slot_positions_world()
        rewards, info = self._compute_rewards(actions, slot_world)
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
            route_len = len(self._route_waypoints_body(int(self.tug_to_slot[i])))
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
        slot_world = self.ship.slot_positions_world()
        obs = np.zeros((self.n_tugs, self.obs_dim), dtype=np.float32)
        ship_u, ship_v, ship_r = self.ship.u, self.ship.v, self.ship.r

        # 预计算所有拖轮的世界系速度和位置（CPA 用）
        tug_world_vx = np.zeros(self.n_tugs, dtype=np.float32)
        tug_world_vy = np.zeros(self.n_tugs, dtype=np.float32)
        has_cpa = getattr(self.cfg, "obs_include_cpa", False)
        has_cpa_ship = getattr(self.cfg, "obs_include_cpa_ship", False)
        ship_vx_w = 0.0
        ship_vy_w = 0.0
        if has_cpa or has_cpa_ship:
            for i, tug in enumerate(self.tugs):
                ci = math.cos(tug.eta.z)
                si = math.sin(tug.eta.z)
                tug_world_vx[i] = ci * tug.nu.x - si * tug.nu.y
                tug_world_vy[i] = si * tug.nu.x + ci * tug.nu.y
        if has_cpa_ship:
            cs = math.cos(self.ship.psi)
            sn = math.sin(self.ship.psi)
            ship_vx_w = cs * ship_u - sn * ship_v
            ship_vy_w = sn * ship_u + cs * ship_v

        for i, tug in enumerate(self.tugs):
            slot = slot_world[self.tug_to_slot[i]]
            dx_w = slot[0] - tug.eta.x
            dy_w = slot[1] - tug.eta.y
            d = math.hypot(dx_w, dy_w)

            # 1) 自身体系速度
            obs[i, 0] = tug.nu.x / 5.0
            obs[i, 1] = tug.nu.y / 5.0
            obs[i, 2] = tug.nu.z / 0.5

            # 2) 自身位置在 slot 局部系下的偏移（slot 朝向 = 大船航向）
            dx_local, dy_local = _world_to_local(dx_w, dy_w, slot[2])
            obs[i, 3] = dx_local / 50.0
            obs[i, 4] = dy_local / 50.0

            # 3) slot 极坐标（log1p 压缩长尾）
            theta = math.atan2(dy_local, dx_local) if d > 1e-6 else 0.0
            obs[i, 5] = math.log1p(d) / 5.0
            obs[i, 6] = math.sin(theta)
            obs[i, 7] = math.cos(theta)

            # 4) 航向误差
            dpsi = _wrap_pi(slot[2] - tug.eta.z)
            obs[i, 8] = math.sin(dpsi)
            obs[i, 9] = math.cos(dpsi)

            # 5) 大船相对自身位置（自身体系下）
            ship_dx_w = self.ship.x - tug.eta.x
            ship_dy_w = self.ship.y - tug.eta.y
            ship_dx_local, ship_dy_local = _world_to_local(ship_dx_w, ship_dy_w, tug.eta.z)
            obs[i, 10] = ship_dx_local / 100.0
            obs[i, 11] = ship_dy_local / 100.0

            # 6) 大船航向相对自身
            dpsi_ship = _wrap_pi(self.ship.psi - tug.eta.z)
            obs[i, 12] = math.sin(dpsi_ship)
            obs[i, 13] = math.cos(dpsi_ship)

            # 7) 大船体系速度
            obs[i, 14] = ship_u / 3.0
            obs[i, 15] = ship_v / 3.0
            obs[i, 16] = ship_r / 0.05

            # 8) 自身执行器实际值（归一化到 [-1, 1]）
            ctrl = tug.get_control_snapshot()
            obs[i, 17] = ctrl["port_rpm_actual"] / tug.rpm_limit
            obs[i, 18] = ctrl["starboard_rpm_actual"] / tug.rpm_limit
            obs[i, 19] = ctrl["port_azimuth_actual_deg"] / tug.azimuth_limit_deg
            obs[i, 20] = ctrl["starboard_azimuth_actual_deg"] / tug.azimuth_limit_deg

            # 9) 上一步动作
            obs[i, 21:25] = self.last_actions[i]

            # 10) 其他 3 个拖轮的相对位置（自身体系下）
            if self.cfg.obs_include_other_tugs:
                other_idx = [j for j in range(self.n_tugs) if j != i]
                for k, j in enumerate(other_idx):
                    other = self.tugs[j]
                    ox_w = other.eta.x - tug.eta.x
                    oy_w = other.eta.y - tug.eta.y
                    ox_local, oy_local = _world_to_local(ox_w, oy_w, tug.eta.z)
                    obs[i, 25 + k * 2] = ox_local / 100.0
                    obs[i, 25 + k * 2 + 1] = oy_local / 100.0

            # 11) slot one-hot（让共享策略前向时知道"我是哪个 slot"）
            obs_idx = _BASE_OBS_DIM
            if self.cfg.obs_include_slot_onehot:
                slot_idx = int(self.tug_to_slot[i])
                obs[i, obs_idx + slot_idx] = 1.0
                obs_idx += _ONEHOT_OBS_DIM

            # 12) 路线特征：下一 waypoint、路线 stage、左/右舷和剩余路线距离
            if getattr(self.cfg, "obs_include_route", False):
                target = self._current_route_target_world(i)
                wp_dx_w = float(target[0] - tug.eta.x)
                wp_dy_w = float(target[1] - tug.eta.y)
                wp_dx_local, wp_dy_local = _world_to_local(wp_dx_w, wp_dy_w, tug.eta.z)
                wp_d = math.hypot(wp_dx_w, wp_dy_w)
                wp_theta = math.atan2(wp_dy_local, wp_dx_local) if wp_d > 1e-6 else 0.0
                route_len = len(self._route_waypoints_body(int(self.tug_to_slot[i])))
                stage_norm = float(self.route_stage[i]) / max(route_len - 1, 1)
                obs[i, obs_idx + 0] = wp_dx_local / 100.0
                obs[i, obs_idx + 1] = wp_dy_local / 100.0
                obs[i, obs_idx + 2] = math.log1p(wp_d) / 5.0
                obs[i, obs_idx + 3] = math.sin(wp_theta)
                obs[i, obs_idx + 4] = math.cos(wp_theta)
                obs[i, obs_idx + 5] = stage_norm
                obs[i, obs_idx + 6] = self._slot_side_sign(int(self.tug_to_slot[i]))
                obs[i, obs_idx + 7] = self._route_remaining_distance(i) / 500.0

            # 13) CPA 特征：显式碰撞风险感知（v32）
            cpa_start = _BASE_OBS_DIM
            if self.cfg.obs_include_slot_onehot:
                cpa_start += _ONEHOT_OBS_DIM
            if getattr(self.cfg, "obs_include_route", False):
                cpa_start += _ROUTE_OBS_DIM
            if has_cpa:
                other_idx = [j for j in range(self.n_tugs) if j != i]
                for k, j in enumerate(other_idx):
                    dx = self.tugs[j].eta.x - tug.eta.x
                    dy = self.tugs[j].eta.y - tug.eta.y
                    vrx = tug_world_vx[j] - tug_world_vx[i]
                    vry = tug_world_vy[j] - tug_world_vy[i]
                    vr_sq = vrx * vrx + vry * vry
                    if vr_sq < 1e-6:
                        dcpa = math.hypot(dx, dy)
                        tcpa = 0.0
                        cpa_dx, cpa_dy = dx, dy
                    else:
                        tcpa = -(dx * vrx + dy * vry) / vr_sq
                        cpa_dx = dx + vrx * tcpa
                        cpa_dy = dy + vry * tcpa
                        dcpa = math.hypot(cpa_dx, cpa_dy) if tcpa >= 0 else math.hypot(dx, dy)
                    if tcpa < 0 or dcpa < 1e-6:
                        tcpa = 0.0
                        cpa_dx, cpa_dy = dx, dy
                    bearing = math.atan2(cpa_dy, cpa_dx) - tug.eta.z if dcpa > 1e-6 else 0.0
                    obs[i, cpa_start + k * 4 + 0] = min(dcpa / 100.0, 1.0)
                    obs[i, cpa_start + k * 4 + 1] = math.tanh(tcpa / 60.0)
                    obs[i, cpa_start + k * 4 + 2] = math.sin(bearing)
                    obs[i, cpa_start + k * 4 + 3] = math.cos(bearing)

            # 14) 拖轮→大船 CPA 特征（v34）
            if has_cpa_ship:
                ship_cpa_start = cpa_start + (_CPA_OBS_DIM if has_cpa else 0)
                dx_s = self.ship.x - tug.eta.x
                dy_s = self.ship.y - tug.eta.y
                vrx_s = ship_vx_w - tug_world_vx[i]
                vry_s = ship_vy_w - tug_world_vy[i]
                vr_sq_s = vrx_s * vrx_s + vry_s * vry_s
                if vr_sq_s < 1e-6:
                    dcpa_s = math.hypot(dx_s, dy_s)
                    tcpa_s = 0.0
                    cpa_dx_s, cpa_dy_s = dx_s, dy_s
                else:
                    tcpa_s = -(dx_s * vrx_s + dy_s * vry_s) / vr_sq_s
                    cpa_dx_s = dx_s + vrx_s * tcpa_s
                    cpa_dy_s = dy_s + vry_s * tcpa_s
                    dcpa_s = math.hypot(cpa_dx_s, cpa_dy_s) if tcpa_s >= 0 else math.hypot(dx_s, dy_s)
                if tcpa_s < 0 or dcpa_s < 1e-6:
                    tcpa_s = 0.0
                    cpa_dx_s, cpa_dy_s = dx_s, dy_s
                bearing_s = math.atan2(cpa_dy_s, cpa_dx_s) - tug.eta.z if dcpa_s > 1e-6 else 0.0
                obs[i, ship_cpa_start + 0] = min(dcpa_s / 100.0, 1.0)
                obs[i, ship_cpa_start + 1] = math.tanh(tcpa_s / 60.0)
                obs[i, ship_cpa_start + 2] = math.sin(bearing_s)
                obs[i, ship_cpa_start + 3] = math.cos(bearing_s)

            # 15) 大船尺度特征：给策略显式几何尺度，避免只适配固定船长/船宽。
            if getattr(self.cfg, "obs_include_ship_size", False):
                ship_size_start = cpa_start
                if has_cpa:
                    ship_size_start += _CPA_OBS_DIM
                if has_cpa_ship:
                    ship_size_start += _CPA_SHIP_OBS_DIM
                base_length = max(float(getattr(self.cfg, "ship_length_m", 200.0)), 1e-6)
                base_beam = max(float(getattr(self.cfg, "ship_beam_m", 30.0)), 1e-6)
                obs[i, ship_size_start + 0] = self.ship.length_m / base_length - 1.0
                obs[i, ship_size_start + 1] = self.ship.beam_m / base_beam - 1.0

            # 16) 自身体系加速度：让 actor 直接感知当前动力学响应。
            if getattr(self.cfg, "obs_include_ego_accel", False):
                ego_accel_start = cpa_start
                if has_cpa:
                    ego_accel_start += _CPA_OBS_DIM
                if has_cpa_ship:
                    ego_accel_start += _CPA_SHIP_OBS_DIM
                if getattr(self.cfg, "obs_include_ship_size", False):
                    ego_accel_start += _SHIP_SIZE_OBS_DIM
                acc = tug.get_last_nu_dot()
                obs[i, ego_accel_start + 0] = acc.x / _TUG_LINEAR_ACCEL_SCALE
                obs[i, ego_accel_start + 1] = acc.y / _TUG_LINEAR_ACCEL_SCALE
                obs[i, ego_accel_start + 2] = acc.z / _TUG_YAW_ACCEL_SCALE

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
            route_len = len(self._route_waypoints_body(slot_idx))
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

    # ---------- 奖励计算 ----------
    def _compute_rewards(
        self, actions: np.ndarray, slot_world: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Any]]:
        cfg = self.cfg
        n = self.n_tugs
        rewards = np.zeros(n, dtype=np.float32)

        # 大船世界系速度（用于速度匹配）
        cs_s = math.cos(self.ship.psi)
        sn_s = math.sin(self.ship.psi)
        ship_vx_w = cs_s * self.ship.u - sn_s * self.ship.v
        ship_vy_w = sn_s * self.ship.u + cs_s * self.ship.v

        comp = {
            "r_progress": np.zeros(n, dtype=np.float32),
            "r_route_progress": np.zeros(n, dtype=np.float32),
            "r_heading": np.zeros(n, dtype=np.float32),
            "r_smooth": np.zeros(n, dtype=np.float32),
            "r_jerk": np.zeros(n, dtype=np.float32),
            "r_mag": np.zeros(n, dtype=np.float32),
            "r_yaw_rate": np.zeros(n, dtype=np.float32),
            "r_speed_match": np.zeros(n, dtype=np.float32),
            "r_chase_speed": np.zeros(n, dtype=np.float32),
            "r_chase_overspeed": np.zeros(n, dtype=np.float32),
            "r_route_speed_limit": np.zeros(n, dtype=np.float32),
            "r_lane": np.zeros(n, dtype=np.float32),
            "r_spacing": np.zeros(n, dtype=np.float32),
            "r_cpa": np.zeros(n, dtype=np.float32),
            "cpa_risk": np.zeros(n, dtype=np.float32),
            "cpa_min_dcpa": np.full(n, np.nan, dtype=np.float32),
            "cpa_min_tcpa": np.full(n, np.nan, dtype=np.float32),
            "r_escort": np.zeros(n, dtype=np.float32),
            "r_safety": np.zeros(n, dtype=np.float32),
            "r_hull_safety": np.zeros(n, dtype=np.float32),
            "r_tug_safety": np.zeros(n, dtype=np.float32),
            "r_ship_future_safety": np.zeros(n, dtype=np.float32),
            "ship_future_risk": np.zeros(n, dtype=np.float32),
            "ship_future_min_hull_dist": np.full(n, np.nan, dtype=np.float32),
            "r_simple": np.zeros(n, dtype=np.float32),
            "r_speed_risk": np.zeros(n, dtype=np.float32),
            "r_safety_risk": np.zeros(n, dtype=np.float32),
            "r_action_pen": np.zeros(n, dtype=np.float32),
            "r_hold": np.zeros(n, dtype=np.float32),
            "hull_dist": np.zeros(n, dtype=np.float32),
            "dist_to_slot": np.zeros(n, dtype=np.float32),
            "route_remaining": np.zeros(n, dtype=np.float32),
            "route_stage": np.zeros(n, dtype=np.int32),
            "heading_err_deg": np.zeros(n, dtype=np.float32),
            "in_zone": np.zeros(n, dtype=np.bool_),
        }

        # 预先计算所有拖轮的世界系位置（用于安全惩罚）
        tug_positions = [(tug.eta.x, tug.eta.y) for tug in self.tugs]
        tug_world_vx = np.zeros(n, dtype=np.float32)
        tug_world_vy = np.zeros(n, dtype=np.float32)
        for i, tug in enumerate(self.tugs):
            ci = math.cos(tug.eta.z)
            si = math.sin(tug.eta.z)
            tug_world_vx[i] = ci * tug.nu.x - si * tug.nu.y
            tug_world_vy[i] = si * tug.nu.x + ci * tug.nu.y
        future_safety_horizon = max(
            float(getattr(cfg, "ship_future_safety_horizon_s", 14.0)),
            float(getattr(cfg, "dt_ctrl", 0.2)),
        )
        future_safety_samples = max(1, int(getattr(cfg, "ship_future_safety_samples", 5)))
        future_safety_times = np.linspace(
            float(getattr(cfg, "dt_ctrl", 0.2)),
            future_safety_horizon,
            future_safety_samples,
            dtype=np.float64,
        )

        for i, tug in enumerate(self.tugs):
            slot = slot_world[self.tug_to_slot[i]]
            d = float(math.hypot(tug.eta.x - slot[0], tug.eta.y - slot[1]))
            dpsi = _wrap_pi(slot[2] - tug.eta.z)
            use_route = self._uses_route_mode()
            if use_route:
                self._advance_route_stage(i)
            route_remaining = self._route_remaining_distance(i)
            route_len = len(self._route_waypoints_body(int(self.tug_to_slot[i])))
            final_route_stage = int(self.route_stage[i]) >= route_len - 1

            # 进度奖励：线性距离差分，每接近 1m 给正奖励。
            # 不用 log 势能——log 在远距离梯度太小，策略会停在远处不靠近。
            # 连续安全惩罚（下方）负责抑制"冲撞"，进度奖励只负责"靠近"。
            r_direct_progress = cfg.reward_progress_w * (self.prev_dist[i] - d)
            route_progress_delta = self.prev_route_remaining[i] - route_remaining
            if use_route:
                route_progress_clip = float(getattr(cfg, "route_progress_step_clip_m", 0.45))
                if route_progress_clip > 0.0:
                    route_progress_delta = float(
                        np.clip(route_progress_delta, -route_progress_clip, route_progress_clip)
                    )
            r_route_progress = cfg.reward_route_progress_w * route_progress_delta
            r_progress = r_route_progress if use_route else r_direct_progress

            # 朝向奖励：cos(dpsi)，朝向对齐时 +w，反向时 -w
            r_heading = cfg.reward_heading_w * math.cos(dpsi)

            # 平滑性惩罚：动作变化平方
            da = actions[i] - self.last_actions[i]
            r_smooth = -cfg.reward_smooth_w * float(np.sum(da * da))

            # 急动度惩罚：动作变化的变化（二阶差分），抑制高频振荡
            dda = da - self.last_action_changes[i]
            r_jerk = -cfg.reward_jerk_w * float(np.sum(dda * dda))

            # 动作幅度惩罚：避免无谓使用满舵满油门
            r_mag = -cfg.reward_mag_w * float(np.sum(actions[i] * actions[i]))

            # 偏航角速度惩罚：拖轮不应在原地疯狂自转
            r_yaw_rate = -cfg.reward_yaw_rate_w * (tug.nu.z * tug.nu.z)

            # 速度匹配（仅在 slot 附近权重大）：拖轮世界速度与大船世界速度差
            tug_vx_w = float(tug_world_vx[i])
            tug_vy_w = float(tug_world_vy[i])
            dvx = tug_vx_w - ship_vx_w
            dvy = tug_vy_w - ship_vy_w
            speed_err_sq = dvx * dvx + dvy * dvy
            speed_err = math.sqrt(speed_err_sq)
            # v25 P3: 用 speed_err * tanh(speed_err/3) 替代 speed_err_sq。
            # 原版 speed_err_sq 在 d→0 时单步惩罚可达 -0.4~-0.8，比 r_zone=+1 还大，
            # 导致策略到 zone 边缘时净奖励为负，反而退回。
            # tanh 压缩让大误差封顶在 ~3，小误差仍接近 speed_err_sq 的线性段。
            speed_err_compressed = speed_err * math.tanh(speed_err / 3.0)
            # 速度匹配：扩大适用范围。原 exp(-d/40) 在 d>70m 时权重已 <0.2，
            # 远距离拖轮拿不到速度匹配梯度；改为 exp(-d/80) 让 d=80m 仍有 ~0.37 权重，
            # 但靠近时权重仍大于远处，保留课程结构。
            zone_factor = math.exp(-d / 80.0)
            if use_route and not final_route_stage:
                r_speed_match = 0.0
            else:
                r_speed_match = -cfg.reward_speed_match_w * zone_factor * speed_err_compressed

            # 船尾追赶阶段不要求速度匹配，而是鼓励相对大船有一个小的正向 closing speed。
            # 到 final stage 后再切回上面的速度匹配，避免"陪走但不到位"。
            rel_u_ship, _ = _world_to_local(dvx, dvy, self.ship.psi)
            r_chase_overspeed = 0.0
            r_route_speed_limit = 0.0
            if use_route and not final_route_stage:
                target_chase = float(getattr(cfg, "route_chase_speed_target_ms", 0.5))
                chase_score = max(0.0, 1.0 - abs(rel_u_ship - target_chase) / max(target_chase, 1e-3))
                r_chase_speed = cfg.reward_chase_speed_w * chase_score
                max_chase = float(getattr(cfg, "route_chase_speed_max_ms", 0.9))
                if rel_u_ship > max_chase:
                    excess = (rel_u_ship - max_chase) / max(max_chase, 1e-3)
                    r_chase_overspeed = -float(getattr(cfg, "reward_chase_overspeed_w", 0.0)) * excess * excess
                speed_soft_limit = float(getattr(cfg, "route_tug_speed_soft_limit_ms", 3.0))
                tug_speed_world = math.hypot(tug_vx_w, tug_vy_w)
                if tug_speed_world > speed_soft_limit:
                    excess = (tug_speed_world - speed_soft_limit) / max(speed_soft_limit, 1e-3)
                    r_route_speed_limit = -float(getattr(cfg, "reward_route_speed_limit_w", 0.0)) * excess * excess
            else:
                r_chase_speed = 0.0

            # 安全惩罚：连续的碰撞预警惩罚（不只是终端惩罚）。
            # 当拖轮进入危险区域时每步给负奖励，让策略在碰撞发生前就学会规避。
            # 使用指数势能：距离越近惩罚越大，在碰撞阈值处约等于 -1.0。
            r_safety = 0.0
            r_hull_safety = 0.0
            r_tug_safety = 0.0
            # 与大船船体的安全惩罚
            d_hull = self.ship.distance_from_hull(tug.eta.x, tug.eta.y)
            safe_hull = float(getattr(cfg, "ship_safety_dist_m", cfg.ship_collision_dist_m * 3.0))
            hull_safety_w = float(cfg.reward_safety_w)
            if final_route_stage:
                safe_hull = max(
                    safe_hull,
                    float(getattr(cfg, "ship_safety_final_dist_m", safe_hull)),
                )
                hull_safety_w *= float(getattr(cfg, "reward_hull_safety_final_multiplier", 1.0))
            if d_hull < safe_hull:
                r_hull_safety -= hull_safety_w * math.exp(
                    -(d_hull - cfg.ship_collision_dist_m) / max(safe_hull - cfg.ship_collision_dist_m, 1e-3)
                )
            r_safety += r_hull_safety

            # 预测性船体安全惩罚：把未来若干控制步的 tug/ship 轨迹投影到当前
            # 运动学平面中，提前压制"现在看着安全、几秒后会贴船"的高速切入。
            r_ship_future_safety = 0.0
            future_min_hull = float("inf")
            future_risk_sum = 0.0
            future_safe_hull = float(getattr(cfg, "ship_future_safety_dist_m", safe_hull))
            if final_route_stage:
                future_safe_hull = max(
                    future_safe_hull,
                    float(getattr(cfg, "ship_future_safety_final_dist_m", future_safe_hull)),
                )
            future_span = max(future_safe_hull - cfg.ship_collision_dist_m, 1e-3)
            for tau in future_safety_times:
                ship_x_f = self.ship.x + ship_vx_w * float(tau)
                ship_y_f = self.ship.y + ship_vy_w * float(tau)
                ship_psi_f = _wrap_pi(self.ship.psi + self.ship.r * float(tau))
                tug_x_f = tug.eta.x + tug_vx_w * float(tau)
                tug_y_f = tug.eta.y + tug_vy_w * float(tau)
                d_future = self._distance_from_ship_hull_pose(
                    tug_x_f, tug_y_f, ship_x_f, ship_y_f, ship_psi_f
                )
                future_min_hull = min(future_min_hull, d_future)
                if d_future >= future_safe_hull:
                    continue
                dist_score = min(
                    1.0,
                    max(0.0, (future_safe_hull - d_future) / future_span),
                )
                time_score = max(0.0, 1.0 - float(tau) / future_safety_horizon)
                future_risk_sum += dist_score * dist_score * time_score
            if future_risk_sum > 0.0:
                future_w = float(getattr(cfg, "reward_ship_future_safety_w", 0.0))
                if final_route_stage:
                    future_w *= float(getattr(cfg, "reward_ship_future_safety_final_multiplier", 1.0))
                future_pen = future_w * future_risk_sum
                future_pen = min(
                    future_pen,
                    float(getattr(cfg, "reward_ship_future_safety_max_penalty", 0.9)),
                )
                r_ship_future_safety = -future_pen

            # 与其他拖轮的安全惩罚
            safe_tug = cfg.tug_collision_dist_m * 2.0
            for j, (ox, oy) in enumerate(tug_positions):
                if j == i:
                    continue
                d_pair = math.hypot(tug.eta.x - ox, tug.eta.y - oy)
                if d_pair < safe_tug:
                    r_tug_safety -= cfg.reward_safety_w * math.exp(
                        -(d_pair - cfg.tug_collision_dist_m) / max(safe_tug - cfg.tug_collision_dist_m, 1e-3)
                    )
            r_safety += r_tug_safety

            # 航道约束：船尾追赶任务中，左舷/右舷角色应沿对应舷侧绕行，
            # 不应从船体后方直接穿越到另一侧或贴着中心线抢最短路径。
            r_lane = 0.0
            if use_route:
                x_b, y_b = self._ship_body_xy(tug.eta.x, tug.eta.y)
                side_coord = self._slot_side_sign(int(self.tug_to_slot[i])) * y_b
                lane_min = float(getattr(cfg, "route_lane_min_lat_m", 32.0))
                l_half = self.ship.length_m / 2.0
                corridor_min_x = -l_half - float(getattr(cfg, "route_stern_gate_dist_m", 60.0)) - 40.0
                corridor_max_x = l_half + float(getattr(cfg, "route_stern_gate_dist_m", 60.0)) + 40.0
                if corridor_min_x <= x_b <= corridor_max_x:
                    lane_violation = max(0.0, lane_min - side_coord) / max(lane_min, 1e-3)
                    r_lane = -cfg.reward_lane_w * lane_violation

            # 同侧拖轮间距：比硬碰撞阈值更早给梯度，抑制同侧船首/船尾拖轮在
            # stern gate 和舷侧航道内追尾或并线挤压。
            r_spacing = 0.0
            if use_route:
                spacing_dist = float(getattr(cfg, "route_tug_spacing_dist_m", 60.0))
                own_side = self._slot_side_sign(int(self.tug_to_slot[i]))
                for j, (ox, oy) in enumerate(tug_positions):
                    if j == i:
                        continue
                    if self._slot_side_sign(int(self.tug_to_slot[j])) != own_side:
                        continue
                    d_pair = math.hypot(tug.eta.x - ox, tug.eta.y - oy)
                    if d_pair < spacing_dist:
                        frac = (spacing_dist - d_pair) / max(spacing_dist, 1e-3)
                        r_spacing -= cfg.reward_spacing_w * frac * frac

            # CPA 风险惩罚：把 v32 的会遇风险特征接入 reward。
            # 距离惩罚只看当前距离，容易漏掉“现在还远但正在快速会遇”的风险；
            # CPA 使用未来短时间窗口内的最近会遇距离，鼓励拖轮提前错峰/减速。
            r_cpa = 0.0
            cpa_risk_sum = 0.0
            min_dcpa = float("inf")
            min_tcpa = float("inf")
            cpa_alert = float(getattr(cfg, "cpa_alert_dist_m", 70.0))
            cpa_horizon = float(getattr(cfg, "cpa_time_horizon_s", 45.0))
            cpa_floor = float(cfg.tug_collision_dist_m)
            cpa_span = max(cpa_alert - cpa_floor, 1e-3)
            for j, (ox, oy) in enumerate(tug_positions):
                if j == i:
                    continue
                dx = ox - tug.eta.x
                dy = oy - tug.eta.y
                vrx = float(tug_world_vx[j] - tug_world_vx[i])
                vry = float(tug_world_vy[j] - tug_world_vy[i])
                vr_sq = vrx * vrx + vry * vry
                if vr_sq < 1e-6:
                    tcpa = 0.0
                    dcpa = math.hypot(dx, dy)
                else:
                    raw_tcpa = -(dx * vrx + dy * vry) / vr_sq
                    if raw_tcpa < 0.0:
                        tcpa = 0.0
                        dcpa = math.hypot(dx, dy)
                    else:
                        tcpa = raw_tcpa
                        dcpa = math.hypot(dx + vrx * tcpa, dy + vry * tcpa)
                min_dcpa = min(min_dcpa, dcpa)
                min_tcpa = min(min_tcpa, tcpa)
                if dcpa >= cpa_alert or tcpa > cpa_horizon:
                    continue
                dcpa_score = min(1.0, max(0.0, (cpa_alert - dcpa) / cpa_span))
                tcpa_score = max(0.0, 1.0 - tcpa / max(cpa_horizon, 1e-3))
                cpa_risk_sum += dcpa_score * dcpa_score * tcpa_score
            if cpa_risk_sum > 0.0:
                cpa_weight = float(getattr(cfg, "reward_cpa_w", 0.18))
                if final_route_stage:
                    cpa_weight *= float(getattr(cfg, "reward_cpa_final_multiplier", 2.0))
                cpa_pen = cpa_weight * cpa_risk_sum
                cpa_pen = min(cpa_pen, float(getattr(cfg, "reward_cpa_max_penalty", 0.75)))
                r_cpa = -cpa_pen

            # in-zone 判定：硬阈值用于累计 hold_time，软评分用于伴航累进奖励。
            # hold_time_s 课程逐步提高，要求稳定伴航而非瞬时到位。
            # r_escort 改为加权和（非乘积），消除"任一维度=0 则总分=0"的梯度悬崖。
            # hold_time 终止判定继续用硬阈值不变。
            in_zone_now = (
                d < cfg.pos_tol_m
                and abs(dpsi) < cfg.heading_tol_rad
                and speed_err < cfg.speed_tol_ms
            )
            pos_score = max(0.0, 1.0 - d / cfg.pos_tol_m)
            hdg_score = max(0.0, 1.0 - abs(dpsi) / cfg.heading_tol_rad)
            spd_score = max(0.0, 1.0 - speed_err / cfg.speed_tol_ms)
            # 伴航累进奖励：v38 将 r_zone 乘积改为带位置门控的加权和。
            # pos_score 作为主门控——距离越近奖励越大，远距离归零防止"远处对齐领奖励"。
            # hdg_score/spd_score 在位置评分基础上提供增量，消除乘积梯度的悬崖效应。
            escort_w = float(cfg.reward_escort_w)
            if final_route_stage:
                escort_w *= float(getattr(cfg, "reward_escort_final_multiplier", 1.0))
            r_escort = escort_w * pos_score * (0.4 + 0.3 * hdg_score + 0.3 * spd_score)
            if in_zone_now:
                self.in_zone_steps[i] += 1
            else:
                self.in_zone_steps[i] = 0

            # v52 staged simple reward:
            # 旧的原子项仍全部计算并写入 comp；真正训练时只使用少量组合项。
            # v53: safety 改为 capped-sum。v52 的 min 只保留最严重风险，长期训练后
            # 同时存在 hull/CPA/spacing 风险时惩罚偏弱，策略容易漂移到碰撞解。
            r_action_pen = r_smooth + 0.5 * r_jerk + 0.25 * r_mag + 0.25 * r_yaw_rate
            r_speed_risk = min(0.0, r_chase_overspeed, r_route_speed_limit)
            hull_cap = float(getattr(cfg, "reward_simple_hull_risk_cap", 1.4))
            tug_cap = float(getattr(cfg, "reward_simple_tug_risk_cap", 1.2))
            safety_cap = float(getattr(cfg, "reward_simple_safety_risk_cap", 1.8))
            r_hull_risk = max(-hull_cap, min(0.0, r_hull_safety) + min(0.0, r_ship_future_safety))
            r_tug_risk = max(-tug_cap, min(0.0, r_tug_safety) + min(0.0, r_spacing) + min(0.0, r_cpa))
            r_safety_risk = max(-safety_cap, r_hull_risk + r_tug_risk)
            hold_steps = max(1, int(round(cfg.hold_time_s / cfg.dt_ctrl)))
            hold_frac = min(1.0, float(self.in_zone_steps[i]) / float(hold_steps))
            r_hold = float(getattr(cfg, "reward_simple_hold_w", 0.0)) * hold_frac if in_zone_now else 0.0

            if getattr(cfg, "reward_use_simple_stage", False):
                if use_route and not final_route_stage:
                    r = (
                        float(getattr(cfg, "reward_simple_nonfinal_progress_w", 1.0)) * r_progress
                        + float(getattr(cfg, "reward_simple_route_chase_w", 0.0)) * r_chase_speed
                        + float(getattr(cfg, "reward_simple_speed_risk_w", 1.0)) * r_speed_risk
                        + float(getattr(cfg, "reward_simple_safety_risk_w", 1.0)) * r_safety_risk
                        + float(getattr(cfg, "reward_simple_lane_w", 1.0)) * r_lane
                        + float(getattr(cfg, "reward_simple_action_w", 1.0)) * r_action_pen
                    )
                else:
                    r = (
                        float(getattr(cfg, "reward_simple_final_escort_w", 1.0)) * r_escort
                        + r_hold
                        + float(getattr(cfg, "reward_simple_final_speed_match_w", 0.0)) * r_speed_match
                        + float(getattr(cfg, "reward_simple_safety_risk_w", 1.0)) * r_safety_risk
                        + float(getattr(cfg, "reward_simple_action_w", 1.0)) * r_action_pen
                    )
            else:
                r = (
                    r_progress + r_heading + r_smooth + r_jerk + r_mag + r_yaw_rate
                    + r_speed_match + r_chase_speed + r_chase_overspeed + r_route_speed_limit
                    + r_lane + r_spacing + r_cpa + r_escort + r_safety + r_ship_future_safety
                )
            rewards[i] = r

            comp["r_progress"][i] = r_progress
            comp["r_route_progress"][i] = r_route_progress
            comp["r_heading"][i] = r_heading
            comp["r_smooth"][i] = r_smooth
            comp["r_jerk"][i] = r_jerk
            comp["r_mag"][i] = r_mag
            comp["r_yaw_rate"][i] = r_yaw_rate
            comp["r_speed_match"][i] = r_speed_match
            comp["r_chase_speed"][i] = r_chase_speed
            comp["r_chase_overspeed"][i] = r_chase_overspeed
            comp["r_route_speed_limit"][i] = r_route_speed_limit
            comp["r_lane"][i] = r_lane
            comp["r_spacing"][i] = r_spacing
            comp["r_cpa"][i] = r_cpa
            comp["cpa_risk"][i] = cpa_risk_sum
            if math.isfinite(min_dcpa):
                comp["cpa_min_dcpa"][i] = min_dcpa
            if math.isfinite(min_tcpa):
                comp["cpa_min_tcpa"][i] = min_tcpa
            comp["r_escort"][i] = r_escort
            comp["r_safety"][i] = r_safety
            comp["r_hull_safety"][i] = r_hull_safety
            comp["r_tug_safety"][i] = r_tug_safety
            comp["r_ship_future_safety"][i] = r_ship_future_safety
            comp["ship_future_risk"][i] = future_risk_sum
            if math.isfinite(future_min_hull):
                comp["ship_future_min_hull_dist"][i] = future_min_hull
            comp["r_simple"][i] = r if getattr(cfg, "reward_use_simple_stage", False) else 0.0
            comp["r_speed_risk"][i] = r_speed_risk
            comp["r_safety_risk"][i] = r_safety_risk
            comp["r_action_pen"][i] = r_action_pen
            comp["r_hold"][i] = r_hold
            comp["hull_dist"][i] = d_hull
            comp["dist_to_slot"][i] = d
            comp["route_remaining"][i] = route_remaining
            comp["route_stage"][i] = int(self.route_stage[i])
            comp["heading_err_deg"][i] = math.degrees(abs(dpsi))
            comp["in_zone"][i] = in_zone_now

        return rewards, {"reward_components": comp}

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
                "r_escort": float(self.last_reward_components.get("r_escort", np.zeros(self.n_tugs))[i]),
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
