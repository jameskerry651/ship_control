"""拖轮编队环境的路径规划。

在船体坐标上进行A*网格搜索、直线可达(LOS)简化、B样条平滑，
以及航点推进跟踪。
"""

from __future__ import annotations

import heapq
import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import splprep, splev

if TYPE_CHECKING:
    from env.formation_env import FormationEnv


class RoutePlanner:
    """Route planner that owns all path-finding and waypoint-tracking logic."""

    def __init__(self, env: FormationEnv) -> None:
        self._env = env

    # ------------------------------------------------------------------ helpers

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
        shrunk = (x_min + 1e-6, x_max - 1e-6, y_min + 1e-6, y_max - 1e-6)
        if shrunk[0] >= shrunk[1] or shrunk[2] >= shrunk[3]:
            return True
        return not self._segment_intersects_closed_rect(p0, p1, shrunk)

    def _simplify_path_los(
        self,
        points: list[tuple[float, float]],
        rect: tuple[float, float, float, float],
    ) -> list[tuple[float, float]]:
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

    # --------------------------------------------------------------- slot utils

    def _slot_side_sign(self, slot_idx: int) -> float:
        return -1.0 if int(slot_idx) in (0, 2) else 1.0

    @staticmethod
    def _slot_is_bow(slot_idx: int) -> bool:
        return int(slot_idx) in (0, 1)

    def _slot_lane_lat_abs(self, slot_idx: int) -> float:
        cfg = self._env.cfg
        if self._slot_is_bow(slot_idx):
            return float(getattr(cfg, "route_bow_lane_lat_m", 90.0))
        return float(getattr(cfg, "route_stern_lane_lat_m", 55.0))

    # --------------------------------------------------------------- hull geom

    def _hull_rect_body(self) -> tuple[float, float, float, float]:
        l_half = self._env.ship.length_m / 2.0
        b_half = self._env.ship.beam_m / 2.0
        return (-l_half, l_half, -b_half, b_half)

    # --------------------------------------------------------------- A*

    def _astar_path_body(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
        rect: tuple[float, float, float, float],
        side: float,
        *,
        allow_los_shortcut: bool = True,
    ) -> list[tuple[float, float]]:
        if allow_los_shortcut and self._body_segment_visible(start, goal, rect):
            return [start, goal]

        cfg = self._env.cfg
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

    # ----------------------------------------------------- waypoint processing

    def _dedupe_route_points(self, points: list[tuple[float, float]]) -> np.ndarray:
        if not points:
            return np.zeros((0, 2), dtype=np.float64)
        min_spacing = float(getattr(self._env.cfg, "route_min_waypoint_spacing_m", 2.0))
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
        if len(points) <= 2:
            return points
        k = min(3, len(points) - 1)
        tck, u = splprep([points[:, 0], points[:, 1]], s=s, k=k)
        if num_points is None:
            num_points = max(len(points) * 3, 12)
        u_new = np.linspace(0, 1, num_points)
        return np.column_stack(splev(u_new, tck))

    # ----------------------------------------------------- route construction

    def _route_at_slot_skip_tol_m(self) -> float:
        cfg = self._env.cfg
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
        return self._route_body_distance(start, goal) <= self._route_at_slot_skip_tol_m()

    def _plan_route_segments_body(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
        slot_idx: int,
    ) -> list[tuple[float, float]]:
        if self._route_already_at_slot(start, goal):
            return [start, goal]

        cfg = self._env.cfg
        side = self._slot_side_sign(slot_idx)
        rect = self._hull_rect_body()
        l_half = self._env.ship.length_m / 2.0
        lane_y = side * self._slot_lane_lat_abs(slot_idx)
        stern_back = float(getattr(cfg, "route_stern_gate_dist_m", 60.0))

        anchors: list[tuple[float, float]] = [start]
        if start[0] < -l_half - 25.0:
            anchors.append((-l_half - stern_back, lane_y))
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
        cached = self._env._route_waypoints_body_cache.get(tug_idx)
        if cached is not None:
            return cached

        cfg = self._env.cfg
        tug = self._env.tugs[tug_idx]
        start_xy = self._env._ship_body_xy(tug.eta.x, tug.eta.y)
        start = (float(start_xy[0]), float(start_xy[1]))
        slot_idx = int(self._env.tug_to_slot[tug_idx])
        slot_arr = self._env.ship.slot_positions_body()[slot_idx, :2]
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

        self._env._route_waypoints_body_cache[tug_idx] = result
        return result

    def _route_waypoints_world_for_tug(self, tug_idx: int) -> np.ndarray:
        points_body = self._route_waypoints_body_for_tug(tug_idx)
        points_world = np.zeros_like(points_body)
        for k, (x_b, y_b) in enumerate(points_body):
            points_world[k] = self._env._ship_body_to_world_xy(float(x_b), float(y_b))
        return points_world

    # --------------------------------------------------- route-state tracking

    def _advance_route_stage(self, tug_idx: int) -> None:
        waypoints = self._route_waypoints_world_for_tug(tug_idx)
        tol = float(getattr(self._env.cfg, "route_waypoint_tol_m", 35.0))
        tug = self._env.tugs[tug_idx]
        while int(self._env.route_stage[tug_idx]) < len(waypoints) - 1:
            target = waypoints[int(self._env.route_stage[tug_idx])]
            if math.hypot(tug.eta.x - target[0], tug.eta.y - target[1]) > tol:
                break
            self._env.route_stage[tug_idx] += 1

    def _route_remaining_distance(self, tug_idx: int) -> float:
        waypoints = self._route_waypoints_world_for_tug(tug_idx)
        stage = int(np.clip(self._env.route_stage[tug_idx], 0, len(waypoints) - 1))
        tug = self._env.tugs[tug_idx]
        rem = math.hypot(tug.eta.x - waypoints[stage, 0], tug.eta.y - waypoints[stage, 1])
        for k in range(stage, len(waypoints) - 1):
            rem += float(np.linalg.norm(waypoints[k + 1] - waypoints[k]))
        return float(rem)

    def _current_route_target_world(self, tug_idx: int) -> np.ndarray:
        waypoints = self._route_waypoints_world_for_tug(tug_idx)
        stage = int(np.clip(self._env.route_stage[tug_idx], 0, len(waypoints) - 1))
        return waypoints[stage]
