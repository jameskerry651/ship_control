"""拖船编队环境的稠密奖励。

奖励结构:
    R = w_dist * R_progress   (距离进度)
      - w_distance * C_distance (目标外距离成本)
      + w_hold * R_hold       (目标内保持)
      + w_vel  * R_vel        (速度匹配)
      - w_coll * P_collision  (碰撞规避；船软碰可走廊软化)
      + R_team                (团队目标外距离成本)

终端奖励由 FormationEnv.step 处理，不参与稠密归一化。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from env.state import MutableEpisodeState, SimState
from physics.large_ship_model import _wrap_pi


class FormationRewardComputer:

    def __init__(self) -> None:
        pass

    @staticmethod
    def _world_velocity(psi: float, u: float, v: float) -> tuple[float, float]:
        c = math.cos(psi)
        s = math.sin(psi)
        return c * u - s * v, s * u + c * v

    @staticmethod
    def _barrier(distance: float, collision_distance: float, safe_distance: float) -> float:
        span = max(safe_distance - collision_distance, 1e-6)
        if distance >= safe_distance:
            return 0.0
        return float(np.clip((safe_distance - distance) / span, 0.0, 1.0))

    @staticmethod
    def _cpa_metrics(rx: float, ry: float, rvx: float, rvy: float, horizon_s: float) -> tuple[float, float, bool]:
        rel_speed_sq = rvx * rvx + rvy * rvy
        closing_dot = rx * rvx + ry * rvy
        if rel_speed_sq <= 1e-9 or closing_dot >= 0.0:
            return horizon_s, math.hypot(rx, ry), False
        tcpa = -closing_dot / rel_speed_sq
        dcpa = math.hypot(rx + rvx * tcpa, ry + rvy * tcpa)
        return tcpa, dcpa, tcpa <= horizon_s

    @classmethod
    def _cpa_risk(cls, dcpa: float, tcpa: float, collision_distance: float, safe_distance: float, horizon_s: float) -> float:
        time_weight = float(np.clip(1.0 - tcpa / max(horizon_s, 1e-6), 0.0, 1.0))
        return cls._barrier(dcpa, collision_distance, safe_distance) * time_weight

    @staticmethod
    def _corridor_gate(
        tx: float,
        ty: float,
        slot_x: float,
        slot_y: float,
        ship_x: float,
        ship_y: float,
        d: float,
        hold_start_m: float,
        half_width_m: float,
        axial_slack_m: float,
    ) -> float:
        """Approach corridor gate in [0, 1] along the ship→slot axis.

        Axis ``e`` is the fixed unit vector from ship through slot. ``r`` is
        slot→tug; ``a = r·e`` is outboard when positive. Overshoot past the slot
        toward the hull is allowed up to ``axial_slack_m`` (``a >= -slack``).
        """
        if d >= hold_start_m:
            return 0.0
        if d <= 1e-6:
            return 1.0
        ax = slot_x - ship_x
        ay = slot_y - ship_y
        axis_norm = math.hypot(ax, ay)
        if axis_norm <= 1e-6:
            return 0.0
        e_x = ax / axis_norm
        e_y = ay / axis_norm
        r_x = tx - slot_x
        r_y = ty - slot_y
        a = r_x * e_x + r_y * e_y
        if a < -axial_slack_m:
            return 0.0
        lat = math.hypot(r_x - a * e_x, r_y - a * e_y)
        lat_n = lat / max(half_width_m, 1e-6)
        if lat_n >= 1.0:
            lat_gate = 0.0
        elif lat_n <= 0.0:
            lat_gate = 1.0
        else:
            u = 1.0 - lat_n
            lat_gate = u * u * (3.0 - 2.0 * u)
        return float(lat_gate)

    @staticmethod
    def _ship_soft_scale(corridor_gate: float, s_min: float) -> float:
        s_min = float(np.clip(s_min, 0.0, 1.0))
        g = float(np.clip(corridor_gate, 0.0, 1.0))
        return 1.0 - (1.0 - s_min) * g

    def compute_rewards(
        self, state: SimState, episode: MutableEpisodeState, actions: np.ndarray, slot_world: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        cfg = state.cfg
        n = state.n_tugs
        rewards = np.zeros(n, dtype=np.float32)

        comp = {
            "r_total": np.zeros(n, dtype=np.float32),
            "r_dist": np.zeros(n, dtype=np.float32),
            "r_hold": np.zeros(n, dtype=np.float32),
            "r_velocity": np.zeros(n, dtype=np.float32),
            "r_team": np.zeros(n, dtype=np.float32),
            "p_distance": np.zeros(n, dtype=np.float32),
            "p_collision": np.zeros(n, dtype=np.float32),
            "p_ship_collision": np.zeros(n, dtype=np.float32),
            "p_tug_collision": np.zeros(n, dtype=np.float32),
            "dist_to_slot": np.zeros(n, dtype=np.float32),
            "heading_err_deg": np.zeros(n, dtype=np.float32),
            "speed_err": np.zeros(n, dtype=np.float32),
            "hull_dist": np.zeros(n, dtype=np.float32),
            "in_zone": np.zeros(n, dtype=np.bool_),
            "corridor_gate": np.zeros(n, dtype=np.float32),
            "ship_soft_scale": np.ones(n, dtype=np.float32),
        }

        ship_vx_w, ship_vy_w = state.ship.world_velocity()
        tug_positions = [(tug.x, tug.y) for tug in state.tugs]
        tug_velocities = [tug.world_velocity() for tug in state.tugs]

        # --- config extraction ---
        w_dist = float(getattr(cfg, "reward_dist_w", 1.0))
        distance_cost_w = max(float(getattr(cfg, "reward_distance_cost_w", 0.2)), 0.0)
        w_hold = float(getattr(cfg, "reward_hold_w", 1.0))
        w_vel = float(getattr(cfg, "reward_velocity_w", 0.25))
        w_coll = float(getattr(cfg, "reward_collision_w", 3.0))
        collision_cap = float(getattr(cfg, "reward_collision_cap", 1.5))
        progress_clip = max(float(getattr(cfg, "reward_dist_progress_clip_m", 1.0)), 1e-6)
        distance_scale = max(float(getattr(cfg, "reward_dist_scale_m", 200.0)), 1e-6)
        target_tol = max(float(cfg.pos_tol_m), 1e-6)
        hold_start_m = max(float(getattr(cfg, "reward_hold_start_m", 120.0)), 1e-6)

        speed_scale = max(float(getattr(cfg, "reward_velocity_speed_scale_ms", cfg.speed_tol_ms)), 1e-6)
        yaw_scale = max(float(getattr(cfg, "reward_velocity_yaw_scale_rads", 0.05)), 1e-6)

        ship_safe_dist = max(float(getattr(cfg, "reward_collision_ship_safe_m", 60.0)), float(cfg.ship_collision_dist_m) + 1e-6)
        tug_safe_dist = max(float(getattr(cfg, "reward_collision_tug_safe_m", 80.0)), float(cfg.tug_collision_dist_m) + 1e-6)
        cpa_horizon_s = max(float(getattr(cfg, "reward_cpa_horizon_s", 60.0)), 1e-6)
        cpa_w = max(float(getattr(cfg, "reward_collision_cpa_w", 2.0)), 0.0)
        corridor_half_w = max(float(getattr(cfg, "reward_corridor_half_width_m", 40.0)), 1e-6)
        corridor_axial_slack = max(float(getattr(cfg, "reward_corridor_axial_slack_m", 30.0)), 0.0)
        ship_soft_min = float(getattr(cfg, "reward_ship_soft_min_scale", 0.15))

        # --- team sync ---
        w_team = float(getattr(cfg, "reward_team_w", 0.0))

        for i, tug in enumerate(state.tugs):
            slot = slot_world[state.tug_to_slot[i]]
            d = float(math.hypot(tug.x - slot[0], tug.y - slot[1]))
            dpsi = _wrap_pi(float(slot[2]) - tug.psi)

            tug_vx_w, tug_vy_w = tug_velocities[i]
            dvx = tug_vx_w - ship_vx_w
            dvy = tug_vy_w - ship_vy_w
            speed_err = math.hypot(dvx, dvy)

            target_x = float(np.clip(1.0 - d / target_tol, 0.0, 1.0))
            target_gate = target_x * target_x * (3.0 - 2.0 * target_x)
            progress = float(np.clip((float(episode.prev_dist[i]) - d) / progress_clip, -1.0, 1.0))
            r_dist = (1.0 - target_gate) * progress
            p_distance = (1.0 - target_gate) * float(np.clip(d / distance_scale, 0.0, 1.0))
            head_score = max(0.0, 1.0 - abs(dpsi) / max(cfg.heading_tol_rad, 1e-6))
            speed_score = max(0.0, 1.0 - speed_err / max(cfg.speed_tol_ms, 1e-6))
            r_hold = target_gate * (0.5 + 0.25 * head_score + 0.25 * speed_score)

            # -- in-zone tracking --
            in_zone_now = d < cfg.pos_tol_m and abs(dpsi) < cfg.heading_tol_rad and speed_err < cfg.speed_tol_ms

            speed_pen = 1.0 - math.exp(-((speed_err / speed_scale) ** 2))
            yaw_err = abs(tug.r - state.ship.r)
            yaw_pen = 1.0 - math.exp(-((yaw_err / yaw_scale) ** 2))
            r_vel = -target_gate * (0.8 * speed_pen + 0.2 * yaw_pen)

            # -- collision --
            d_hull = state.ship.distance_from_hull(tug.x, tug.y)
            p_ship_prox = self._barrier(d_hull, cfg.ship_collision_dist_m, ship_safe_dist)
            ship_tcpa, _, ship_cpa_active = self._cpa_metrics(
                tug.x - state.ship.x, tug.y - state.ship.y, tug_vx_w - ship_vx_w, tug_vy_w - ship_vy_w, cpa_horizon_s,
            )
            p_ship_cpa = 0.0
            if ship_cpa_active:
                future_tug_x = tug.x + tug_vx_w * ship_tcpa
                future_tug_y = tug.y + tug_vy_w * ship_tcpa
                future_ship_x = state.ship.x + ship_vx_w * ship_tcpa
                future_ship_y = state.ship.y + ship_vy_w * ship_tcpa
                future_ship_psi = state.ship.psi + state.ship.r * ship_tcpa
                ship_dcpa_hull = state.ship.distance_from_hull_pose(
                    future_tug_x, future_tug_y, future_ship_x, future_ship_y, future_ship_psi,
                )
                p_ship_cpa = self._cpa_risk(ship_dcpa_hull, ship_tcpa, cfg.ship_collision_dist_m, ship_safe_dist, cpa_horizon_s)
            p_ship = p_ship_prox + cpa_w * p_ship_cpa
            c_gate = self._corridor_gate(
                tug.x,
                tug.y,
                float(slot[0]),
                float(slot[1]),
                float(state.ship.x),
                float(state.ship.y),
                d,
                hold_start_m,
                corridor_half_w,
                corridor_axial_slack,
            )
            soft = self._ship_soft_scale(c_gate, ship_soft_min)
            p_ship *= soft

            p_tug_prox = 0.0
            p_tug_cpa = 0.0
            for j, (other_x, other_y) in enumerate(tug_positions):
                if j == i:
                    continue
                d_pair = math.hypot(tug.x - other_x, tug.y - other_y)
                p_tug_prox += self._barrier(d_pair, cfg.tug_collision_dist_m, tug_safe_dist)
                other_vx, other_vy = tug_velocities[j]
                tug_tcpa, tug_dcpa, tug_cpa_active = self._cpa_metrics(
                    other_x - tug.x, other_y - tug.y, other_vx - tug_vx_w, other_vy - tug_vy_w, cpa_horizon_s,
                )
                if tug_cpa_active:
                    p_tug_cpa += self._cpa_risk(tug_dcpa, tug_tcpa, cfg.tug_collision_dist_m, tug_safe_dist, cpa_horizon_s)
            p_tug = p_tug_prox + cpa_w * p_tug_cpa
            p_coll = min(p_ship + p_tug, collision_cap)

            # -- total --
            r_total = (
                w_dist * r_dist
                - distance_cost_w * p_distance
                + w_hold * r_hold
                + w_vel * r_vel
                - w_coll * p_coll
            )
            rewards[i] = r_total

            # -- in-zone step tracking --
            if in_zone_now:
                episode.in_zone_steps[i] = int(episode.in_zone_steps[i]) + 1
            else:
                episode.in_zone_steps[i] = max(0, int(episode.in_zone_steps[i]) - 2)

            # -- component logging --
            comp["r_total"][i] = r_total
            comp["r_dist"][i] = r_dist
            comp["r_hold"][i] = r_hold
            comp["r_velocity"][i] = r_vel
            comp["p_distance"][i] = p_distance
            comp["p_collision"][i] = p_coll
            comp["p_ship_collision"][i] = p_ship
            comp["p_tug_collision"][i] = p_tug
            comp["dist_to_slot"][i] = d
            comp["heading_err_deg"][i] = math.degrees(abs(dpsi))
            comp["speed_err"][i] = speed_err
            comp["hull_dist"][i] = d_hull
            comp["in_zone"][i] = in_zone_now
            comp["corridor_gate"][i] = c_gate
            comp["ship_soft_scale"][i] = soft

        # -- team distance cost --
        if w_team > 0.0:
            beta = max(float(getattr(cfg, "reward_team_softmin_beta", 4.0)), 1e-6)
            logits = beta * np.asarray(comp["p_distance"], dtype=np.float64)
            weights = np.exp(logits - float(np.max(logits)))
            team_cost = float(np.dot(weights, comp["p_distance"]) / max(float(weights.sum()), 1e-12))
            team_reward = np.float32(-w_team * team_cost)
            rewards += team_reward
            comp["r_total"] += team_reward
            comp["r_team"][:] = team_reward

        return rewards, {"reward_components": comp}
