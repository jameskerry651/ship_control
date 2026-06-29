"""拖船编队环境的稠密奖励。

训练奖励有一个有意设计的统一结构：

    R = w_target * R_target + w_velocity * R_velocity
        - w_collision * P_collision

终止的成功奖励和严重碰撞惩罚仍然由
``FormationEnv.step`` 处理，因此稠密奖励的归一化并不会稀释这些奖励。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from env.state import MutableEpisodeState, SimState
from physics.large_ship_model import _wrap_pi


class FormationRewardComputer:
    """Reward calculator for ``FormationEnv``.

    不持有 ``FormationEnv`` 引用，所有数据通过 ``SimState`` 和 ``MutableEpisodeState`` 传入。
    """

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
        frac = np.clip((safe_distance - distance) / span, 0.0, 1.0)
        return float(frac)

    @staticmethod
    def _cpa_metrics(
        rx: float,
        ry: float,
        rvx: float,
        rvy: float,
        horizon_s: float,
    ) -> tuple[float, float, bool]:
        """Return TCPA/DCPA for a closing pair within the configured horizon."""
        rel_speed_sq = rvx * rvx + rvy * rvy
        closing_dot = rx * rvx + ry * rvy
        if rel_speed_sq <= 1e-9 or closing_dot >= 0.0:
            return horizon_s, math.hypot(rx, ry), False

        tcpa = -closing_dot / rel_speed_sq
        dcpa = math.hypot(rx + rvx * tcpa, ry + rvy * tcpa)
        return tcpa, dcpa, tcpa <= horizon_s

    @classmethod
    def _cpa_risk(
        cls,
        dcpa: float,
        tcpa: float,
        collision_distance: float,
        safe_distance: float,
        horizon_s: float,
    ) -> float:
        time_weight = float(np.clip(1.0 - tcpa / max(horizon_s, 1e-6), 0.0, 1.0))
        return cls._barrier(dcpa, collision_distance, safe_distance) * time_weight

    def compute_rewards(
        self,
        state: SimState,
        episode: MutableEpisodeState,
        actions: np.ndarray,
        slot_world: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        cfg = state.cfg
        n = state.n_tugs
        rewards = np.zeros(n, dtype=np.float32)

        comp = {
            "r_total": np.zeros(n, dtype=np.float32),
            "r_target": np.zeros(n, dtype=np.float32),
            "r_chase": np.zeros(n, dtype=np.float32),
            "r_hold": np.zeros(n, dtype=np.float32),
            "r_velocity": np.zeros(n, dtype=np.float32),
            "r_team": np.zeros(n, dtype=np.float32),
            "r_route": np.zeros(n, dtype=np.float32),
            "r_shape": np.zeros(n, dtype=np.float32),
            "r_precision": np.zeros(n, dtype=np.float32),
            "r_near_hold": np.zeros(n, dtype=np.float32),
            "r_hold_streak": np.zeros(n, dtype=np.float32),
            "p_collision": np.zeros(n, dtype=np.float32),
            "p_ship_collision": np.zeros(n, dtype=np.float32),
            "p_tug_collision": np.zeros(n, dtype=np.float32),
            "p_cpa_collision": np.zeros(n, dtype=np.float32),
            "p_ship_cpa": np.zeros(n, dtype=np.float32),
            "p_tug_cpa": np.zeros(n, dtype=np.float32),
            "min_tcpa": np.zeros(n, dtype=np.float32),
            "min_dcpa": np.zeros(n, dtype=np.float32),
            "dist_to_slot": np.zeros(n, dtype=np.float32),
            "heading_err_deg": np.zeros(n, dtype=np.float32),
            "speed_err": np.zeros(n, dtype=np.float32),
            "hull_dist": np.zeros(n, dtype=np.float32),
            "chase_closing_speed": np.zeros(n, dtype=np.float32),
            "hold_gate": np.zeros(n, dtype=np.float32),
            "far_gate": np.zeros(n, dtype=np.float32),
            "in_zone": np.zeros(n, dtype=np.bool_),
            "route_stage": np.zeros(n, dtype=np.float32),
            "route_remaining": np.zeros(n, dtype=np.float32),
            "route_target_dist": np.zeros(n, dtype=np.float32),
            "route_progress": np.zeros(n, dtype=np.float32),
            "route_final": np.zeros(n, dtype=np.bool_),
            "route_chase": np.zeros(n, dtype=np.bool_),
        }

        ship_vx_w, ship_vy_w = state.ship.world_velocity()
        tug_positions = [(tug.x, tug.y) for tug in state.tugs]
        tug_velocities = [tug.world_velocity() for tug in state.tugs]

        w_target = float(getattr(cfg, "reward_target_w", 1.0))
        w_velocity = float(getattr(cfg, "reward_velocity_w", 0.5))
        w_collision = float(getattr(cfg, "reward_collision_w", 2.0))
        # P2 团队同步：收集每艇在区度 z_i∈[0,1]（pos×heading×speed），循环后取 softmin。
        w_team = float(getattr(cfg, "reward_team_w", 0.0))
        team_softmin_beta = max(float(getattr(cfg, "reward_team_softmin_beta", 4.0)), 1e-6)
        z_in_zone = np.zeros(n, dtype=np.float64)
        # P3 势函数式接近 shaping 参数。
        w_shape = float(getattr(cfg, "reward_shape_w", 0.0))
        shape_gamma = float(getattr(cfg, "reward_shape_gamma", 0.99))
        shape_d_ref = max(float(getattr(cfg, "reward_shape_d_ref_m", 200.0)), 1e-6)
        shape_clip = max(float(getattr(cfg, "reward_shape_clip", 1.0)), 1e-6)
        w_precision = float(getattr(cfg, "reward_precision_w", 0.0))
        precision_scale = max(
            float(getattr(cfg, "reward_precision_scale_m", 40.0)),
            1e-6,
        )
        w_near_hold = float(getattr(cfg, "reward_near_hold_w", 0.0))
        near_hold_scale = max(
            float(getattr(cfg, "reward_near_hold_scale_m", 80.0)),
            1e-6,
        )
        _speed_tol = max(float(cfg.speed_tol_ms), 1e-6)
        _heading_tol = max(float(cfg.heading_tol_rad), 1e-6)
        # P4 hold 连续性参数。
        hold_steps = max(1, int(round(float(cfg.hold_time_s) / max(float(cfg.dt_ctrl), 1e-6))))
        streak_decay = max(int(getattr(cfg, "reward_hold_streak_decay", 1)), 1)
        w_streak = float(getattr(cfg, "reward_hold_streak_w", 0.0))

        def _potential(dist: float, spd_err: float, head_err: float) -> float:
            return -(
                0.6 * dist / shape_d_ref
                + 0.25 * spd_err / _speed_tol
                + 0.15 * head_err / _heading_tol
            )

        target_progress_clip = max(
            float(getattr(cfg, "reward_target_progress_clip_m", 1.5)),
            1e-6,
        )
        chase_speed_target = max(
            float(getattr(cfg, "reward_chase_speed_target_ms", 0.8)),
            1e-6,
        )
        w_route = float(getattr(cfg, "reward_route_w", 0.0))

        hold_start_m = max(
            float(getattr(cfg, "reward_hold_start_m", cfg.pos_tol_m * 2.0)),
            1e-6,
        )
        hold_full_m = max(
            float(getattr(cfg, "reward_hold_full_m", cfg.pos_tol_m)),
            1e-6,
        )
        if hold_start_m < hold_full_m:
            hold_start_m, hold_full_m = hold_full_m, hold_start_m

        velocity_gate_scale = max(
            float(getattr(cfg, "reward_velocity_gate_m", 120.0)),
            1e-6,
        )
        velocity_speed_scale = max(
            float(getattr(cfg, "reward_velocity_speed_scale_ms", cfg.speed_tol_ms)),
            1e-6,
        )
        velocity_yaw_scale = max(
            float(getattr(cfg, "reward_velocity_yaw_scale_rads", 0.05)),
            1e-6,
        )

        ship_safe_dist = max(
            float(getattr(cfg, "reward_collision_ship_safe_m", 30.0)),
            float(cfg.ship_collision_dist_m) + 1e-6,
        )
        tug_safe_dist = max(
            float(getattr(cfg, "reward_collision_tug_safe_m", 60.0)),
            float(cfg.tug_collision_dist_m) + 1e-6,
        )
        cpa_horizon_s = max(float(getattr(cfg, "reward_cpa_horizon_s", 60.0)), 1e-6)
        cpa_w = max(float(getattr(cfg, "reward_collision_cpa_w", 2.0)), 0.0)
        collision_cap = float(getattr(cfg, "reward_collision_cap", 1e9))
        comp["min_tcpa"].fill(cpa_horizon_s)
        comp["min_dcpa"].fill(max(ship_safe_dist, tug_safe_dist))

        for i, tug in enumerate(state.tugs):
            slot = slot_world[state.tug_to_slot[i]]
            d = float(math.hypot(tug.x - slot[0], tug.y - slot[1]))
            dpsi = _wrap_pi(float(slot[2]) - tug.psi)

            route_len = len(state.route_waypoints_body_for_tug(i))
            route_stage = int(np.clip(state.route_stage[i], 0, max(route_len - 1, 0)))
            route_final = route_stage >= route_len - 1
            route_target = state.current_route_target_world(i)
            route_dx = float(route_target[0]) - tug.x
            route_dy = float(route_target[1]) - tug.y
            route_target_dist = float(math.hypot(route_dx, route_dy))
            route_remaining = float(state.route_remaining_distance(i))

            tug_vx_w, tug_vy_w = tug_velocities[i]
            dvx = tug_vx_w - ship_vx_w
            dvy = tug_vy_w - ship_vy_w
            speed_err = math.hypot(dvx, dvy)

            progress_score = float(
                np.clip((float(episode.prev_dist[i]) - d) / target_progress_clip, -1.0, 1.0)
            )

            if d > 1e-6:
                los_x = (float(slot[0]) - tug.x) / d
                los_y = (float(slot[1]) - tug.y) / d
            else:
                los_x = math.cos(float(slot[2]))
                los_y = math.sin(float(slot[2]))
            closing_speed = dvx * los_x + dvy * los_y
            closing_score = float(np.clip(closing_speed / chase_speed_target, -1.0, 1.0))

            use_route_chase = bool(
                state.uses_route_mode
                and not route_final
                and d > float(cfg.pos_tol_m)
            )
            if use_route_chase and route_target_dist > 1e-6:
                route_progress_score = float(
                    np.clip(
                        (float(episode.prev_route_remaining[i]) - route_remaining)
                        / target_progress_clip,
                        -1.0,
                        1.0,
                    )
                )
                route_los_x = route_dx / route_target_dist
                route_los_y = route_dy / route_target_dist
                route_closing_speed = dvx * route_los_x + dvy * route_los_y
                route_closing_score = float(
                    np.clip(route_closing_speed / chase_speed_target, -1.0, 1.0)
                )
            else:
                route_progress_score = 0.0
                route_closing_score = 0.0

            if d <= hold_full_m:
                hold_gate = 1.0
            elif d >= hold_start_m:
                hold_gate = 0.0
            else:
                blend = (hold_start_m - d) / max(hold_start_m - hold_full_m, 1e-6)
                hold_gate = float(blend * blend * (3.0 - 2.0 * blend))
            far_gate = 1.0 - hold_gate

            r_chase = far_gate * (
                0.6 * progress_score
                + 0.4 * closing_score
            )
            r_route = far_gate * w_route * (
                0.7 * route_progress_score
                + 0.3 * route_closing_score
            )

            pos_hold_score = max(0.0, 1.0 - d / max(cfg.pos_tol_m, 1e-6))
            heading_hold_score = max(0.0, 1.0 - abs(dpsi) / max(cfg.heading_tol_rad, 1e-6))
            speed_hold_score = max(0.0, 1.0 - speed_err / max(cfg.speed_tol_ms, 1e-6))
            # z_i：三项同时满足才接近 1（与成功的 AND 条件一致），用于团队 softmin。
            z_in_zone[i] = pos_hold_score * heading_hold_score * speed_hold_score
            hold_score = pos_hold_score * (
                0.5
                + 0.25 * heading_hold_score
                + 0.25 * speed_hold_score
            )

            in_zone_now = (
                d < cfg.pos_tol_m
                and abs(dpsi) < cfg.heading_tol_rad
                and speed_err < cfg.speed_tol_ms
            )
            if in_zone_now:
                next_in_zone_steps = int(episode.in_zone_steps[i]) + 1
            else:
                next_in_zone_steps = max(0, int(episode.in_zone_steps[i]) - streak_decay)
            r_hold = hold_gate * hold_score
            # P4 streak 稠密奖励：仅在区帧按连续步数占比给奖，激励逼近并维持 2s hold。
            if w_streak > 0.0 and in_zone_now:
                r_hold_streak = w_streak * min(next_in_zone_steps / hold_steps, 1.0)
            else:
                r_hold_streak = 0.0
            r_target = r_chase + r_hold

            velocity_gate = hold_gate + (1.0 - hold_gate) * math.exp(-d / velocity_gate_scale)
            speed_pen = 1.0 - math.exp(-((speed_err / velocity_speed_scale) ** 2))
            yaw_err = abs(tug.r - state.ship.r)
            yaw_pen = 1.0 - math.exp(-((yaw_err / velocity_yaw_scale) ** 2))
            r_velocity = -velocity_gate * (0.8 * speed_pen + 0.2 * yaw_pen)
            if w_precision > 0.0:
                precision_pos = 1.0 - math.tanh(d / precision_scale)
                precision_match = 0.5 + 0.25 * heading_hold_score + 0.25 * speed_hold_score
                r_precision = w_precision * precision_pos * precision_match
            else:
                r_precision = 0.0
            if w_near_hold > 0.0:
                near_pos_score = math.exp(-((d / near_hold_scale) ** 2))
                near_match = 0.4 + 0.3 * heading_hold_score + 0.3 * speed_hold_score
                r_near_hold = w_near_hold * near_pos_score * near_match
            else:
                r_near_hold = 0.0

            d_hull = state.ship.distance_from_hull(tug.x, tug.y)
            p_ship_proximity = self._barrier(
                d_hull, cfg.ship_collision_dist_m, ship_safe_dist
            )
            ship_tcpa, _, ship_cpa_active = self._cpa_metrics(
                tug.x - state.ship.x,
                tug.y - state.ship.y,
                tug_vx_w - ship_vx_w,
                tug_vy_w - ship_vy_w,
                cpa_horizon_s,
            )
            p_ship_cpa = 0.0
            if ship_cpa_active:
                future_tug_x = tug.x + tug_vx_w * ship_tcpa
                future_tug_y = tug.y + tug_vy_w * ship_tcpa
                future_ship_x = state.ship.x + ship_vx_w * ship_tcpa
                future_ship_y = state.ship.y + ship_vy_w * ship_tcpa
                future_ship_psi = state.ship.psi + state.ship.r * ship_tcpa
                ship_dcpa_hull = state.ship.distance_from_hull_pose(
                    future_tug_x,
                    future_tug_y,
                    future_ship_x,
                    future_ship_y,
                    future_ship_psi,
                )
                p_ship_cpa = self._cpa_risk(
                    ship_dcpa_hull,
                    ship_tcpa,
                    cfg.ship_collision_dist_m,
                    ship_safe_dist,
                    cpa_horizon_s,
                )
                comp["min_tcpa"][i] = min(comp["min_tcpa"][i], ship_tcpa)
                comp["min_dcpa"][i] = min(comp["min_dcpa"][i], ship_dcpa_hull)
            p_ship = p_ship_proximity + cpa_w * p_ship_cpa
            p_tug = 0.0
            p_tug_cpa = 0.0
            for j, (other_x, other_y) in enumerate(tug_positions):
                if j == i:
                    continue
                d_pair = math.hypot(tug.x - other_x, tug.y - other_y)
                p_tug += self._barrier(d_pair, cfg.tug_collision_dist_m, tug_safe_dist)
                other_vx, other_vy = tug_velocities[j]
                tug_tcpa, tug_dcpa, tug_cpa_active = self._cpa_metrics(
                    other_x - tug.x,
                    other_y - tug.y,
                    other_vx - tug_vx_w,
                    other_vy - tug_vy_w,
                    cpa_horizon_s,
                )
                if tug_cpa_active:
                    risk = self._cpa_risk(
                        tug_dcpa,
                        tug_tcpa,
                        cfg.tug_collision_dist_m,
                        tug_safe_dist,
                        cpa_horizon_s,
                    )
                    p_tug_cpa += risk
                    comp["min_tcpa"][i] = min(comp["min_tcpa"][i], tug_tcpa)
                    comp["min_dcpa"][i] = min(comp["min_dcpa"][i], tug_dcpa)
            p_tug += cpa_w * p_tug_cpa
            p_collision = min(p_ship + p_tug, collision_cap)
            p_cpa = cpa_w * (p_ship_cpa + p_tug_cpa)

            if w_shape > 0.0:
                phi_cur = _potential(d, speed_err, abs(dpsi))
                phi_prev = _potential(
                    float(episode.prev_dist[i]),
                    float(episode.prev_speed_err[i]),
                    float(episode.prev_heading_err[i]),
                )
                shaping = float(
                    np.clip(shape_gamma * phi_cur - phi_prev, -shape_clip, shape_clip)
                )
                r_shape = w_shape * shaping
            else:
                r_shape = 0.0

            r_total = (
                w_target * r_target
                + w_velocity * r_velocity
                - w_collision * p_collision
                + r_route
                + r_shape
                + r_precision
                + r_near_hold
                + r_hold_streak
            )
            rewards[i] = r_total

            # next_in_zone_steps 已折入 streak 衰减（在区+1，违例-decay 不清零）。
            episode.in_zone_steps[i] = next_in_zone_steps

            comp["r_total"][i] = r_total
            comp["r_target"][i] = r_target
            comp["r_chase"][i] = r_chase
            comp["r_hold"][i] = r_hold
            comp["r_velocity"][i] = r_velocity
            comp["r_route"][i] = r_route
            comp["r_shape"][i] = r_shape
            comp["r_precision"][i] = r_precision
            comp["r_near_hold"][i] = r_near_hold
            comp["r_hold_streak"][i] = r_hold_streak
            comp["p_collision"][i] = p_collision
            comp["p_ship_collision"][i] = p_ship
            comp["p_tug_collision"][i] = p_tug
            comp["p_cpa_collision"][i] = p_cpa
            comp["p_ship_cpa"][i] = p_ship_cpa
            comp["p_tug_cpa"][i] = p_tug_cpa
            comp["dist_to_slot"][i] = d
            comp["heading_err_deg"][i] = math.degrees(abs(dpsi))
            comp["speed_err"][i] = speed_err
            comp["hull_dist"][i] = d_hull
            comp["chase_closing_speed"][i] = closing_speed
            comp["hold_gate"][i] = hold_gate
            comp["far_gate"][i] = far_gate
            comp["in_zone"][i] = in_zone_now
            comp["route_stage"][i] = float(route_stage)
            comp["route_remaining"][i] = route_remaining
            comp["route_target_dist"][i] = route_target_dist
            comp["route_progress"][i] = float(episode.prev_route_remaining[i]) - route_remaining
            comp["route_final"][i] = route_final
            comp["route_chase"][i] = use_route_chase

        if w_team > 0.0:
            # softmin_β(z) = -1/β·log(mean(exp(-β·z)))，聚焦最弱一艇；z∈[0,1] 无溢出风险。
            team_softmin = float(
                -np.log(np.mean(np.exp(-team_softmin_beta * z_in_zone))) / team_softmin_beta
            )
            team_scalar = w_team * team_softmin
            rewards += np.float32(team_scalar)
            comp["r_total"] += np.float32(team_scalar)
            comp["r_team"][:] = np.float32(team_scalar)

        return rewards, {"reward_components": comp}
