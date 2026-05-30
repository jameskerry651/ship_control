"""Dense reward for the tug formation environment.

The training reward intentionally has a single shape:

    R = w_target * R_target + w_velocity * R_velocity
        + w_control * R_control - w_collision * P_collision

Terminal success bonuses and hard collision penalties are still handled by
``FormationEnv.step`` so dense reward normalization does not dilute them.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from physics.large_ship_model import _wrap_pi

if TYPE_CHECKING:
    from env.formation_env import FormationEnv


class FormationRewardComputer:
    """Reward calculator for ``FormationEnv``."""

    def __init__(self, env: FormationEnv) -> None:
        self._env = env

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
        return float(frac * frac)

    def compute_rewards(
        self, actions: np.ndarray, slot_world: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Any]]:
        cfg = self._env.cfg
        n = self._env.n_tugs
        rewards = np.zeros(n, dtype=np.float32)

        comp = {
            "r_total": np.zeros(n, dtype=np.float32),
            "r_target": np.zeros(n, dtype=np.float32),
            "r_chase": np.zeros(n, dtype=np.float32),
            "r_hold": np.zeros(n, dtype=np.float32),
            "r_velocity": np.zeros(n, dtype=np.float32),
            "r_control": np.zeros(n, dtype=np.float32),
            "p_collision": np.zeros(n, dtype=np.float32),
            "p_ship_collision": np.zeros(n, dtype=np.float32),
            "p_tug_collision": np.zeros(n, dtype=np.float32),
            "dist_to_slot": np.zeros(n, dtype=np.float32),
            "heading_err_deg": np.zeros(n, dtype=np.float32),
            "speed_err": np.zeros(n, dtype=np.float32),
            "hull_dist": np.zeros(n, dtype=np.float32),
            "chase_closing_speed": np.zeros(n, dtype=np.float32),
            "hold_gate": np.zeros(n, dtype=np.float32),
            "far_gate": np.zeros(n, dtype=np.float32),
            "in_zone": np.zeros(n, dtype=np.bool_),
        }

        ship_vx_w, ship_vy_w = self._world_velocity(
            self._env.ship.psi,
            self._env.ship.u,
            self._env.ship.v,
        )
        tug_positions = [(tug.eta.x, tug.eta.y) for tug in self._env.tugs]

        w_target = float(getattr(cfg, "reward_target_w", 1.0))
        w_velocity = float(getattr(cfg, "reward_velocity_w", 0.5))
        w_control = float(getattr(cfg, "reward_control_w", 0.08))
        w_collision = float(getattr(cfg, "reward_collision_w", 2.0))

        target_progress_clip = max(
            float(getattr(cfg, "reward_target_progress_clip_m", 1.5)),
            1e-6,
        )
        chase_speed_target = max(
            float(getattr(cfg, "reward_chase_speed_target_ms", 0.8)),
            1e-6,
        )

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

        control_delta_w = float(getattr(cfg, "reward_control_delta_w", 1.0))
        control_jerk_w = float(getattr(cfg, "reward_control_jerk_w", 0.5))
        control_mag_w = float(getattr(cfg, "reward_control_mag_w", 0.05))

        ship_safe_dist = max(
            float(getattr(cfg, "reward_collision_ship_safe_m", 30.0)),
            float(cfg.ship_collision_dist_m) + 1e-6,
        )
        tug_safe_dist = max(
            float(getattr(cfg, "reward_collision_tug_safe_m", 60.0)),
            float(cfg.tug_collision_dist_m) + 1e-6,
        )

        for i, tug in enumerate(self._env.tugs):
            slot = slot_world[self._env.tug_to_slot[i]]
            d = float(math.hypot(tug.eta.x - slot[0], tug.eta.y - slot[1]))
            dpsi = _wrap_pi(float(slot[2]) - tug.eta.z)

            tug_vx_w, tug_vy_w = self._world_velocity(tug.eta.z, tug.nu.x, tug.nu.y)
            dvx = tug_vx_w - ship_vx_w
            dvy = tug_vy_w - ship_vy_w
            speed_err = math.hypot(dvx, dvy)

            progress_score = float(
                np.clip((float(self._env.prev_dist[i]) - d) / target_progress_clip, -1.0, 1.0)
            )

            if d > 1e-6:
                los_x = (float(slot[0]) - tug.eta.x) / d
                los_y = (float(slot[1]) - tug.eta.y) / d
            else:
                los_x = math.cos(float(slot[2]))
                los_y = math.sin(float(slot[2]))
            closing_speed = dvx * los_x + dvy * los_y
            closing_score = float(np.clip(closing_speed / chase_speed_target, -1.0, 1.0))

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

            pos_hold_score = max(0.0, 1.0 - d / max(cfg.pos_tol_m, 1e-6))
            heading_hold_score = max(0.0, 1.0 - abs(dpsi) / max(cfg.heading_tol_rad, 1e-6))
            speed_hold_score = max(0.0, 1.0 - speed_err / max(cfg.speed_tol_ms, 1e-6))
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
            next_in_zone_steps = int(self._env.in_zone_steps[i]) + 1 if in_zone_now else 0
            r_hold = hold_gate * hold_score
            r_target = r_chase + r_hold

            velocity_gate = hold_gate + (1.0 - hold_gate) * math.exp(-d / velocity_gate_scale)
            speed_pen = 1.0 - math.exp(-((speed_err / velocity_speed_scale) ** 2))
            yaw_err = abs(tug.nu.z - self._env.ship.r)
            yaw_pen = 1.0 - math.exp(-((yaw_err / velocity_yaw_scale) ** 2))
            r_velocity = -velocity_gate * (0.8 * speed_pen + 0.2 * yaw_pen)

            da = actions[i] - self._env.last_actions[i]
            dda = da - self._env.last_action_changes[i]
            delta_pen = float(np.mean(da * da))
            jerk_pen = float(np.mean(dda * dda))
            mag_pen = float(np.mean(actions[i] * actions[i]))
            r_control = -(
                control_delta_w * delta_pen
                + control_jerk_w * jerk_pen
                + control_mag_w * mag_pen
            )

            d_hull = self._env.ship.distance_from_hull(tug.eta.x, tug.eta.y)
            p_ship = self._barrier(d_hull, cfg.ship_collision_dist_m, ship_safe_dist)
            p_tug = 0.0
            for j, (other_x, other_y) in enumerate(tug_positions):
                if j == i:
                    continue
                d_pair = math.hypot(tug.eta.x - other_x, tug.eta.y - other_y)
                p_tug += self._barrier(d_pair, cfg.tug_collision_dist_m, tug_safe_dist)
            p_collision = p_ship + p_tug

            r_total = (
                w_target * r_target
                + w_velocity * r_velocity
                + w_control * r_control
                - w_collision * p_collision
            )
            rewards[i] = r_total

            if in_zone_now:
                self._env.in_zone_steps[i] = next_in_zone_steps
            else:
                self._env.in_zone_steps[i] = 0

            comp["r_total"][i] = r_total
            comp["r_target"][i] = r_target
            comp["r_chase"][i] = r_chase
            comp["r_hold"][i] = r_hold
            comp["r_velocity"][i] = r_velocity
            comp["r_control"][i] = r_control
            comp["p_collision"][i] = p_collision
            comp["p_ship_collision"][i] = p_ship
            comp["p_tug_collision"][i] = p_tug
            comp["dist_to_slot"][i] = d
            comp["heading_err_deg"][i] = math.degrees(abs(dpsi))
            comp["speed_err"][i] = speed_err
            comp["hull_dist"][i] = d_hull
            comp["chase_closing_speed"][i] = closing_speed
            comp["hold_gate"][i] = hold_gate
            comp["far_gate"][i] = far_gate
            comp["in_zone"][i] = in_zone_now

        return rewards, {"reward_components": comp}
