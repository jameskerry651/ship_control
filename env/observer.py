"""拖轮编队环境的观测与全局状态构建。

构建每个智能体的71维观测向量，以及用于中心化评论器的121维标准全局状态。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from physics.large_ship_model import _wrap_pi

if TYPE_CHECKING:
    from env.formation_env import FormationEnv

# --------  observation dimension constants  --------
ACTION_DIM: int = 4

_EGO_MOTION_OBS_DIM = 7
_ACTION_HISTORY_OBS_DIM = ACTION_DIM
_SHIP_REL_OBS_DIM = 6
_SHIP_PREVIEW_POINT_DIM = 2
_NEIGHBOR_COUNT = 3
_NEIGHBOR_OBS_DIM = 5

# --------  global-state dimension constants  --------
_GLOBAL_SHIP_DIM = 5
_GLOBAL_PER_TUG_DIM = 23
_GLOBAL_ACCEL_PER_TUG_DIM = 6

_TUG_LINEAR_ACCEL_SCALE = 1.0
_TUG_YAW_ACCEL_SCALE = 0.1
_SHIP_LINEAR_ACCEL_SCALE = 0.2
_SHIP_YAW_ACCEL_SCALE = 0.01


def world_to_local(dx: float, dy: float, psi_local: float) -> tuple[float, float]:
    c = math.cos(psi_local)
    s = math.sin(psi_local)
    x_local = c * dx + s * dy
    y_local = -s * dx + c * dy
    return x_local, y_local


def local_to_world(dx_local: float, dy_local: float, psi_local: float) -> tuple[float, float]:
    c = math.cos(psi_local)
    s = math.sin(psi_local)
    x_world = c * dx_local - s * dy_local
    y_world = s * dx_local + c * dy_local
    return x_world, y_world


class Observer:
    """Builds per-agent observations and the canonical global state."""

    def __init__(self, env: FormationEnv) -> None:
        self._env = env

    def _motion_frame(self, tug, prev_nu) -> np.ndarray:
        """tug is TugboatDynamicsModel; prev_nu is Vec3 | None."""
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

    def fill_obs_history(self, actions: np.ndarray) -> None:
        env = self._env
        env._ensure_history_buffers()
        actions = np.clip(actions, -1.0, 1.0).astype(np.float32, copy=False)
        for i, tug in enumerate(env.tugs):
            env.motion_history[i, :, :] = self._motion_frame(tug, prev_nu=None)
            env.action_history[i, :, :] = actions[i]

    def append_obs_history(self, actions: np.ndarray, prev_nu: np.ndarray) -> None:
        env = self._env
        env._ensure_history_buffers()
        env.motion_history[:, 1:, :] = env.motion_history[:, :-1, :].copy()
        env.action_history[:, 1:, :] = env.action_history[:, :-1, :].copy()
        from physics.tugboat_dynamics_model import Vec3
        for i, tug in enumerate(env.tugs):
            prev = Vec3(float(prev_nu[i, 0]), float(prev_nu[i, 1]), float(prev_nu[i, 2]))
            env.motion_history[i, 0, :] = self._motion_frame(tug, prev_nu=prev)
        env.action_history[:, 0, :] = np.clip(actions, -1.0, 1.0).astype(
            np.float32, copy=False
        )

    def build_obs(self) -> np.ndarray:
        env = self._env
        env._ensure_history_buffers()
        obs = np.zeros((env.n_tugs, env.obs_dim), dtype=np.float32)
        ship_u, ship_v, ship_r = env.ship.u, env.ship.v, env.ship.r
        preview_times = tuple(getattr(env.cfg, "obs_ship_preview_times_s", (5.0, 10.0, 15.0)))

        tug_world_vx = np.zeros(env.n_tugs, dtype=np.float32)
        tug_world_vy = np.zeros(env.n_tugs, dtype=np.float32)
        for i, tug in enumerate(env.tugs):
            ci = math.cos(tug.eta.z)
            si = math.sin(tug.eta.z)
            tug_world_vx[i] = ci * tug.nu.x - si * tug.nu.y
            tug_world_vy[i] = si * tug.nu.x + ci * tug.nu.y

        for i, tug in enumerate(env.tugs):
            idx = 0
            motion_len = env.motion_history.shape[1] * _EGO_MOTION_OBS_DIM
            obs[i, idx:idx + motion_len] = env.motion_history[i].reshape(-1)
            idx += motion_len

            action_len = env.action_history.shape[1] * ACTION_DIM
            obs[i, idx:idx + action_len] = env.action_history[i].reshape(-1)
            idx += action_len

            ship_dx_w = env.ship.x - tug.eta.x
            ship_dy_w = env.ship.y - tug.eta.y
            ship_dx_local, ship_dy_local = world_to_local(ship_dx_w, ship_dy_w, tug.eta.z)
            dpsi_ship = _wrap_pi(env.ship.psi - tug.eta.z)
            obs[i, idx + 0] = ship_dx_local / 100.0
            obs[i, idx + 1] = ship_dy_local / 100.0
            obs[i, idx + 2] = ship_u / 3.0
            obs[i, idx + 3] = ship_v / 3.0
            obs[i, idx + 4] = math.sin(dpsi_ship)
            obs[i, idx + 5] = math.cos(dpsi_ship)
            idx += _SHIP_REL_OBS_DIM

            cs = math.cos(env.ship.psi)
            sn = math.sin(env.ship.psi)
            ship_vx_w = cs * ship_u - sn * ship_v
            ship_vy_w = sn * ship_u + cs * ship_v
            for tau in preview_times:
                t = float(tau)
                if abs(ship_r) < 1e-6:
                    ship_x_f = env.ship.x + ship_vx_w * t
                    ship_y_f = env.ship.y + ship_vy_w * t
                else:
                    dx_body = (ship_u * math.sin(ship_r * t) + ship_v * (math.cos(ship_r * t) - 1.0)) / ship_r
                    dy_body = (ship_u * (1.0 - math.cos(ship_r * t)) + ship_v * math.sin(ship_r * t)) / ship_r
                    dx_world_1, dy_world_1 = local_to_world(dx_body, dy_body, env.ship.psi)
                    ship_x_f = env.ship.x + dx_world_1
                    ship_y_f = env.ship.y + dy_world_1
                dx_local, dy_local = world_to_local(
                    ship_x_f - tug.eta.x,
                    ship_y_f - tug.eta.y,
                    tug.eta.z,
                )
                obs[i, idx + 0] = dx_local / 100.0
                obs[i, idx + 1] = dy_local / 100.0
                idx += _SHIP_PREVIEW_POINT_DIM

            other_idx = [j for j in range(env.n_tugs) if j != i]
            for j in other_idx:
                other = env.tugs[j]
                dx_w = other.eta.x - tug.eta.x
                dy_w = other.eta.y - tug.eta.y
                dx_local, dy_local = world_to_local(dx_w, dy_w, tug.eta.z)
                du_local, dv_local = world_to_local(
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

        np.clip(obs, -10.0, 10.0, out=obs)
        return obs

    def get_global_state(self) -> np.ndarray:
        env = self._env
        cfg = env.cfg
        n = env.n_tugs
        state = np.zeros(env.global_state_dim, dtype=np.float32)

        state[0] = float(env.ship.u) / 5.0
        state[1] = float(env.ship.v) / 5.0
        state[2] = float(env.ship.r) / 0.05
        base_length = max(float(getattr(cfg, "ship_length_m", 200.0)), 1e-6)
        base_beam = max(float(getattr(cfg, "ship_beam_m", 30.0)), 1e-6)
        state[3] = float(env.ship.length_m) / base_length - 1.0
        state[4] = float(env.ship.beam_m) / base_beam - 1.0

        cs_s = math.cos(env.ship.psi)
        sn_s = math.sin(env.ship.psi)
        acc_tail_start = _GLOBAL_SHIP_DIM + _GLOBAL_PER_TUG_DIM * n

        hold_steps = max(1, int(round(cfg.hold_time_s / cfg.dt_ctrl)))

        for i, tug in enumerate(env.tugs):
            base = _GLOBAL_SHIP_DIM + i * _GLOBAL_PER_TUG_DIM

            x_b, y_b = env._ship_body_xy(tug.eta.x, tug.eta.y)
            state[base + 0] = float(x_b) / 100.0
            state[base + 1] = float(y_b) / 100.0

            ci = math.cos(tug.eta.z)
            si = math.sin(tug.eta.z)
            vx_w = ci * tug.nu.x - si * tug.nu.y
            vy_w = si * tug.nu.x + ci * tug.nu.y
            u_b = cs_s * vx_w + sn_s * vy_w
            v_b = -sn_s * vx_w + cs_s * vy_w
            state[base + 2] = float(u_b) / 5.0
            state[base + 3] = float(v_b) / 5.0

            dpsi_ship = _wrap_pi(tug.eta.z - env.ship.psi)
            state[base + 4] = math.sin(dpsi_ship)
            state[base + 5] = math.cos(dpsi_ship)
            state[base + 6] = float(tug.nu.z) / 0.5

            ctrl = tug.get_control_snapshot()
            state[base + 7] = ctrl["port_rpm_actual"] / tug.rpm_limit
            state[base + 8] = ctrl["starboard_rpm_actual"] / tug.rpm_limit
            state[base + 9] = ctrl["port_azimuth_actual_deg"] / tug.azimuth_limit_deg
            state[base + 10] = ctrl["starboard_azimuth_actual_deg"] / tug.azimuth_limit_deg

            state[base + 11:base + 15] = env.last_actions[i]

            slot_idx = int(env.tug_to_slot[i])
            state[base + 15 + slot_idx] = 1.0

            route_len = len(env._route._route_waypoints_body_for_tug(i))
            stage_norm = float(env.route_stage[i]) / max(route_len - 1, 1)
            state[base + 19] = float(np.clip(stage_norm, 0.0, 1.0))
            state[base + 20] = float(env._route._route_remaining_distance(i)) / 500.0

            state[base + 21] = float(env.in_zone_steps[i]) / float(hold_steps)

            d_hull = env.ship.distance_from_hull(tug.eta.x, tug.eta.y)
            state[base + 22] = float(d_hull) / 50.0

            acc = tug.get_last_nu_dot()
            ax_w = ci * acc.x - si * acc.y
            ay_w = si * acc.x + ci * acc.y
            ax_b = cs_s * ax_w + sn_s * ay_w
            ay_b = -sn_s * ax_w + cs_s * ay_w
            acc_base = acc_tail_start + i * _GLOBAL_ACCEL_PER_TUG_DIM
            state[acc_base + 0] = float(ax_b) / _TUG_LINEAR_ACCEL_SCALE
            state[acc_base + 1] = float(ay_b) / _TUG_LINEAR_ACCEL_SCALE
            state[acc_base + 2] = float(acc.z) / _TUG_YAW_ACCEL_SCALE
            state[acc_base + 3] = float(env.ship.u_dot) / _SHIP_LINEAR_ACCEL_SCALE
            state[acc_base + 4] = float(env.ship.v_dot) / _SHIP_LINEAR_ACCEL_SCALE
            state[acc_base + 5] = float(env.ship.r_dot) / _SHIP_YAW_ACCEL_SCALE

        np.clip(state, -10.0, 10.0, out=state)
        return state
