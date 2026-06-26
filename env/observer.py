"""拖轮编队环境的观测与全局状态构建。

构建每个智能体的93维观测向量，以及用于中心化评论器的90维标准全局状态。
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

_EGO_MOTION_OBS_DIM = 6
_ACTION_HISTORY_OBS_DIM = ACTION_DIM
_SHIP_REL_OBS_DIM = 5
_SHIP_PREVIEW_POINT_DIM = 2
# 本 agent 目标槽位（在自身坐标系下）的相对量：[dx/100, dy/100, dist, sin(dψ), cos(dψ)]
# 关键：actor 4 个 agent 参数共享，但每个 tug 的目标槽位不同。若 own_obs 不含目标槽位，
# 共享策略在相近位姿下会输出相近动作、把多船挤向同一区域，碰撞率随训练上升。
# 槽位 one-hot 之前只喂给 critic（get_global_state），actor 看不到，故必须在此补上。
_SLOT_TARGET_OBS_DIM = 5
# 当前路径目标和进度：[route_dx/100, route_dy/100, stage_norm, remaining/500]
_ROUTE_TARGET_OBS_DIM = 4
# 最近船体边界向量与距离：[closest_hull_dx/50, closest_hull_dy/50, d_hull/50]
_HULL_CLEARANCE_OBS_DIM = 3
_NEIGHBOR_COUNT = 3
# 单个邻居观测维度：从纯几何状态升级为碰撞风险特征
# [dx, dy, distance, sin(bearing), cos(bearing), du, dv, range_rate, tcpa, dcpa]
_NEIGHBOR_OBS_DIM = 10

# TCPA（最近会遇时刻）归一化尺度，单位秒；超过该窗口的会遇被视为低风险
_NEIGHBOR_TCPA_SCALE_S = 60.0
# 相对速度模长下限，低于此值认为两船相对静止，TCPA/DCPA 退化为当前距离
_NEIGHBOR_REL_SPEED_EPS = 1e-6

# --------  global-state dimension constants  --------
_GLOBAL_SHIP_DIM = 2
_GLOBAL_PER_TUG_DIM = 19
_GLOBAL_ACCEL_PER_TUG_DIM = 3

_TUG_LINEAR_ACCEL_SCALE = 1.0
_TUG_YAW_ACCEL_SCALE = 0.1
_SHIP_LINEAR_ACCEL_SCALE = 0.2


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


def _closest_hull_point_body(x_body: float, y_body: float, length_m: float, beam_m: float) -> tuple[float, float]:
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
                float(tug.nu.z) / 0.5,
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
        slot_world = env.ship.slot_positions_world()

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
            obs[i, idx + 3] = math.sin(dpsi_ship)
            obs[i, idx + 4] = math.cos(dpsi_ship)
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

            # 本 agent 目标槽位（自身坐标系下的相对量）。让共享策略能区分"我该去哪个槽位"，
            # 否则多船会挤向同一区域。slot_world 由大船当前位姿给出，与 reward 用的目标一致。
            slot = slot_world[env.tug_to_slot[i]]
            slot_dx_w = float(slot[0]) - tug.eta.x
            slot_dy_w = float(slot[1]) - tug.eta.y
            slot_dx_local, slot_dy_local = world_to_local(slot_dx_w, slot_dy_w, tug.eta.z)
            slot_dist = math.hypot(slot_dx_local, slot_dy_local)
            dpsi_slot = _wrap_pi(float(slot[2]) - tug.eta.z)
            obs[i, idx + 0] = slot_dx_local / 100.0
            obs[i, idx + 1] = slot_dy_local / 100.0
            obs[i, idx + 2] = min(slot_dist / 100.0, 10.0)
            obs[i, idx + 3] = math.sin(dpsi_slot)
            obs[i, idx + 4] = math.cos(dpsi_slot)
            idx += _SLOT_TARGET_OBS_DIM

            route_target = env._route._current_route_target_world(i)
            route_dx_w = float(route_target[0]) - tug.eta.x
            route_dy_w = float(route_target[1]) - tug.eta.y
            route_dx_local, route_dy_local = world_to_local(route_dx_w, route_dy_w, tug.eta.z)
            route_len = len(env._route._route_waypoints_body_for_tug(i))
            stage_norm = float(env.route_stage[i]) / max(route_len - 1, 1)
            obs[i, idx + 0] = route_dx_local / 100.0
            obs[i, idx + 1] = route_dy_local / 100.0
            obs[i, idx + 2] = float(np.clip(stage_norm, 0.0, 1.0))
            obs[i, idx + 3] = float(env._route._route_remaining_distance(i)) / 500.0
            idx += _ROUTE_TARGET_OBS_DIM

            tug_x_body, tug_y_body = env._ship_body_xy(tug.eta.x, tug.eta.y)
            hull_x_body, hull_y_body = _closest_hull_point_body(
                float(tug_x_body),
                float(tug_y_body),
                float(env.ship.length_m),
                float(env.ship.beam_m),
            )
            hull_x_world, hull_y_world = env._ship_body_to_world_xy(hull_x_body, hull_y_body)
            hull_dx_local, hull_dy_local = world_to_local(
                hull_x_world - tug.eta.x,
                hull_y_world - tug.eta.y,
                tug.eta.z,
            )
            obs[i, idx + 0] = hull_dx_local / 50.0
            obs[i, idx + 1] = hull_dy_local / 50.0
            obs[i, idx + 2] = float(env.ship.distance_from_hull(tug.eta.x, tug.eta.y)) / 50.0
            idx += _HULL_CLEARANCE_OBS_DIM

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
                # 相对位置与相对速度（拖轮 i 自身坐标系）
                dist = math.hypot(dx_local, dy_local)
                # 方位角：邻居相对本船船首的夹角，用 sin/cos 表示避免角度回绕
                bearing = math.atan2(dy_local, dx_local)
                # 距离变化率（range rate）= d(distance)/dt，沿视线方向的相对速度投影
                # 负值代表正在接近，正值代表正在远离
                if dist > _NEIGHBOR_REL_SPEED_EPS:
                    range_rate = (dx_local * du_local + dy_local * dv_local) / dist
                else:
                    range_rate = 0.0
                # TCPA / DCPA：基于当前相对位置与相对速度的匀速外推
                rel_speed_sq = du_local * du_local + dv_local * dv_local
                if rel_speed_sq > _NEIGHBOR_REL_SPEED_EPS:
                    # t* = -(r·v) / |v|^2，会遇时刻；过去的会遇（t*<0）裁剪为 0
                    tcpa = -(dx_local * du_local + dy_local * dv_local) / rel_speed_sq
                    tcpa = max(tcpa, 0.0)
                    cpa_x = dx_local + du_local * tcpa
                    cpa_y = dy_local + dv_local * tcpa
                    dcpa = math.hypot(cpa_x, cpa_y)
                else:
                    # 相对静止：无会遇趋势，DCPA 退化为当前距离，TCPA 视为窗口外
                    tcpa = _NEIGHBOR_TCPA_SCALE_S
                    dcpa = dist
                obs[i, idx + 0] = dx_local / 100.0
                obs[i, idx + 1] = dy_local / 100.0
                obs[i, idx + 2] = min(dist / 100.0, 10.0)
                obs[i, idx + 3] = math.sin(bearing)
                obs[i, idx + 4] = math.cos(bearing)
                obs[i, idx + 5] = du_local / 5.0
                obs[i, idx + 6] = dv_local / 5.0
                obs[i, idx + 7] = range_rate / 5.0
                obs[i, idx + 8] = min(tcpa / _NEIGHBOR_TCPA_SCALE_S, 10.0)
                obs[i, idx + 9] = min(dcpa / 100.0, 10.0)
                idx += _NEIGHBOR_OBS_DIM

        np.clip(obs, -10.0, 10.0, out=obs)
        return obs

    def get_global_state(self) -> np.ndarray:
        env = self._env
        cfg = env.cfg
        n = env.n_tugs
        state = np.zeros(env.global_state_dim, dtype=np.float32)

        state[0] = float(env.ship.u) / 5.0
        state[1] = float(env.ship.u_dot) / _SHIP_LINEAR_ACCEL_SCALE

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

            route_len = len(env._route._route_waypoints_body_for_tug(i))
            stage_norm = float(env.route_stage[i]) / max(route_len - 1, 1)
            state[base + 15] = float(np.clip(stage_norm, 0.0, 1.0))
            state[base + 16] = float(env._route._route_remaining_distance(i)) / 500.0

            state[base + 17] = float(env.in_zone_steps[i]) / float(hold_steps)

            d_hull = env.ship.distance_from_hull(tug.eta.x, tug.eta.y)
            state[base + 18] = float(d_hull) / 50.0

            acc = tug.get_last_nu_dot()
            ax_w = ci * acc.x - si * acc.y
            ay_w = si * acc.x + ci * acc.y
            ax_b = cs_s * ax_w + sn_s * ay_w
            ay_b = -sn_s * ax_w + cs_s * ay_w
            acc_base = acc_tail_start + i * _GLOBAL_ACCEL_PER_TUG_DIM
            state[acc_base + 0] = float(ax_b) / _TUG_LINEAR_ACCEL_SCALE
            state[acc_base + 1] = float(ay_b) / _TUG_LINEAR_ACCEL_SCALE
            state[acc_base + 2] = float(acc.z) / _TUG_YAW_ACCEL_SCALE

        np.clip(state, -10.0, 10.0, out=state)
        return state
