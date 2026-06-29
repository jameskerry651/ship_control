"""拖轮编队环境的观测与全局状态构建。

构建每个智能体的93维观测向量，以及用于中心化评论器的90维标准全局状态。

子模块不持有 ``FormationEnv`` 引用，改为通过 ``SimState`` 和显式参数接收数据。
维度常量统一来自 ``env/obs_spec.py``，坐标变换移入 ``env/state.py``。
"""

from __future__ import annotations

import math

import numpy as np

from env.obs_spec import (
    ACTION_DIM,
    _EGO_MOTION_OBS_DIM,
    _ACTION_HISTORY_OBS_DIM,
    _SHIP_REL_OBS_DIM,
    _SHIP_PREVIEW_POINT_DIM,
    _SLOT_TARGET_OBS_DIM,
    _ROUTE_TARGET_OBS_DIM,
    _HULL_CLEARANCE_OBS_DIM,
    _NEIGHBOR_COUNT,
    _NEIGHBOR_OBS_DIM,
    _NEIGHBOR_TCPA_SCALE_S,
    _NEIGHBOR_REL_SPEED_EPS,
)
from env.state import SimState, TugSnapshot, _closest_hull_point_body, local_to_world, world_to_local
from physics.large_ship_model import _wrap_pi


class Observer:
    """Builds per-agent observations and the canonical global state.

    无状态设计：不持有 ``FormationEnv`` 引用，所有数据通过参数传入。
    """

    def __init__(self) -> None:
        pass

    # ----------------------------------------------------------- motion frame

    @staticmethod
    def _motion_frame(
        tug_u: float, tug_v: float, tug_r: float,
        prev_u: float | None, prev_v: float | None, prev_r: float | None,
    ) -> np.ndarray:
        """单帧运动特征（6 维），用于填充历史缓冲区。"""
        if prev_u is None:
            du = dv = dr = 0.0
        else:
            du = float(tug_u - prev_u)
            dv = float(tug_v - prev_v)
            dr = float(tug_r - prev_r)
        return np.asarray(
            [
                float(tug_u) / 5.0,
                float(tug_v) / 5.0,
                float(tug_r) / 0.5,
                du / 5.0,
                dv / 5.0,
                dr / 0.5,
            ],
            dtype=np.float32,
        )

    # --------------------------------------------------- history-buffer helpers

    @staticmethod
    def fill_obs_history(
        motion_history: np.ndarray,
        action_history: np.ndarray,
        tugs: list,
        actions: np.ndarray,
    ) -> None:
        """初始化历史缓冲区（reset 时调用）。"""
        actions = np.clip(actions, -1.0, 1.0).astype(np.float32, copy=False)
        for i, tug in enumerate(tugs):
            motion_history[i, :, :] = Observer._motion_frame(
                tug.nu.x, tug.nu.y, tug.nu.z, None, None, None,
            )
            action_history[i, :, :] = actions[i]

    @staticmethod
    def append_obs_history(
        motion_history: np.ndarray,
        action_history: np.ndarray,
        tugs: list,
        actions: np.ndarray,
        prev_nu: np.ndarray,
    ) -> None:
        """追加一帧到历史缓冲区（step 后调用）。"""
        motion_history[:, 1:, :] = motion_history[:, :-1, :].copy()
        action_history[:, 1:, :] = action_history[:, :-1, :].copy()
        for i, tug in enumerate(tugs):
            motion_history[i, 0, :] = Observer._motion_frame(
                tug.nu.x, tug.nu.y, tug.nu.z,
                float(prev_nu[i, 0]), float(prev_nu[i, 1]), float(prev_nu[i, 2]),
            )
        action_history[:, 0, :] = np.clip(actions, -1.0, 1.0).astype(
            np.float32, copy=False
        )

    # --------------------------------------------------------- observation

    @staticmethod
    def build_obs(
        state: SimState,
        motion_history: np.ndarray,
        action_history: np.ndarray,
        obs_dim: int,
    ) -> np.ndarray:
        """构建每个 agent 的观测向量。

        参数：
            state: 当前仿真状态快照
            motion_history: (n_tugs, hist_len, _EGO_MOTION_OBS_DIM)
            action_history: (n_tugs, hist_len, ACTION_DIM)
            obs_dim: 单个 agent 的总观测维度
        返回：
            obs: (n_tugs, obs_dim) float32
        """
        n_tugs = state.n_tugs
        obs = np.zeros((n_tugs, obs_dim), dtype=np.float32)
        ship = state.ship
        preview_times = tuple(getattr(state.cfg, "obs_ship_preview_times_s", (5.0, 10.0, 15.0)))

        # 预先计算所有拖轮的世界系速度
        tug_world_vx = np.zeros(n_tugs, dtype=np.float32)
        tug_world_vy = np.zeros(n_tugs, dtype=np.float32)
        for i, tug in enumerate(state.tugs):
            tug_world_vx[i], tug_world_vy[i] = tug.world_velocity()

        for i, tug in enumerate(state.tugs):
            idx = 0
            motion_len = motion_history.shape[1] * _EGO_MOTION_OBS_DIM
            obs[i, idx:idx + motion_len] = motion_history[i].reshape(-1)
            idx += motion_len

            action_len = action_history.shape[1] * ACTION_DIM
            obs[i, idx:idx + action_len] = action_history[i].reshape(-1)
            idx += action_len

            # -- 大船相对状态 (5 维) --
            ship_dx_w = ship.x - tug.x
            ship_dy_w = ship.y - tug.y
            ship_dx_local, ship_dy_local = world_to_local(ship_dx_w, ship_dy_w, tug.psi)
            dpsi_ship = _wrap_pi(ship.psi - tug.psi)
            obs[i, idx + 0] = ship_dx_local / 100.0
            obs[i, idx + 1] = ship_dy_local / 100.0
            obs[i, idx + 2] = ship.u / 3.0
            obs[i, idx + 3] = math.sin(dpsi_ship)
            obs[i, idx + 4] = math.cos(dpsi_ship)
            idx += _SHIP_REL_OBS_DIM

            # -- 大船预瞄点 (3×2=6 维) --
            ship_vx_w, ship_vy_w = ship.world_velocity()
            for tau in preview_times:
                t = float(tau)
                if abs(ship.r) < 1e-6:
                    ship_x_f = ship.x + ship_vx_w * t
                    ship_y_f = ship.y + ship_vy_w * t
                else:
                    dx_body = (ship.u * math.sin(ship.r * t) + ship.v * (math.cos(ship.r * t) - 1.0)) / ship.r
                    dy_body = (ship.u * (1.0 - math.cos(ship.r * t)) + ship.v * math.sin(ship.r * t)) / ship.r
                    dx_world_1, dy_world_1 = local_to_world(dx_body, dy_body, ship.psi)
                    ship_x_f = ship.x + dx_world_1
                    ship_y_f = ship.y + dy_world_1
                dx_local, dy_local = world_to_local(
                    ship_x_f - tug.x, ship_y_f - tug.y, tug.psi,
                )
                obs[i, idx + 0] = dx_local / 100.0
                obs[i, idx + 1] = dy_local / 100.0
                idx += _SHIP_PREVIEW_POINT_DIM

            # -- 目标槽位 (5 维) --
            slot = state.slot_positions_world[state.tug_to_slot[i]]
            slot_dx_w = float(slot[0]) - tug.x
            slot_dy_w = float(slot[1]) - tug.y
            slot_dx_local, slot_dy_local = world_to_local(slot_dx_w, slot_dy_w, tug.psi)
            slot_dist = math.hypot(slot_dx_local, slot_dy_local)
            dpsi_slot = _wrap_pi(float(slot[2]) - tug.psi)
            obs[i, idx + 0] = slot_dx_local / 100.0
            obs[i, idx + 1] = slot_dy_local / 100.0
            obs[i, idx + 2] = min(slot_dist / 100.0, 10.0)
            obs[i, idx + 3] = math.sin(dpsi_slot)
            obs[i, idx + 4] = math.cos(dpsi_slot)
            idx += _SLOT_TARGET_OBS_DIM

            # -- 路径目标 (4 维) --
            route_target = state.current_route_target_world(i)
            route_dx_w = float(route_target[0]) - tug.x
            route_dy_w = float(route_target[1]) - tug.y
            route_dx_local, route_dy_local = world_to_local(route_dx_w, route_dy_w, tug.psi)
            route_len = len(state.route_waypoints_body_for_tug(i))
            stage_norm = float(state.route_stage[i]) / max(route_len - 1, 1)
            obs[i, idx + 0] = route_dx_local / 100.0
            obs[i, idx + 1] = route_dy_local / 100.0
            obs[i, idx + 2] = float(np.clip(stage_norm, 0.0, 1.0))
            obs[i, idx + 3] = float(state.route_remaining_distance(i)) / 500.0
            idx += _ROUTE_TARGET_OBS_DIM

            # -- 船体间隙 (3 维) --
            tug_x_body, tug_y_body = ship.world_to_body(tug.x, tug.y)
            hull_x_body, hull_y_body = _closest_hull_point_body(
                float(tug_x_body), float(tug_y_body),
                ship.length_m, ship.beam_m,
            )
            hull_x_world, hull_y_world = ship.body_to_world(hull_x_body, hull_y_body)
            hull_dx_local, hull_dy_local = world_to_local(
                hull_x_world - tug.x, hull_y_world - tug.y, tug.psi,
            )
            obs[i, idx + 0] = hull_dx_local / 50.0
            obs[i, idx + 1] = hull_dy_local / 50.0
            obs[i, idx + 2] = float(ship.distance_from_hull(tug.x, tug.y)) / 50.0
            idx += _HULL_CLEARANCE_OBS_DIM

            # -- 邻居特征 (3×10=30 维) --
            other_idx = [j for j in range(n_tugs) if j != i]
            for j in other_idx:
                other = state.tugs[j]
                dx_w = other.x - tug.x
                dy_w = other.y - tug.y
                dx_local, dy_local = world_to_local(dx_w, dy_w, tug.psi)
                du_local, dv_local = world_to_local(
                    float(tug_world_vx[j] - tug_world_vx[i]),
                    float(tug_world_vy[j] - tug_world_vy[i]),
                    tug.psi,
                )
                dist = math.hypot(dx_local, dy_local)
                bearing = math.atan2(dy_local, dx_local)
                if dist > _NEIGHBOR_REL_SPEED_EPS:
                    range_rate = (dx_local * du_local + dy_local * dv_local) / dist
                else:
                    range_rate = 0.0
                rel_speed_sq = du_local * du_local + dv_local * dv_local
                if rel_speed_sq > _NEIGHBOR_REL_SPEED_EPS:
                    tcpa = -(dx_local * du_local + dy_local * dv_local) / rel_speed_sq
                    tcpa = max(tcpa, 0.0)
                    cpa_x = dx_local + du_local * tcpa
                    cpa_y = dy_local + dv_local * tcpa
                    dcpa = math.hypot(cpa_x, cpa_y)
                else:
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

    # ----------------------------------------------------------- global state

    @staticmethod
    def get_global_state(state: SimState, in_zone_steps: np.ndarray) -> np.ndarray:
        """构建中心化评论器的全局状态向量。

        已委托给 ``SimState.build_global_state_array()`` 以避免逻辑重复。
        """
        return state.build_global_state_array(in_zone_steps)
