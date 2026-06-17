"""多智能体拖轮编队环境。

任务：4 艘拖轮 + 1 艘移动的大船。每艘拖轮被分配到大船周围一个固定的 slot
（船首左/右、船尾左/右），需要平滑驶入 slot 并在大船前进过程中保持就位。

设计要点：
- 单环境一次接受 (n_tugs, action_dim) 的动作，返回 (n_tugs, obs_dim) 的观察。
- 4 个智能体共享同一份策略网络（参数共享），观察都用"以自身为参考系"的相对量。
- 奖励是逐 agent 计算，但碰撞/成功这种全局事件所有 agent 同步收到。
- 初始化时 slot 角色固定为 tug i → slot i。

模块拆分：
- init.py       初始位置/状态采样
- route_planner.py  A* 路径规划、waypoint、route 状态追踪
- observer.py    观察构造、全局状态构造
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from config import EnvConfig
from env.init import InitSampler
from env.observer import (
    ACTION_DIM,
    _EGO_MOTION_OBS_DIM,
    _ACTION_HISTORY_OBS_DIM,
    _SHIP_REL_OBS_DIM,
    _SHIP_PREVIEW_POINT_DIM,
    _NEIGHBOR_COUNT,
    _NEIGHBOR_OBS_DIM,
    _GLOBAL_SHIP_DIM,
    _GLOBAL_PER_TUG_DIM,
    _GLOBAL_ACCEL_PER_TUG_DIM,
    Observer,
    local_to_world,
    world_to_local,
)
from env.reward import FormationRewardComputer
from env.route_planner import RoutePlanner
from physics.large_ship_model import LargeShipModel, _wrap_pi
from physics.tugboat_dynamics_model import TugboatDynamicsModel, Vec3


def _world_to_local(dx: float, dy: float, psi_local: float) -> tuple[float, float]:
    """把世界系下的相对向量旋转到本地坐标系（朝向 psi_local 的物体的体系）。"""
    return world_to_local(dx, dy, psi_local)


def _local_to_world(dx_local: float, dy_local: float, psi_local: float) -> tuple[float, float]:
    """把局部坐标系向量旋转到世界系。"""
    return local_to_world(dx_local, dy_local, psi_local)


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
    last_action_changes: np.ndarray = field(init=False)
    motion_history: np.ndarray = field(init=False)
    action_history: np.ndarray = field(init=False)
    last_reward_components: dict = field(init=False)
    _route_waypoints_body_cache: dict[int, np.ndarray] = field(init=False)
    _mixed_ready_tugs: set[int] = field(init=False)
    prev_dist: np.ndarray = field(init=False)
    in_zone_steps: np.ndarray = field(init=False)
    route_stage: np.ndarray = field(init=False)
    tug_to_slot: np.ndarray = field(init=False)

    # 组合模块（在 __post_init__ 中创建）
    _route: RoutePlanner = field(init=False)
    _obs: Observer = field(init=False)
    _init: InitSampler = field(init=False)
    _reward: FormationRewardComputer = field(init=False)

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

        self._route = RoutePlanner(self)
        self._obs = Observer(self)
        self._init = InitSampler(self)
        self._reward = FormationRewardComputer(self)

    # ----------------------------------------------------------- properties

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
        return (
            _GLOBAL_SHIP_DIM
            + _GLOBAL_PER_TUG_DIM * self.n_tugs
            + _GLOBAL_ACCEL_PER_TUG_DIM * self.n_tugs
        )

    @property
    def action_dim(self) -> int:
        return ACTION_DIM

    # ------------------------------------------------ history-buffer helpers

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

    # -------------------------------------------------- coordinate transforms

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

    # ------------------------------------------------------- mode / ship size

    def _init_mode(self) -> str:
        return str(getattr(self.cfg, "tug_init_mode", "mixed_slot_approach"))

    def _uses_route_mode(self, mode: str | None = None) -> bool:
        init_mode = self._init_mode() if mode is None else str(mode)
        return init_mode == "mixed_slot_approach"

    def _sample_ship_size(self) -> None:
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
        self._route_waypoints_body_cache.clear()

    # -------------------------------------------------------- route delegates

    def _route_waypoints_body_for_tug(self, tug_idx: int) -> np.ndarray:
        return self._route._route_waypoints_body_for_tug(tug_idx)

    def _route_waypoints_world_for_tug(self, tug_idx: int) -> np.ndarray:
        return self._route._route_waypoints_world_for_tug(tug_idx)

    def _advance_route_stage(self, tug_idx: int) -> None:
        self._route._advance_route_stage(tug_idx)

    def _route_remaining_distance(self, tug_idx: int) -> float:
        return self._route._route_remaining_distance(tug_idx)

    def _current_route_target_world(self, tug_idx: int) -> np.ndarray:
        return self._route._current_route_target_world(tug_idx)

    # ------------------------------------------------------------- reset

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
        self._route_waypoints_body_cache.clear()

        self._sample_ship_size()
        self.ship.reset(self.rng)

        slot_world = self.ship.slot_positions_world()
        init_mode = self._init_mode()
        tug_psi = np.zeros(self.n_tugs, dtype=np.float64)
        tug_nu = np.zeros((self.n_tugs, 3), dtype=np.float64)
        init_actions = np.zeros((self.n_tugs, ACTION_DIM), dtype=np.float32)

        self._mixed_ready_tugs = set()
        if init_mode == "mixed_slot_approach":
            self.tug_to_slot = np.arange(self.n_tugs, dtype=np.int32)
            tug_xy, tug_psi, tug_nu, init_actions = self._init.sample_mixed_slot_approach_states()
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
        self._obs.fill_obs_history(init_actions)

        for i in range(self.n_tugs):
            self._route._route_waypoints_body_for_tug(i)

        for i in range(self.n_tugs):
            route_len = len(self._route._route_waypoints_body_for_tug(i))
            if not self._uses_route_mode(init_mode):
                self.route_stage[i] = max(route_len - 1, 0)
            elif init_mode == "mixed_slot_approach" and i in self._mixed_ready_tugs:
                self.route_stage[i] = max(route_len - 1, 0)
            else:
                self.route_stage[i] = 0

        for i, tug in enumerate(self.tugs):
            slot = slot_world[self.tug_to_slot[i]]
            self.prev_dist[i] = float(math.hypot(tug.eta.x - slot[0], tug.eta.y - slot[1]))

        return self._obs.build_obs()

    # -------------------------------------------------------------- step

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

        for i, tug in enumerate(self.tugs):
            port_rpm = float(actions[i, 0]) * tug.rpm_limit
            stbd_rpm = float(actions[i, 1]) * tug.rpm_limit
            port_az_deg = float(actions[i, 2]) * tug.azimuth_limit_deg
            stbd_az_deg = float(actions[i, 3]) * tug.azimuth_limit_deg
            tug.set_control_commands(port_rpm, stbd_rpm, port_az_deg, stbd_az_deg)
            tug.step(self.cfg.dt_ctrl)

        self.ship.step(self.cfg.dt_ctrl)
        self.step_count += 1

        if self._uses_route_mode():
            for i in range(self.n_tugs):
                self._route._advance_route_stage(i)

        slot_world = self.ship.slot_positions_world()
        rewards, info = self._reward.compute_rewards(actions, slot_world)
        self.last_reward_components = info.get("reward_components", {})

        dones, term_info = self._check_termination(slot_world)
        info.update(term_info)

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

        self.last_action_changes = actions - self.last_actions
        self.last_actions = actions.copy()
        self._obs.append_obs_history(actions, prev_nu)
        for i, tug in enumerate(self.tugs):
            slot = slot_world[self.tug_to_slot[i]]
            self.prev_dist[i] = float(math.hypot(tug.eta.x - slot[0], tug.eta.y - slot[1]))

        obs = self._obs.build_obs()
        return obs, rewards + terminal_reward, dones, info

    def _apply_route_speed_governor(self, actions: np.ndarray) -> np.ndarray:
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
            route_len = len(self._route._route_waypoints_body_for_tug(i))
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

    # ----------------------------------------------------- observation / state

    def _build_obs(self) -> np.ndarray:
        return self._obs.build_obs()

    def get_global_state(self) -> np.ndarray:
        return self._obs.get_global_state()

    # -------------------------------------------------------- termination

    def _check_termination(self, slot_world: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
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

        for i, tug in enumerate(self.tugs):
            d_to_hull = self.ship.distance_from_hull(tug.eta.x, tug.eta.y)
            if d_to_hull < cfg.ship_collision_dist_m:
                dones[:] = True
                info["collision"] = True
                info["collision_kind"] = "tug_vs_ship"
                info["collision_tug"] = int(i)
                info["terminated"] = True
                return dones, info

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

        hold_steps = int(round(cfg.hold_time_s / cfg.dt_ctrl))
        if all(int(self.in_zone_steps[i]) >= hold_steps for i in range(n)):
            dones[:] = True
            info["success"] = True
            info["terminated"] = True
            return dones, info

        if self.step_count >= cfg.max_episode_steps:
            dones[:] = True
            info["timeout"] = True
            info["truncated"] = True

        return dones, info

    # --------------------------------------------------------- render snapshot

    def render_snapshot(self) -> dict[str, Any]:
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
                "route_remaining": float(self._route._route_remaining_distance(i)),
                "route_target": self._route._current_route_target_world(i).copy(),
                "in_zone_steps": int(self.in_zone_steps[i]),
                "r_target": float(self.last_reward_components.get("r_target", np.zeros(self.n_tugs))[i]),
                "r_velocity": float(self.last_reward_components.get("r_velocity", np.zeros(self.n_tugs))[i]),
                "dist_to_slot": float(self.last_reward_components.get("dist_to_slot", np.zeros(self.n_tugs))[i]),
            })

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
