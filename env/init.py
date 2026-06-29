"""拖轮编队环境的初始化逻辑。
处理多种场景下的拖轮初始布放（mixed_slot_approach）：
部分拖轮已在泊位，其余从尾流航道、侧向航道、船尾入口、
外侧泊位或对侧等位置出发。

子模块不持有 ``FormationEnv`` 引用，状态通过参数传入。
"""

from __future__ import annotations

import math

import numpy as np

from config import EnvConfig
from env.obs_spec import ACTION_DIM
from env.state import (
    MutableEpisodeState,
    ShipSnapshot,
    local_to_world,
    world_to_local,
)
from physics.large_ship_model import _wrap_pi


class InitSampler:
    """Owns all tugboat initial-state sampling logic.

    无状态设计：不持有 ``FormationEnv`` 引用，rng / cfg / ship 通过参数传入。
    """

    def __init__(self) -> None:
        pass

    # -- helpers that need route-planner info (imported lazily to avoid circ) --

    @staticmethod
    def _slot_is_bow(slot_idx: int) -> bool:
        return int(slot_idx) in (0, 1)

    @staticmethod
    def _slot_side_sign(slot_idx: int) -> float:
        return -1.0 if int(slot_idx) in (0, 2) else 1.0

    @staticmethod
    def _slot_rear_start_dist_range(cfg: EnvConfig, slot_idx: int) -> tuple[float, float]:
        if InitSampler._slot_is_bow(slot_idx):
            lo = float(getattr(cfg, "tug_init_rear_bow_slot_dist_min_m", 150.0))
            hi = float(getattr(cfg, "tug_init_rear_bow_slot_dist_max_m", 230.0))
        else:
            lo = float(getattr(cfg, "tug_init_rear_stern_slot_dist_min_m", 60.0))
            hi = float(getattr(cfg, "tug_init_rear_stern_slot_dist_max_m", 100.0))
        return lo, max(lo, hi)

    @staticmethod
    def _slot_lane_lat_abs(cfg: EnvConfig, slot_idx: int) -> float:
        if InitSampler._slot_is_bow(slot_idx):
            return float(getattr(cfg, "route_bow_lane_lat_m", 90.0))
        return float(getattr(cfg, "route_stern_lane_lat_m", 55.0))

    # ----------------------------------------------------------- motion helper

    @staticmethod
    def _sample_ship_tracking_motion(
        rng: np.random.Generator,
        ship_u: float,
        ship_v: float,
        ship_psi: float,
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
        speed_offset = float(rng.uniform(speed_offset_min, speed_offset_max))
        sway = float(rng.uniform(-sway_noise_ms, sway_noise_ms))
        ship_body_vx = ship_u + speed_offset
        ship_body_vy = ship_v + sway

        if position_body is not None and target_body is not None:
            dx = float(target_body[0] - position_body[0])
            dy = float(target_body[1] - position_body[1])
            dist = math.hypot(dx, dy)
            if dist > 1e-6:
                lo = speed_offset_min if approach_speed_min is None else float(approach_speed_min)
                hi = speed_offset_max if approach_speed_max is None else float(approach_speed_max)
                lo, hi = sorted((max(0.0, lo), max(0.0, hi)))
                approach_speed = float(rng.uniform(lo, hi))
                ship_body_vx = ship_u + approach_speed * dx / dist
                ship_body_vy = ship_v + approach_speed * dy / dist + sway

        vx_w, vy_w = local_to_world(ship_body_vx, ship_body_vy, ship_psi)
        if math.hypot(vx_w, vy_w) > 1e-4:
            heading_base = math.atan2(vy_w, vx_w)
        else:
            heading_base = ship_psi
        psi = _wrap_pi(
            heading_base + float(rng.uniform(-heading_noise_rad, heading_noise_rad))
        )
        u_tug, v_tug = world_to_local(vx_w, vy_w, psi)
        r_tug = float(rng.uniform(-yaw_rate_noise_rads, yaw_rate_noise_rads))

        action = np.zeros(ACTION_DIM, dtype=np.float32)
        forward = float(
            np.clip(
                forward_action + rng.uniform(-forward_action_jitter, forward_action_jitter),
                -1.0,
                1.0,
            )
        )
        action[:] = (forward, forward, 0.0, 0.0)
        return psi, np.asarray((u_tug, v_tug, r_tug), dtype=np.float64), action

    # -------------------------------------------------------------- placement

    @staticmethod
    def _init_position_is_safe(
        ship: ShipSnapshot,
        x_w: float,
        y_w: float,
        placed: list[tuple[float, float]],
        min_pair_dist: float,
        min_hull_dist: float,
    ) -> bool:
        if ship.distance_from_hull(x_w, y_w) < min_hull_dist:
            return False
        for px, py in placed:
            if math.hypot(x_w - px, y_w - py) < min_pair_dist:
                return False
        return True

    @staticmethod
    def _mixed_init_safety_margins(cfg: EnvConfig) -> tuple[float, float]:
        min_pair_dist = max(
            2.0 * cfg.tug_collision_dist_m,
            float(getattr(cfg, "tug_init_mixed_pair_min_dist_m", 120.0)),
        )
        min_hull_dist = max(
            2.0 * cfg.ship_collision_dist_m,
            float(getattr(cfg, "ship_safety_dist_m", cfg.ship_collision_dist_m * 3.0)),
        )
        return min_pair_dist, min_hull_dist

    # --------------------------------------------------------- ready-slot init

    @staticmethod
    def _sample_ready_slot_state(
        rng: np.random.Generator,
        ship: ShipSnapshot,
        cfg: EnvConfig,
        slot_idx: int,
        placed: list[tuple[float, float]] | None = None,
    ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        slot_body = ship.slot_positions_body()[int(slot_idx), :2]
        side = InitSampler._slot_side_sign(slot_idx)
        outward = float(getattr(cfg, "tug_init_ready_outward_offset_m", 0.0))
        jitter = float(getattr(cfg, "tug_init_ready_pos_jitter_m", 0.0))
        placed_safe = [] if placed is None else placed
        min_pair_dist, min_hull_dist = InitSampler._mixed_init_safety_margins(cfg)

        chosen_xy: tuple[float, float] | None = None
        for _ in range(120):
            x_b = float(slot_body[0] + rng.uniform(-jitter, jitter))
            y_b = float(slot_body[1] + side * outward + rng.uniform(-jitter, jitter))
            x_w, y_w = ship.body_to_world(x_b, y_b)
            if InitSampler._init_position_is_safe(ship, x_w, y_w, placed_safe, min_pair_dist, min_hull_dist):
                chosen_xy = (x_w, y_w)
                break

        if chosen_xy is None:
            for k in range(16):
                x_b = float(slot_body[0])
                y_b = float(slot_body[1] + side * (outward + 10.0 * (k + 1)))
                x_w, y_w = ship.body_to_world(x_b, y_b)
                if InitSampler._init_position_is_safe(ship, x_w, y_w, placed_safe, min_pair_dist, min_hull_dist):
                    chosen_xy = (x_w, y_w)
                    break
        if chosen_xy is None:
            x_b = float(slot_body[0])
            y_b = float(slot_body[1] + side * (outward + 180.0))
            chosen_xy = ship.body_to_world(x_b, y_b)

        speed_noise = float(getattr(cfg, "tug_init_ready_speed_noise_ms", 0.1))
        psi, nu, action = InitSampler._sample_ship_tracking_motion(
            rng,
            ship.u,
            ship.v,
            ship.psi,
            speed_offset_min=-speed_noise,
            speed_offset_max=speed_noise,
            heading_noise_rad=float(getattr(cfg, "tug_init_ready_heading_noise_rad", math.radians(5.0))),
            sway_noise_ms=float(getattr(cfg, "tug_init_ready_sway_noise_ms", 0.03)),
            yaw_rate_noise_rads=float(getattr(cfg, "tug_init_ready_yaw_rate_noise_rads", 0.004)),
            forward_action=float(getattr(cfg, "tug_init_ready_forward_action", 0.22)),
            forward_action_jitter=float(getattr(cfg, "tug_init_action_jitter", 0.04)) * 0.5,
        )
        return np.asarray(chosen_xy, dtype=np.float64), psi, nu, action

    # ---------------------------------------------------- route-candidate init

    @staticmethod
    def _sample_mixed_route_body_candidate(
        rng: np.random.Generator,
        ship: ShipSnapshot,
        cfg: EnvConfig,
        slot_idx: int,
        zone: str,
    ) -> tuple[float, float, tuple[float, float]]:
        side = InitSampler._slot_side_sign(slot_idx)
        slot_body_arr = ship.slot_positions_body()[int(slot_idx), :2]
        slot_body = (float(slot_body_arr[0]), float(slot_body_arr[1]))
        l_half = ship.length_m / 2.0
        lane_min = float(getattr(cfg, "route_lane_min_lat_m", 32.0))
        lane_abs = max(InitSampler._slot_lane_lat_abs(cfg, slot_idx), lane_min + 5.0)
        gate_dist = float(getattr(cfg, "route_stern_gate_dist_m", 60.0))
        lon_jitter = float(getattr(cfg, "tug_init_mixed_route_longitudinal_jitter_m", 18.0))
        lat_jitter = float(getattr(cfg, "tug_init_mixed_route_lateral_jitter_m", 20.0))

        if zone == "rear_lane":
            dist_min, dist_max = InitSampler._slot_rear_start_dist_range(cfg, slot_idx)
            x_b = float(
                -l_half
                - rng.uniform(dist_min, dist_max)
                + rng.uniform(-lon_jitter, lon_jitter)
            )
            y_b = float(side * lane_abs + rng.uniform(-lat_jitter, lat_jitter))
            target = (-l_half - gate_dist, side * lane_abs)
        elif zone == "stern_gate":
            gate_lo = max(20.0, gate_dist * 0.4)
            gate_hi = max(gate_lo, gate_dist * 1.6)
            x_b = float(
                -l_half
                - rng.uniform(gate_lo, gate_hi)
                + rng.uniform(-0.5 * lon_jitter, 0.5 * lon_jitter)
            )
            y_b = float(side * lane_abs + rng.uniform(-0.6 * lat_jitter, 0.6 * lat_jitter))
            target = slot_body
        elif zone == "side_lane":
            side_lat = max(
                lane_abs,
                abs(slot_body[1]) + float(getattr(cfg, "route_outer_holding_extra_m", 18.0)),
            )
            if InitSampler._slot_is_bow(slot_idx):
                x_lo = -l_half - 0.5 * gate_dist
                x_hi = min(slot_body[0] - 20.0, l_half + 0.5 * float(getattr(cfg, "slot_lon_offset_m", 30.0)))
            else:
                x_lo = slot_body[0] - gate_dist
                x_hi = -l_half - 8.0
            x_lo, x_hi = sorted((x_lo, x_hi))
            x_b = float(
                rng.uniform(x_lo, x_hi)
                + rng.uniform(-0.3 * lon_jitter, 0.3 * lon_jitter)
            )
            y_b = float(side * side_lat + rng.uniform(-0.6 * lat_jitter, 0.6 * lat_jitter))
            target = slot_body
        elif zone == "outer_slot":
            outward_min = max(25.0, float(getattr(cfg, "route_outer_holding_extra_m", 18.0)) + 15.0)
            outward_max = outward_min + 70.0
            x_span = max(20.0, lon_jitter)
            x_b = float(slot_body[0] + rng.uniform(-x_span, x_span))
            y_b = float(
                slot_body[1]
                + side * rng.uniform(outward_min, outward_max)
                + rng.uniform(-0.5 * lat_jitter, 0.5 * lat_jitter)
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
                - rng.uniform(dist_lo, dist_hi)
                + rng.uniform(-lon_jitter, lon_jitter)
            )
            y_b = float(-side * opposite_lane_abs + rng.uniform(-lat_jitter, lat_jitter))
            target = (-l_half - gate_dist, side * lane_abs)
        else:
            raise ValueError(f"unknown mixed init zone: {zone!r}")

        return x_b, y_b, target

    @staticmethod
    def _mixed_route_fallback_candidates(
        ship: ShipSnapshot,
        cfg: EnvConfig,
        slot_idx: int,
        *,
        force_opposite_side: bool,
    ) -> list[tuple[float, float, tuple[float, float]]]:
        side = InitSampler._slot_side_sign(slot_idx)
        l_half = ship.length_m / 2.0
        lane_min = float(getattr(cfg, "route_lane_min_lat_m", 32.0))
        lane_abs = max(InitSampler._slot_lane_lat_abs(cfg, slot_idx), lane_min + 5.0)
        gate_dist = float(getattr(cfg, "route_stern_gate_dist_m", 60.0))
        _, dist_max = InitSampler._slot_rear_start_dist_range(cfg, slot_idx)
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
                else tuple(float(v) for v in ship.slot_positions_body()[int(slot_idx), :2])
            )
            for k in range(48):
                x_b = float(-l_half - base_dist - 80.0 - 70.0 * k)
                y_b = float(fallback_side * fallback_lane_abs)
                candidates.append((x_b, y_b, target))
        return candidates

    # --------------------------------------------------------- full route init

    @staticmethod
    def _sample_random_route_state(
        rng: np.random.Generator,
        ship: ShipSnapshot,
        cfg: EnvConfig,
        slot_idx: int,
        placed: list[tuple[float, float]],
        *,
        force_opposite_side: bool = False,
    ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        side = InitSampler._slot_side_sign(slot_idx)
        min_pair_dist, min_hull_dist = InitSampler._mixed_init_safety_margins(cfg)
        lane_min = float(getattr(cfg, "route_lane_min_lat_m", 32.0))
        all_zones = ["rear_lane", "stern_gate", "side_lane", "outer_slot", "opposite_stern"]
        if force_opposite_side:
            zones = ["opposite_stern"]
        else:
            configured = tuple(getattr(cfg, "tug_init_mixed_zones", tuple(all_zones)))
            zones = [str(zone) for zone in configured if str(zone) in all_zones]
            if not zones:
                zones = list(all_zones)
        zones = list(rng.permutation(np.asarray(zones, dtype=object)))

        chosen_xy: tuple[float, float] | None = None
        chosen_body: tuple[float, float] | None = None
        target_body: tuple[float, float] | None = None
        for zone in zones:
            for _ in range(80):
                x_b, y_b, target = InitSampler._sample_mixed_route_body_candidate(
                    rng, ship, cfg, slot_idx, str(zone),
                )
                if str(zone) == "opposite_stern":
                    if side * y_b > -lane_min:
                        continue
                elif side * y_b < lane_min:
                    continue

                x_w, y_w = ship.body_to_world(x_b, y_b)
                if not InitSampler._init_position_is_safe(ship, x_w, y_w, placed, min_pair_dist, min_hull_dist):
                    continue
                chosen_xy = (x_w, y_w)
                chosen_body = (x_b, y_b)
                target_body = target
                break
            if chosen_xy is not None:
                break

        if chosen_xy is None:
            for x_b, y_b, target in InitSampler._mixed_route_fallback_candidates(
                ship, cfg, slot_idx,
                force_opposite_side=force_opposite_side,
            ):
                x_w, y_w = ship.body_to_world(x_b, y_b)
                if InitSampler._init_position_is_safe(ship, x_w, y_w, placed, min_pair_dist, min_hull_dist):
                    chosen_xy = (x_w, y_w)
                    chosen_body = (x_b, y_b)
                    target_body = target
                    break

        if chosen_xy is None or chosen_body is None or target_body is None:
            raise RuntimeError("failed to sample a safe mixed tug initialization position")

        psi, nu, action = InitSampler._sample_ship_tracking_motion(
            rng,
            ship.u,
            ship.v,
            ship.psi,
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

    @staticmethod
    def sample_mixed_slot_approach_states(
        rng: np.random.Generator,
        n_tugs: int,
        ship: ShipSnapshot,
        cfg: EnvConfig,
        episode: MutableEpisodeState,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = n_tugs
        ready_counts_raw = getattr(cfg, "tug_init_mixed_ready_counts", (2, 3))
        ready_counts = [int(v) for v in ready_counts_raw if 0 <= int(v) <= n]
        if not ready_counts:
            ready_counts = [max(0, n - 1)]
        ready_count = int(rng.choice(np.asarray(ready_counts, dtype=np.int32)))
        ready_slots = set(int(v) for v in rng.choice(n, size=ready_count, replace=False))
        episode.mixed_ready_tugs = ready_slots

        positions = np.zeros((n, 2), dtype=np.float64)
        psis = np.zeros(n, dtype=np.float64)
        nus = np.zeros((n, 3), dtype=np.float64)
        actions = np.zeros((n, ACTION_DIM), dtype=np.float32)

        placed: list[tuple[float, float]] = []
        for i in range(n):
            if i not in ready_slots:
                continue
            pos, psi, nu, action = InitSampler._sample_ready_slot_state(
                rng, ship, cfg, i, placed,
            )
            positions[i] = pos
            psis[i] = psi
            nus[i] = nu
            actions[i] = action
            placed.append((float(pos[0]), float(pos[1])))

        free_slots = [i for i in range(n) if i not in ready_slots]
        force_single_opposite = (
            ready_count == n - 1
            and len(free_slots) == 1
            and bool(getattr(cfg, "tug_init_force_single_opposite", True))
        )
        for i in rng.permutation(free_slots):
            pos, psi, nu, action = InitSampler._sample_random_route_state(
                rng, ship, cfg, int(i), placed,
                force_opposite_side=force_single_opposite,
            )
            positions[i] = pos
            psis[i] = psi
            nus[i] = nu
            actions[i] = action
            placed.append((float(pos[0]), float(pos[1])))

        return positions, psis, nus, actions
