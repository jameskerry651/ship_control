"""拖轮编队环境的安全初始化逻辑。"""

from __future__ import annotations

import math
from itertools import permutations
import warnings

import numpy as np

from config import EnvConfig
from env.obs_spec import ACTION_DIM
from physics.large_ship_model import distance_from_rectangular_hull_pose


class InitializationError(RuntimeError):
    """初始化参数无解或在有界尝试次数内未能生成合法场景。"""


def assign_tugs_to_slots(
    tug_positions: np.ndarray,
    slot_positions: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, dict[str, object]]:
    """为拖轮分配唯一 slot；minimax 优先降低最难单艇航程，再降低总航程。"""
    tugs = np.asarray(tug_positions, dtype=np.float64)
    slots = np.asarray(slot_positions, dtype=np.float64)
    if tugs.ndim != 2 or tugs.shape[1] != 2:
        raise InitializationError(
            f"tug_positions must have shape (n_tugs, 2), got {tugs.shape}"
        )
    if slots.ndim != 2 or slots.shape[1] < 2:
        raise InitializationError(
            f"slot_positions must have shape (n_slots, >=2), got {slots.shape}"
        )
    if not np.isfinite(tugs).all() or not np.isfinite(slots[:, :2]).all():
        raise InitializationError("tug and slot positions must all be finite")
    n_tugs = int(tugs.shape[0])
    n_slots = int(slots.shape[0])
    if n_tugs <= 0:
        raise InitializationError(f"n_tugs must be > 0, got {n_tugs}")
    if n_tugs > n_slots:
        raise InitializationError(
            f"cannot assign {n_tugs} tugs to only {n_slots} unique slots"
        )

    mode_key = str(mode).strip().lower()
    if mode_key not in {"fixed", "minimax"}:
        raise InitializationError(
            f"unknown tug_slot_assignment_mode={mode!r}; expected 'fixed' or 'minimax'"
        )

    distances = np.linalg.norm(tugs[:, None, :] - slots[None, :, :2], axis=2)
    if mode_key == "fixed":
        selected = tuple(range(n_tugs))
    else:
        # permutations 按字典序生成，因此完全相同的浮点代价也有确定性结果。
        selected = min(
            permutations(range(n_slots), n_tugs),
            key=lambda candidate: (
                max(float(distances[i, candidate[i]]) for i in range(n_tugs)),
                sum(float(distances[i, candidate[i]]) for i in range(n_tugs)),
                candidate,
            ),
        )

    assigned_distances = tuple(
        float(distances[i, selected[i]]) for i in range(n_tugs)
    )
    mapping = np.asarray(selected, dtype=np.int32)
    diagnostics: dict[str, object] = {
        "assignment_mode": mode_key,
        "sample_to_slot": tuple(int(v) for v in mapping),
        "assignment_distances_m": assigned_distances,
        "assignment_max_distance_m": float(max(assigned_distances)),
        "assignment_total_distance_m": float(sum(assigned_distances)),
    }
    return mapping, diagnostics


def _validated_init_parameters(
    cfg: EnvConfig, n_tugs: int
) -> tuple[float, float, float, int, bool]:
    """校验安全圆环参数并返回派生约束。"""
    radius = float(cfg.tug_init_radius_m)
    ship_margin = float(cfg.tug_init_ship_margin_m)
    pair_margin = float(cfg.tug_init_pair_margin_m)
    max_attempts = int(cfg.tug_init_max_attempts)

    values = {
        "tug_init_radius_m": radius,
        "tug_init_ship_margin_m": ship_margin,
        "tug_init_pair_margin_m": pair_margin,
    }
    for name, value in values.items():
        if not math.isfinite(value):
            raise InitializationError(f"{name} must be finite, got {value!r}")
    if radius <= 0.0:
        raise InitializationError(f"tug_init_radius_m must be > 0, got {radius}")
    if ship_margin < 0.0 or pair_margin < 0.0:
        raise InitializationError(
            "initialization margins must be >= 0, got "
            f"ship={ship_margin}, pair={pair_margin}"
        )
    if max_attempts <= 0:
        raise InitializationError(
            f"tug_init_max_attempts must be > 0, got {max_attempts}"
        )
    if n_tugs <= 0:
        raise InitializationError(f"n_tugs must be > 0, got {n_tugs}")

    ship_clearance = float(cfg.ship_collision_dist_m) + ship_margin
    pair_separation = float(cfg.tug_collision_dist_m) + pair_margin
    if ship_clearance < 0.0 or pair_separation < 0.0:
        raise InitializationError(
            "derived initialization clearances must be >= 0, got "
            f"ship={ship_clearance}, pair={pair_separation}"
        )

    # 圆上离矩形船体最远的方向是沿较短半轴方向；达不到时必然无解。
    shortest_half_extent = min(cfg.ship_length_m, cfg.ship_beam_m) / 2.0
    if radius + 1e-12 < shortest_half_extent + ship_clearance:
        raise InitializationError(
            "safe circle has no ship-clearance solution: "
            f"radius={radius:.3f}, required_ship_clearance={ship_clearance:.3f}, "
            f"shortest_ship_half_extent={shortest_half_extent:.3f}"
        )

    # n 个圆周点的最大可能最小弦长由正 n 边形取得。
    if n_tugs > 1:
        max_pair_separation = 2.0 * radius * math.sin(math.pi / n_tugs)
        if pair_separation > max_pair_separation + 1e-12:
            raise InitializationError(
                "safe circle has no pair-separation solution: "
                f"n_tugs={n_tugs}, radius={radius:.3f}, "
                f"required_pair_separation={pair_separation:.3f}, "
                f"maximum_possible={max_pair_separation:.3f}"
            )

    full_angle_radius = (
        math.hypot(cfg.ship_length_m / 2.0, cfg.ship_beam_m / 2.0)
        + ship_clearance
    )
    biased_angles = radius + 1e-12 < full_angle_radius
    return radius, ship_clearance, pair_separation, max_attempts, biased_angles


def validate_tug_init_positions(
    positions: np.ndarray,
    ship_x: float,
    ship_y: float,
    ship_psi: float,
    cfg: EnvConfig,
) -> None:
    """验证一组初始位置的安全后置条件；非法时抛出 ``InitializationError``。"""
    points = np.asarray(positions, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise InitializationError(
            f"initial positions must have shape (n_tugs, 2), got {points.shape}"
        )
    _, ship_clearance, pair_separation, _, _ = _validated_init_parameters(
        cfg, int(points.shape[0])
    )
    if not np.isfinite(points).all():
        raise InitializationError("initial positions must all be finite")

    for i, (x_world, y_world) in enumerate(points):
        d_hull = distance_from_rectangular_hull_pose(
            float(x_world),
            float(y_world),
            ship_x,
            ship_y,
            ship_psi,
            cfg.ship_length_m,
            cfg.ship_beam_m,
        )
        if d_hull + 1e-12 < ship_clearance:
            raise InitializationError(
                f"tug {i} violates initial ship clearance: "
                f"distance={d_hull:.6f}, required={ship_clearance:.6f}"
            )

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distance = float(np.linalg.norm(points[i] - points[j]))
            if distance + 1e-12 < pair_separation:
                raise InitializationError(
                    f"tugs {i} and {j} violate initial pair separation: "
                    f"distance={distance:.6f}, required={pair_separation:.6f}"
                )


def sample_tug_init_states(
    rng: np.random.Generator,
    n_tugs: int,
    ship_x: float,
    ship_y: float,
    ship_psi: float,
    cfg: EnvConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """在固定圆环上逐艇拒绝采样，保证 reset 后满足硬碰撞阈值与余量。

    返回:
        positions: (n_tugs, 2) 世界坐标
        psis: (n_tugs,) 航向角，均为 0
        nus: (n_tugs, 3) 船体系速度 (u, v, r)，均为 0
        actions: (n_tugs, 4) 初始动作 [port_rpm, stbd_rpm, port_az, stbd_az]，均为 0
        diagnostics: 采样次数、安全阈值及是否为条件角度分布
    """
    radius, ship_clearance, pair_separation, max_attempts, biased_angles = (
        _validated_init_parameters(cfg, n_tugs)
    )
    if biased_angles:
        full_angle_radius = (
            math.hypot(cfg.ship_length_m / 2.0, cfg.ship_beam_m / 2.0)
            + ship_clearance
        )
        warnings.warn(
            "tug_init_radius_m is below the full-angle safe radius; rejection "
            "sampling remains collision-free but biases accepted positions away "
            f"from blocked directions (radius={radius:.3f}, "
            f"full_angle_safe_radius={full_angle_radius:.3f})",
            RuntimeWarning,
            stacklevel=2,
        )

    positions = np.zeros((n_tugs, 2), dtype=np.float64)
    accepted: list[tuple[float, float]] = []
    attempts_per_tug = np.zeros(n_tugs, dtype=np.int32)
    attempts_total = 0
    sample_order = rng.permutation(n_tugs)
    cos_ship = math.cos(ship_psi)
    sin_ship = math.sin(ship_psi)

    for raw_idx in sample_order:
        tug_idx = int(raw_idx)
        accepted_current = False
        while attempts_total < max_attempts:
            attempts_total += 1
            attempts_per_tug[tug_idx] += 1
            angle_body = float(rng.uniform(0.0, 2.0 * math.pi))
            x_body = radius * math.cos(angle_body)
            y_body = radius * math.sin(angle_body)
            x_world = ship_x + cos_ship * x_body - sin_ship * y_body
            y_world = ship_y + sin_ship * x_body + cos_ship * y_body

            d_hull = distance_from_rectangular_hull_pose(
                x_world,
                y_world,
                ship_x,
                ship_y,
                ship_psi,
                cfg.ship_length_m,
                cfg.ship_beam_m,
            )
            if d_hull + 1e-12 < ship_clearance:
                continue
            if any(
                math.hypot(x_world - other_x, y_world - other_y) + 1e-12
                < pair_separation
                for other_x, other_y in accepted
            ):
                continue

            positions[tug_idx] = (x_world, y_world)
            accepted.append((x_world, y_world))
            accepted_current = True
            break

        if not accepted_current:
            raise InitializationError(
                "failed to sample a collision-free tug formation within the "
                f"attempt budget: n_tugs={n_tugs}, placed={len(accepted)}, "
                f"radius={radius:.3f}, ship_clearance={ship_clearance:.3f}, "
                f"pair_separation={pair_separation:.3f}, "
                f"attempts={attempts_total}, max_attempts={max_attempts}"
            )

    psis = np.zeros(n_tugs, dtype=np.float64)
    nus = np.zeros((n_tugs, 3), dtype=np.float64)
    actions = np.zeros((n_tugs, ACTION_DIM), dtype=np.float32)
    validate_tug_init_positions(positions, ship_x, ship_y, ship_psi, cfg)
    diagnostics: dict[str, object] = {
        "schema": str(cfg.tug_init_schema),
        "attempts_total": int(attempts_total),
        "attempts_per_tug": tuple(int(v) for v in attempts_per_tug),
        "sample_order": tuple(int(v) for v in sample_order),
        "ship_clearance_m": float(ship_clearance),
        "pair_separation_m": float(pair_separation),
        "biased_angles": bool(biased_angles),
    }
    return positions, psis, nus, actions, diagnostics
