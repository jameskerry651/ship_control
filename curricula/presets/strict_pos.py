"""Strict position-tolerance ladder built on top of c04."""

from __future__ import annotations

from curricula.presets.helpers import make_course, merge_overrides
from curricula.presets.shared import (
    C04_FINAL_MIXED_READY_INIT,
    C04_ZERO_READY_INIT,
    FINAL_SLOT_GUIDANCE_OVERRIDES,
    SAFE_JOIN_REWARD_OVERRIDES,
    STRICT_POS_FINAL_REWARD_OVERRIDES,
    STRICT_POS_REWARD_OVERRIDES,
)

# (pos_tol_m, total_steps) — ordered from loose to strict.
STRICT_POS_LADDER: tuple[tuple[float, int], ...] = (
    (140.0, 800_000),
    (120.0, 800_000),
    (100.0, 800_000),
    (80.0, 1_000_000),
    (60.0, 1_000_000),
    (40.0, 1_500_000),
    (20.0, 2_000_000),
    (10.0, 3_000_000),
)


def _pos_suffix(pos_tol_m: float) -> str:
    text = f"{pos_tol_m:g}".replace(".", "p")
    return f"pos{text}m"


def strict_pos_course(pos_tol_m: float, total_steps: int) -> dict:
    suffix = _pos_suffix(pos_tol_m)
    final = pos_tol_m <= 10.0
    init_overrides = C04_FINAL_MIXED_READY_INIT if final else C04_ZERO_READY_INIT
    reward_overrides = (
        STRICT_POS_FINAL_REWARD_OVERRIDES if final else STRICT_POS_REWARD_OVERRIDES
    )
    return make_course(
        name=f"c04_{suffix}",
        description=(
            f"{'Mixed-ready' if final else 'Zero-ready'} c04 strict-position "
            f"{'final' if final else 'ladder'} stage "
            f"with pos_tol={pos_tol_m:g} m."
        ),
        total_steps=total_steps,
        env_overrides=merge_overrides(
            SAFE_JOIN_REWARD_OVERRIDES,
            init_overrides,
            FINAL_SLOT_GUIDANCE_OVERRIDES,
            reward_overrides,
            {"pos_tol_m": float(pos_tol_m)},
        ),
    )


STRICT_POS_COURSES = {
    f"c04_{_pos_suffix(pos_tol_m)}": strict_pos_course(pos_tol_m, total_steps)
    for pos_tol_m, total_steps in STRICT_POS_LADDER
}


def strict_pos_filename(pos_tol_m: float) -> str:
    return f"c04_{_pos_suffix(pos_tol_m)}.py"
