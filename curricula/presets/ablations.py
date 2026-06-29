"""C4 reward-redesign ablation courses (incremental P1..P5)."""

from __future__ import annotations

from curricula.presets.helpers import make_course, merge_overrides
from curricula.presets.shared import (
    C04_ABLATION_INIT,
    C04_ABLATION_REWARD_OFF,
)

_P1 = {
    "reward_collision_pen_culprit": 80.0,
    "reward_collision_pen_bystander": 15.0,
}

_P2 = {
    "reward_team_w": 0.5,
    "reward_team_softmin_beta": 4.0,
}

_P3 = {
    "reward_shape_w": 0.3,
    "reward_shape_gamma": 0.99,
    "reward_shape_d_ref_m": 200.0,
    "reward_shape_clip": 1.0,
}

_P4 = {
    "reward_hold_streak_w": 0.3,
    "reward_hold_streak_decay": 2,
}

_P5 = {
    "reward_collision_w": 3.0,
    "reward_collision_cap": 1.5,
}

_ABLATION_SPECS: tuple[tuple[str, str, tuple[dict[str, float], ...]], ...] = (
    (
        "c04_abl_p1",
        "C4 ablation: P1 per-culprit terminal penalty only.",
        (_P1,),
    ),
    (
        "c04_abl_p1p2",
        "C4 ablation: P1 + P2 team softmin.",
        (_P1, _P2),
    ),
    (
        "c04_abl_p1p2p3",
        "C4 ablation: P1 + P2 + P3 potential-based approach shaping.",
        (_P1, _P2, _P3),
    ),
    (
        "c04_abl_p1p2p3p4",
        "C4 ablation: P1 + P2 + P3 + P4 hold-streak robustness.",
        (_P1, _P2, _P3, _P4),
    ),
    (
        "c04_abl_full",
        "C4 ablation: full P1..P5 redesign.",
        (_P1, _P2, _P3, _P4, _P5),
    ),
)


def _ablation_course(name: str, description: str, reward_layers: tuple[dict, ...]) -> dict:
    return make_course(
        name=name,
        description=description,
        total_steps=1_500_000,
        env_overrides=merge_overrides(C04_ABLATION_INIT, C04_ABLATION_REWARD_OFF, *reward_layers),
    )


ABLATION_COURSES = {
    name: _ablation_course(name, description, layers)
    for name, description, layers in _ABLATION_SPECS
}
