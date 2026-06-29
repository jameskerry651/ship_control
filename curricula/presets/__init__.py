"""Reusable course presets and builders."""

from curricula.presets.ablations import ABLATION_COURSES
from curricula.presets.main import MAIN_COURSES
from curricula.presets.strict_pos import STRICT_POS_COURSES, strict_pos_course

__all__ = [
    "ABLATION_COURSES",
    "MAIN_COURSES",
    "STRICT_POS_COURSES",
    "strict_pos_course",
]
