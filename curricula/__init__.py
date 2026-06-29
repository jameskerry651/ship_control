"""Curriculum course definitions for staged MAPPO training."""

from curricula.loader import CourseSpec, apply_course, load_course
from curricula.registry import (
    ABLATION_SEQUENCE,
    ALL_COURSE_ENTRIES,
    MAIN_SEQUENCE,
    STRICT_POS_SEQUENCE,
    CourseEntry,
)

__all__ = [
    "ABLATION_SEQUENCE",
    "ALL_COURSE_ENTRIES",
    "CourseEntry",
    "CourseSpec",
    "MAIN_SEQUENCE",
    "STRICT_POS_SEQUENCE",
    "apply_course",
    "load_course",
]
