"""Utilities for loading staged training course files."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from config import EnvConfig

from curricula.spec import CourseSpec, parse_course_dict

__all__ = ["CourseSpec", "apply_course", "load_course", "load_course_dict"]


def load_course_dict(path: str | Path) -> Mapping:
    course_path = Path(path).expanduser().resolve()
    if not course_path.exists():
        raise FileNotFoundError(f"course file not found: {course_path}")
    if not course_path.is_file():
        raise ValueError(f"course path is not a file: {course_path}")

    module_name = f"_ship_control_course_{course_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, course_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import course file: {course_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    raw = getattr(module, "COURSE", None)
    if not isinstance(raw, Mapping):
        raise ValueError(f"{course_path} must define COURSE as a mapping")
    return raw


def load_course(path: str | Path) -> CourseSpec:
    course_path = Path(path).expanduser().resolve()
    raw = load_course_dict(course_path)
    return parse_course_dict(raw, path=course_path)


def apply_course(env_cfg: EnvConfig, course: CourseSpec) -> None:
    for key, value in course.env_overrides.items():
        setattr(env_cfg, key, value)
