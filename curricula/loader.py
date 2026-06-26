"""Utilities for loading staged training course files."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from config import EnvConfig


@dataclass(frozen=True)
class CourseSpec:
    name: str
    description: str
    env_overrides: Mapping[str, Any]
    total_steps: int | None
    path: str

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "env_overrides": dict(self.env_overrides),
            "total_steps": self.total_steps,
            "path": self.path,
        }


def load_course(path: str | Path) -> CourseSpec:
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

    env_overrides = raw.get("env_overrides", {})
    if not isinstance(env_overrides, Mapping):
        raise ValueError(f"{course_path}: COURSE['env_overrides'] must be a mapping")

    valid_env_keys = {field.name for field in fields(EnvConfig)}
    invalid = sorted(str(key) for key in env_overrides if str(key) not in valid_env_keys)
    if invalid:
        raise ValueError(
            f"{course_path}: unknown EnvConfig override key(s): {', '.join(invalid)}"
        )

    total_steps_raw = raw.get("total_steps", None)
    total_steps: int | None
    if total_steps_raw is None:
        total_steps = None
    else:
        total_steps = int(total_steps_raw)
        if total_steps <= 0:
            raise ValueError(f"{course_path}: total_steps must be positive")

    name = str(raw.get("name") or course_path.stem)
    description = str(raw.get("description") or "")
    return CourseSpec(
        name=name,
        description=description,
        env_overrides=MappingProxyType(dict(env_overrides)),
        total_steps=total_steps,
        path=str(course_path),
    )


def apply_course(env_cfg: EnvConfig, course: CourseSpec) -> None:
    for key, value in course.env_overrides.items():
        setattr(env_cfg, key, value)

