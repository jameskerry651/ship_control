"""Course specification types and validation."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from config import EnvConfig

CourseDict = Mapping[str, Any]


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


def validate_env_overrides(
    env_overrides: Mapping[str, Any],
    *,
    source: str,
) -> None:
    valid_env_keys = {field.name for field in fields(EnvConfig)}
    invalid = sorted(str(key) for key in env_overrides if str(key) not in valid_env_keys)
    if invalid:
        raise ValueError(f"{source}: unknown EnvConfig override key(s): {', '.join(invalid)}")


def parse_course_dict(raw: CourseDict, *, path: str | Path) -> CourseSpec:
    course_path = Path(path).expanduser().resolve()

    env_overrides = raw.get("env_overrides", {})
    if not isinstance(env_overrides, Mapping):
        raise ValueError(f"{course_path}: COURSE['env_overrides'] must be a mapping")
    validate_env_overrides(env_overrides, source=str(course_path))

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
