"""Central catalog of curriculum courses and file paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from curricula.presets.ablations import ABLATION_COURSES
from curricula.presets.main import MAIN_COURSES
from curricula.presets.strict_pos import STRICT_POS_COURSES, STRICT_POS_LADDER, strict_pos_filename

CURRICULA_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class CourseEntry:
    key: str
    relative_path: str
    run_name: str

    @property
    def path(self) -> Path:
        return CURRICULA_ROOT / self.relative_path

    @property
    def project_relative_path(self) -> str:
        return str(Path("curricula") / self.relative_path)


MAIN_SEQUENCE: tuple[CourseEntry, ...] = (
    CourseEntry("c01", "c01_three_ready.py", "c01_three_ready"),
    CourseEntry("c02", "c02_two_ready.py", "c02_two_ready"),
    CourseEntry("c03", "c03_one_ready.py", "c03_one_ready"),
    CourseEntry("c04a", "c04_one_ready_bridge.py", "c04_one_ready_bridge"),
    CourseEntry("c04b", "c04_zero_route_bridge.py", "c04_zero_route_bridge"),
    CourseEntry("c04", "c04_zero_ready.py", "c04_zero_ready"),
)

STRICT_POS_SEQUENCE: tuple[CourseEntry, ...] = tuple(
    CourseEntry(
        f"pos{int(pos_tol_m) if pos_tol_m == int(pos_tol_m) else pos_tol_m}",
        f"strict_pos/{strict_pos_filename(pos_tol_m)}",
        f"c04_{strict_pos_filename(pos_tol_m).removesuffix('.py')}",
    )
    for pos_tol_m, _ in STRICT_POS_LADDER
)

ABLATION_SEQUENCE: tuple[CourseEntry, ...] = tuple(
    CourseEntry(
        name.removeprefix("c04_abl_"),
        f"ablations/{name}.py",
        name,
    )
    for name in ABLATION_COURSES
)

ALL_COURSE_ENTRIES: tuple[CourseEntry, ...] = (
    MAIN_SEQUENCE + STRICT_POS_SEQUENCE + ABLATION_SEQUENCE
)


def all_course_paths() -> tuple[Path, ...]:
    return tuple(entry.path for entry in ALL_COURSE_ENTRIES)


def main_sequence_paths() -> tuple[str, ...]:
    return tuple(entry.relative_path for entry in MAIN_SEQUENCE)


def strict_pos_path_for_tol(pos_tol_m: float) -> Path | None:
    filename = strict_pos_filename(pos_tol_m)
    candidate = CURRICULA_ROOT / "strict_pos" / filename
    return candidate if candidate.exists() else None
