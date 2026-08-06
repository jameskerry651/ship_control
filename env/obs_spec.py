"""观测空间契约与全局状态维度。

``ObservationSpec`` 是环境、Observer、Actor 和 checkpoint 共同使用的单一真相源。
规格在一次运行创建时由配置确定，运行过程中保持不变。

观测结构总览（默认 93 维 / agent）::

    ┌─────────────────────────────────────────────────────────────┐
    │  历史帧（4 帧 × 6 维）  = 24 维  (_EGO_MOTION_OBS_DIM)      │
    │  动作历史（4 帧 × 4 维）= 16 维  (_ACTION_HISTORY_OBS_DIM)  │
    │  大船相对状态              =  5 维  (_SHIP_REL_OBS_DIM)      │
    │  大船预瞄点（3 点 × 2 维）=  6 维  (_SHIP_PREVIEW_POINT_DIM)│
    │  目标槽位                  =  5 维  (_SLOT_TARGET_OBS_DIM)   │
    │  船体间隙                  =  3 维  (_HULL_CLEARANCE_OBS_DIM)│
    │  实际推进器状态            =  4 维  (_THRUSTER_STATE_OBS_DIM)│
    │  邻居特征（3 邻 ×10 维）  = 30 维  (_NEIGHBOR_OBS_DIM)      │
    └─────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

# -------- action dimension --------
ACTION_DIM: int = 4

# -------- per-agent observation sub-dimensions --------
_EGO_MOTION_OBS_DIM = 6
_ACTION_HISTORY_OBS_DIM = ACTION_DIM
_THRUSTER_STATE_OBS_DIM = 4
_SHIP_REL_OBS_DIM = 5
_SHIP_PREVIEW_POINT_DIM = 2
_SLOT_TARGET_OBS_DIM = 5
_HULL_CLEARANCE_OBS_DIM = 3
_NEIGHBOR_COUNT = 3
_NEIGHBOR_OBS_DIM = 10

_NEIGHBOR_TCPA_SCALE_S = 60.0
_NEIGHBOR_REL_SPEED_EPS = 1e-6

# -------- configurable observation contract --------
_DEFAULT_HIST_LEN = 4
_DEFAULT_PREVIEW_POINTS = 3
_HISTORY_TOKEN_DIM = _EGO_MOTION_OBS_DIM + _ACTION_HISTORY_OBS_DIM


@dataclass(frozen=True)
class ObservationSpec:
    """一次训练运行中固定的扁平观测布局。

    可配置的是历史帧数、母船预瞄点数和邻居槽位数；单个字段的语义维度由
    当前 schema 固定。所有 slice 都从这些字段派生，避免 Actor 使用手写偏移量。
    """

    schema_version: int = 2
    history_len: int = _DEFAULT_HIST_LEN
    preview_count: int = _DEFAULT_PREVIEW_POINTS
    neighbor_count: int = _NEIGHBOR_COUNT
    motion_dim: int = _EGO_MOTION_OBS_DIM
    action_history_dim: int = _ACTION_HISTORY_OBS_DIM
    thruster_state_dim: int = _THRUSTER_STATE_OBS_DIM
    ship_relative_dim: int = _SHIP_REL_OBS_DIM
    preview_point_dim: int = _SHIP_PREVIEW_POINT_DIM
    slot_target_dim: int = _SLOT_TARGET_OBS_DIM
    hull_clearance_dim: int = _HULL_CLEARANCE_OBS_DIM
    neighbor_dim: int = _NEIGHBOR_OBS_DIM

    def __post_init__(self) -> None:
        if self.schema_version not in (1, 2):
            raise ValueError(f"unsupported observation schema_version={self.schema_version}")
        if self.history_len < 1:
            raise ValueError("history_len must be >= 1")
        if self.preview_count < 0:
            raise ValueError("preview_count must be >= 0")
        if self.neighbor_count < 1:
            raise ValueError("neighbor_count must be >= 1")
        fixed_dims = {
            "motion_dim": (self.motion_dim, _EGO_MOTION_OBS_DIM),
            "action_history_dim": (self.action_history_dim, _ACTION_HISTORY_OBS_DIM),
            "ship_relative_dim": (self.ship_relative_dim, _SHIP_REL_OBS_DIM),
            "preview_point_dim": (self.preview_point_dim, _SHIP_PREVIEW_POINT_DIM),
            "slot_target_dim": (self.slot_target_dim, _SLOT_TARGET_OBS_DIM),
            "hull_clearance_dim": (self.hull_clearance_dim, _HULL_CLEARANCE_OBS_DIM),
            "neighbor_dim": (self.neighbor_dim, _NEIGHBOR_OBS_DIM),
        }
        for name, (actual, expected) in fixed_dims.items():
            if actual != expected:
                raise ValueError(
                    f"schema v{self.schema_version} requires {name}={expected}, got {actual}"
                )
        expected_thruster_dim = 0 if self.schema_version == 1 else _THRUSTER_STATE_OBS_DIM
        if self.thruster_state_dim != expected_thruster_dim:
            raise ValueError(
                f"schema v{self.schema_version} requires "
                f"thruster_state_dim={expected_thruster_dim}, got "
                f"{self.thruster_state_dim}"
            )

    @classmethod
    def from_config(
        cls,
        cfg: Any,
        *,
        neighbor_count: int | None = None,
    ) -> "ObservationSpec":
        history_len = int(getattr(cfg, "obs_history_k", _DEFAULT_HIST_LEN - 1)) + 1
        preview_times = tuple(
            getattr(cfg, "obs_ship_preview_times_s", (5.0, 10.0, 15.0))
        )
        if neighbor_count is None:
            n_tugs = int(getattr(cfg, "n_tugs", _NEIGHBOR_COUNT + 1))
            # 单艇调试时保留一个零填充邻居槽位，避免空集合 attention。
            neighbor_count = max(1, n_tugs - 1)
        return cls(
            history_len=history_len,
            preview_count=len(preview_times),
            neighbor_count=int(neighbor_count),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ObservationSpec":
        allowed = set(cls.__dataclass_fields__)
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown observation spec fields: {sorted(unknown)}")
        parsed = {key: int(raw) for key, raw in value.items()}
        # schema v1 是加入实际推进器反馈之前的 89 维布局。
        if parsed.get("schema_version") == 1:
            parsed.setdefault("thruster_state_dim", 0)
        return cls(**parsed)

    def to_dict(self) -> dict[str, int]:
        return {key: int(value) for key, value in asdict(self).items()}

    @property
    def history_token_dim(self) -> int:
        return self.motion_dim + self.action_history_dim

    @property
    def motion_history_size(self) -> int:
        return self.history_len * self.motion_dim

    @property
    def action_history_size(self) -> int:
        return self.history_len * self.action_history_dim

    @property
    def ship_preview_size(self) -> int:
        return self.preview_count * self.preview_point_dim

    @property
    def own_context_dim(self) -> int:
        return (
            self.ship_relative_dim
            + self.ship_preview_size
            + self.slot_target_dim
            + self.hull_clearance_dim
            + self.thruster_state_dim
        )

    @property
    def own_dim(self) -> int:
        return self.motion_history_size + self.action_history_size + self.own_context_dim

    @property
    def attention_dim(self) -> int:
        return self.neighbor_count * self.neighbor_dim

    @property
    def total_dim(self) -> int:
        return self.own_dim + self.attention_dim

    @staticmethod
    def _slice(start: int, size: int) -> slice:
        return slice(start, start + size)

    @property
    def motion_history_slice(self) -> slice:
        return self._slice(0, self.motion_history_size)

    @property
    def action_history_slice(self) -> slice:
        return self._slice(self.motion_history_slice.stop, self.action_history_size)

    @property
    def ship_relative_slice(self) -> slice:
        return self._slice(self.action_history_slice.stop, self.ship_relative_dim)

    @property
    def ship_preview_slice(self) -> slice:
        return self._slice(self.ship_relative_slice.stop, self.ship_preview_size)

    @property
    def slot_target_slice(self) -> slice:
        return self._slice(self.ship_preview_slice.stop, self.slot_target_dim)

    @property
    def hull_clearance_slice(self) -> slice:
        return self._slice(self.slot_target_slice.stop, self.hull_clearance_dim)

    @property
    def thruster_state_slice(self) -> slice:
        return self._slice(self.hull_clearance_slice.stop, self.thruster_state_dim)

    @property
    def own_slice(self) -> slice:
        return slice(0, self.own_dim)

    @property
    def neighbor_slice(self) -> slice:
        return self._slice(self.own_dim, self.attention_dim)

    def neighbor_item_slice(self, index: int) -> slice:
        if not 0 <= index < self.neighbor_count:
            raise IndexError(
                f"neighbor index {index} outside [0, {self.neighbor_count})"
            )
        return self._slice(self.own_dim + index * self.neighbor_dim, self.neighbor_dim)

    def differences(self, other: "ObservationSpec") -> dict[str, tuple[int, int]]:
        mine = self.to_dict()
        theirs = other.to_dict()
        return {
            key: (mine[key], theirs[key])
            for key in mine
            if mine[key] != theirs[key]
        }

    def is_thruster_feedback_upgrade_from(self, older: "ObservationSpec") -> bool:
        """是否仅从 v1 增加了 4 维实际推进器反馈。"""
        return older.differences(self) == {
            "schema_version": (1, 2),
            "thruster_state_dim": (0, _THRUSTER_STATE_OBS_DIM),
        }


DEFAULT_OBSERVATION_SPEC = ObservationSpec()
LEGACY_OBSERVATION_SPEC_V1 = ObservationSpec(
    schema_version=1,
    thruster_state_dim=0,
)

# Dimension constants remain available for direct imports. New code should
# receive an ObservationSpec instance instead.
_OWN_CONTEXT_DIM = DEFAULT_OBSERVATION_SPEC.own_context_dim
_OWN_OBS_DIM = DEFAULT_OBSERVATION_SPEC.own_dim
_ATTENTION_OBS_DIM = DEFAULT_OBSERVATION_SPEC.attention_dim

# --------  global-state dimension constants  --------
_GLOBAL_SHIP_DIM = 2
_GLOBAL_PER_TUG_DIM = 17
_GLOBAL_ACCEL_PER_TUG_DIM = 3

# -------- normalization scales (shared between observer and global state) --------
_TUG_LINEAR_ACCEL_SCALE = 1.0
_TUG_YAW_ACCEL_SCALE = 0.1
_SHIP_LINEAR_ACCEL_SCALE = 0.2
