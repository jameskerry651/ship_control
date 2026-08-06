"""Batch termination via CPU ``FormationEnv._check_termination`` (parity-first)."""

from __future__ import annotations

from typing import Any

import numpy as np

from env.formation_env import FormationEnv


def check_termination_for_env(env: FormationEnv) -> tuple[np.ndarray, dict[str, Any]]:
    slot_world = env.ship.slot_positions_world()
    return env._check_termination(slot_world)
