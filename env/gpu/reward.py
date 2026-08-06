"""Batch reward via CPU ``FormationRewardComputer`` on synced snapshots (parity-first).

Dynamics run on GPU; reward formulas stay identical to ``FormationEnv`` by
reusing the oracle computer after ``push_env_from_gpu``.
"""

from __future__ import annotations

import numpy as np

from env.formation_env import FormationEnv


def compute_rewards_for_env(
    env: FormationEnv,
    actions: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Compute dense rewards for one env after dynamics have been synced in."""
    slot_world = env.ship.slot_positions_world()
    state = env._build_sim_state()
    return env._reward.compute_rewards(state, env._episode, actions, slot_world)
