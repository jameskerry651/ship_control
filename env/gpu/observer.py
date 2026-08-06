"""Batch observations / global state via CPU ``Observer`` (parity-first)."""

from __future__ import annotations

import numpy as np

from env.formation_env import FormationEnv
from env.observer import Observer


def build_obs_for_env(env: FormationEnv) -> np.ndarray:
    state = env._build_sim_state()
    return Observer.build_obs(
        state, env.motion_history, env.action_history, env.observation_spec
    )


def get_global_state_for_env(env: FormationEnv) -> np.ndarray:
    return env.get_global_state()
