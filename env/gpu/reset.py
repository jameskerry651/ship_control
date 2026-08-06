"""Reset helpers: reuse CPU ``FormationEnv.reset``, then upload to GPU batch."""

from __future__ import annotations

import numpy as np

from env.formation_env import FormationEnv
from env.gpu.state import GpuEnvBatch, pull_env_to_gpu


def reset_env(
    env: FormationEnv,
    batch: GpuEnvBatch,
    env_idx: int,
    seed: int | None = None,
) -> np.ndarray:
    obs = env.reset(seed=seed)
    pull_env_to_gpu(env, batch, env_idx)
    return obs
