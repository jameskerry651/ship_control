"""拖轮编队环境的初始化逻辑。

将所有拖轮随机放置在大船周围半径为 tug_init_radius_m 的圆环上。
所有拖轮初始速度 / 转速 / 方位角均为 0。
"""

from __future__ import annotations

import math

import numpy as np

from config import EnvConfig
from env.obs_spec import ACTION_DIM


def sample_tug_init_states(
    rng: np.random.Generator,
    n_tugs: int,
    ship_x: float,
    ship_y: float,
    cfg: EnvConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """在大船周围半径为 tug_init_radius_m 的圆环上均匀随机放置拖轮。

    返回:
        positions: (n_tugs, 2) 世界坐标
        psis: (n_tugs,) 航向角，均为 0
        nus: (n_tugs, 3) 船体系速度 (u, v, r)，均为 0
        actions: (n_tugs, 4) 初始动作 [port_rpm, stbd_rpm, port_az, stbd_az]，均为 0
    """
    radius = float(getattr(cfg, "tug_init_radius_m", 200.0))
    positions = np.zeros((n_tugs, 2), dtype=np.float64)
    for i in range(n_tugs):
        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        positions[i, 0] = ship_x + radius * math.cos(angle)
        positions[i, 1] = ship_y + radius * math.sin(angle)
    psis = np.zeros(n_tugs, dtype=np.float64)
    nus = np.zeros((n_tugs, 3), dtype=np.float64)
    actions = np.zeros((n_tugs, ACTION_DIM), dtype=np.float32)
    return positions, psis, nus, actions
