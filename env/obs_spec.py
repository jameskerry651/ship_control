"""观测空间与全局状态的维度常量 —— 单一真相源。

所有维度常量集中定义在此，由 ``env/observer.py``、``rl/actor.py``、``env/formation_env.py``
等模块共同导入，从根源消除常量不一致风险。

观测结构总览 (93 维 / agent)::

    ┌─────────────────────────────────────────────────────────────┐
    │  历史帧（4 帧 × 6 维）  = 24 维  (_EGO_MOTION_OBS_DIM)      │
    │  动作历史（4 帧 × 4 维）= 16 维  (_ACTION_HISTORY_OBS_DIM)  │
    │  大船相对状态              =  5 维  (_SHIP_REL_OBS_DIM)      │
    │  大船预瞄点（3 点 × 2 维）=  6 维  (_SHIP_PREVIEW_POINT_DIM)│
    │  目标槽位                  =  5 维  (_SLOT_TARGET_OBS_DIM)   │
    │  路径目标                  =  4 维  (_ROUTE_TARGET_OBS_DIM)  │
    │  船体间隙                  =  3 维  (_HULL_CLEARANCE_OBS_DIM)│
    │  邻居特征（3 邻 ×10 维）  = 30 维  (_NEIGHBOR_OBS_DIM)      │
    └─────────────────────────────────────────────────────────────┘

全局状态结构 (90 维)::

    ┌─────────────────────────────────────────────────────────────┐
    │  大船状态                  =  2 维  (_GLOBAL_SHIP_DIM)       │
    │  每艇状态（4 × 19 维）     = 76 维  (_GLOBAL_PER_TUG_DIM)   │
    │  每艇加速度（4 × 3 维）    = 12 维  (_GLOBAL_ACCEL_PER_TUG_DIM)│
    └─────────────────────────────────────────────────────────────┘
"""

# -------- action dimension --------
ACTION_DIM: int = 4

# -------- per-agent observation sub-dimensions --------
_EGO_MOTION_OBS_DIM = 6
_ACTION_HISTORY_OBS_DIM = ACTION_DIM
_SHIP_REL_OBS_DIM = 5
_SHIP_PREVIEW_POINT_DIM = 2
# 本 agent 目标槽位（在自身坐标系下）的相对量：[dx/100, dy/100, dist, sin(dψ), cos(dψ)]
# 关键：actor 4 个 agent 参数共享，但每个 tug 的目标槽位不同。若 own_obs 不含目标槽位，
# 共享策略在相近位姿下会输出相近动作、把多船挤向同一区域，碰撞率随训练上升。
# 槽位 one-hot 之前只喂给 critic（get_global_state），actor 看不到，故必须在此补上。
_SLOT_TARGET_OBS_DIM = 5
# 当前路径目标和进度：[route_dx/100, route_dy/100, stage_norm, remaining/500]
_ROUTE_TARGET_OBS_DIM = 4
# 最近船体边界向量与距离：[closest_hull_dx/50, closest_hull_dy/50, d_hull/50]
_HULL_CLEARANCE_OBS_DIM = 3
_NEIGHBOR_COUNT = 3
# 单个邻居观测维度：从纯几何状态升级为碰撞风险特征
# [dx, dy, distance, sin(bearing), cos(bearing), du, dv, range_rate, tcpa, dcpa]
_NEIGHBOR_OBS_DIM = 10

# TCPA（最近会遇时刻）归一化尺度，单位秒；超过该窗口的会遇被视为低风险
_NEIGHBOR_TCPA_SCALE_S = 60.0
# 相对速度模长下限，低于此值认为两船相对静止，TCPA/DCPA 退化为当前距离
_NEIGHBOR_REL_SPEED_EPS = 1e-6

# -------- actor network dimension constants (derived from observation specs) --------
# 本船历史观测维度（不含邻居信息）：
# motion(4帧×6=24) + action(4帧×4=16) + ship_rel(5) + preview(3×2=6)
# + slot_target(5) + route_target(4) + hull_clearance(3) = 63
# 与 env/observer.py 的 build_obs 组装顺序严格对应
_OWN_OBS_DIM = 63
# 邻居观测总维度，供 attention 模块输入
_ATTENTION_OBS_DIM = _NEIGHBOR_COUNT * _NEIGHBOR_OBS_DIM  # = 30

# --------  global-state dimension constants  --------
_GLOBAL_SHIP_DIM = 2
_GLOBAL_PER_TUG_DIM = 19
_GLOBAL_ACCEL_PER_TUG_DIM = 3

# -------- normalization scales (shared between observer and global state) --------
_TUG_LINEAR_ACCEL_SCALE = 1.0
_TUG_YAW_ACCEL_SCALE = 0.1
_SHIP_LINEAR_ACCEL_SCALE = 0.2
