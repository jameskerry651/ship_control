"""集中管理强化学习训练所需的全部超参数。

把训练、环境、网络、可视化四类参数分开放，调参时只改这一处。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


# ---------- 环境参数 ----------
@dataclass
class EnvConfig:
    # 仿真步长与回合长度
    dt_ctrl: float = 0.2          # 控制周期（秒）
    max_episode_steps: int = 1500  # 单回合最大步数，300 秒；先稳定 hold=10s，再逐步拉长 episode

    # 拖轮数量
    n_tugs: int = 4

    # 大船尺度（典型集装箱船）
    ship_length_m: float = 200.0
    ship_beam_m: float = 30.0
    ship_size_randomize: bool = False
    ship_length_min_m: float = 180.0
    ship_length_max_m: float = 240.0
    ship_beam_min_m: float = 26.0
    ship_beam_max_m: float = 40.0

    # 大船速度/航向变化范围
    ship_speed_min: float = 0.5
    ship_speed_max: float = 1.0
    ship_yaw_rate_max: float = 0.02       # rad/s
    ship_speed_tau_s: float = 15.0        # 速度变化的一阶时间常数
    ship_yaw_tau_s: float = 20.0          # 航向角速度变化的一阶时间常数
    ship_target_resample_min_s: float = 20.0
    ship_target_resample_max_s: float = 40.0

    # Slot 在大船船体系下的偏移（船首/船尾 + 左舷/右舷）
    slot_lon_offset_m: float = 20.0       # slot 距船首/船尾的纵向距离
    slot_lat_offset_m: float = 25.0       # slot 距船舷的横向净距；缓解 stern/starboard final-slot 船体碰撞

    # 拖轮初始位置模式：当前仅保留 mixed_slot_approach。
    # 它覆盖 0/1/2/3 艘拖轮已就位，以及未就位拖轮从多种安全区域起步。
    tug_init_mode: str = "mixed_slot_approach"

    # 未就位拖轮在船尾后方区域（rear_lane）采样时的纵向距离（大船船体系，按目标 slot 分层）。
    # 船尾 slot 起得更近，船首 slot 起得更远，减少同侧拖轮开局共线汇聚。
    tug_init_rear_stern_slot_dist_min_m: float = 80.0   # 目标 slot 为船尾：距船尾端面纵向距离下限（米）
    tug_init_rear_stern_slot_dist_max_m: float = 200.0   # 目标 slot 为船尾：距船尾端面纵向距离上限（米）
    tug_init_rear_bow_slot_dist_min_m: float = 260.0       # 目标 slot 为船首：距船尾端面纵向距离下限（米）
    tug_init_rear_bow_slot_dist_max_m: float = 380.0       # 目标 slot 为船首：距船尾端面纵向距离上限（米）
    tug_init_speed_boost_min_ms: float = 0.1               # 未就位拖轮纵向速度相对大船的下限增益（m/s）
    tug_init_speed_boost_max_ms: float = 0.45              # 同上上限；过大易诱发远距离高速追赶
    tug_init_heading_noise_rad: float = math.radians(3.0) # 初始航向相对速度方向的随机偏差（弧度）
    tug_init_sway_noise_ms: float = 0.0                   # 船体系横向速度扰动幅度（m/s）
    tug_init_yaw_rate_noise_rads: float = 0.0             # 初始艏摇角速度扰动幅度（rad/s）
    tug_init_forward_action: float = 0.25                  # 初始推进器前进指令，避免"有速度但油门为零"

    # mixed_slot_approach：0/1/2/3 艘拖轮已经以接近 slot 保持状态起步，
    # 剩余拖轮从船尾、舷侧、目标 slot 外侧或对侧船尾等区域随机起步，
    # 用于提升策略对非统一初态的鲁棒性。
    tug_init_mixed_ready_counts: tuple[int, ...] = (0, 1, 2, 3)  # reset 时随机抽取的已就位拖轮数量候选
    tug_init_mixed_pair_min_dist_m: float = 120.0              # 初态采样时拖轮间最小间距（米）；与 2×tug_collision_dist 取较大值
    tug_init_ready_outward_offset_m: float = 18.0              # 已就位拖轮相对目标 slot 向舷外侧基准偏移（船体系横向，米）
    tug_init_ready_pos_jitter_m: float = 2.0                   # 已就位拖轮在 slot 附近的平面位置随机扰动幅度（米）
    tug_init_ready_heading_noise_rad: float = math.radians(5.0) # 已就位拖轮航向相对跟踪方向的随机偏差（弧度）
    tug_init_ready_speed_noise_ms: float = 0.12                  # 已就位拖轮纵向速度相对大船的扰动幅度（m/s）
    tug_init_ready_sway_noise_ms: float = 0.03                   # 已就位拖轮船体系横向速度扰动幅度（m/s）
    tug_init_ready_yaw_rate_noise_rads: float = 0.004            # 已就位拖轮艏摇角速度扰动幅度（rad/s）
    tug_init_ready_forward_action: float = 0.22                  # 已就位拖轮初始前进油门；其 jitter 为 tug_init_action_jitter 的一半
    tug_init_mixed_route_longitudinal_jitter_m: float = 18.0   # 未就位拖轮在 rear/gate/opposite 等区域的纵向采样扰动（米）
    tug_init_mixed_route_lateral_jitter_m: float = 20.0        # 未就位拖轮在上述区域的横向采样扰动（米）
    tug_init_mixed_zones: tuple[str, ...] = (
        "rear_lane",
        "stern_gate",
        "side_lane",
        "outer_slot",
        "opposite_stern",
    )
    tug_init_force_single_opposite: bool = True
    tug_init_mixed_approach_speed_min_ms: float = 0.20         # 未就位拖轮朝 route 目标点缓慢接近时的速度下限（m/s）
    tug_init_mixed_approach_speed_max_ms: float = 0.80         # 同上上限；过大易开局远距离高速追赶
    tug_init_mixed_opposite_stern_dist_min_m: float = 220.0    # opposite_stern 区：距船尾端面纵向距离下限（米）
    tug_init_mixed_opposite_stern_dist_max_m: float = 420.0    # opposite_stern 区：距船尾端面纵向距离上限（米）
    tug_init_mixed_opposite_lateral_extra_m: float = 35.0      # opposite_stern 区：相对本侧航道横向额外外扩（米）
    tug_init_action_jitter: float = 0.04                       # 混合初态前进油门的均匀随机扰动幅度（未就位用全量，已就位用一半）

    # 每艘拖轮的 route：从 reset 后拖轮位置 A* 到目标 slot（船体系，绕开船体）。
    route_bow_lane_lat_m: float = 100.0
    route_stern_lane_lat_m: float = 60.0
    route_stern_gate_dist_m: float = 60.0
    route_waypoint_tol_m: float = 35.0
    route_lane_min_lat_m: float = 32.0
    route_chase_speed_max_ms: float = 0.9       # 非 final 阶段相对大船的软追赶速度上限
    route_tug_speed_soft_limit_ms: float = 3.0  # 非 final 阶段拖轮世界速度软上限，抑制远处满油门
    route_speed_governor: bool = False          # 可选安全层；强限幅会破坏当前策略到达节奏，默认关闭
    route_nonfinal_forward_action_cap: float = 0.45
    route_speed_governor_min_forward_action: float = 0.05
    route_speed_governor_cap_slope: float = 0.30
    route_outer_holding_extra_m: float = 18.0 # final approach 前保持在 slot 外侧的额外横距
    route_astar_cell_m: float = 4.0           # A* 网格分辨率 (m)
    route_astar_margin_m: float = 10.0        # A* 搜索区域相对起终点的额外边距
    route_astar_lane_penalty: float = 10_000.0  # 偏离同舷 lane 的软惩罚
    route_visibility_node_margin_m: float = 10.0  # 兼容旧配置名
    route_min_waypoint_spacing_m: float = 2.0
    route_num_waypoints: int = 12            # 每条 route 固定点数（弧长等距重采样）；便于 route_stage 归一化
    route_at_slot_skip_tol_m: float = 30.0   # 起终点距 slot 小于此值时不再绕路规划（避免已就位拖轮出现环形 route）
    route_spline_smooth: bool = True         # B-spline 平滑 + 等距重采样

    # 到位判定阈值
    pos_tol_m: float = 140.0              # 
    heading_tol_rad: float = math.radians(30.0)
    speed_tol_ms: float = 3.0             # 
    hold_time_s: float = 2.0              # curriculum: 1s→2s→5s 逐步提升

    # 安全距离
    tug_collision_dist_m: float = 20.0
    ship_collision_dist_m: float = 6.0

    # ---------- 奖励权重 ----------
    # Dense reward:
    # R = w1 * R_target + w2 * R_velocity - w3 * P_collision.
    reward_target_w: float = 1.0
    reward_velocity_w: float = 0.25
    reward_collision_w: float = 3.0
    # P5 每步 P_collision 上限：防单次近距 barrier（可叠加 ship+3 邻居+CPA）压垮接近梯度，
    # 缓解过度保守；硬碰撞威慑改由 P1 的按责终端 −80 承担。
    reward_collision_cap: float = 1.5
    # P2 团队同步项：以"最弱一艇"的在区度（softmin）为优化对象，给全员同一份稠密奖励，
    # 弥补"4 艘同时在区"只由稀疏终端 +80 驱动、稠密层缺同步梯度的问题。
    reward_team_w: float = 0.5
    # softmin 锐度：越大越接近 min（更聚焦最弱一艇），越小越接近均值。
    reward_team_softmin_beta: float = 4.0
    # P3 势函数式接近 shaping：F = γ·Φ(s') − Φ(s)，Φ = −(0.6·d/d_ref + 0.25·spd/spd_tol + 0.15·head/head_tol)。
    # 理论上策略不变、对 reward 归一化鲁棒、提供一路到底的稠密接近梯度，缓解过度保守（dist 卡在 145–242m）。
    reward_shape_w: float = 0.3
    reward_shape_gamma: float = 0.99
    reward_shape_d_ref_m: float = 200.0
    reward_shape_clip: float = 1.0
    # Strict-position curriculum uses this optional dense term to keep a
    # learnable gradient inside the near-slot region while the terminal
    # success threshold is tightened to 20 m. Default 0 preserves old runs.
    reward_precision_w: float = 0.0
    reward_precision_scale_m: float = 40.0
    # Optional near-slot shaping for strict curricula. Unlike r_hold, this does
    # not go to zero immediately outside pos_tol_m, so it can teach the policy
    # to move deterministic mean actions from 60 m toward a 20 m terminal gate.
    reward_near_hold_w: float = 0.0
    reward_near_hold_scale_m: float = 80.0

    # R_target：远场 chase + 近场 hold。
    reward_target_progress_clip_m: float = 1.5
    reward_chase_speed_target_ms: float = 0.8
    reward_hold_start_m: float = 140.0
    reward_hold_full_m: float = 20.0
    # 额外 route 辅助项：保留直接 slot progress 作为主目标，只在非 final
    # route 阶段小权重奖励沿规划路径减少 remaining distance。
    reward_route_w: float = 0.25
    # P4 hold 连续性鲁棒化：
    # - streak_decay：单帧违例时 in_zone_steps 减此值而非清零，容忍 ship 机动引起的瞬时闪断，
    #   让 2s 连续 hold 在多艇同步下可达；decay>1 仍能防"闪进闪出"刷成功。
    # - streak_w：按 in_zone_steps/hold_steps 的连续值给稠密奖励，直接激励"逼近并维持 2s hold"。
    reward_hold_streak_decay: int = 2
    reward_hold_streak_w: float = 0.3

    # R_velocity：近场强匹配大船线速度和角速度，远场只给弱约束。
    reward_velocity_gate_m: float = 120.0
    reward_velocity_speed_scale_ms: float = 3.0
    reward_velocity_yaw_scale_rads: float = 0.05

    # P_collision：当前距离 barrier + CPA 预判 barrier；硬碰撞仍由终端惩罚处理。
    reward_collision_ship_safe_m: float = 60.0
    reward_collision_tug_safe_m: float = 80.0
    ship_safety_dist_m: float = 18.0      # 初始化采样的最小船体安全距离。
    reward_cpa_horizon_s: float = 60.0    # CPA 风险预判窗口；窗口内越早会遇惩罚越强。
    reward_collision_cpa_w: float = 2.0   # CPA 风险在 P_collision 内的额外权重。

    # 终端信号：不参与 dense reward 归一化。
    # 碰撞惩罚与到达奖励同量级，避免策略学会"赌博式逼近"：
    # 当 pen << bonus 时，只要 P(到达) > P(碰撞)·pen/bonus 激进贴近就 EV 为正，
    # 碰撞率会随 chase 能力增强而单调上升。pen≈bonus 时撞船不再是可接受代价。
    reward_arrival_bonus: float = 80.0
    reward_collision_pen: float = 80.0
    # P1 终端碰撞惩罚按责分配：肇事 tug 重罚、其余 bystander 轻罚。
    # 现状"全员共担 −80"使 critic 无法归因（实测 EV_collision 为负、value_collision 爆炸），
    # advantage 噪声大、策略不稳。按责分配后无辜 tug 不再吃满惩罚，value 可学、梯度更干净。
    reward_collision_pen_culprit: float = 80.0
    reward_collision_pen_bystander: float = 15.0

    # Actor 观察：4 帧历史（当前 + 过去 3 帧）与 3 个大船中心未来前瞻点。
    obs_history_k: int = 3
    obs_ship_preview_times_s: tuple[float, float, float] = (5.0, 10.0, 15.0)


# ---------- PPO 网络与训练参数 ----------
@dataclass
class PPOConfig:
    # PPO 超参数
    gamma: float = 0.99
    gae_lambda: float = 0.98
    clip_eps: float = 0.2
    value_clip_eps: float = 0.2
    entropy_coef: float = 0.005           # entropy 系数，从 0.01 降低以加速策略收敛
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.015              # 更保守的 KL 早停阈值，抑制后期策略漂移

    # 数据收集
    rollout_steps: int = 512              # 每个并行环境每次 rollout 收集的步数，从 256 翻倍以覆盖更多完整 episode
    num_envs: int = 8                     # 并行环境数（顺序执行的 vector env）
    minibatch_size: int = 1024            # 一次梯度更新的 mini-batch 大小
    update_epochs: int = 4                # 降低重复更新次数，减少后期策略漂移

    # 优化器（torch.optim.lr_scheduler.CosineAnnealingLR）
    learning_rate: float = 1e-4
    lr_anneal: bool = True                # 是否启用余弦学习率退火
    lr_min_factor: float = 0.05           # eta_min = learning_rate * lr_min_factor

    # 总训练量
    total_steps: int = 5_000_000          # 全局环境步数（含所有 envs 与所有 tugs）

    # 日志/保存
    log_interval: int = 1
    save_interval: int = 25               # 每 N 次 update 保存一次最近权重
    eval_interval: int = 10               # 每 N 次 update 跑一次评估
    eval_episodes: int = 64               

    # 设备
    device: str = "cpu"                   # 一般 CPU 比 MPS 快（小网络）

    # 训练种子
    seed: int = 42


# ---------- 可视化参数 ----------
@dataclass
class VizConfig:
    meters_per_pixel: float = 0.6         # 缩放：1 像素对应多少米（小=放大）
    follow_ship: bool = True              # 视角是否跟随大船
    show_thrust: bool = True              # 是否绘制推进器力矢量
