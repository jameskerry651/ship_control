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
    ship_size_randomize: bool = True
    ship_length_min_m: float = 180.0
    ship_length_max_m: float = 240.0
    ship_beam_min_m: float = 26.0
    ship_beam_max_m: float = 40.0

    # 大船速度/航向变化范围
    ship_speed_min: float = 0.5
    ship_speed_max: float = 2.0
    ship_yaw_rate_max: float = 0.02       # rad/s
    ship_speed_tau_s: float = 15.0        # 速度变化的一阶时间常数
    ship_yaw_tau_s: float = 20.0          # 航向角速度变化的一阶时间常数
    ship_target_resample_min_s: float = 20.0
    ship_target_resample_max_s: float = 40.0

    # Slot 在大船船体系下的偏移（船首/船尾 + 左舷/右舷）
    slot_lon_offset_m: float = 30.0       # slot 距船首/船尾的纵向距离
    slot_lat_offset_m: float = 35.0       # slot 距船舷的横向净距；缓解 stern/starboard final-slot 船体碰撞

    # 拖轮初始位置模式：
    # - astern_approach：真实任务主模式，拖轮从大船船尾后方带初速追赶，再绕行就位
    # - mixed_slot_approach：鲁棒性课程；随机 1/2/3 艘已合理就位，其余从路线周围随机绕行
    tug_init_mode: str = "astern_approach"

    # astern_approach：初始点在船尾后方、左右舷航道上，单位都是大船船体系。
    # 同侧两艘拖轮分层：船尾 slot 走内侧/更靠前，船首 slot 走外侧/更靠后，
    # 避免开局和第一个 stern gate 共线汇聚造成拖轮互撞。
    tug_init_astern_stern_dist_min_m: float = 120.0
    tug_init_astern_stern_dist_max_m: float = 180.0
    tug_init_astern_bow_dist_min_m: float = 260.0
    tug_init_astern_bow_dist_max_m: float = 380.0
    tug_init_astern_lateral_jitter_m: float = 6.0
    tug_init_pair_min_dist_m: float = 110.0
    tug_init_speed_boost_min_ms: float = 0.1     # 初始速度比大船快的范围；过大会诱导远距离高速追赶
    tug_init_speed_boost_max_ms: float = 0.45
    tug_init_heading_noise_rad: float = math.radians(12.0)
    tug_init_sway_noise_ms: float = 0.08
    tug_init_yaw_rate_noise_rads: float = 0.01
    tug_init_forward_action: float = 0.25        # 初始推进器前进指令，避免"有速度但油门为零"

    # mixed_slot_approach：部分拖轮已经以接近 slot 保持状态起步，剩余拖轮从
    # 船尾/舷侧路线不同阶段随机起步，用于提升策略对非统一初态的鲁棒性。
    # ready_count=3 时，剩余 1 艘会被放到目标 slot 的对侧船尾外侧，迫使其学习绕行。
    tug_init_mixed_ready_counts: tuple[int, ...] = (1, 2, 3)
    tug_init_mixed_pair_min_dist_m: float = 120.0
    tug_init_ready_outward_offset_m: float = 18.0
    tug_init_ready_pos_jitter_m: float = 2.0
    tug_init_ready_heading_noise_rad: float = math.radians(5.0)
    tug_init_ready_speed_noise_ms: float = 0.12
    tug_init_ready_sway_noise_ms: float = 0.03
    tug_init_ready_yaw_rate_noise_rads: float = 0.004
    tug_init_ready_forward_action: float = 0.22
    tug_init_mixed_route_longitudinal_jitter_m: float = 18.0
    tug_init_mixed_route_lateral_jitter_m: float = 20.0
    tug_init_mixed_opposite_stern_dist_min_m: float = 220.0
    tug_init_mixed_opposite_stern_dist_max_m: float = 420.0
    tug_init_mixed_opposite_lateral_extra_m: float = 35.0

    # astern_approach 路线：默认用船体系 visibility planner 生成 waypoint；
    # 设为 "manual" 使用手写几何模板，便于 ablation。
    route_planner: str = "visibility"
    route_bow_lane_lat_m: float = 100.0
    route_stern_lane_lat_m: float = 60.0
    route_stern_gate_dist_m: float = 60.0
    route_waypoint_tol_m: float = 35.0
    route_lane_min_lat_m: float = 32.0
    route_chase_speed_target_ms: float = 0.35
    route_chase_speed_max_ms: float = 0.9       # 非 final 阶段相对大船的软追赶速度上限
    route_tug_speed_soft_limit_ms: float = 3.0  # 非 final 阶段拖轮世界速度软上限，抑制远处满油门
    route_progress_step_clip_m: float = 0.45    # 每控制步最多按该距离计 route progress 奖励
    route_speed_governor: bool = False          # 可选安全层；强限幅会破坏当前策略到达节奏，默认关闭
    route_nonfinal_forward_action_cap: float = 0.45
    route_speed_governor_min_forward_action: float = 0.05
    route_speed_governor_cap_slope: float = 0.30
    route_tug_spacing_dist_m: float = 90.0
    route_hull_clearance_m: float = 18.0      # visibility planner 的膨胀船体硬避障距离
    route_outer_holding_extra_m: float = 18.0 # final approach 前保持在 slot 外侧的额外横距
    route_final_entry_lat_extra_m: float = 22.0
    route_final_entry_lon_offset_m: float = 12.0
    route_visibility_node_margin_m: float = 10.0
    route_min_waypoint_spacing_m: float = 2.0
    # v39: waypoint 路径泛化
    route_anchor_jitter: bool = True         # P0: 每 episode 对锚点加随机抖动
    route_anchor_jitter_m: float = 10.0      # P0: 抖动幅度基准 (m)，按船体尺寸等比例缩放
    route_spline_smooth: bool = True         # P1: B-spline 平滑 + 等距重采样
    route_stagger: bool = True               # P2: 同舷船首/船尾 stern_gate 错峰
    route_stagger_dist_m: float = 50.0       # P2: 错峰距离 (m)，正=船首靠前, 负=船尾靠后

    # 到位判定阈值（v20 进一步放宽：让首批 succ 信号能出现）
    pos_tol_m: float = 60.0               # 50m → 60m，不少拖轮卡在 50-60m 区间
    heading_tol_rad: float = math.radians(30.0)
    speed_tol_ms: float = 3.0             # 2.0 → 3.0，靠近时 speed_err 经常卡在 2 边缘
    hold_time_s: float = 10.0               # curriculum: 5s -> 10s -> 20s -> 30s 逐步提升稳定伴航时间

    # 安全距离
    tug_collision_dist_m: float = 20.0
    ship_collision_dist_m: float = 6.0

    # ---------- 奖励权重 ----------
    reward_progress_w: float = 0.05       # 从 0.02 提高到 0.05，增强靠近激励
    reward_route_progress_w: float = 0.06
    reward_lane_w: float = 0.2
    reward_chase_speed_w: float = 0.08
    reward_chase_overspeed_w: float = 0.05  # v57: 0.18→0.05，audit 显示 quadratic 项每步 -2，长 ep 累积 -1200 完全主导 dense reward
    reward_route_speed_limit_w: float = 0.02  # v57: 0.08→0.02，同上
    reward_spacing_w: float = 0.25
    reward_cpa_w: float = 0.18
    reward_cpa_final_multiplier: float = 2.0
    reward_cpa_max_penalty: float = 0.75
    reward_heading_w: float = 0.1
    reward_smooth_w: float = 0.1
    reward_jerk_w: float = 0.05
    reward_mag_w: float = 0.01
    reward_yaw_rate_w: float = 0.1
    reward_speed_match_w: float = 0.15    # v25 P3: 0.2→0.15，配合 tanh 压缩降低主导性
    reward_escort_w: float = 0.5             # v38: 伴航累进奖励，替代乘积 r_zone + arrival_bonus
    reward_escort_final_multiplier: float = 1.5  # final stage 加强持续伴航，避免只短暂进入 slot
    reward_arrival_bonus: float = 80.0       # v57: 30→80，拉开 success vs timeout 累积 reward 差距（v50 audit 显示 timeout cum +238 > success cum +15）
    reward_collision_pen: float = 20.0    # v4 原值（碰撞终止）
    reward_safety_w: float = 0.3
    ship_safety_dist_m: float = 18.0      # 非 final 阶段：3 倍船体碰撞阈值开始预警
    ship_safety_final_dist_m: float = 40.0
    reward_hull_safety_final_multiplier: float = 3.0

    # 预测船体安全奖励：在短时未来窗口里检查 tug 轨迹与船体的最小距离，
    # 让策略更早学会避开"看起来还没撞、但几秒后会贴船"的高速切入。
    ship_future_safety_horizon_s: float = 14.0
    ship_future_safety_samples: int = 5
    ship_future_safety_dist_m: float = 26.0
    ship_future_safety_final_dist_m: float = 45.0
    reward_ship_future_safety_w: float = 0.25
    reward_ship_future_safety_final_multiplier: float = 1.8
    reward_ship_future_safety_max_penalty: float = 0.9

    # v52: 简化 staged reward。保留上方各原子项用于诊断，但训练目标改为：
    # - 非 final route stage：route progress - overspeed/safety/lane/action risk
    # - final stage：escort/hold - hull/tug/action risk
    reward_use_simple_stage: bool = True
    reward_simple_nonfinal_progress_w: float = 1.0
    reward_simple_final_escort_w: float = 1.0
    reward_simple_hold_w: float = 0.2
    reward_simple_route_chase_w: float = 0.5
    reward_simple_final_speed_match_w: float = 0.25
    reward_simple_speed_risk_w: float = 1.0
    reward_simple_safety_risk_w: float = 1.0
    reward_simple_hull_risk_cap: float = 1.4
    reward_simple_tug_risk_cap: float = 1.2
    reward_simple_safety_risk_cap: float = 1.8
    reward_simple_lane_w: float = 1.0
    reward_simple_action_w: float = 1.0

    # CPA 风险奖励参数：用于把 v32 的 CPA 观测真正接入训练目标。
    # DCPA 小且 TCPA 短时给连续惩罚；final approach 阶段用 multiplier 加强。
    cpa_alert_dist_m: float = 70.0
    cpa_time_horizon_s: float = 45.0

    # 是否在观察中加入"其他拖轮相对位置"
    obs_include_other_tugs: bool = True
    # 是否在观察中加入 slot one-hot（4 维）
    obs_include_slot_onehot: bool = True
    # 是否加入路线 waypoint/stage 特征
    obs_include_route: bool = True

    # 是否在观察中加入 CPA 特征（v32）
    obs_include_cpa: bool = True
    # 是否在观察中加入 拖轮→大船 CPA 特征（v34）
    obs_include_cpa_ship: bool = True
    # 是否在观察中加入大船尺度特征（length/beam 相对基准尺度的偏差；v36）
    obs_include_ship_size: bool = True
    # 是否在 actor 观察中加入自身体系加速度（u_dot/v_dot/r_dot；v58）
    obs_include_ego_accel: bool = True


# ---------- PPO 网络与训练参数 ----------
@dataclass
class PPOConfig:
    # Actor-Critic 网络结构
    hidden_dims: tuple[int, ...] = (256, 256)
    critic_hidden_dims: tuple[int, ...] = (512, 512, 512)   # 集中式 critic 更深更宽，处理 121 维 canonical global state
    activation: str = "tanh"
    log_std_init: float = -0.5            # 初始 log std，对应 std≈0.6

    # PPO 超参数
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_clip_eps: float = 0.2
    entropy_coef: float = 0.01            # entropy 系数：0.005 太低易早收敛，0.02 太高阻止收敛
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.03               # 早停的 KL 阈值

    # 数据收集
    rollout_steps: int = 256              # 每个并行环境每次 rollout 收集的步数
    num_envs: int = 8                     # 并行环境数（顺序执行的 vector env）
    minibatch_size: int = 1024            # 一次梯度更新的 mini-batch 大小
    update_epochs: int = 8                # 一份 rollout 数据上的更新轮数

    # 优化器
    learning_rate: float = 3e-4
    lr_anneal: bool = True                # 线性退火到 0

    # 总训练量
    total_steps: int = 5_000_000          # 全局环境步数（含所有 envs 与所有 tugs）

    # 日志/保存
    log_interval: int = 1
    save_interval: int = 25               # 每 N 次 update 保存一次最近权重
    eval_interval: int = 10               # 每 N 次 update 跑一次评估
    eval_episodes: int = 64               # v56: 16→64，best 选择更可靠（v55 eval 抖动剧烈）

    # 设备
    device: str = "cpu"                   # 一般 CPU 比 MPS 快（小网络）

    # 训练种子
    seed: int = 42


# ---------- 可视化参数 ----------
@dataclass
class VizConfig:
    window_w: int = 1280
    window_h: int = 800
    fps: int = 30
    meters_per_pixel: float = 0.6         # 缩放：1 像素对应多少米（小=放大）
    follow_ship: bool = True              # 视角是否跟随大船
    show_trail: bool = True
    trail_max_len: int = 800
    show_thrust: bool = True              # 是否绘制推进器力矢量
