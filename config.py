"""集中管理强化学习训练所需的全部超参数。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


# ---------- 环境参数 ----------
@dataclass
class EnvConfig:
    # 仿真
    dt_ctrl: float = 0.2
    max_episode_steps: int = 2000
    n_tugs: int = 4

    # 大船（固定）
    ship_length_m: float = 200.0
    ship_beam_m: float = 30.0
    ship_size_randomize: bool = False  # kept for CLI compat with train.py until train WIP lands
    ship_speed_min: float = 1.0
    ship_speed_max: float = 1.0
    ship_yaw_rate_max: float = 0.0
    ship_speed_tau_s: float = 15.0
    ship_yaw_tau_s: float = 20.0
    ship_target_resample_min_s: float = 20.0
    ship_target_resample_max_s: float = 40.0
    slot_lon_offset_m: float = 20.0
    slot_lat_offset_m: float = 25.0

    # 初始化（圆环半径；远距复现可用 --init-radius 200）
    tug_init_mode: str = "circle"
    tug_init_radius_m: float = 100.0

    # 到位判定（Capture）与持续跟随（Track）
    # Capture：全体同时 in-zone 满 hold_time_s → 发捕获奖励，进入 Track，不结束
    # Track：在 Capture 后再连续 in-zone 满 track_horizon_s → success 终止
    pos_tol_m: float = 10.0
    heading_tol_rad: float = math.radians(30.0)
    speed_tol_ms: float = 3.0
    hold_time_s: float = 2.0
    track_horizon_s: float = 30.0

    # 安全
    tug_collision_dist_m: float = 20.0
    ship_collision_dist_m: float = 6.0

    # ---------- 奖励 ----------
    # R = w_dist*R_dist + w_hold*R_hold + w_vel*R_vel - w_coll*P_coll
    #     - w_stall*P_stall + R_shape + R_team
    reward_dist_w: float = 3.0
    reward_hold_w: float = 2.0
    reward_velocity_w: float = 0.0
    reward_collision_w: float = 1.0
    reward_collision_cap: float = 2.0
    reward_stall_w: float = 0.5

    # 距离
    reward_dist_progress_clip_m: float = 5.0
    reward_dist_progress_frac: float = 0.7
    reward_dist_scale_m: float = 500.0
    reward_hold_start_m: float = 150.0
    reward_hold_full_m: float = 20.0

    # 碰撞
    reward_collision_ship_safe_m: float = 80.0
    reward_collision_tug_safe_m: float = 120.0
    reward_cpa_horizon_s: float = 60.0
    reward_collision_cpa_w: float = 2.0

    # 进槽走廊软化（仅船软碰）
    reward_corridor_half_width_m: float = 40.0
    reward_corridor_axial_slack_m: float = 30.0
    reward_ship_soft_min_scale: float = 0.15

    # 停滞（防外围刷分）
    reward_stall_window_s: float = 5.0
    reward_stall_min_progress_m: float = 2.0
    reward_stall_floor: float = 0.2

    # 势函数 shaping
    reward_shape_w: float = 0.3
    reward_shape_gamma: float = 0.99
    reward_shape_d_ref_m: float = 200.0
    reward_shape_clip: float = 1.0

    # 团队同步（多艇用）
    reward_team_w: float = 0.2
    reward_team_softmin_beta: float = 4.0

    # 终端
    reward_arrival_bonus: float = 80.0
    reward_collision_pen: float = 80.0
    reward_collision_pen_culprit: float = 80.0
    reward_collision_pen_bystander: float = 15.0

    # 观测
    obs_history_k: int = 3
    obs_ship_preview_times_s: tuple[float, float, float] = (5.0, 10.0, 15.0)


REWARD_PRESETS: dict[str, dict[str, float]] = {
    "rw_baseline": {},
    "rw_dist_up": {"reward_dist_w": 6.0},
    "rw_ship_safe_dn": {"reward_collision_ship_safe_m": 60.0},
    "rw_coll_soft": {
        "reward_collision_w": 0.5,
        "reward_collision_cpa_w": 1.0,
    },
    "rw_shape_up": {"reward_shape_w": 0.8},
    "rw_combo": {
        "reward_dist_w": 6.0,
        "reward_collision_ship_safe_m": 60.0,
    },
}


def list_reward_presets() -> list[str]:
    return sorted(REWARD_PRESETS)


def apply_reward_preset(env_cfg: EnvConfig, preset_id: str | None) -> str | None:
    if preset_id is None:
        return None
    key = str(preset_id).strip()
    if not key:
        return None
    if key not in REWARD_PRESETS:
        known = ", ".join(list_reward_presets())
        raise ValueError(f"Unknown reward preset {key!r}. Known: {known}")
    for field_name, value in REWARD_PRESETS[key].items():
        setattr(env_cfg, field_name, value)
    return key


# ---------- PPO ----------
@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.98
    clip_eps: float = 0.2
    value_clip_eps: float = 0.2
    entropy_coef: float = 0.005
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.015

    rollout_steps: int = 512
    num_envs: int = 8
    minibatch_size: int = 1024
    update_epochs: int = 4

    learning_rate: float = 1e-4
    lr_anneal: bool = True
    lr_min_factor: float = 0.05
    total_steps: int = 5_000_000

    # Actor 时序架构：mlp | transformer（gru/lstm 预留）
    actor_arch: str = "mlp"
    tf_d_model: int = 64
    tf_nhead: int = 4
    tf_num_layers: int = 2
    tf_ffn_dim: int = 128
    tf_dropout: float = 0.0

    log_interval: int = 1
    save_interval: int = 25
    eval_interval: int = 10
    eval_episodes: int = 64
    device: str = "cpu"
    seed: int = 42


# ---------- 可视化 ----------
@dataclass
class VizConfig:
    meters_per_pixel: float = 0.6
    follow_ship: bool = True
    show_thrust: bool = True
