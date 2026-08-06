"""Capture → Track 阶段：入位后不立刻成功终止，需持续跟随满 track_horizon。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from config import EnvConfig
from env.formation_env import FormationEnv
from physics.tugboat_dynamics_model import Vec3


def _cfg(**kwargs) -> EnvConfig:
    base = dict(
        n_tugs=4,
        dt_ctrl=0.2,
        hold_time_s=1.0,  # 5 steps
        track_horizon_s=2.0,  # 10 steps after capture
        max_episode_steps=500,
        ship_speed_min=1.0,
        ship_speed_max=1.0,
        ship_yaw_rate_max=0.0,
        reward_arrival_bonus=80.0,
    )
    base.update(kwargs)
    return EnvConfig(**base)


def _place_all_in_slots(env: FormationEnv) -> None:
    """把全部拖轮放到 slot 上，速度/航向对齐大船，便于立刻 in-zone。"""
    slot_world = env.ship.slot_positions_world()
    cs = math.cos(env.ship.psi)
    sn = math.sin(env.ship.psi)
    # 船体坐标速度 ≈ (u, 0)；转到世界系给拖轮 body 速度近似
    for i, tug in enumerate(env.tugs):
        slot = slot_world[env.tug_to_slot[i]]
        tug.set_state(
            Vec3(float(slot[0]), float(slot[1]), float(slot[2])),
            Vec3(float(env.ship.u), 0.0, 0.0),
        )
    # 清零 in_zone，让奖励逻辑重新累计
    env.in_zone_steps[:] = 0


def _zero_actions(env: FormationEnv) -> np.ndarray:
    return np.zeros((env.n_tugs, env.action_dim), dtype=np.float32)


def test_capture_does_not_terminate_immediately() -> None:
    env = FormationEnv(cfg=_cfg(), seed=0)
    env.reset(seed=0)
    _place_all_in_slots(env)

    hold_steps = int(round(env.cfg.hold_time_s / env.cfg.dt_ctrl))
    capture_seen = False
    for _ in range(hold_steps + 2):
        _, rew, done, info = env.step(_zero_actions(env))
        if info.get("capture"):
            capture_seen = True
            assert not bool(done.any()), "capture 时不应结束 episode"
            assert info.get("phase") == "track"
            assert not info.get("success")
            # 捕获奖励只应在该步出现
            assert float(np.min(info["terminal_reward"])) >= env.cfg.reward_arrival_bonus - 1e-5
            break
    assert capture_seen, "应在 hold_time 后触发 capture"


def test_success_requires_track_horizon_after_capture() -> None:
    env = FormationEnv(cfg=_cfg(), seed=1)
    env.reset(seed=1)
    _place_all_in_slots(env)

    hold_steps = int(round(env.cfg.hold_time_s / env.cfg.dt_ctrl))
    track_steps = int(round(env.cfg.track_horizon_s / env.cfg.dt_ctrl))
    success_at = None
    for t in range(hold_steps + track_steps + 5):
        _, _, done, info = env.step(_zero_actions(env))
        if info.get("success"):
            success_at = t + 1
            assert bool(done.any())
            assert info.get("capture")
            assert info.get("phase") == "track"
            break
    assert success_at is not None
    # 成功步数应接近 capture + track（允许 in_zone 计数从第 1 步开始）
    assert success_at >= hold_steps + track_steps


def test_leaving_zone_during_track_delays_success() -> None:
    env = FormationEnv(cfg=_cfg(track_horizon_s=2.0, hold_time_s=1.0), seed=2)
    env.reset(seed=2)
    _place_all_in_slots(env)

    hold_steps = int(round(env.cfg.hold_time_s / env.cfg.dt_ctrl))
    # 先跑到 capture
    for _ in range(hold_steps + 1):
        _, _, done, info = env.step(_zero_actions(env))
        if info.get("capture") and not info.get("success"):
            break
    else:
        pytest.fail("未能进入 capture")

    assert not bool(done.any())

    # 把一艘拖轮拖离 slot，打断连续 in-zone
    tug0 = env.tugs[0]
    tug0.set_state(
        Vec3(tug0.eta.x + 80.0, tug0.eta.y + 80.0, tug0.eta.z),
        Vec3(0.0, 0.0, 0.0),
    )
    env.in_zone_steps[0] = 0

    _, _, done, info = env.step(_zero_actions(env))
    assert not info.get("success")
    assert not bool(done.any())
    assert info.get("capture") is True  # 已捕获过，不撤销
    assert info.get("phase") == "track"
