from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import EnvConfig
from env.formation_env import FormationEnv
from physics.tugboat_dynamics_model import Vec3


def _sync_prev_dist(env: FormationEnv) -> None:
    slot_world = env.ship.slot_positions_world()
    for i, tug in enumerate(env.tugs):
        slot = slot_world[env.tug_to_slot[i]]
        env.prev_dist[i] = math.hypot(tug.eta.x - slot[0], tug.eta.y - slot[1])


def test_four_term_reward_components_are_reported() -> None:
    env = FormationEnv(cfg=EnvConfig(), seed=10)
    env.reset()

    actions = np.zeros((env.n_tugs, env.action_dim), dtype=np.float32)
    _, reward, done, info = env.step(actions)

    assert reward.shape == (env.n_tugs,)
    assert done.shape == (env.n_tugs,)
    assert np.isfinite(reward).all()

    comp = info["reward_components"]
    expected = {
        "r_total",
        "r_target",
        "r_chase",
        "r_hold",
        "r_velocity",
        "r_control",
        "p_collision",
        "p_ship_collision",
        "p_tug_collision",
        "dist_to_slot",
        "heading_err_deg",
        "speed_err",
        "hull_dist",
        "chase_closing_speed",
        "hold_gate",
        "far_gate",
        "in_zone",
    }
    assert expected.issubset(comp)
    assert "r_simple" not in comp
    assert "phi" not in comp
    assert "r_cpa" not in comp


def test_target_reward_prefers_near_aligned_slot_state() -> None:
    env = FormationEnv(cfg=EnvConfig(), seed=11)
    env.reset()
    actions = np.zeros((env.n_tugs, env.action_dim), dtype=np.float32)
    slot_world = env.ship.slot_positions_world()
    slot = slot_world[env.tug_to_slot[0]]

    env.tugs[0].set_state(
        Vec3(float(slot[0] + 300.0), float(slot[1] + 300.0), float(slot[2] + math.pi)),
        Vec3(float(env.ship.u), float(env.ship.v), float(env.ship.r)),
    )
    _sync_prev_dist(env)
    _, info_far = env._reward.compute_rewards(actions, slot_world)
    far_target = float(info_far["reward_components"]["r_target"][0])

    env.tugs[0].set_state(
        Vec3(float(slot[0]), float(slot[1]), float(slot[2])),
        Vec3(float(env.ship.u), float(env.ship.v), float(env.ship.r)),
    )
    _sync_prev_dist(env)
    _, info_near = env._reward.compute_rewards(actions, slot_world)
    near_target = float(info_near["reward_components"]["r_target"][0])

    assert near_target > far_target


def test_control_reward_penalizes_action_changes() -> None:
    env = FormationEnv(cfg=EnvConfig(), seed=12)
    env.reset()
    slot_world = env.ship.slot_positions_world()
    env.last_actions[:] = 0.0
    env.last_action_changes[:] = 0.0

    zero_actions = np.zeros((env.n_tugs, env.action_dim), dtype=np.float32)
    large_actions = np.ones((env.n_tugs, env.action_dim), dtype=np.float32)

    _, info_zero = env._reward.compute_rewards(zero_actions, slot_world)
    _, info_large = env._reward.compute_rewards(large_actions, slot_world)

    zero_control = float(info_zero["reward_components"]["r_control"][0])
    large_control = float(info_large["reward_components"]["r_control"][0])
    assert large_control < zero_control


def test_far_field_chase_rewards_positive_closing_speed() -> None:
    env = FormationEnv(cfg=EnvConfig(), seed=15)
    env.reset()
    actions = np.zeros((env.n_tugs, env.action_dim), dtype=np.float32)
    slot_world = env.ship.slot_positions_world()
    slot = slot_world[env.tug_to_slot[0]]

    ship_vx = math.cos(env.ship.psi) * env.ship.u - math.sin(env.ship.psi) * env.ship.v
    ship_vy = math.sin(env.ship.psi) * env.ship.u + math.cos(env.ship.psi) * env.ship.v
    start = Vec3(float(slot[0] - 300.0), float(slot[1]), 0.0)

    env.tugs[0].set_state(start, Vec3(ship_vx + 1.0, ship_vy, env.ship.r))
    _sync_prev_dist(env)
    _, info_toward = env._reward.compute_rewards(actions, slot_world)
    toward = float(info_toward["reward_components"]["r_chase"][0])
    toward_closing = float(info_toward["reward_components"]["chase_closing_speed"][0])

    env.tugs[0].set_state(start, Vec3(ship_vx - 1.0, ship_vy, env.ship.r))
    _sync_prev_dist(env)
    _, info_away = env._reward.compute_rewards(actions, slot_world)
    away = float(info_away["reward_components"]["r_chase"][0])
    away_closing = float(info_away["reward_components"]["chase_closing_speed"][0])

    assert toward_closing > 0.0
    assert away_closing < 0.0
    assert toward > away


def test_near_field_hold_rewards_stable_slot_state() -> None:
    env = FormationEnv(cfg=EnvConfig(), seed=16)
    env.reset()
    actions = np.zeros((env.n_tugs, env.action_dim), dtype=np.float32)
    slot_world = env.ship.slot_positions_world()
    slot = slot_world[env.tug_to_slot[0]]

    env.tugs[0].set_state(
        Vec3(float(slot[0]), float(slot[1]), float(slot[2])),
        Vec3(float(env.ship.u), float(env.ship.v), float(env.ship.r)),
    )
    _sync_prev_dist(env)
    _, info = env._reward.compute_rewards(actions, slot_world)
    comp = info["reward_components"]

    assert comp["hold_gate"][0] == 1.0
    assert comp["r_hold"][0] > 0.0
    assert comp["in_zone"][0]


def test_collision_barrier_triggers_before_hard_collision() -> None:
    cfg = EnvConfig()
    env = FormationEnv(cfg=cfg, seed=13)
    env.reset()

    base_x = env.ship.x + 500.0
    base_y = env.ship.y + 500.0
    env.tugs[0].set_state(Vec3(base_x, base_y, 0.0), Vec3.zero())
    env.tugs[1].set_state(
        Vec3(base_x + cfg.tug_collision_dist_m + 5.0, base_y, 0.0),
        Vec3.zero(),
    )
    _sync_prev_dist(env)

    actions = np.zeros((env.n_tugs, env.action_dim), dtype=np.float32)
    _, info = env._reward.compute_rewards(actions, env.ship.slot_positions_world())
    comp = info["reward_components"]

    assert comp["p_tug_collision"][0] > 0.0
    assert comp["p_collision"][0] >= comp["p_tug_collision"][0]


def test_terminal_collision_penalty_is_preserved() -> None:
    cfg = EnvConfig()
    env = FormationEnv(cfg=cfg, seed=14)
    env.reset()

    base_x = env.ship.x + 500.0
    base_y = env.ship.y + 500.0
    env.tugs[0].set_state(Vec3(base_x, base_y, 0.0), Vec3.zero())
    env.tugs[1].set_state(Vec3(base_x + cfg.tug_collision_dist_m * 0.5, base_y, 0.0), Vec3.zero())

    actions = np.zeros((env.n_tugs, env.action_dim), dtype=np.float32)
    _, _, done, info = env.step(actions)

    assert done.all()
    assert info["collision"] is True
    assert info["collision_kind"] == "tug_vs_tug"
    assert np.min(info["terminal_reward"]) <= -cfg.reward_collision_pen
