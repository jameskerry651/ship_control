import numpy as np

from config import EnvConfig
from env.formation_env import FormationEnv


def _make_route_reward_env() -> FormationEnv:
    cfg = EnvConfig()
    cfg.tug_init_mixed_ready_counts = (0,)
    cfg.reward_velocity_w = 0.0
    cfg.reward_collision_w = 0.0
    cfg.reward_team_w = 0.0
    cfg.reward_shape_w = 0.0
    cfg.reward_hold_streak_w = 0.0
    env = FormationEnv(cfg=cfg, seed=23)
    env.reset()
    return env


def test_nonfinal_route_progress_controls_route_reward() -> None:
    env = _make_route_reward_env()
    actions = np.zeros((env.cfg.n_tugs, env.action_dim), dtype=np.float32)
    slot_world = env.ship.slot_positions_world()
    for i in range(env.cfg.n_tugs):
        env._advance_route_stage(i)

    tug_idx = None
    for i, tug in enumerate(env.tugs):
        route_len = len(env._route._route_waypoints_body_for_tug(i))
        slot = slot_world[env.tug_to_slot[i]]
        dist_to_slot = np.hypot(tug.eta.x - slot[0], tug.eta.y - slot[1])
        if int(env.route_stage[i]) < route_len - 1 and dist_to_slot > env.cfg.pos_tol_m:
            tug_idx = i
            break
    assert tug_idx is not None

    route_remaining = env._route._route_remaining_distance(tug_idx)
    clip = float(env.cfg.reward_target_progress_clip_m)

    env.prev_route_remaining[tug_idx] = route_remaining + clip
    _, info_forward = env._reward.compute_rewards(actions, slot_world)

    env.in_zone_steps[:] = 0
    env.prev_route_remaining[tug_idx] = route_remaining - clip
    _, info_backward = env._reward.compute_rewards(actions, slot_world)

    forward = info_forward["reward_components"]
    backward = info_backward["reward_components"]

    assert bool(forward["route_chase"][tug_idx])
    assert forward["route_progress"][tug_idx] > 0.0
    assert backward["route_progress"][tug_idx] < 0.0
    assert forward["r_route"][tug_idx] > backward["r_route"][tug_idx]


def test_prev_route_remaining_tracks_current_route_after_step() -> None:
    env = _make_route_reward_env()
    actions = np.zeros((env.cfg.n_tugs, env.action_dim), dtype=np.float32)

    env.step(actions)

    current = np.asarray(
        [env._route._route_remaining_distance(i) for i in range(env.cfg.n_tugs)],
        dtype=np.float32,
    )
    np.testing.assert_allclose(env.prev_route_remaining, current, rtol=1e-5, atol=1e-5)
