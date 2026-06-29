from pathlib import Path

from config import EnvConfig
from curricula.loader import apply_course, load_course
from curricula.registry import ALL_COURSE_ENTRIES, MAIN_SEQUENCE, STRICT_POS_SEQUENCE
from env.formation_env import FormationEnv


def test_curriculum_files_load_and_apply() -> None:
    for entry in ALL_COURSE_ENTRIES:
        course = load_course(entry.path)
        cfg = EnvConfig()
        apply_course(cfg, course)

        assert course.name
        assert course.total_steps is not None and course.total_steps > 0
        assert tuple(cfg.tug_init_mixed_ready_counts) == tuple(
            course.env_overrides["tug_init_mixed_ready_counts"]
        )
        if entry in STRICT_POS_SEQUENCE:
            assert cfg.reward_precision_w > 0.0
            assert cfg.reward_near_hold_w > 0.0
            assert cfg.hold_time_s == 2.0


def test_curriculum_ready_counts_match_reset() -> None:
    for entry in ALL_COURSE_ENTRIES:
        course = load_course(entry.path)
        cfg = EnvConfig()
        apply_course(cfg, course)

        env = FormationEnv(cfg=cfg, seed=7)
        obs = env.reset()

        expected = tuple(int(v) for v in course.env_overrides["tug_init_mixed_ready_counts"])
        assert obs.shape == (cfg.n_tugs, env.obs_dim)
        assert len(env._mixed_ready_tugs) in expected


def test_main_sequence_order() -> None:
    keys = [entry.key for entry in MAIN_SEQUENCE]
    assert keys == ["c01", "c02", "c03", "c04a", "c04b", "c04"]


def test_main_joining_stages_use_safety_biased_routing() -> None:
    for entry in MAIN_SEQUENCE[1:]:
        course = load_course(entry.path)
        cfg = EnvConfig()
        apply_course(cfg, course)

        assert cfg.route_speed_governor is True
        assert cfg.route_nonfinal_forward_action_cap <= 0.32
        assert cfg.route_chase_speed_max_ms <= 0.55
        assert cfg.reward_collision_w >= 4.0
        assert cfg.reward_collision_tug_safe_m >= 100.0
        assert cfg.reward_collision_cpa_w >= 3.0


def test_strict_pos_sequence_ends_with_mixed_ready_10m() -> None:
    final_course = load_course(STRICT_POS_SEQUENCE[-1].path)
    cfg = EnvConfig()
    apply_course(cfg, final_course)

    assert cfg.pos_tol_m == 10.0
    assert tuple(cfg.tug_init_mixed_ready_counts) == (0, 1, 2, 3, 4)
    assert cfg.route_speed_governor is True
    assert cfg.reward_team_w > 0.5
    assert cfg.reward_precision_w > 0.0
    assert cfg.reward_near_hold_w > 0.0
