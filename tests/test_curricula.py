from pathlib import Path

from config import EnvConfig
from curricula.loader import apply_course, load_course
from env.formation_env import FormationEnv


COURSE_FILES = (
    Path("c01_three_ready.py"),
    Path("c02_two_ready.py"),
    Path("c03_one_ready.py"),
    Path("c04_zero_ready.py"),
    Path("strict_pos/c04_pos140m.py"),
    Path("strict_pos/c04_pos120m.py"),
    Path("strict_pos/c04_pos100m.py"),
    Path("strict_pos/c04_pos80m.py"),
    Path("strict_pos/c04_pos60m.py"),
    Path("strict_pos/c04_pos40m.py"),
    Path("strict_pos/c04_pos20m.py"),
)


def test_curriculum_files_load_and_apply() -> None:
    for filename in COURSE_FILES:
        course = load_course(Path("curricula") / filename)
        cfg = EnvConfig()
        apply_course(cfg, course)

        assert course.name
        assert course.total_steps is not None and course.total_steps > 0
        assert tuple(cfg.tug_init_mixed_ready_counts) == tuple(
            course.env_overrides["tug_init_mixed_ready_counts"]
        )
        if filename.parts[0] == "strict_pos":
            assert cfg.reward_precision_w > 0.0
            assert cfg.reward_near_hold_w > 0.0
            assert cfg.hold_time_s == 2.0


def test_curriculum_ready_counts_match_reset() -> None:
    for filename in COURSE_FILES:
        course = load_course(Path("curricula") / filename)
        cfg = EnvConfig()
        apply_course(cfg, course)

        env = FormationEnv(cfg=cfg, seed=7)
        obs = env.reset()

        expected = int(tuple(course.env_overrides["tug_init_mixed_ready_counts"])[0])
        assert obs.shape == (cfg.n_tugs, env.obs_dim)
        assert len(env._mixed_ready_tugs) == expected
