"""Train a position-tolerance curriculum down to a strict final threshold.

The default schedule first bridges the c04 handoff from a 1 s hold to a 2 s
hold at the loose 140 m gate, then tightens the success distance from 140 m to
10 m.  The final 10 m stage switches to mixed-ready starts so reset covers
0..4 ready tugs. Each stage resumes from the previous stage's best checkpoint
and is evaluated with the same success threshold used for training.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import EnvConfig
from curricula.loader import apply_course, load_course
from env.formation_env import FormationEnv
from rl.ppo import MAPPOActorCritic
from scripts.train import _load_checkpoint, evaluate_policy

DEFAULT_POS_TOLS = [140.0, 140.0, 140.0, 120.0, 100.0, 80.0, 60.0, 40.0, 20.0, 10.0]
DEFAULT_HOLD_TIMES = [1.0, 1.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
DEFAULT_SINGLE_HOLD_TIME = 2.0


def _env_with_project_pythonpath() -> dict[str, str]:
    env = os.environ.copy()
    old = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(PROJECT_ROOT) if not old else f"{PROJECT_ROOT}{os.pathsep}{old}"
    return env


def _checkpoint_path(run_name: str) -> Path:
    return PROJECT_ROOT / "checkpoints" / run_name / "best.pt"


def _parse_float_list(raw: str) -> list[float]:
    vals = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("list must contain at least one value")
    return vals


def _stage_suffix(pos_tol: float, hold_time: float) -> str:
    pos_text = f"{pos_tol:g}".replace(".", "p")
    hold_text = f"{hold_time:g}".replace(".", "p")
    return f"p{pos_text}m_h{hold_text}s"


def _course_path_for_pos(args: argparse.Namespace, pos_tol: float) -> str:
    if not args.no_strict_course_files:
        text = f"{pos_tol:g}".replace(".", "p")
        candidate = PROJECT_ROOT / args.strict_course_dir / f"c04_pos{text}m.py"
        if candidate.exists():
            return str(candidate.relative_to(PROJECT_ROOT))
    return str(args.course)


def _build_env_cfg(
    course_path: str,
    *,
    pos_tol: float,
    hold_time: float,
    heading_tol_deg: float,
    speed_tol: float,
    reward_precision_w: float | None,
    reward_precision_scale: float | None,
    reward_near_hold_w: float | None,
    reward_near_hold_scale: float | None,
) -> EnvConfig:
    course = load_course(PROJECT_ROOT / course_path)
    env_cfg = EnvConfig()
    apply_course(env_cfg, course)
    env_cfg.pos_tol_m = float(pos_tol)
    env_cfg.hold_time_s = float(hold_time)
    env_cfg.heading_tol_rad = math.radians(float(heading_tol_deg))
    env_cfg.speed_tol_ms = float(speed_tol)
    if reward_precision_w is not None:
        env_cfg.reward_precision_w = float(reward_precision_w)
    if reward_precision_scale is not None:
        env_cfg.reward_precision_scale_m = float(reward_precision_scale)
    if reward_near_hold_w is not None:
        env_cfg.reward_near_hold_w = float(reward_near_hold_w)
    if reward_near_hold_scale is not None:
        env_cfg.reward_near_hold_scale_m = float(reward_near_hold_scale)
    return env_cfg


def _evaluate_checkpoint(
    checkpoint: Path,
    course_path: str,
    *,
    pos_tol: float,
    hold_time: float,
    heading_tol_deg: float,
    speed_tol: float,
    reward_precision_w: float | None,
    reward_precision_scale: float | None,
    reward_near_hold_w: float | None,
    reward_near_hold_scale: float | None,
    episodes: int,
    seed: int,
    eval_workers: int,
) -> dict[str, float]:
    env_cfg = _build_env_cfg(
        course_path,
        pos_tol=pos_tol,
        hold_time=hold_time,
        heading_tol_deg=heading_tol_deg,
        speed_tol=speed_tol,
        reward_precision_w=reward_precision_w,
        reward_precision_scale=reward_precision_scale,
        reward_near_hold_w=reward_near_hold_w,
        reward_near_hold_scale=reward_near_hold_scale,
    )
    probe = FormationEnv(env_cfg, seed=seed)
    model_kwargs = {
        "obs_dim": probe.obs_dim,
        "action_dim": probe.action_dim,
        "n_agents": env_cfg.n_tugs,
        "global_state_dim": probe.global_state_dim,
    }
    model = MAPPOActorCritic(**model_kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _load_checkpoint(model, optimizer, ckpt, reset_progress=True)
    model.eval()
    return evaluate_policy(
        model,
        env_cfg,
        n_episodes=episodes,
        device=torch.device("cpu"),
        seed=seed,
        eval_workers=eval_workers,
        model_kwargs=model_kwargs,
    )


def _run_train_stage(
    args: argparse.Namespace,
    *,
    course_path: str,
    pos_tol: float,
    hold_time: float,
    run_name: str,
    resume: Path | None,
    final_stage: bool,
) -> Path:
    best_path = _checkpoint_path(run_name)
    if args.skip_existing and best_path.exists():
        print(f"[skip] pos_tol={pos_tol:g}, hold={hold_time:g}: found {best_path}")
        return best_path

    total_steps = args.final_total_steps if final_stage else args.stage_total_steps
    learning_rate = args.final_learning_rate if final_stage else args.learning_rate
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train.py"),
        "--course",
        course_path,
        "--run-name",
        run_name,
        "--seed",
        str(args.seed),
        "--torch-threads",
        str(args.torch_threads),
        "--eval-workers",
        str(args.eval_workers),
        "--env-backend",
        args.env_backend,
        "--learning-rate",
        str(learning_rate),
        "--entropy-coef",
        str(args.entropy_coef),
        "--target-kl",
        str(args.target_kl),
        "--pos-tol",
        str(pos_tol),
        "--hold-time",
        str(hold_time),
        "--heading-tol-deg",
        str(args.heading_tol_deg),
        "--speed-tol",
        str(args.speed_tol),
        "--success-bc-coef",
        str(args.success_bc_coef),
    ]
    if total_steps is not None:
        cmd.extend(["--total-steps", str(total_steps)])
    optional_reward_args = (
        ("--reward-precision-w", args.reward_precision_w),
        ("--reward-precision-scale", args.reward_precision_scale),
        ("--reward-near-hold-w", args.reward_near_hold_w),
        ("--reward-near-hold-scale", args.reward_near_hold_scale),
    )
    for flag, value in optional_reward_args:
        if value is not None:
            cmd.extend([flag, str(value)])
    if args.set_log_std is not None:
        cmd.extend(["--set-log-std", str(args.set_log_std)])
    if args.env_workers is not None:
        cmd.extend(["--env-workers", str(args.env_workers)])
    if resume is not None:
        if not args.dry_run and not resume.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume}")
        cmd.extend(
            [
                "--resume",
                str(resume),
                "--reset-progress",
                "--critic-warmup-updates",
                str(args.critic_warmup_updates),
            ]
        )

    print("\n[run]", " ".join(cmd), flush=True)
    if args.dry_run:
        return best_path

    subprocess.run(cmd, cwd=PROJECT_ROOT, env=_env_with_project_pythonpath(), check=True)
    if not best_path.exists():
        raise FileNotFoundError(f"stage finished but best.pt was not created: {best_path}")
    return best_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train c04 with progressively tighter pos_tol, ending at 10 m."
    )
    parser.add_argument("--course", default="curricula/c04_zero_ready.py")
    parser.add_argument("--strict-course-dir", default="curricula/strict_pos")
    parser.add_argument(
        "--no-strict-course-files",
        action="store_true",
        help="Use --course plus CLI overrides instead of curricula/strict_pos files.",
    )
    parser.add_argument("--run-prefix", default=time.strftime("strict_pos_%Y%m%d_%H%M%S"))
    parser.add_argument("--initial-resume", type=str, default=None)
    parser.add_argument(
        "--pos-tols",
        type=_parse_float_list,
        default=DEFAULT_POS_TOLS,
    )
    parser.add_argument(
        "--hold-times",
        type=_parse_float_list,
        default=None,
        help=(
            "Comma-separated hold_time_s per stage. Default uses the built-in "
            "140 m hold bridge: 1.0,1.5,2.0 before tightening pos_tol."
        ),
    )
    parser.add_argument(
        "--hold-time",
        type=float,
        default=None,
        help=(
            "Use one hold_time_s for every stage. If omitted with the default "
            "--pos-tols, the built-in hold bridge is used."
        ),
    )
    parser.add_argument("--heading-tol-deg", type=float, default=30.0)
    parser.add_argument("--speed-tol", type=float, default=3.0)
    parser.add_argument(
        "--reward-precision-w",
        type=float,
        default=None,
        help="Override EnvConfig.reward_precision_w; default uses each course file.",
    )
    parser.add_argument(
        "--reward-precision-scale",
        type=float,
        default=None,
        help="Override EnvConfig.reward_precision_scale_m; default uses each course file.",
    )
    parser.add_argument(
        "--reward-near-hold-w",
        type=float,
        default=None,
        help="Override EnvConfig.reward_near_hold_w; default uses each course file.",
    )
    parser.add_argument(
        "--reward-near-hold-scale",
        type=float,
        default=None,
        help="Override EnvConfig.reward_near_hold_scale_m; default uses each course file.",
    )
    parser.add_argument("--success-bc-coef", type=float, default=0.10)
    parser.add_argument(
        "--stage-total-steps",
        type=int,
        default=None,
        help="Override non-final strict stage steps; default uses each course file.",
    )
    parser.add_argument(
        "--final-total-steps",
        type=int,
        default=None,
        help="Override final strict stage steps; default uses the final course file.",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--final-learning-rate", type=float, default=1e-5)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--target-kl", type=float, default=0.01)
    parser.add_argument("--set-log-std", type=float, default=-1.0)
    parser.add_argument("--critic-warmup-updates", type=int, default=3)
    parser.add_argument("--stage-attempts", type=int, default=2)
    parser.add_argument("--stage-threshold", type=float, default=0.80)
    parser.add_argument("--target-success", type=float, default=0.90)
    parser.add_argument("--max-stage-collision", type=float, default=0.10)
    parser.add_argument("--max-final-collision", type=float, default=0.08)
    parser.add_argument("--stage-eval-episodes", type=int, default=64)
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument(
        "--no-pre-eval-resume",
        action="store_true",
        help="Do not evaluate the resume checkpoint before training each stage.",
    )
    parser.add_argument(
        "--no-intermediate-gates",
        action="store_true",
        help="Always continue through non-final stages after one training attempt.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--eval-workers", type=int, default=1)
    parser.add_argument("--env-backend", choices=("subproc", "sync"), default="subproc")
    parser.add_argument("--env-workers", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _resolve_hold_times(args: argparse.Namespace, pos_tols: list[float]) -> list[float]:
    if args.hold_times is not None:
        hold_times = [float(v) for v in args.hold_times]
        if len(hold_times) != len(pos_tols):
            raise ValueError(
                f"--hold-times length ({len(hold_times)}) must match "
                f"--pos-tols length ({len(pos_tols)})"
            )
        return hold_times
    if args.hold_time is not None:
        return [float(args.hold_time) for _ in pos_tols]
    if len(pos_tols) == len(DEFAULT_POS_TOLS) and all(
        abs(a - b) <= 1e-9 for a, b in zip(pos_tols, DEFAULT_POS_TOLS)
    ):
        return list(DEFAULT_HOLD_TIMES)
    return [DEFAULT_SINGLE_HOLD_TIME for _ in pos_tols]


def main() -> int:
    args = parse_args()
    pos_tols = [float(v) for v in args.pos_tols]
    hold_times = _resolve_hold_times(args, pos_tols)
    previous = Path(args.initial_resume).expanduser().resolve() if args.initial_resume else None
    final_ckpt: Path | None = None

    for stage_idx, (pos_tol, stage_hold_time) in enumerate(zip(pos_tols, hold_times)):
        stage_course = _course_path_for_pos(args, pos_tol)
        final_stage = stage_idx == len(pos_tols) - 1
        threshold = args.target_success if final_stage else args.stage_threshold
        max_collision = args.max_final_collision if final_stage else args.max_stage_collision
        stage_input = previous
        stage_passed = False
        best_attempt_success = -1.0
        best_attempt_collision = 1.0
        best_attempt_score = -float("inf")
        best_attempt_ckpt: Path | None = None
        episodes = args.eval_episodes if final_stage else args.stage_eval_episodes

        if stage_input is not None and not args.dry_run and not args.no_pre_eval_resume:
            stats = _evaluate_checkpoint(
                stage_input,
                stage_course,
                pos_tol=pos_tol,
                hold_time=stage_hold_time,
                heading_tol_deg=args.heading_tol_deg,
                speed_tol=args.speed_tol,
                reward_precision_w=args.reward_precision_w,
                reward_precision_scale=args.reward_precision_scale,
                reward_near_hold_w=args.reward_near_hold_w,
                reward_near_hold_scale=args.reward_near_hold_scale,
                episodes=episodes,
                seed=args.seed + 40_000 + stage_idx * 1000,
                eval_workers=args.eval_workers,
            )
            success_rate = stats["eval/success_rate"]
            collision_rate = stats["eval/collision_rate"]
            best_attempt_success = success_rate
            best_attempt_collision = collision_rate
            best_attempt_score = success_rate - 0.5 * collision_rate
            best_attempt_ckpt = stage_input
            print(
                f"\n[pre-eval] pos_tol={pos_tol:g} "
                f"hold={stage_hold_time:g} resume={stage_input}"
            )
            for key, value in stats.items():
                print(f"{key}: {value:.6f}")
            if success_rate >= threshold and collision_rate <= max_collision:
                print(
                    f"[stage-pass] pos_tol={pos_tol:g}, hold={stage_hold_time:g}: "
                    f"success={success_rate:.4f} >= {threshold:.4f}, "
                    f"collision={collision_rate:.4f} <= {max_collision:.4f}"
                )
                final_ckpt = stage_input
                stage_passed = True
                continue

        for attempt in range(1, max(1, int(args.stage_attempts)) + 1):
            suffix = _stage_suffix(pos_tol, stage_hold_time)
            if attempt > 1:
                suffix = f"{suffix}_a{attempt}"
            run_name = f"{args.run_prefix}_{suffix}"
            final_ckpt = _run_train_stage(
                args,
                pos_tol=pos_tol,
                hold_time=stage_hold_time,
                course_path=stage_course,
                run_name=run_name,
                resume=stage_input,
                final_stage=final_stage,
            )
            if args.dry_run:
                previous = final_ckpt
                stage_passed = True
                break

            stats = _evaluate_checkpoint(
                final_ckpt,
                stage_course,
                pos_tol=pos_tol,
                hold_time=stage_hold_time,
                heading_tol_deg=args.heading_tol_deg,
                speed_tol=args.speed_tol,
                reward_precision_w=args.reward_precision_w,
                reward_precision_scale=args.reward_precision_scale,
                reward_near_hold_w=args.reward_near_hold_w,
                reward_near_hold_scale=args.reward_near_hold_scale,
                episodes=episodes,
                seed=args.seed + 50_000 + stage_idx * 1000 + attempt * 100,
                eval_workers=args.eval_workers,
            )
            success_rate = stats["eval/success_rate"]
            collision_rate = stats["eval/collision_rate"]
            print(
                f"\n[stage-eval] pos_tol={pos_tol:g} "
                f"hold={stage_hold_time:g} attempt={attempt}"
            )
            for key, value in stats.items():
                print(f"{key}: {value:.6f}")

            stage_score = success_rate - 0.5 * collision_rate
            if stage_score > best_attempt_score:
                best_attempt_score = stage_score
                best_attempt_success = success_rate
                best_attempt_collision = collision_rate
                best_attempt_ckpt = final_ckpt
            if args.no_intermediate_gates and not final_stage:
                print(
                    f"[stage-continue] pos_tol={pos_tol:g}, hold={stage_hold_time:g}: "
                    f"success={success_rate:.4f}, threshold={threshold:.4f}, "
                    f"collision={collision_rate:.4f}, max_collision={max_collision:.4f}"
                )
                previous = final_ckpt
                stage_passed = True
                break
            if success_rate >= threshold and collision_rate <= max_collision:
                print(
                    f"[stage-pass] pos_tol={pos_tol:g}, hold={stage_hold_time:g}: "
                    f"success={success_rate:.4f} >= {threshold:.4f}, "
                    f"collision={collision_rate:.4f} <= {max_collision:.4f}"
                )
                previous = final_ckpt
                stage_passed = True
                break

            print(
                f"[stage-retry] pos_tol={pos_tol:g}, hold={stage_hold_time:g}: "
                f"success={success_rate:.4f}/{threshold:.4f}, "
                f"collision={collision_rate:.4f}/{max_collision:.4f}"
            )

        if not stage_passed:
            assert best_attempt_ckpt is not None
            print(
                f"[stage-fail] pos_tol={pos_tol:g}, hold={stage_hold_time:g}: "
                f"best_success={best_attempt_success:.4f}, "
                f"best_collision={best_attempt_collision:.4f}, "
                f"best_ckpt={best_attempt_ckpt}",
                file=sys.stderr,
            )
            return 3 if not final_stage else 2

    if args.dry_run:
        print("\n[dry-run] commands printed; no training or evaluation was run.")
        return 0

    assert final_ckpt is not None
    print(f"[pass] final checkpoint: {final_ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
