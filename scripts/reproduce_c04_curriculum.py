"""Reproduce the curriculum handoff needed for c04 >= 80% success.

This script trains c01 -> c02 -> c03 -> c04 in sequence. Each stage resumes
from the previous stage's best checkpoint, resets optimizer/progress, and uses
a short critic warmup. After c04 finishes, it runs a deterministic evaluation
and exits non-zero if the target success rate is not reached.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import EnvConfig
from curricula.loader import apply_course, load_course
from env.formation_env import FormationEnv
from rl.ppo import MAPPOActorCritic
from scripts.train import _load_checkpoint, evaluate_policy


from curricula.registry import MAIN_SEQUENCE

STAGES = tuple(
    (entry.key, entry.project_relative_path, entry.run_name) for entry in MAIN_SEQUENCE
)
STAGE_INDEX = {entry.key: idx for idx, entry in enumerate(MAIN_SEQUENCE)}


def _env_with_project_pythonpath() -> dict[str, str]:
    env = os.environ.copy()
    old = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(PROJECT_ROOT) if not old else f"{PROJECT_ROOT}{os.pathsep}{old}"
    return env


def _stage_total_steps(args: argparse.Namespace, key: str) -> int | None:
    return getattr(args, f"total_steps_{key}")


def _checkpoint_path(run_name: str) -> Path:
    return PROJECT_ROOT / "checkpoints" / run_name / "best.pt"


def _stage_threshold(args: argparse.Namespace, key: str) -> float:
    return float(getattr(args, f"stage_threshold_{key}"))


def _stage_max_collision(args: argparse.Namespace, key: str) -> float:
    return float(getattr(args, f"max_stage_collision_{key}"))


def _stage_learning_rate(args: argparse.Namespace, resume: Path | None) -> float:
    return (
        float(args.learning_rate_transfer)
        if resume is not None
        else float(args.learning_rate_initial)
    )


def _stage_success_bc_coef(args: argparse.Namespace, resume: Path | None) -> float:
    return (
        float(args.success_bc_coef_transfer)
        if resume is not None
        else float(args.success_bc_coef_initial)
    )


def _run_stage(
    args: argparse.Namespace,
    *,
    key: str,
    course_path: str,
    run_name: str,
    resume: Path | None,
) -> Path:
    best_path = _checkpoint_path(run_name)
    if args.skip_existing and best_path.exists():
        print(f"[skip] {key}: found {best_path}")
        return best_path

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
        str(_stage_learning_rate(args, resume)),
        "--entropy-coef",
        str(args.entropy_coef),
        "--target-kl",
        str(args.target_kl),
        "--success-bc-coef",
        str(_stage_success_bc_coef(args, resume)),
    ]
    if args.env_workers is not None:
        cmd.extend(["--env-workers", str(args.env_workers)])
    total_steps = _stage_total_steps(args, key)
    if total_steps is not None:
        cmd.extend(["--total-steps", str(total_steps)])
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
        if args.set_log_std_transfer is not None:
            cmd.extend(["--set-log-std", str(args.set_log_std_transfer)])

    print("\n[run]", " ".join(cmd), flush=True)
    if args.dry_run:
        return best_path

    subprocess.run(cmd, cwd=PROJECT_ROOT, env=_env_with_project_pythonpath(), check=True)
    if not best_path.exists():
        raise FileNotFoundError(f"stage {key} finished but best.pt was not created: {best_path}")
    return best_path


def _run_c04_refine(args: argparse.Namespace, *, resume: Path, attempt: int) -> Path:
    suffix = "c04_zero_ready_refine" if attempt == 1 else f"c04_zero_ready_refine_a{attempt}"
    run_name = f"{args.run_prefix}_{suffix}"
    best_path = _checkpoint_path(run_name)
    if args.skip_existing and best_path.exists():
        print(f"[skip] c04-refine: found {best_path}")
        return best_path

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train.py"),
        "--course",
        "curricula/c04_zero_ready.py",
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
        "--total-steps",
        str(args.total_steps_c04_refine),
        "--learning-rate",
        str(args.learning_rate_c04_refine),
        "--entropy-coef",
        str(args.entropy_coef),
        "--target-kl",
        str(args.target_kl),
        "--success-bc-coef",
        str(args.success_bc_coef_transfer),
        "--resume",
        str(resume),
        "--reset-progress",
        "--critic-warmup-updates",
        str(args.critic_warmup_updates_c04_refine),
    ]
    if args.set_log_std_transfer is not None:
        cmd.extend(["--set-log-std", str(args.set_log_std_transfer)])
    if args.env_workers is not None:
        cmd.extend(["--env-workers", str(args.env_workers)])

    print("\n[run]", " ".join(cmd), flush=True)
    if args.dry_run:
        return best_path

    subprocess.run(cmd, cwd=PROJECT_ROOT, env=_env_with_project_pythonpath(), check=True)
    if not best_path.exists():
        raise FileNotFoundError(f"c04 refine finished but best.pt was not created: {best_path}")
    return best_path


def _evaluate_checkpoint(
    checkpoint: Path,
    course_path: str,
    *,
    episodes: int,
    seed: int,
    eval_workers: int,
) -> dict[str, float]:
    course = load_course(PROJECT_ROOT / course_path)
    env_cfg = EnvConfig()
    apply_course(env_cfg, course)

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


def _evaluate_final(
    checkpoint: Path,
    *,
    episodes: int,
    seed: int,
    eval_workers: int,
) -> dict[str, float]:
    return _evaluate_checkpoint(
        checkpoint,
        "curricula/c04_zero_ready.py",
        episodes=episodes,
        seed=seed,
        eval_workers=eval_workers,
    )


def _canonical_final_checkpoint(args: argparse.Namespace) -> Path:
    return _checkpoint_path(f"{args.run_prefix}_c04_zero_ready_refine")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train c01->c04 curriculum and verify c04 success rate."
    )
    parser.add_argument(
        "--run-prefix",
        default=time.strftime("repro_%Y%m%d_%H%M%S"),
        help="Prefix for run/checkpoint directories.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--env-backend", choices=("subproc", "sync"), default="subproc")
    parser.add_argument("--env-workers", type=int, default=None)
    parser.add_argument("--critic-warmup-updates", type=int, default=5)
    parser.add_argument("--learning-rate-initial", type=float, default=1e-4)
    parser.add_argument("--learning-rate-transfer", type=float, default=5e-5)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--target-kl", type=float, default=0.01)
    parser.add_argument("--set-log-std-transfer", type=float, default=-1.0)
    parser.add_argument("--success-bc-coef-initial", type=float, default=0.0)
    parser.add_argument("--success-bc-coef-transfer", type=float, default=0.10)
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--target-success", type=float, default=0.80)
    parser.add_argument(
        "--start-stage",
        choices=tuple(STAGE_INDEX),
        default="c01",
        help="First curriculum stage to run.",
    )
    parser.add_argument(
        "--initial-resume",
        type=str,
        default=None,
        help="Checkpoint used as resume input for --start-stage.",
    )
    parser.add_argument(
        "--stage-attempts",
        type=int,
        default=1,
        help="Train/eval attempts per stage before giving up.",
    )
    parser.add_argument(
        "--stage-eval-episodes",
        type=int,
        default=32,
        help="Episodes used for gate evaluation after each stage.",
    )
    parser.add_argument("--stage-threshold-c01", type=float, default=0.60)
    parser.add_argument("--stage-threshold-c02", type=float, default=0.60)
    parser.add_argument("--stage-threshold-c03", type=float, default=0.50)
    parser.add_argument("--stage-threshold-c04a", type=float, default=0.70)
    parser.add_argument("--stage-threshold-c04b", type=float, default=0.70)
    parser.add_argument("--stage-threshold-c04", type=float, default=0.80)
    parser.add_argument("--max-stage-collision-c01", type=float, default=0.12)
    parser.add_argument("--max-stage-collision-c02", type=float, default=0.12)
    parser.add_argument("--max-stage-collision-c03", type=float, default=0.12)
    parser.add_argument("--max-stage-collision-c04a", type=float, default=0.10)
    parser.add_argument("--max-stage-collision-c04b", type=float, default=0.10)
    parser.add_argument("--max-stage-collision-c04", type=float, default=0.10)
    parser.add_argument("--max-final-collision", type=float, default=0.08)
    parser.add_argument(
        "--no-c04-refine",
        action="store_true",
        help="Disable the low-learning-rate c04 refinement pass.",
    )
    parser.add_argument("--total-steps-c04-refine", type=int, default=700_000)
    parser.add_argument("--learning-rate-c04-refine", type=float, default=2e-5)
    parser.add_argument("--critic-warmup-updates-c04-refine", type=int, default=3)
    parser.add_argument(
        "--no-stage-gates",
        action="store_true",
        help="Run all stages once without stopping on stage thresholds.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--total-steps-c01", type=int, default=None)
    parser.add_argument("--total-steps-c02", type=int, default=None)
    parser.add_argument("--total-steps-c03", type=int, default=None)
    parser.add_argument("--total-steps-c04a", type=int, default=None)
    parser.add_argument("--total-steps-c04b", type=int, default=None)
    parser.add_argument("--total-steps-c04", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    previous: Path | None = Path(args.initial_resume).expanduser().resolve() if args.initial_resume else None
    final_ckpt: Path | None = None

    for key, course_path, label in STAGES[STAGE_INDEX[args.start_stage]:]:
        stage_input = previous
        stage_passed = False
        best_stage_ckpt: Path | None = None
        best_stage_success = -1.0
        best_stage_collision = 1.0
        best_stage_score = -float("inf")
        for attempt in range(1, max(1, int(args.stage_attempts)) + 1):
            run_name = f"{args.run_prefix}_{label}" if attempt == 1 else f"{args.run_prefix}_{label}_a{attempt}"
            final_ckpt = _run_stage(
                args,
                key=key,
                course_path=course_path,
                run_name=run_name,
                resume=stage_input,
            )
            if key == "c04" and not args.no_c04_refine:
                final_ckpt = _run_c04_refine(args, resume=final_ckpt, attempt=attempt)
            if args.dry_run:
                previous = final_ckpt
                stage_passed = True
                break

            stats = _evaluate_checkpoint(
                final_ckpt,
                course_path,
                episodes=args.stage_eval_episodes,
                seed=args.seed + 10_000 + STAGE_INDEX[key] * 1000 + attempt * 100,
                eval_workers=args.eval_workers,
            )
            success_rate = stats["eval/success_rate"]
            collision_rate = stats["eval/collision_rate"]
            print(f"\n[stage-eval] {key} attempt={attempt}")
            for stat_key, value in stats.items():
                print(f"{stat_key}: {value:.6f}")
            stage_score = success_rate - 0.5 * collision_rate
            if stage_score > best_stage_score:
                best_stage_score = stage_score
                best_stage_success = success_rate
                best_stage_collision = collision_rate
                best_stage_ckpt = final_ckpt
            threshold = _stage_threshold(args, key)
            max_collision = _stage_max_collision(args, key)
            if args.no_stage_gates:
                print(
                    f"[stage-continue] {key}: "
                    f"success={success_rate:.4f}, threshold={threshold:.4f}, "
                    f"collision={collision_rate:.4f}, max_collision={max_collision:.4f}"
                )
                previous = final_ckpt
                stage_passed = True
                break
            success_ok = success_rate >= threshold
            collision_ok = collision_rate <= max_collision
            if success_ok and collision_ok:
                print(
                    f"[stage-pass] {key}: success={success_rate:.4f} >= {threshold:.4f}, "
                    f"collision={collision_rate:.4f} <= {max_collision:.4f}"
                )
                previous = final_ckpt
                stage_passed = True
                break
            print(
                f"[stage-retry] {key}: success={success_rate:.4f}/{threshold:.4f}, "
                f"collision={collision_rate:.4f}/{max_collision:.4f}"
            )

        if not stage_passed:
            assert best_stage_ckpt is not None
            print(
                f"[stage-fail] {key}: best_success={best_stage_success:.4f}, "
                f"best_collision={best_stage_collision:.4f}, "
                f"best_ckpt={best_stage_ckpt}",
                file=sys.stderr,
            )
            return 3

    if args.dry_run:
        print("\n[dry-run] commands printed; no training or evaluation was run.")
        return 0

    assert final_ckpt is not None
    stats = _evaluate_final(
        final_ckpt,
        episodes=args.eval_episodes,
        seed=args.seed + 50_000,
        eval_workers=args.eval_workers,
    )
    print("\n[final-eval]")
    for key, value in stats.items():
        print(f"{key}: {value:.6f}")

    success_rate = stats["eval/success_rate"]
    collision_rate = stats["eval/collision_rate"]
    if success_rate < args.target_success or collision_rate > args.max_final_collision:
        print(
            f"[fail] success_rate={success_rate:.4f}/{args.target_success:.4f}, "
            f"collision_rate={collision_rate:.4f}/{args.max_final_collision:.4f}",
            file=sys.stderr,
        )
        return 2
    print(
        f"[pass] success_rate={success_rate:.4f} >= target={args.target_success:.4f}, "
        f"collision_rate={collision_rate:.4f} <= max={args.max_final_collision:.4f}"
    )
    canonical_ckpt = _canonical_final_checkpoint(args)
    if final_ckpt.resolve() != canonical_ckpt.resolve():
        canonical_ckpt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(final_ckpt, canonical_ckpt)
        print(f"[artifact] canonical final checkpoint: {canonical_ckpt}")
    print(f"[artifact] final checkpoint: {final_ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
