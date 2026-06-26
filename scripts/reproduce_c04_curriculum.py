"""Reproduce the curriculum handoff needed for c04 >= 80% success.

This script trains c01 -> c02 -> c03 -> c04 in sequence. Each stage resumes
from the previous stage's best checkpoint, resets optimizer/progress, and uses
a short critic warmup. After c04 finishes, it runs a deterministic evaluation
and exits non-zero if the target success rate is not reached.
"""

from __future__ import annotations

import argparse
import os
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


STAGES = (
    ("c01", "curricula/c01_three_ready.py", "c01_three_ready"),
    ("c02", "curricula/c02_two_ready.py", "c02_two_ready"),
    ("c03", "curricula/c03_one_ready.py", "c03_one_ready"),
    ("c04", "curricula/c04_zero_ready.py", "c04_zero_ready"),
)

STAGE_INDEX = {key: idx for idx, (key, _, _) in enumerate(STAGES)}


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
        "--resume",
        str(resume),
        "--reset-progress",
        "--critic-warmup-updates",
        str(args.critic_warmup_updates_c04_refine),
    ]
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
    parser.add_argument("--stage-threshold-c04", type=float, default=0.80)
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
    parser.add_argument("--total-steps-c04", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    previous: Path | None = Path(args.initial_resume).expanduser().resolve() if args.initial_resume else None
    final_ckpt: Path | None = None

    for key, course_path, label in STAGES[STAGE_INDEX[args.start_stage]:]:
        stage_passed = False
        best_stage_ckpt: Path | None = None
        best_stage_success = -1.0
        for attempt in range(1, max(1, int(args.stage_attempts)) + 1):
            run_name = f"{args.run_prefix}_{label}" if attempt == 1 else f"{args.run_prefix}_{label}_a{attempt}"
            final_ckpt = _run_stage(
                args,
                key=key,
                course_path=course_path,
                run_name=run_name,
                resume=previous,
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
            print(f"\n[stage-eval] {key} attempt={attempt}")
            for stat_key, value in stats.items():
                print(f"{stat_key}: {value:.6f}")
            if success_rate > best_stage_success:
                best_stage_success = success_rate
                best_stage_ckpt = final_ckpt
            threshold = _stage_threshold(args, key)
            if args.no_stage_gates or success_rate >= threshold:
                print(f"[stage-pass] {key}: {success_rate:.4f} >= {threshold:.4f}")
                previous = final_ckpt
                stage_passed = True
                break
            print(f"[stage-retry] {key}: {success_rate:.4f} < {threshold:.4f}")
            previous = final_ckpt

        if not stage_passed:
            assert best_stage_ckpt is not None
            print(
                f"[stage-fail] {key}: best_success={best_stage_success:.4f}, "
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
    if success_rate < args.target_success:
        print(
            f"[fail] success_rate={success_rate:.4f} < target={args.target_success:.4f}",
            file=sys.stderr,
        )
        return 2
    print(f"[pass] success_rate={success_rate:.4f} >= target={args.target_success:.4f}")
    print(f"[artifact] final checkpoint: {final_ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
