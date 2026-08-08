#!/usr/bin/env python3
"""清理 runs/ 下的训练日志目录，仅保留最近修改的 N 个文件夹。"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def list_run_dirs(runs_dir: Path) -> list[Path]:
    if not runs_dir.exists():
        return []
    return [p for p in runs_dir.iterdir() if p.is_dir()]


def select_for_cleanup(run_dirs: list[Path], keep: int) -> tuple[list[Path], list[Path]]:
    ranked = sorted(run_dirs, key=lambda p: p.stat().st_mtime, reverse=True)
    keep_n = max(0, keep)
    return ranked[:keep_n], ranked[keep_n:]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=_ROOT / "runs",
        help="训练日志根目录（默认：仓库根下 runs/）",
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=3,
        help="保留最近修改的目录数（默认：3）",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="真正删除；省略时仅预览",
    )
    args = parser.parse_args(argv)

    if args.keep < 0:
        print("error: --keep 必须 >= 0", file=sys.stderr)
        return 2

    runs_dir = args.runs_dir.resolve()
    run_dirs = list_run_dirs(runs_dir)
    keep_dirs, delete_dirs = select_for_cleanup(run_dirs, args.keep)

    print(f"runs_dir: {runs_dir}")
    print(f"total: {len(run_dirs)}  keep: {args.keep}  delete: {len(delete_dirs)}")
    print()

    if keep_dirs:
        print("保留:")
        for path in keep_dirs:
            print(f"  + {path.name}")
    else:
        print("保留: (无)")

    if delete_dirs:
        print("删除:")
        for path in delete_dirs:
            print(f"  - {path.name}")
    else:
        print("删除: (无)")

    if not delete_dirs:
        return 0

    if not args.apply:
        print()
        print("dry-run：未删除任何内容。加 --apply 执行删除。")
        return 0

    for path in delete_dirs:
        shutil.rmtree(path)
        print(f"removed: {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
