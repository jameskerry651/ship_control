"""命令行入口：``python -m simulator``。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulator.app import run
from simulator.config import SimConfig


def _build_config(args: argparse.Namespace) -> SimConfig:
    cfg = SimConfig()
    cfg.sim_speed = args.speed
    cfg.meters_per_pixel = args.mpp
    cfg.use_wheel = not args.no_wheel
    if args.steer_axis is not None:
        cfg.steer_axis = args.steer_axis
    if args.throttle_axis is not None:
        cfg.throttle_axis = args.throttle_axis
    if args.brake_axis is not None:
        cfg.brake_axis = args.brake_axis
    if args.invert_steer:
        cfg.steer_invert = not cfg.steer_invert
    if args.paddle_left_button is not None:
        cfg.paddle_left_button = args.paddle_left_button
    if args.paddle_right_button is not None:
        cfg.paddle_right_button = args.paddle_right_button
    cfg.show_ship = not args.no_ship
    cfg.ship_speed_ms = args.ship_speed
    cfg.ship_heading_deg = args.ship_heading
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="拖轮手动驾驶仿真（罗技 G29 方向盘 + 双踏板）",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="初始仿真倍速")
    parser.add_argument("--mpp", type=float, default=0.6, help="缩放：每像素对应米数(小=放大)")
    parser.add_argument("--no-wheel", action="store_true", help="强制使用键盘控制")
    parser.add_argument("--steer-axis", type=int, default=None, help="覆盖方向盘转向轴索引")
    parser.add_argument("--throttle-axis", type=int, default=None, help="覆盖油门(左桨)轴索引")
    parser.add_argument("--brake-axis", type=int, default=None, help="覆盖刹车(右桨)轴索引")
    parser.add_argument("--invert-steer", action="store_true", help="翻转方向盘转向(相对默认)")
    parser.add_argument("--paddle-left-button", type=int, default=None, help="左拨片按钮索引(左桨反转)")
    parser.add_argument("--paddle-right-button", type=int, default=None, help="右拨片按钮索引(右桨反转)")
    parser.add_argument("--no-ship", action="store_true", help="不显示匀速直线大船")
    parser.add_argument("--ship-speed", type=float, default=3.0, help="大船匀速直线速度(m/s)")
    parser.add_argument("--ship-heading", type=float, default=0.0, help="大船航向(度, 0=正北)")
    args = parser.parse_args()
    run(_build_config(args))
    # 正常退出后立即终止进程，绕开 macOS 上 SDL/Cocoa 在解释器 atexit
    # 阶段偶发的原生段错误（此时仿真已干净结束）。
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
