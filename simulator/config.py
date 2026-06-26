"""仿真器配置：窗口/相机、仿真倍速、罗技 G29 轴映射。

所有字段都可被命令行覆盖（见 ``simulator/__main__.py``）。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SimConfig:
    # ---------- 窗口与相机 ----------
    window_w: int = 1280
    window_h: int = 800
    meters_per_pixel: float = 0.6      # 越小越放大
    follow_ship: bool = True           # 相机是否跟随本船
    grid_spacing_m: float = 100.0
    fps: int = 60                      # 渲染/逻辑帧率

    # ---------- 仿真倍速 ----------
    sim_speed: float = 1.0             # 初始倍速：sim_dt = real_dt * sim_speed
    sim_speed_min: float = 0.25
    sim_speed_max: float = 64.0
    sim_speed_step: float = 1.5        # +/- 键的乘法步进

    # ---------- 罗技 G29 轴映射 ----------
    # G29/G920/G923：转向轴 0；踏板各占独立轴，松开=+1、踩到底=-1。
    # 不同平台/驱动下踏板轴序可能不同，可用 CLI 覆盖或屏幕调试面板(D)确认。
    steer_axis: int = 0
    throttle_axis: int = 2             # 油门踏板 -> 左桨
    brake_axis: int = 3               # 刹车踏板 -> 右桨

    # 转向：方向盘原始 [-1,1]，invert 后映射到方位角。
    steer_invert: bool = True          # 左打方向盘 -> 船左转
    steer_deadzone: float = 0.02       # 中位死区，避免抖动

    # 踏板原始范围：松开 rest -> 0 油门，踩到底 full -> 1 油门。
    # 默认按 G29：rest=+1, full=-1，归一化 (rest - raw)/(rest - full)。
    pedal_rest_raw: float = 1.0
    pedal_full_raw: float = -1.0
    pedal_deadzone: float = 0.03       # 归一化后 [0,1] 的低端死区

    # 换挡拨片按钮：按住左拨片让左桨反转、右拨片让右桨反转。
    # G29 拨片按钮索引各平台略有差异，可用 CLI 覆盖或按 D 看按钮调试。
    paddle_left_button: int = 5        # 左拨片 -> 左桨反转
    paddle_right_button: int = 4       # 右拨片 -> 右桨反转

    # ---------- 输入设备 ----------
    use_wheel: bool = True             # False 强制使用键盘回退

    # ---------- 测试用大船（匀速直线运动，用于伴航控制测试）----------
    show_ship: bool = True
    ship_length_m: float = 200.0
    ship_beam_m: float = 30.0
    ship_speed_ms: float = 3.0         # 大船匀速直线速度
    ship_heading_deg: float = 0.0      # 大船航向（0=正北，与拖轮初始航向一致）
    # 大船初始位置（世界系，相对拖轮初始原点）：默认在拖轮右前方，
    # 让拖轮需要先追上再贴到船舷侧保持伴航。
    ship_init_north_m: float = 60.0
    ship_init_east_m: float = 60.0
