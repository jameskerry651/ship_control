"""输入设备抽象：罗技 G29 方向盘（含双踏板）与键盘回退。

统一输出 :class:`ControlState`：
    steer    ∈ [-1, 1]  方向盘转角，正=右打
    throttle ∈ [0, 1]   油门踏板 -> 左桨转速
    brake    ∈ [0, 1]   刹车踏板 -> 右桨转速
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pygame

from .config import SimConfig


@dataclass
class ControlState:
    steer: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    port_reverse: bool = False        # 左拨片：左桨反转
    starboard_reverse: bool = False   # 右拨片：右桨反转
    source: str = "keyboard"
    connected: bool = False


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _apply_deadzone_signed(value: float, deadzone: float) -> float:
    if deadzone <= 0.0:
        return _clamp(value, -1.0, 1.0)
    value = _clamp(value, -1.0, 1.0)
    if abs(value) < deadzone:
        return 0.0
    sign = math.copysign(1.0, value)
    return sign * (abs(value) - deadzone) / (1.0 - deadzone)


def _normalize_pedal(raw: float, rest: float, full: float, deadzone: float) -> float:
    span = rest - full
    if abs(span) < 1e-6:
        return 0.0
    value = _clamp((rest - raw) / span, 0.0, 1.0)
    if deadzone > 0.0:
        if value < deadzone:
            return 0.0
        value = (value - deadzone) / (1.0 - deadzone)
    return value


class WheelInput:
    """读取罗技 G29 系方向盘：转向轴 + 两个踏板轴。"""

    def __init__(self, cfg: SimConfig) -> None:
        self.cfg = cfg
        if not pygame.joystick.get_init():
            pygame.joystick.init()
        self.joystick: pygame.joystick.Joystick | None = None
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
        # 每个踏板轴独立的校准范围（rest/full），默认取自配置。
        self._pedal_cal: dict[int, tuple[float, float]] = {
            cfg.throttle_axis: (cfg.pedal_rest_raw, cfg.pedal_full_raw),
            cfg.brake_axis: (cfg.pedal_rest_raw, cfg.pedal_full_raw),
        }
        self._calibrating = False
        self._cal_min: dict[int, float] = {}
        self._cal_max: dict[int, float] = {}

    @property
    def connected(self) -> bool:
        return self.joystick is not None

    @property
    def name(self) -> str:
        if self.joystick is None:
            return "<no wheel>"
        return self.joystick.get_name()

    def _axis(self, idx: int) -> float:
        if self.joystick is None:
            return 0.0
        if idx < 0 or idx >= self.joystick.get_numaxes():
            return 0.0
        return float(self.joystick.get_axis(idx))

    def raw_axes(self) -> list[float]:
        if self.joystick is None:
            return []
        return [float(self.joystick.get_axis(i)) for i in range(self.joystick.get_numaxes())]

    def _button(self, idx: int) -> bool:
        if self.joystick is None or idx < 0 or idx >= self.joystick.get_numbuttons():
            return False
        return bool(self.joystick.get_button(idx))

    def pressed_buttons(self) -> list[int]:
        if self.joystick is None:
            return []
        return [i for i in range(self.joystick.get_numbuttons()) if self.joystick.get_button(i)]

    # ---------- 踏板校准：采集 min/max 后写回 rest/full ----------
    def begin_calibration(self) -> None:
        self._calibrating = True
        self._cal_min = {}
        self._cal_max = {}

    @property
    def calibrating(self) -> bool:
        return self._calibrating

    def _track_calibration(self) -> None:
        for idx in (self.cfg.throttle_axis, self.cfg.brake_axis):
            raw = self._axis(idx)
            self._cal_min[idx] = min(self._cal_min.get(idx, raw), raw)
            self._cal_max[idx] = max(self._cal_max.get(idx, raw), raw)

    def end_calibration(self) -> None:
        for idx in (self.cfg.throttle_axis, self.cfg.brake_axis):
            lo = self._cal_min.get(idx)
            hi = self._cal_max.get(idx)
            if lo is not None and hi is not None and abs(hi - lo) > 0.1:
                # 松开时停在 max（rest），踩到底为 min（full）。
                self._pedal_cal[idx] = (hi, lo)
        self._calibrating = False

    def poll(self, dt: float) -> ControlState:
        if self.joystick is None:
            return ControlState(source="wheel", connected=False)
        if self._calibrating:
            self._track_calibration()

        steer_raw = self._axis(self.cfg.steer_axis)
        if self.cfg.steer_invert:
            steer_raw = -steer_raw
        steer = _apply_deadzone_signed(steer_raw, self.cfg.steer_deadzone)

        rest_t, full_t = self._pedal_cal[self.cfg.throttle_axis]
        rest_b, full_b = self._pedal_cal[self.cfg.brake_axis]
        throttle = _normalize_pedal(
            self._axis(self.cfg.throttle_axis), rest_t, full_t, self.cfg.pedal_deadzone
        )
        brake = _normalize_pedal(
            self._axis(self.cfg.brake_axis), rest_b, full_b, self.cfg.pedal_deadzone
        )
        return ControlState(
            steer=steer,
            throttle=throttle,
            brake=brake,
            port_reverse=self._button(self.cfg.paddle_left_button),
            starboard_reverse=self._button(self.cfg.paddle_right_button),
            source="wheel",
            connected=True,
        )


class KeyboardInput:
    """无方向盘时的回退：方向键转向，Q/Z 控左桨，E/C 控右桨。"""

    STEER_RATE = 2.5      # 每秒满量程
    STEER_CENTER_RATE = 3.0
    PEDAL_RATE = 1.5

    def __init__(self, cfg: SimConfig) -> None:
        self.cfg = cfg
        self._steer = 0.0
        self._throttle = 0.0
        self._brake = 0.0

    @property
    def connected(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "keyboard"

    def raw_axes(self) -> list[float]:
        return []

    def poll(self, dt: float) -> ControlState:
        keys = pygame.key.get_pressed()
        left = keys[pygame.K_LEFT]
        right = keys[pygame.K_RIGHT]
        if left and not right:
            self._steer -= self.STEER_RATE * dt
        elif right and not left:
            self._steer += self.STEER_RATE * dt
        else:
            # 自动回中
            if self._steer > 0:
                self._steer = max(0.0, self._steer - self.STEER_CENTER_RATE * dt)
            else:
                self._steer = min(0.0, self._steer + self.STEER_CENTER_RATE * dt)
        self._steer = _clamp(self._steer, -1.0, 1.0)

        if keys[pygame.K_q]:
            self._throttle += self.PEDAL_RATE * dt
        if keys[pygame.K_z]:
            self._throttle -= self.PEDAL_RATE * dt
        if keys[pygame.K_e]:
            self._brake += self.PEDAL_RATE * dt
        if keys[pygame.K_c]:
            self._brake -= self.PEDAL_RATE * dt
        self._throttle = _clamp(self._throttle, 0.0, 1.0)
        self._brake = _clamp(self._brake, 0.0, 1.0)

        return ControlState(
            steer=self._steer,
            throttle=self._throttle,
            brake=self._brake,
            port_reverse=bool(keys[pygame.K_LSHIFT]),
            starboard_reverse=bool(keys[pygame.K_RSHIFT]),
            source="keyboard",
            connected=True,
        )


def create_input(cfg: SimConfig) -> WheelInput | KeyboardInput:
    """优先返回方向盘，未检测到或禁用时回退键盘。"""
    if cfg.use_wheel:
        wheel = WheelInput(cfg)
        if wheel.connected:
            return wheel
    return KeyboardInput(cfg)
