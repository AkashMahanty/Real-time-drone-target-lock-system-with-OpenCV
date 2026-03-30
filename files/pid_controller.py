"""
pid_controller.py
=================
4-axis PID controller for drone target centering.

Control mapping (corrected):
  X error (target left/right of center):
    -> YAW   : primary  -- rotate drone to face target horizontally
    -> ROLL  : secondary blend -- strafe laterally for faster correction

  Y error (target above/below center):
    -> THROTTLE : primary  -- climb/descend to match target altitude
    -> PITCH    : secondary blend -- nose up/down for faster vertical response
                  (target moves up -> drone pitches up; down -> pitches down)

  Distance error (bbox too small/large vs reference):
    -> PITCH : primary -- fly forward to approach, back to retreat

Sign conventions (image Y increases downward):
  x_error > 0  : target is RIGHT  -> yaw right (+), roll right (+)
  x_error < 0  : target is LEFT   -> yaw left  (-), roll left  (-)
  y_error > 0  : target is BELOW  -> throttle down (-), pitch down (-)
  y_error < 0  : target is ABOVE  -> throttle up   (+), pitch up   (+)
  dist_error>0 : too far away     -> pitch forward  (+)
  dist_error<0 : too close        -> pitch backward (-)
"""

import time
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PIDConfig:
    kp: float = 0.4
    ki: float = 0.05
    kd: float = 0.1
    dead_zone: float = 20.0
    integral_limit: float = 100.0
    output_limit: float = 1.0
    derivative_filter_alpha: float = 0.7


class PIDController:
    def __init__(self, config: PIDConfig, name: str = "PID"):
        self.cfg  = config
        self.name = name
        self._integral          = 0.0
        self._prev_error        = 0.0
        self._prev_derivative   = 0.0
        self._last_time: Optional[float] = None

    def compute(self, error: float) -> float:
        now = time.monotonic()
        if self._last_time is None:
            self._last_time = now
            return 0.0
        dt = now - self._last_time
        self._last_time = now
        if dt <= 0 or dt > 0.5:
            dt = 0.033

        # Dead zone
        if abs(error) < self.cfg.dead_zone:
            error = 0.0
            self._integral = 0.0

        p = self.cfg.kp * error

        self._integral += error * dt
        self._integral  = max(-self.cfg.integral_limit,
                              min(self.cfg.integral_limit, self._integral))
        i = self.cfg.ki * self._integral

        raw_d = (error - self._prev_error) / dt
        a     = self.cfg.derivative_filter_alpha
        fd    = a * self._prev_derivative + (1.0 - a) * raw_d
        self._prev_derivative = fd
        d = self.cfg.kd * fd

        self._prev_error = error

        out = p + i + d
        return max(-self.cfg.output_limit, min(self.cfg.output_limit, out))

    def reset(self):
        self._integral        = 0.0
        self._prev_error      = 0.0
        self._prev_derivative = 0.0
        self._last_time       = None


@dataclass
class ControlCommand:
    """
    Normalized drone control [-1, 1] on all axes.
      yaw      : -1=left,     +1=right
      pitch    : -1=backward, +1=forward (approach)
      throttle : -1=descend,  +1=ascend
      roll     : -1=left,     +1=right
    """
    yaw:      float = 0.0
    pitch:    float = 0.0
    throttle: float = 0.0
    roll:     float = 0.0

    def to_dict(self):
        return {
            "yaw":      round(self.yaw,      4),
            "pitch":    round(self.pitch,    4),
            "throttle": round(self.throttle, 4),
            "roll":     round(self.roll,     4),
        }

    def is_zero(self):
        return self.yaw == 0.0 and self.pitch == 0.0 \
            and self.throttle == 0.0 and self.roll == 0.0


class DroneAxisController:
    """
    Full 4-axis PID for target centering + approach.

    YAW      <- x_error        (primary horizontal)
    ROLL     <- x_error        (secondary, scaled by roll_blend)
    THROTTLE <- y_error        (primary vertical)
    PITCH    <- dist_error     (primary approach)
              + y_error        (secondary tilt, scaled by vertical_pitch_blend)

    Tune blend coefficients to taste:
      roll_blend=0.0           -> pure yaw, no strafe
      vertical_pitch_blend=0.0 -> pure throttle, no pitch tilt on vertical
    """

    # Secondary blend ratios -- adjust these for your airframe
    ROLL_BLEND:            float = 0.30
    VERTICAL_PITCH_BLEND:  float = 0.25

    YAW_DEFAULT      = PIDConfig(kp=0.35, ki=0.02, kd=0.08, dead_zone=15.0)
    ROLL_DEFAULT     = PIDConfig(kp=0.20, ki=0.01, kd=0.05, dead_zone=15.0)
    THROTTLE_DEFAULT = PIDConfig(kp=0.35, ki=0.02, kd=0.08, dead_zone=15.0)
    APPROACH_DEFAULT = PIDConfig(kp=0.30, ki=0.01, kd=0.06, dead_zone=0.08,
                                 output_limit=0.6)

    def __init__(
        self,
        yaw_config=None, roll_config=None,
        throttle_config=None, approach_config=None,
        frame_width=640, frame_height=480,
        roll_blend=None, vertical_pitch_blend=None,
    ):
        self.frame_width  = frame_width
        self.frame_height = frame_height
        self.roll_blend           = roll_blend           if roll_blend           is not None else self.ROLL_BLEND
        self.vertical_pitch_blend = vertical_pitch_blend if vertical_pitch_blend is not None else self.VERTICAL_PITCH_BLEND

        self.pid_yaw      = PIDController(yaw_config      or self.YAW_DEFAULT,      name="YAW")
        self.pid_roll     = PIDController(roll_config      or self.ROLL_DEFAULT,     name="ROLL")
        self.pid_throttle = PIDController(throttle_config  or self.THROTTLE_DEFAULT, name="THROT")
        self.pid_approach = PIDController(approach_config  or self.APPROACH_DEFAULT, name="APPRCH")

    def compute(
        self,
        target_cx: float,
        target_cy: float,
        distance_ratio: float = 1.0,
        desired_distance_ratio: float = 1.0,
    ) -> ControlCommand:
        """
        Compute 4-axis command from target pixel position and distance.
        """
        frame_cx = self.frame_width  / 2.0
        frame_cy = self.frame_height / 2.0

        # Signed pixel errors
        x_error    = target_cx - frame_cx   # +ve = target is RIGHT
        y_error    = target_cy - frame_cy   # +ve = target is BELOW (image Y down)
        dist_error = desired_distance_ratio - distance_ratio  # +ve = too far

        # YAW: rotate to face target horizontally
        yaw_cmd = self.pid_yaw.compute(x_error)

        # ROLL: lateral strafe blend (same error, lower gain)
        roll_cmd = self.pid_roll.compute(x_error) * self.roll_blend

        # THROTTLE: climb/descend -- y_error > 0 (below) -> descend (negative)
        throttle_cmd = -self.pid_throttle.compute(y_error)

        # PITCH primary: approach/retreat from distance error
        approach_cmd = self.pid_approach.compute(dist_error)

        # PITCH secondary: tilt nose up/down with vertical error
        # y_error < 0 (above) -> pitch up (+); y_error > 0 (below) -> pitch down (-)
        vertical_tilt = -(y_error / max(frame_cy, 1.0))          # normalized [-1,1]
        vertical_tilt  = max(-1.0, min(1.0, vertical_tilt))

        pitch_cmd = approach_cmd + vertical_tilt * self.vertical_pitch_blend
        pitch_cmd = max(-1.0, min(1.0, pitch_cmd))

        return ControlCommand(
            yaw=yaw_cmd, roll=roll_cmd,
            throttle=throttle_cmd, pitch=pitch_cmd,
        )

    def reset_all(self):
        self.pid_yaw.reset()
        self.pid_roll.reset()
        self.pid_throttle.reset()
        self.pid_approach.reset()
