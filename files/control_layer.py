"""
control_layer.py
================
Abstract drone command dispatch layer.

Separates control *computation* from control *execution*.
The same PID outputs can be routed to:

  1. SimulatedDrone   — Print/log to console (default, for development)
  2. MAVLinkDrone     — ArduPilot / PX4 via pymavlink (RC_CHANNELS_OVERRIDE)
  3. BetaflightDrone  — Betaflight via MSP (Multiwii Serial Protocol)
  4. (Future) ROS2Drone, DJISDKDrone, etc.

Architecture:
  DroneController (ABC)
    ├── SimulatedDrone
    ├── MAVLinkDrone
    └── BetaflightDrone

Usage:
    controller = SimulatedDrone()
    # or:
    controller = MAVLinkDrone(connection_string="udp:127.0.0.1:14550")

    controller.send_command(command)
    controller.arm()
    controller.disarm()
"""

import logging
import time
import struct
from abc import ABC, abstractmethod
from typing import Optional

from pid_controller import ControlCommand

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base class
# ─────────────────────────────────────────────────────────────────────────────

class DroneController(ABC):
    """
    Abstract interface for drone command dispatch.
    Subclass this to add new flight controller backends.
    """

    def __init__(self):
        self._armed = False
        self._command_count = 0

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to drone/FC. Returns True on success."""
        ...

    @abstractmethod
    def send_command(self, command: ControlCommand) -> bool:
        """Send a control command. Returns True on success."""
        ...

    @abstractmethod
    def arm(self) -> bool:
        """Arm the drone motors."""
        ...

    @abstractmethod
    def disarm(self) -> bool:
        """Disarm the drone motors."""
        ...

    @abstractmethod
    def disconnect(self):
        """Clean shutdown."""
        ...

    @property
    def is_armed(self) -> bool:
        return self._armed

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()


# ─────────────────────────────────────────────────────────────────────────────
# Backend 1: Simulated (Development / Testing)
# ─────────────────────────────────────────────────────────────────────────────

class SimulatedDrone(DroneController):
    """
    Prints control commands to console.
    Use this during development before connecting to real hardware.
    """

    def __init__(self, log_interval: float = 0.1):
        """
        Args:
            log_interval : Minimum seconds between log prints (rate-limit verbose output)
        """
        super().__init__()
        self.log_interval = log_interval
        self._last_log_time = 0.0

    def connect(self) -> bool:
        logger.info("[SIM] Simulated drone connected.")
        return True

    def send_command(self, command: ControlCommand) -> bool:
        self._command_count += 1
        now = time.monotonic()

        # Rate-limit console output
        if now - self._last_log_time >= self.log_interval:
            self._last_log_time = now
            cmd_dict = command.to_dict()
            print(
                f"[DRONE CMD #{self._command_count:05d}] "
                f"YAW={cmd_dict['yaw']:+.3f}  "
                f"PITCH={cmd_dict['pitch']:+.3f}  "
                f"THROTTLE={cmd_dict['throttle']:+.3f}  "
                f"ROLL={cmd_dict['roll']:+.3f}"
            )
        return True

    def arm(self) -> bool:
        self._armed = True
        logger.info("[SIM] Motors ARMED")
        return True

    def disarm(self) -> bool:
        self._armed = False
        logger.info("[SIM] Motors DISARMED")
        return True

    def disconnect(self):
        logger.info("[SIM] Disconnected.")


# ─────────────────────────────────────────────────────────────────────────────
# Backend 2: MAVLink (ArduPilot / PX4)
# ─────────────────────────────────────────────────────────────────────────────

class MAVLinkDrone(DroneController):
    """
    Sends RC_CHANNELS_OVERRIDE via MAVLink.

    Requires: pip install pymavlink

    RC Channel Mapping (ArduPilot default):
      CH1 = Roll
      CH2 = Pitch
      CH3 = Throttle
      CH4 = Yaw

    Commands are mapped from [-1, 1] to RC PWM range [1000, 2000].
    Center (hover/neutral) = 1500 µs.
    """

    RC_MIN = 1000
    RC_MAX = 2000
    RC_MID = 1500

    def __init__(self, connection_string: str = "udp:127.0.0.1:14550"):
        """
        Args:
            connection_string : MAVLink connection string.
                                Examples:
                                  "udp:127.0.0.1:14550"   (SITL / UDP)
                                  "/dev/ttyUSB0,57600"    (Serial)
                                  "tcp:192.168.1.1:5760"  (TCP)
        """
        super().__init__()
        self.connection_string = connection_string
        self._mav = None  # pymavlink MAVLink connection

    def connect(self) -> bool:
        try:
            from pymavlink import mavutil
            logger.info(f"[MAVLink] Connecting to {self.connection_string}...")
            self._mav = mavutil.mavlink_connection(self.connection_string)
            self._mav.wait_heartbeat(timeout=10)
            logger.info(
                f"[MAVLink] Heartbeat received from system "
                f"{self._mav.target_system}, component {self._mav.target_component}"
            )
            return True
        except ImportError:
            logger.error("[MAVLink] pymavlink not installed: pip install pymavlink")
            return False
        except Exception as e:
            logger.error(f"[MAVLink] Connection failed: {e}")
            return False

    def _normalized_to_pwm(self, value: float) -> int:
        """Convert [-1.0, +1.0] → [1000, 2000] µs PWM."""
        value = max(-1.0, min(1.0, value))
        return int(self.RC_MID + value * 500)

    def send_command(self, command: ControlCommand) -> bool:
        if not self._mav:
            return False
        try:
            # Map PID outputs to RC channels
            ch_roll     = self._normalized_to_pwm(command.roll)
            ch_pitch    = self._normalized_to_pwm(command.pitch)
            ch_throttle = self._normalized_to_pwm(command.throttle)
            ch_yaw      = self._normalized_to_pwm(command.yaw)

            # RC_CHANNELS_OVERRIDE: override RC input to autopilot
            # Channels not used: set to 0 (means "don't override")
            self._mav.mav.rc_channels_override_send(
                self._mav.target_system,
                self._mav.target_component,
                ch_roll,       # CH1
                ch_pitch,      # CH2
                ch_throttle,   # CH3
                ch_yaw,        # CH4
                0, 0, 0, 0     # CH5-8: not overridden
            )
            self._command_count += 1
            return True
        except Exception as e:
            logger.error(f"[MAVLink] send_command failed: {e}")
            return False

    def arm(self) -> bool:
        if not self._mav:
            return False
        try:
            from pymavlink import mavutil
            self._mav.mav.command_long_send(
                self._mav.target_system,
                self._mav.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0, 1, 0, 0, 0, 0, 0, 0  # 1 = ARM
            )
            self._armed = True
            logger.info("[MAVLink] ARM command sent")
            return True
        except Exception as e:
            logger.error(f"[MAVLink] ARM failed: {e}")
            return False

    def disarm(self) -> bool:
        if not self._mav:
            return False
        try:
            from pymavlink import mavutil
            self._mav.mav.command_long_send(
                self._mav.target_system,
                self._mav.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0, 0, 0, 0, 0, 0, 0, 0  # 0 = DISARM
            )
            self._armed = False
            logger.info("[MAVLink] DISARM command sent")
            return True
        except Exception as e:
            logger.error(f"[MAVLink] DISARM failed: {e}")
            return False

    def disconnect(self):
        if self._mav:
            self._mav.close()
            logger.info("[MAVLink] Disconnected.")


# ─────────────────────────────────────────────────────────────────────────────
# Backend 3: Betaflight / MSP (Multiwii Serial Protocol)
# ─────────────────────────────────────────────────────────────────────────────

class BetaflightDrone(DroneController):
    """
    Sends RC commands to Betaflight FC via MSP (MSP_SET_RAW_RC).

    Requires: pip install pyserial

    MSP RC Channel Order (Betaflight default):
      [ROLL, PITCH, THROTTLE, YAW, AUX1, AUX2, AUX3, AUX4]

    Throttle range: 1000–2000 µs
    """

    MSP_SET_RAW_RC = 200
    MSP_HEADER = b"$M<"

    def __init__(self, port: str = "/dev/ttyUSB0", baud: int = 115200):
        """
        Args:
            port : Serial port connected to FC (e.g., /dev/ttyUSB0, COM3)
            baud : Baud rate (typically 115200 for Betaflight)
        """
        super().__init__()
        self.port = port
        self.baud = baud
        self._serial = None

    def connect(self) -> bool:
        try:
            import serial
            self._serial = serial.Serial(self.port, self.baud, timeout=1)
            logger.info(f"[MSP] Connected to {self.port} @ {self.baud}")
            return True
        except ImportError:
            logger.error("[MSP] pyserial not installed: pip install pyserial")
            return False
        except Exception as e:
            logger.error(f"[MSP] Connection failed: {e}")
            return False

    def _build_msp_packet(self, cmd: int, data: bytes) -> bytes:
        """
        Build MSP packet:
          $ M < [size] [cmd] [data...] [checksum]
        Checksum = XOR of size, cmd, and all data bytes.
        """
        size = len(data)
        checksum = size ^ cmd
        for byte in data:
            checksum ^= byte
        return self.MSP_HEADER + bytes([size, cmd]) + data + bytes([checksum])

    def _normalized_to_pwm(self, value: float) -> int:
        value = max(-1.0, min(1.0, value))
        return int(1500 + value * 500)

    def send_command(self, command: ControlCommand) -> bool:
        if not self._serial:
            return False
        try:
            channels = [
                self._normalized_to_pwm(command.roll),       # Roll
                self._normalized_to_pwm(command.pitch),      # Pitch
                self._normalized_to_pwm(command.throttle),   # Throttle
                self._normalized_to_pwm(command.yaw),        # Yaw
                1000, 1000, 1000, 1000                        # AUX1-4
            ]
            # Pack as little-endian unsigned shorts
            data = struct.pack("<" + "H" * len(channels), *channels)
            packet = self._build_msp_packet(self.MSP_SET_RAW_RC, data)
            self._serial.write(packet)
            self._command_count += 1
            return True
        except Exception as e:
            logger.error(f"[MSP] send_command failed: {e}")
            return False

    def arm(self) -> bool:
        # Betaflight arms via AUX channel or stick input — not via MSP directly.
        # This is a placeholder; actual arming logic depends on FC config.
        logger.warning("[MSP] ARM: Set AUX1 channel high in FC configurator to arm.")
        self._armed = True
        return True

    def disarm(self) -> bool:
        self._armed = False
        return True

    def disconnect(self):
        if self._serial:
            self._serial.close()
            logger.info("[MSP] Serial port closed.")


# ─────────────────────────────────────────────────────────────────────────────
# Factory function
# ─────────────────────────────────────────────────────────────────────────────

def create_controller(backend: str = "sim", **kwargs) -> DroneController:
    """
    Convenience factory.

    Args:
        backend : "sim", "mavlink", or "betaflight"
        **kwargs: Passed to the backend constructor

    Returns:
        DroneController instance
    """
    backends = {
        "sim": SimulatedDrone,
        "mavlink": MAVLinkDrone,
        "betaflight": BetaflightDrone,
    }
    cls = backends.get(backend.lower())
    if not cls:
        raise ValueError(f"Unknown backend: {backend}. Choose from {list(backends)}")
    return cls(**kwargs)
