"""
main.py
=======
Drone Target Lock System — Main Entry Point

Orchestrates:
 VideoSource → TrackerManager → DroneAxisController → DroneController
                    ↓
                HUDOverlay ← ROISelector (mouse)

Architecture flow:
  1. Open video source
  2. Wait for user to draw ROI (bounding box) with mouse
  3. Initialize CSRT tracker on selected ROI
  4. Each frame:
     a. Read frame from source
     b. Update tracker → get bbox + center
     c. Compute pixel error (target center - frame center)
     d. Run PID → get normalized [yaw, pitch, throttle]
     e. Send commands to drone controller
     f. Draw HUD overlay
     g. Display frame

Controls:
  - Left-click + drag : Select target
  - R                 : Reset tracker
  - Q / Esc           : Quit

Usage:
    # Webcam simulation:
    python main.py

    # RTSP drone camera:
    python main.py --source rtsp://192.168.1.100:554/live

    # UDP stream (e.g., GStreamer from companion computer):
    python main.py --source "udp://@:5600"

    # MAVLink backend:
    python main.py --backend mavlink --mavlink-conn "udp:127.0.0.1:14550"
"""

import sys
import os
import argparse
import logging
import time
import cv2

# ── Path setup (allows running from project root) ─────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_input      import VideoSource
from tracker          import TrackerManager, TrackerState
from pid_controller   import DroneAxisController, PIDConfig
from control_layer    import create_controller, ControlCommand
from ui               import ROISelector, HUDOverlay, SelectionState

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("DroneTargetLock")

WINDOW_NAME = "Drone Target Lock"


# ─────────────────────────────────────────────────────────────────────────────
# Main application class
# ─────────────────────────────────────────────────────────────────────────────

class DroneTargetLockSystem:
    """
    Main system class.
    Keeps all components in one place and runs the main loop.
    """

    def __init__(self, args):
        self.args = args

        # ── Components (initialized in setup()) ────────────────────────────
        self.video:     VideoSource        = None
        self.tracker:   TrackerManager     = None
        self.axis_ctrl: DroneAxisController = None
        self.drone:     object             = None
        self.selector:  ROISelector        = None
        self.hud:       HUDOverlay         = None

        # ── State ──────────────────────────────────────────────────────────
        self._last_command             = ControlCommand()
        self._last_distance_ratio      = 1.0
        self._target_distance_ratio    = 1.0

        # Performance profiling
        self._frame_count = 0
        self._start_time  = time.monotonic()

    # ------------------------------------------------------------------ #
    #  Setup                                                               #
    # ------------------------------------------------------------------ #

    def setup(self):
        logger.info("=== Drone Target Lock System Starting ===")

        # ── 1. Video source ────────────────────────────────────────────────
        self.video = VideoSource(
            source=self.args.source,
            width=self.args.width,
            height=self.args.height,
            fps=self.args.fps,
        )

        fw = self.video.actual_width
        fh = self.video.actual_height
        logger.info(f"Frame size: {fw}x{fh}")

        # ── 2. Single-target tracker ───────────────────────────────────────
        self.tracker = TrackerManager(
            tracker_type=self.args.tracker,
            use_kalman=not self.args.no_kalman,
            frame_width=fw,
            frame_height=fh,
        )

        # ── 3. PID / axis controller ───────────────────────────────────────
        yaw_cfg = PIDConfig(
            kp=self.args.kp, ki=self.args.ki, kd=self.args.kd,
            dead_zone=self.args.dead_zone
        )
        self.axis_ctrl = DroneAxisController(
            yaw_config=yaw_cfg,
            frame_width=fw,
            frame_height=fh,
        )

        # ── 4. Drone controller backend ────────────────────────────────────
        backend_kwargs = {}
        if self.args.backend == "mavlink":
            backend_kwargs["connection_string"] = self.args.mavlink_conn
        elif self.args.backend == "betaflight":
            backend_kwargs["port"] = self.args.serial_port
            backend_kwargs["baud"] = self.args.serial_baud

        self.drone = create_controller(self.args.backend, **backend_kwargs)
        self.drone.connect()

        # ── 5. UI ──────────────────────────────────────────────────────────
        self.selector = ROISelector()
        self.hud      = HUDOverlay(
            fw, fh,
            dead_zone_px=self.args.dead_zone,
            tracker_type=self.args.tracker,
            source_str=self.args.source,
        )

        # ── 6. OpenCV window + mouse callback ─────────────────────────────
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, fw, fh)
        cv2.setMouseCallback(WINDOW_NAME, self.selector.mouse_callback)

        logger.info("Setup complete. Draw a bounding box to select target.")

    # ------------------------------------------------------------------ #
    #  Main loop                                                           #
    # ------------------------------------------------------------------ #

    def run(self):
        """Main processing loop."""
        current_frame = None

        while True:
            # ── Read frame ────────────────────────────────────────────────
            ret, frame = self.video.read()
            if not ret or frame is None:
                logger.warning("No frame received, retrying...")
                time.sleep(0.033)
                continue

            current_frame = frame.copy()
            self._frame_count += 1

            # ── New ROI selection → initialize tracker ────────────────────
            if self.selector.state == SelectionState.SELECTED:
                bbox = self.selector.consume()
                if bbox and current_frame is not None:
                    self.tracker.reset()
                    ok = self.tracker.initialize(current_frame, bbox)
                    if ok is not False:
                        logger.info(f"Tracker initialized at {bbox}")
                        self._target_distance_ratio = 1.0
                        self.axis_ctrl.reset_all()
                    else:
                        logger.error("Tracker initialization failed")

            # ── Update tracker ────────────────────────────────────────────
            result = self.tracker.update(frame)

            # ── PID from tracker result ───────────────────────────────────
            command        = ControlCommand()
            error_x        = 0.0
            error_y        = 0.0
            distance_ratio = self._last_distance_ratio

            if result.state == TrackerState.TRACKING and result.bbox:
                tcx, tcy = result.smoothed_center or result.center or (
                    self.video.actual_width / 2, self.video.actual_height / 2
                )
                error_x        = tcx - self.video.actual_width  / 2
                error_y        = tcy - self.video.actual_height / 2
                distance_ratio = self.tracker.estimate_distance_ratio(result.bbox)
                self._last_distance_ratio = distance_ratio
                command = self.axis_ctrl.compute(
                    target_cx=tcx,
                    target_cy=tcy,
                    distance_ratio=distance_ratio,
                    desired_distance_ratio=self._target_distance_ratio,
                )
            elif result.state in (TrackerState.LOST, TrackerState.SEARCHING):
                self.axis_ctrl.reset_all()
                if result.smoothed_center:
                    px, py  = result.smoothed_center
                    error_x = px - self.video.actual_width  / 2
                    error_y = py - self.video.actual_height / 2

            self._last_command = command
            self.drone.send_command(command)

            # ── Draw HUD ──────────────────────────────────────────────────
            self.hud.draw(
                frame,
                bbox=result.bbox,
                state_str=result.state.name,
                confidence=result.confidence,
                distance_ratio=distance_ratio,
                error_x=error_x,
                error_y=error_y,
                command_dict=command.to_dict(),
                selector=self.selector,
            )

            cv2.imshow(WINDOW_NAME, frame)

            # ── Keyboard input ────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                logger.info("Quit key pressed.")
                break

            elif key == ord('r'):
                logger.info("Tracker reset by user.")
                self.tracker.reset()
                self.axis_ctrl.reset_all()

            elif key == ord('p'):
                cmd = command.to_dict()
                print(f"\n[PID STATE] Error: ({error_x:.1f}px, {error_y:.1f}px) "
                      f"Dist: {distance_ratio:.2f}x  State: {result.state.name}")
                print(f"  Command: {cmd}")

            elif key == ord('+') or key == ord('='):
                self._target_distance_ratio = min(self._target_distance_ratio + 0.1, 3.0)
                logger.info(f"Target dist ratio: {self._target_distance_ratio:.1f}x")

            elif key == ord('-'):
                self._target_distance_ratio = max(self._target_distance_ratio - 0.1, 0.1)
                logger.info(f"Target dist ratio: {self._target_distance_ratio:.1f}x")

    # ------------------------------------------------------------------ #
    #  Teardown                                                            #
    # ------------------------------------------------------------------ #

    def teardown(self):
        total_time = time.monotonic() - self._start_time
        avg_fps = self._frame_count / max(total_time, 1)
        logger.info(
            f"Session stats: {self._frame_count} frames, "
            f"{total_time:.1f}s, avg {avg_fps:.1f} FPS"
        )
        self.drone.disconnect()
        self.video.release()
        cv2.destroyAllWindows()
        logger.info("=== System shut down cleanly ===")


# ─────────────────────────────────────────────────────────────────────────────
# CLI argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Drone Target Lock System",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Video
    parser.add_argument(
        "--source", default=0,
        help=(
            "Video source:\n"
            "  0          : Default webcam\n"
            "  1, 2 ...   : Other webcam index\n"
            "  rtsp://... : RTSP stream\n"
            "  udp://@:5600: UDP stream"
        )
    )
    parser.add_argument("--width",  type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps",    type=int, default=30)

    # Tracker
    parser.add_argument(
        "--tracker", default="CSRT", choices=["CSRT","KCF","MOSSE","MIL"],
        help="OpenCV tracker type (default: CSRT)"
    )
    parser.add_argument(
        "--no-kalman", action="store_true",
        help="Disable Kalman filter smoothing"
    )

    # PID
    parser.add_argument("--kp",        type=float, default=0.35)
    parser.add_argument("--ki",        type=float, default=0.02)
    parser.add_argument("--kd",        type=float, default=0.08)
    parser.add_argument("--dead-zone", type=float, default=20.0,
                        help="Dead zone radius in pixels")

    # Drone backend
    parser.add_argument(
        "--backend", default="sim", choices=["sim", "mavlink", "betaflight"],
        help="Drone control backend (default: sim)"
    )
    parser.add_argument("--mavlink-conn", default="udp:127.0.0.1:14550")
    parser.add_argument("--serial-port",  default="/dev/ttyUSB0")
    parser.add_argument("--serial-baud",  type=int, default=115200)

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    # Allow --source 0 as integer
    try:
        args.source = int(args.source)
    except (ValueError, TypeError):
        pass  # Keep as string (RTSP/UDP URL)

    system = DroneTargetLockSystem(args)
    try:
        system.setup()
        system.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
    finally:
        system.teardown()
