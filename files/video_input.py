"""
video_input.py
==============
Abstracts all video sources: webcam, RTSP stream, UDP stream, or local file.
Provides a unified VideoSource interface used by the main loop.

Usage:
    src = VideoSource(source=0)           # webcam
    src = VideoSource(source="rtsp://...")  # RTSP drone camera
    src = VideoSource(source="udp://@:5600")  # UDP (e.g., MAVLink companion)
"""

import cv2
import time
import logging

logger = logging.getLogger(__name__)


class VideoSource:
    """
    Unified video source abstraction.
    Handles reconnection on stream drop (important for RTSP).
    """

    def __init__(
        self,
        source=0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        reconnect_delay: float = 2.0,
    ):
        """
        Args:
            source          : int (webcam index), str (RTSP/UDP URL or file path)
            width           : Desired capture width in pixels
            height          : Desired capture height in pixels
            fps             : Desired capture FPS (hint to backend, may not apply to RTSP)
            reconnect_delay : Seconds to wait before reconnecting on failure
        """
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.reconnect_delay = reconnect_delay

        self._cap: cv2.VideoCapture | None = None
        self._connected = False

        # Track actual frame dimensions after first successful read
        self.actual_width = width
        self.actual_height = height

        self._open()

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    #  Open                                                                #
    # ------------------------------------------------------------------ #

    def _open(self) -> bool:
        """Open the capture device / stream."""
        if self._cap and self._cap.isOpened():
            self._cap.release()

        # For RTSP: prefer TCP transport to reduce packet loss
        if isinstance(self.source, str) and self.source.startswith("rtsp"):
            self._cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            self._cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            self._cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        else:
            self._cap = cv2.VideoCapture(self.source)

        if not self._cap.isOpened():
            logger.error(f"[VideoSource] Cannot open source: {self.source}")
            self._connected = False
            return False

        # Apply hints — ignored by many RTSP backends but help webcams
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS,          self.fps)

        self._finalize_open()
        return True

    def _finalize_open(self) -> None:
        """Read back actual dimensions and minimize buffer after any successful open."""
        # Minimize internal buffer to reduce latency (critical for real-time control)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.actual_width  = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._connected    = True
        logger.info(
            f"[VideoSource] Opened {self.source} @ "
            f"{self.actual_width}x{self.actual_height}"
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def read(self):
        """
        Read next frame. Attempts one reconnection on failure.

        Returns:
            (ret: bool, frame: np.ndarray | None)
        """
        if not self._connected or self._cap is None:
            time.sleep(self.reconnect_delay)
            self._open()
            return False, None

        ret, frame = self._cap.read()

        if not ret:
            logger.warning("[VideoSource] Frame read failed — attempting reconnect...")
            self._connected = False
            time.sleep(self.reconnect_delay)
            self._open()
            return False, None

        return True, frame

    def get_frame_center(self) -> tuple[int, int]:
        """Returns pixel coordinate of the frame center."""
        return self.actual_width // 2, self.actual_height // 2

    def release(self):
        """Release capture resources."""
        if self._cap:
            self._cap.release()
        logger.info("[VideoSource] Released.")

    @property
    def is_connected(self) -> bool:
        return self._connected

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()
