"""
Urban Safety AI — Webcam Thread
High-FPS live capture with graceful error handling and frame skipping.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger("urban_safety.webcam")

try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False


class WebcamThread(threading.Thread):
    """
    Daemon thread that reads frames from a webcam and calls `frame_callback`
    for each new frame. Runs until `stop()` is called.

    Args:
        cam_index:       OpenCV camera index (0 = default).
        frame_callback:  Called with (numpy BGR frame,) in this thread.
        error_callback:  Called with (str error message,) on failure.
        target_fps:      Desired FPS cap (default 30).
    """

    def __init__(
        self,
        cam_index = 0,
        frame_callback: Optional[Callable] = None,
        error_callback: Optional[Callable] = None,
        target_fps: int = 30,
    ):
        super().__init__(daemon=True, name=f"webcam-{cam_index}")
        self._cam_index     = cam_index
        self._frame_cb      = frame_callback
        self._error_cb      = error_callback
        self._target_fps    = target_fps
        self._interval      = 1.0 / target_fps
        self._stop_event    = threading.Event()
        self._cap           = None

    def run(self):
        if not CV2_OK:
            self._emit_error("OpenCV (cv2) not installed. Run: pip install opencv-python")
            return

        logger.info("Opening camera index %d", self._cam_index)
        try:
            if isinstance(self._cam_index, str):
                cap = cv2.VideoCapture(self._cam_index)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)  
            else:
                cap = cv2.VideoCapture(self._cam_index, cv2.CAP_DSHOW if self._is_windows() else cv2.CAP_ANY)
        except Exception as e:
            self._emit_error(f"Failed to open camera {self._cam_index}: {e}")
            return

        if not cap.isOpened():
            cap.release()
            self._emit_error(f"Camera {self._cam_index} could not be opened.")
            return

        # Optimize capture settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS,          self._target_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)  # Minimize buffer lag

        self._cap = cap
        logger.info("Camera %d opened successfully", self._cam_index)

        consecutive_errors = 0
        max_errors = 10

        while not self._stop_event.is_set():
            t0 = time.perf_counter()
            ret, frame = cap.read()

            if not ret or frame is None:
                consecutive_errors += 1
                logger.warning("Frame read failed (%d/%d)", consecutive_errors, max_errors)
                if consecutive_errors >= max_errors:
                    self._emit_error(f"Camera {self._cam_index} lost connection after {max_errors} errors.")
                    break
                time.sleep(0.05)
                continue

            consecutive_errors = 0

            if self._frame_cb:
                try:
                    self._frame_cb(frame)
                except Exception as e:
                    logger.debug("Frame callback error: %s", e)

            # FPS limiter
            elapsed = time.perf_counter() - t0
            sleep_for = self._interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

        cap.release()
        logger.info("Camera %d released", self._cam_index)

    def stop(self):
        self._stop_event.set()
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass

    def _emit_error(self, msg: str):
        logger.error(msg)
        if self._error_cb:
            try:
                self._error_cb(msg)
            except Exception:
                pass

    @staticmethod
    def _is_windows() -> bool:
        import platform
        return platform.system() == "Windows"
