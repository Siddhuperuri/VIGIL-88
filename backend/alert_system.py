"""
Urban Safety AI — Alert System
Sound alerts via winsound (Windows) or playsound/beep fallbacks.
Debounced: same alert won't fire more than once per 5 seconds.
"""
from __future__ import annotations

import logging
import os
import sys
import time
import threading
from pathlib import Path

logger = logging.getLogger("urban_safety.alerts")

ALERT_COOLDOWN = 5.0  # seconds between same-label alerts


class AlertSystem:
    """
    Fires auditory alerts on threat detection.
    Non-blocking (runs in background thread).
    """

    def __init__(self):
        self._last_alert: dict[str, float] = {}
        self._lock = threading.Lock()
        self._is_windows = sys.platform.startswith("win")
        logger.info("AlertSystem initialised | platform=%s", sys.platform)

    def trigger(self, label: str, confidence: float, sound_enabled: bool = True):
        """
        Fire alert for the given label (fire / accident).
        Debounced per label — won't repeat within ALERT_COOLDOWN seconds.
        """
        now = time.time()
        with self._lock:
            last = self._last_alert.get(label, 0.0)
            if now - last < ALERT_COOLDOWN:
                return  # debounce
            self._last_alert[label] = now

        logger.info("ALERT: %s (conf=%.2f%%)", label.upper(), confidence * 100)

        if sound_enabled:
            threading.Thread(target=self._play_sound, args=(label,), daemon=True).start()

    def _play_sound(self, label: str):
        try:
            if self._is_windows:
                self._play_windows(label)
            else:
                self._play_posix(label)
        except Exception as e:
            logger.debug("Sound alert failed: %s", e)

    def _play_windows(self, label: str):
        import winsound
        if label == "fire":
            # Urgent ascending beeps
            for freq, dur in [(880, 120), (1040, 120), (1200, 200), (1040, 120), (1200, 300)]:
                winsound.Beep(freq, dur)
        elif label == "accident":
            # Double low beep
            for freq, dur in [(660, 200), (660, 200)]:
                winsound.Beep(freq, dur)
                time.sleep(0.08)

    def _play_posix(self, label: str):
        # Fallback: try playsound, then terminal bell
        try:
            import playsound
            sounds_dir = Path(__file__).resolve().parent.parent / "assets" / "sounds"
            sound_file = sounds_dir / f"{label}.wav"
            if sound_file.exists():
                playsound.playsound(str(sound_file), block=False)
                return
        except Exception:
            pass

        # Terminal bell fallback
        try:
            os.system('printf "\\a"')
        except Exception:
            pass
