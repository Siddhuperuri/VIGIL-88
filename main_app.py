#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║     VIGIL-88 AI — REAL-TIME THREAT DETECTION SYSTEM         ║
║              PyQt6 Desktop Application  v4.0                     ║
║         Fire · Accident · Normal  |  Live Webcam + Image         ║
╚══════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import sys
import os
import time
import math
import random
import threading
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

# ── Fix imports before Qt ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("urban_safety.app")

# ── Qt imports ────────────────────────────────────────────────────────────────
try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QFrame, QFileDialog, QProgressBar,
        QSizePolicy, QStackedWidget, QGraphicsDropShadowEffect,
        QScrollArea, QTextEdit, QSlider, QComboBox, QCheckBox
    )
    from PyQt6.QtCore import (
        Qt, QTimer, QThread, pyqtSignal, QPropertyAnimation,
        QEasingCurve, QPoint, QSize, QRect, QSequentialAnimationGroup,
        QParallelAnimationGroup, QVariantAnimation
    )
    from PyQt6.QtGui import (
        QColor, QFont, QPixmap, QImage, QIcon, QPainter, QPen,
        QBrush, QLinearGradient, QRadialGradient, QPalette,
        QFontDatabase, QConicalGradient
    )
except ImportError as e:
    print(f"[ERROR] PyQt6 not installed: {e}")
    print("Run: pip install PyQt6")
    sys.exit(1)

# ── Optional OpenCV ───────────────────────────────────────────────────────────
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not found — webcam disabled. pip install opencv-python")

# ── Backend imports ───────────────────────────────────────────────────────────
from backend.inference_engine import InferenceEngine, DummyEngine
from backend.webcam_thread import WebcamThread
from backend.alert_system import AlertSystem

# ═══════════════════════════════════════════════════════════════════════════════
# DESIGN TOKENS
# ═══════════════════════════════════════════════════════════════════════════════
DARK_BG       = "#0A0E1A"
PANEL_BG      = "#0F1526"
CARD_BG       = "#131929"
BORDER_COLOR  = "#1E2D4A"
ACCENT_BLUE   = "#2979FF"
ACCENT_CYAN   = "#00E5FF"
ACCENT_GLOW   = "#1565C0"
TEXT_PRIMARY  = "#E8EDF5"
TEXT_SECONDARY= "#8899AA"
TEXT_DIM      = "#4A5568"

FIRE_COLOR    = "#FF3D00"
FIRE_GLOW     = "#BF360C"
ACCIDENT_CLR  = "#FF8F00"
ACCIDENT_GLOW = "#E65100"
NORMAL_COLOR  = "#00E676"
NORMAL_GLOW   = "#1B5E20"

STATUS_COLORS = {
    "fire":     (FIRE_COLOR,     FIRE_GLOW,     "🔥 FIRE DETECTED"),
    "accident": (ACCIDENT_CLR,   ACCIDENT_GLOW, "⚠️ ACCIDENT DETECTED"),
    "normal":   (NORMAL_COLOR,   NORMAL_GLOW,   "✅ SCENE NORMAL"),
    "idle":     (TEXT_DIM,       DARK_BG,       "● AWAITING INPUT"),
}

# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM WIDGETS
# ═══════════════════════════════════════════════════════════════════════════════

class GlowLabel(QLabel):
    """Label with animated glow effect."""

    def __init__(self, text="", glow_color=ACCENT_CYAN, parent=None):
        super().__init__(text, parent)
        self._glow_color = glow_color
        self._glow_anim = QVariantAnimation(self)
        self._glow_anim.setDuration(1200)
        self._glow_anim.setStartValue(0.3)
        self._glow_anim.setEndValue(1.0)
        self._glow_anim.setEasingCurve(QEasingCurve.Type.SineCurve)
        self._glow_anim.setLoopCount(-1)
        self._glow_anim.valueChanged.connect(self._update_glow)
        self._shadow = QGraphicsDropShadowEffect(self)
        self._shadow.setOffset(0, 0)
        self._shadow.setBlurRadius(20)
        self._shadow.setColor(QColor(glow_color))
        self.setGraphicsEffect(self._shadow)

    def start_glow(self):
        self._glow_anim.start()

    def stop_glow(self):
        self._glow_anim.stop()

    def set_glow_color(self, color: str):
        self._glow_color = color
        self._shadow.setColor(QColor(color))

    def _update_glow(self, v):
        self._shadow.setBlurRadius(int(10 + v * 30))


class AnimatedBar(QWidget):
    """Animated confidence progress bar with gradient fill."""

    def __init__(self, label="", color=ACCENT_BLUE, parent=None):
        super().__init__(parent)
        self._value = 0.0
        self._target = 0.0
        self._color = color
        self._label = label
        self.setFixedHeight(48)
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(16)
        self._anim_timer.timeout.connect(self._animate_step)

    def set_value(self, v: float):
        self._target = max(0.0, min(1.0, v))
        self._anim_timer.start()

    def set_color(self, c: str):
        self._color = c
        self.update()

    def _animate_step(self):
        diff = self._target - self._value
        if abs(diff) < 0.002:
            self._value = self._target
            self._anim_timer.stop()
        else:
            self._value += diff * 0.15
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # Label
        p.setPen(QColor(TEXT_SECONDARY))
        p.setFont(QFont("Consolas", 9))
        p.drawText(QRect(0, 0, w, 18), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self._label)

        # Track
        track_y = 24
        track_h = 14
        bar_rect = QRect(0, track_y, w, track_h)
        p.setBrush(QColor(BORDER_COLOR))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(bar_rect, 7, 7)

        # Fill gradient
        fill_w = int(w * self._value)
        if fill_w > 0:
            grad = QLinearGradient(0, track_y, fill_w, track_y)
            base = QColor(self._color)
            light = base.lighter(130)
            grad.setColorAt(0, base)
            grad.setColorAt(1, light)
            p.setBrush(QBrush(grad))
            p.drawRoundedRect(QRect(0, track_y, fill_w, track_h), 7, 7)

            # Shimmer
            shimmer_pos = int((time.time() * 200) % (w + 60)) - 30
            shimmer_grad = QLinearGradient(shimmer_pos - 20, 0, shimmer_pos + 20, 0)
            shimmer_grad.setColorAt(0, QColor(255, 255, 255, 0))
            shimmer_grad.setColorAt(0.5, QColor(255, 255, 255, 50))
            shimmer_grad.setColorAt(1, QColor(255, 255, 255, 0))
            p.setBrush(QBrush(shimmer_grad))
            p.drawRoundedRect(QRect(0, track_y, fill_w, track_h), 7, 7)

        # Percentage text
        pct_text = f"{self._value * 100:.1f}%"
        p.setPen(QColor(TEXT_PRIMARY))
        p.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        p.drawText(QRect(0, track_y, w, track_h),
                   Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, pct_text)

        p.end()


class PulsingCircle(QWidget):
    """Animated pulsing status indicator."""

    def __init__(self, color=NORMAL_COLOR, parent=None):
        super().__init__(parent)
        self._color = color
        self._pulse = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        self.setFixedSize(24, 24)
        self._active = True

    def set_color(self, c: str):
        self._color = c
        self.update()

    def set_active(self, a: bool):
        self._active = a

    def _tick(self):
        if self._active:
            self._pulse = (self._pulse + 0.05) % (2 * math.pi)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c = QColor(self._color)
        cx, cy = self.width() // 2, self.height() // 2

        if self._active:
            # Outer pulse ring
            alpha = int(80 + 80 * math.sin(self._pulse))
            ring_col = QColor(c)
            ring_col.setAlpha(alpha)
            pulse_radius = int(9 + 3 * math.sin(self._pulse))
            p.setPen(QPen(ring_col, 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(cx - pulse_radius, cy - pulse_radius, pulse_radius * 2, pulse_radius * 2)

        # Inner core
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(c))
        p.drawEllipse(cx - 5, cy - 5, 10, 10)
        p.end()


class ThreatBadge(QWidget):
    """Large animated threat status display."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._label = "STANDBY"
        self._color = TEXT_DIM
        self._glow_alpha = 0
        self._pulse = 0.0
        self._glow_timer = QTimer(self)
        self._glow_timer.setInterval(30)
        self._glow_timer.timeout.connect(self._tick)
        self._glow_timer.start()
        self.setMinimumHeight(80)

    def set_status(self, label: str, color: str):
        self._label = label
        self._color = color
        self.update()

    def _tick(self):
        self._pulse = (self._pulse + 0.04) % (2 * math.pi)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        c = QColor(self._color)

        # Glow background
        glow_alpha = int(15 + 10 * math.sin(self._pulse))
        bg_col = QColor(c)
        bg_col.setAlpha(glow_alpha)
        p.setBrush(QBrush(bg_col))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(0, 0, w, h, 12, 12)

        # Border
        border_col = QColor(c)
        border_alpha = int(60 + 40 * math.sin(self._pulse))
        border_col.setAlpha(border_alpha)
        pen = QPen(border_col, 2)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(1, 1, w - 2, h - 2, 12, 12)

        # Text
        p.setPen(QColor(c))
        font = QFont("Consolas", 18, QFont.Weight.Bold)
        p.setFont(font)
        p.drawText(QRect(0, 0, w, h), Qt.AlignmentFlag.AlignCenter, self._label)
        p.end()


class VideoDisplay(QLabel):
    """Webcam / image display with overlay."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._overlay_text = ""
        self._overlay_color = ACCENT_CYAN
        self._border_color = BORDER_COLOR
        self._border_pulse = 0.0
        self._pulse_active = False
        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(30)
        self._pulse_timer.timeout.connect(self._tick_pulse)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(480, 360)
        self.setStyleSheet(f"background: {DARK_BG}; border-radius: 12px;")
        self._draw_idle_screen()

    def set_threat_glow(self, color: str, active: bool):
        self._border_color = color
        self._pulse_active = active
        if active:
            self._pulse_timer.start()
        else:
            self._pulse_timer.stop()
            self.setStyleSheet(f"""
                background: {DARK_BG};
                border: 2px solid {BORDER_COLOR};
                border-radius: 12px;
            """)

    def _tick_pulse(self):
        self._border_pulse = (self._border_pulse + 0.06) % (2 * math.pi)
        alpha = int(100 + 100 * math.sin(self._border_pulse))
        w = int(2 + 4 * abs(math.sin(self._border_pulse)))
        c = QColor(self._border_color)
        c.setAlpha(alpha)
        self.setStyleSheet(f"""
            background: {DARK_BG};
            border: {w}px solid rgba({c.red()},{c.green()},{c.blue()},{alpha});
            border-radius: 12px;
        """)

    def _draw_idle_screen(self):
        w, h = self.width() or 480, self.height() or 360
        pixmap = QPixmap(w, h)
        pixmap.fill(QColor(DARK_BG))
        p = QPainter(pixmap)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Grid lines
        pen = QPen(QColor(BORDER_COLOR), 1)
        p.setPen(pen)
        for x in range(0, w, 40):
            p.drawLine(x, 0, x, h)
        for y in range(0, h, 40):
            p.drawLine(0, y, w, y)

        # Center icon placeholder
        cx, cy = w // 2, h // 2
        p.setPen(QPen(QColor(TEXT_DIM), 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(cx - 40, cy - 40, 80, 80)
        p.setPen(QPen(QColor(TEXT_DIM), 3))
        p.drawLine(cx - 15, cy, cx + 15, cy)
        p.drawLine(cx, cy - 15, cx, cy + 15)

        p.setPen(QColor(TEXT_DIM))
        p.setFont(QFont("Consolas", 11))
        p.drawText(QRect(0, cy + 55, w, 30), Qt.AlignmentFlag.AlignCenter,
                   "LOAD IMAGE  ·  START WEBCAM")
        p.end()
        self.setPixmap(pixmap)


class GlassCard(QFrame):
    """Glassmorphism card container."""

    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self._title = title
        self.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(19,25,41,0.95), stop:1 rgba(15,21,38,0.85));
                border: 1px solid {BORDER_COLOR};
                border-radius: 14px;
            }}
        """)
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setOffset(0, 4)
        shadow.setBlurRadius(24)
        shadow.setColor(QColor(0, 0, 0, 80))
        self.setGraphicsEffect(shadow)

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._title:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QColor(TEXT_DIM))
        p.setFont(QFont("Consolas", 8))
        p.drawText(QRect(16, 8, self.width(), 16),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                   self._title.upper())
        p.end()


class CoolButton(QPushButton):
    """Animated button with hover glow."""

    def __init__(self, text="", icon_text="", primary=True, parent=None):
        super().__init__(parent)
        self._icon_text = icon_text
        self._is_primary = primary
        self._hover = False
        self._press_anim = 0.0
        self.setText(f"  {icon_text}  {text}  " if icon_text else f"  {text}  ")
        self.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(44)
        self._apply_style(False)

    def _apply_style(self, hover: bool):
        if self._is_primary:
            bg = "#1E44C8" if hover else "#1565C0"
            border = ACCENT_CYAN if hover else ACCENT_BLUE
        else:
            bg = "#1A2540" if hover else "#131929"
            border = ACCENT_BLUE if hover else BORDER_COLOR

        self.setStyleSheet(f"""
            QPushButton {{
                background: {bg};
                color: {TEXT_PRIMARY};
                border: 1.5px solid {border};
                border-radius: 10px;
                padding: 0 16px;
                letter-spacing: 1px;
            }}
            QPushButton:pressed {{
                background: #0D2F8F;
                color: {TEXT_SECONDARY};
            }}
        """)
        if hover and self._is_primary:
            shadow = QGraphicsDropShadowEffect(self)
            shadow.setOffset(0, 0)
            shadow.setBlurRadius(20)
            shadow.setColor(QColor(ACCENT_BLUE))
            self.setGraphicsEffect(shadow)
        else:
            self.setGraphicsEffect(None)

    def enterEvent(self, event):
        self._hover = True
        self._apply_style(True)

    def leaveEvent(self, event):
        self._hover = False
        self._apply_style(False)


class LogPanel(QTextEdit):
    """Scrolling event log."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(QFont("Consolas", 8))
        self.setStyleSheet(f"""
            QTextEdit {{
                background: {DARK_BG};
                color: {TEXT_SECONDARY};
                border: 1px solid {BORDER_COLOR};
                border-radius: 8px;
                padding: 8px;
            }}
        """)
        self.setMaximumHeight(140)

    def log(self, msg: str, level="INFO"):
        ts = datetime.now().strftime("%H:%M:%S")
        colors = {"INFO": TEXT_SECONDARY, "WARN": ACCIDENT_CLR,
                  "ERROR": FIRE_COLOR, "OK": NORMAL_COLOR}
        color = colors.get(level, TEXT_SECONDARY)
        self.append(f'<span style="color:{TEXT_DIM}">[{ts}]</span> '
                    f'<span style="color:{color}">{msg}</span>')
        scrollbar = self.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING DIALOG
# ═══════════════════════════════════════════════════════════════════════════════

class TrainingWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str)

    def __init__(self, dataset_path: str, epochs: int, lr: float, parent=None):
        super().__init__(parent)
        self.dataset_path = dataset_path
        self.epochs = epochs
        self.lr = lr

    def run(self):
        try:
            from backend.model_trainer import ModelTrainer
            trainer = ModelTrainer(
                dataset_path=self.dataset_path,
                epochs=self.epochs,
                lr=self.lr,
                progress_callback=self._on_progress,
            )
            model_path = trainer.train()
            self.finished.emit(True, str(model_path))
        except Exception as e:
            logger.exception("Training error")
            self.finished.emit(False, str(e))

    def _on_progress(self, pct: int, msg: str):
        self.progress.emit(pct, msg)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════════

class UrbanSafetyApp(QMainWindow):

    # ── Signals from worker threads ──────────────────────────────────────────
    _signal_prediction = pyqtSignal(dict)
    _signal_frame      = pyqtSignal(object)
    _signal_webcam_err = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._engine: Optional[InferenceEngine] = None
        self._webcam_thread: Optional[WebcamThread] = None
        self._alert_system = AlertSystem()
        self._sound_enabled = True
        self._webcam_active = False
        self._frame_skip = 0
        self._last_label = "idle"
        self._fps_counter = 0
        self._fps_last_time = time.time()
        self._fps_display = 0.0
        self._training_worker: Optional[TrainingWorker] = None

        # Connect internal signals
        self._signal_prediction.connect(self._on_prediction_received)
        self._signal_frame.connect(self._on_frame_received)
        self._signal_webcam_err.connect(self._on_webcam_error)

        self._build_ui()
        self._load_engine_async()
        self._start_ui_timers()

    # ── UI CONSTRUCTION ───────────────────────────────────────────────────────

    def _build_ui(self):
        self.setWindowTitle("VIGIL-88 AI  ·  Real-Time Threat Detection  v4.0")
        self.setMinimumSize(1200, 780)
        self.setWindowIcon(QIcon("assets/vigil88.ico"))
        self.resize(1340, 840)
        self.setStyleSheet(f"QMainWindow {{ background: {DARK_BG}; }}")

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(16, 12, 16, 12)
        root_layout.setSpacing(10)

        # ── Header ──
        root_layout.addWidget(self._build_header())

        # ── Body (left + right columns) ──
        body = QHBoxLayout()
        body.setSpacing(12)
        body.addWidget(self._build_left_panel(), stretch=6)
        body.addWidget(self._build_right_panel(), stretch=4)
        root_layout.addLayout(body)

        # ── Footer log ──
        root_layout.addWidget(self._build_footer())

    def _build_header(self) -> QWidget:
        w = QWidget()
        w.setFixedHeight(64)
        w.setStyleSheet(f"""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 {PANEL_BG}, stop:0.5 #0D1830, stop:1 {PANEL_BG});
            border: 1px solid {BORDER_COLOR};
            border-radius: 12px;
        """)
        layout = QHBoxLayout(w)
        layout.setContentsMargins(20, 0, 20, 0)

        # Logo + title
        title_layout = QHBoxLayout()
        title_layout.setSpacing(10)

        logo = PulsingCircle(ACCENT_CYAN)
        logo.setFixedSize(28, 28)
        self._header_pulse = logo
        title_layout.addWidget(logo)

        title = QLabel("VIGIL-88 AI")
        title.setFont(QFont("Consolas", 16, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {TEXT_PRIMARY}; background: transparent; border: none;")
        title_layout.addWidget(title)

        subtitle = QLabel("REAL-TIME THREAT DETECTION")
        subtitle.setFont(QFont("Consolas", 9))
        subtitle.setStyleSheet(f"color: {ACCENT_CYAN}; background: transparent; border: none; letter-spacing: 3px;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignBottom)

        layout.addLayout(title_layout)
        layout.addWidget(subtitle)
        layout.addStretch()

        # Status strip
        self._status_label = QLabel("⬤  INITIALIZING…")
        self._status_label.setFont(QFont("Consolas", 9))
        self._status_label.setStyleSheet(f"color: {TEXT_DIM}; background: transparent; border: none;")
        layout.addWidget(self._status_label)

        layout.addSpacing(20)

        # FPS counter
        self._fps_label = QLabel("FPS: --")
        self._fps_label.setFont(QFont("Consolas", 9))
        self._fps_label.setStyleSheet(f"color: {TEXT_DIM}; background: transparent; border: none;")
        layout.addWidget(self._fps_label)

        layout.addSpacing(20)

        # Sound toggle
        self._sound_btn = CoolButton("SOUND", "🔊", primary=False)
        self._sound_btn.setFixedWidth(110)
        self._sound_btn.clicked.connect(self._toggle_sound)
        layout.addWidget(self._sound_btn)

        return w

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # ── Video display ──
        video_card = GlassCard("LIVE FEED")
        video_layout = QVBoxLayout(video_card)
        video_layout.setContentsMargins(12, 24, 12, 12)

        self._video_display = VideoDisplay()
        self._video_display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        video_layout.addWidget(self._video_display)

        # Webcam controls row
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(8)

        self._webcam_btn = CoolButton("START WEBCAM", "🎥", primary=True)
        self._webcam_btn.clicked.connect(self._toggle_webcam)
        if not CV2_AVAILABLE:
            self._webcam_btn.setEnabled(False)
            self._webcam_btn.setToolTip("Install opencv-python to enable webcam")
        ctrl_row.addWidget(self._webcam_btn)

        self._upload_btn = CoolButton("LOAD IMAGE", "📁", primary=False)
        self._upload_btn.clicked.connect(self._load_image)
        ctrl_row.addWidget(self._upload_btn)

        # RTSP input
        from PyQt6.QtWidgets import QLineEdit
        self._rtsp_input = QLineEdit()
        self._rtsp_input.setPlaceholderText("rtsp://user:pass@192.168.1.1:554/stream")
        self._rtsp_input.setFont(QFont("Consolas", 8))
        self._rtsp_input.setStyleSheet(f"""
            QLineEdit {{
                background: {CARD_BG};
                color: {TEXT_PRIMARY};
                border: 1px solid {BORDER_COLOR};
                border-radius: 6px;
                padding: 4px 10px;
            }}
            QLineEdit:focus {{
                border: 1px solid {ACCENT_CYAN};
            }}
        """)
        ctrl_row.addWidget(self._rtsp_input, stretch=3)

        self._rtsp_btn = CoolButton("CONNECT", "📡", primary=False)
        self._rtsp_btn.setFixedWidth(130)
        self._rtsp_btn.clicked.connect(self._connect_rtsp)
        ctrl_row.addWidget(self._rtsp_btn)

        self._cam_index_label = QLabel("Cam:")
        self._cam_index_label.setFont(QFont("Consolas", 9))
        self._cam_index_label.setStyleSheet(f"color: {TEXT_DIM}; background: transparent;")
        ctrl_row.addWidget(self._cam_index_label)

        self._cam_combo = QComboBox()
        self._cam_combo.addItems(["0", "1", "2", "3"])
        self._cam_combo.setFixedWidth(60)
        self._cam_combo.setStyleSheet(f"""
            QComboBox {{
                background: {CARD_BG};
                color: {TEXT_PRIMARY};
                border: 1px solid {BORDER_COLOR};
                border-radius: 6px;
                padding: 4px 8px;
                font-family: Consolas;
                font-size: 10px;
            }}
            QComboBox::drop-down {{ border: none; }}
        """)
        ctrl_row.addWidget(self._cam_combo)

        video_layout.addLayout(ctrl_row)
        layout.addWidget(video_card)

        # ── Confidence bars ──
        conf_card = GlassCard("CLASSIFICATION PROBABILITIES")
        conf_layout = QVBoxLayout(conf_card)
        conf_layout.setContentsMargins(16, 28, 16, 16)
        conf_layout.setSpacing(4)

        self._bar_fire     = AnimatedBar("FIRE",     FIRE_COLOR)
        self._bar_accident = AnimatedBar("ACCIDENT", ACCIDENT_CLR)
        self._bar_normal   = AnimatedBar("NORMAL",   NORMAL_COLOR)

        for bar in (self._bar_fire, self._bar_accident, self._bar_normal):
            conf_layout.addWidget(bar)

        layout.addWidget(conf_card)
        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # ── Threat badge ──
        badge_card = GlassCard("THREAT STATUS")
        badge_layout = QVBoxLayout(badge_card)
        badge_layout.setContentsMargins(16, 28, 16, 16)

        self._threat_badge = ThreatBadge()
        badge_layout.addWidget(self._threat_badge)

        # Confidence text
        self._conf_text = QLabel("CONFIDENCE: --")
        self._conf_text.setFont(QFont("Consolas", 11, QFont.Weight.Bold))
        self._conf_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._conf_text.setStyleSheet(f"color: {TEXT_SECONDARY}; background: transparent; letter-spacing: 2px;")
        badge_layout.addWidget(self._conf_text)

        self._latency_text = QLabel("LATENCY: --")
        self._latency_text.setFont(QFont("Consolas", 9))
        self._latency_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._latency_text.setStyleSheet(f"color: {TEXT_DIM}; background: transparent;")
        badge_layout.addWidget(self._latency_text)

        layout.addWidget(badge_card)

        # ── Stats panel ──
        stats_card = GlassCard("SESSION METRICS")
        stats_layout = QVBoxLayout(stats_card)
        stats_layout.setContentsMargins(16, 28, 16, 12)
        stats_layout.setSpacing(6)

        self._stat_widgets = {}
        stats_data = [
            ("frames_analyzed", "Frames Analyzed", "0"),
            ("threats_detected", "Threats Detected", "0"),
            ("avg_latency",      "Avg Latency",      "--"),
            ("model_device",     "Device",            "Loading…"),
        ]
        for key, label, default in stats_data:
            row = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setFont(QFont("Consolas", 9))
            lbl.setStyleSheet(f"color: {TEXT_DIM}; background: transparent;")
            val = QLabel(default)
            val.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
            val.setStyleSheet(f"color: {ACCENT_CYAN}; background: transparent;")
            val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(lbl)
            row.addStretch()
            row.addWidget(val)
            stats_layout.addLayout(row)
            self._stat_widgets[key] = val

            sep = QFrame()
            sep.setFixedHeight(1)
            sep.setStyleSheet(f"background: {BORDER_COLOR};")
            stats_layout.addWidget(sep)

        layout.addWidget(stats_card)

        # ── Model / Training panel ──
        model_card = GlassCard("MODEL CONTROL")
        model_layout = QVBoxLayout(model_card)
        model_layout.setContentsMargins(16, 28, 16, 16)
        model_layout.setSpacing(8)

        self._model_status = QLabel("No model loaded")
        self._model_status.setFont(QFont("Consolas", 9))
        self._model_status.setStyleSheet(f"color: {ACCIDENT_CLR}; background: transparent;")
        self._model_status.setWordWrap(True)
        model_layout.addWidget(self._model_status)

        self._load_model_btn = CoolButton("LOAD MODEL", "💾", primary=False)
        self._load_model_btn.clicked.connect(self._load_model_dialog)
        model_layout.addWidget(self._load_model_btn)

        self._train_btn = CoolButton("TRAIN MODEL", "🧠", primary=True)
        self._train_btn.clicked.connect(self._open_training_dialog)
        model_layout.addWidget(self._train_btn)

        # Training progress (hidden initially)
        self._train_progress = QProgressBar()
        self._train_progress.setRange(0, 100)
        self._train_progress.setValue(0)
        self._train_progress.setVisible(False)
        self._train_progress.setStyleSheet(f"""
            QProgressBar {{
                background: {DARK_BG};
                border: 1px solid {BORDER_COLOR};
                border-radius: 6px;
                text-align: center;
                color: {TEXT_PRIMARY};
                font-family: Consolas;
                font-size: 9px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {ACCENT_BLUE}, stop:1 {ACCENT_CYAN});
                border-radius: 6px;
            }}
        """)
        model_layout.addWidget(self._train_progress)

        self._train_status_lbl = QLabel("")
        self._train_status_lbl.setFont(QFont("Consolas", 8))
        self._train_status_lbl.setStyleSheet(f"color: {TEXT_DIM}; background: transparent;")
        self._train_status_lbl.setVisible(False)
        model_layout.addWidget(self._train_status_lbl)

        layout.addWidget(model_card)
        layout.addStretch()
        return panel

    def _build_footer(self) -> QWidget:
        footer = QWidget()
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        log_card = GlassCard("SYSTEM LOG")
        log_layout = QVBoxLayout(log_card)
        log_layout.setContentsMargins(12, 28, 12, 8)
        self._log_panel = LogPanel()
        log_layout.addWidget(self._log_panel)

        layout.addWidget(log_card)
        return footer

    # ── TIMERS ────────────────────────────────────────────────────────────────

    def _start_ui_timers(self):
        self._stats = {
            "frames_analyzed": 0,
            "threats_detected": 0,
            "total_latency": 0.0,
        }
        # Refresh display every 500ms
        self._display_timer = QTimer(self)
        self._display_timer.setInterval(500)
        self._display_timer.timeout.connect(self._refresh_stats)
        self._display_timer.start()

        # Shimmer repaint
        self._shimmer_timer = QTimer(self)
        self._shimmer_timer.setInterval(50)
        self._shimmer_timer.timeout.connect(self._repaint_bars)
        self._shimmer_timer.start()

    def _repaint_bars(self):
        self._bar_fire.update()
        self._bar_accident.update()
        self._bar_normal.update()

    def _refresh_stats(self):
        now = time.time()
        elapsed = now - self._fps_last_time
        if elapsed >= 1.0:
            self._fps_display = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_last_time = now

        self._fps_label.setText(f"FPS: {self._fps_display:.1f}" if self._webcam_active else "FPS: --")

        fa = self._stats["frames_analyzed"]
        td = self._stats["threats_detected"]
        avg_lat = (self._stats["total_latency"] / max(1, fa) * 1000)

        self._stat_widgets["frames_analyzed"].setText(str(fa))
        self._stat_widgets["threats_detected"].setText(str(td))
        self._stat_widgets["avg_latency"].setText(f"{avg_lat:.1f} ms")

    # ── ENGINE LOADING ────────────────────────────────────────────────────────

    def _load_engine_async(self):
        self._set_status("LOADING AI ENGINE…", TEXT_DIM)

        def _load():
            try:
                model_path = ROOT / "models" / "urban_safety_model.pt"
                engine = InferenceEngine(str(model_path) if model_path.exists() else None)
                self._engine = engine
                device_str = engine.device_name
                self._signal_prediction.emit({"_init": True, "device": device_str})
            except Exception as e:
                logger.exception("Engine load error")
                self._engine = DummyEngine()
                self._signal_prediction.emit({"_init": True, "device": "CPU (fallback)", "error": str(e)})

        threading.Thread(target=_load, daemon=True).start()

    def _on_engine_initialized(self, device: str, error: Optional[str] = None):
        if error:
            self._set_status(f"ENGINE ERROR — {error[:50]}", FIRE_COLOR)
            self._log_panel.log(f"Engine error: {error}", "ERROR")
        else:
            model_loaded = self._engine is not None and self._engine.model_ready
            if model_loaded:
                self._set_status("AI ENGINE READY", NORMAL_COLOR)
                self._model_status.setText(f"✓ Model loaded  ({device})")
                self._model_status.setStyleSheet(f"color: {NORMAL_COLOR}; background: transparent;")
                self._log_panel.log(f"Model loaded on {device}", "OK")
            else:
                self._set_status("ENGINE READY — NO MODEL", ACCIDENT_CLR)
                self._model_status.setText(f"⚠ No model — train or load one")
                self._log_panel.log("No model found. Train a model or load a .pt file.", "WARN")

        self._stat_widgets["model_device"].setText(device)

    # ── WEBCAM ────────────────────────────────────────────────────────────────

    def _toggle_webcam(self):
        if self._webcam_active:
            self._stop_webcam()
        else:
            self._start_webcam()

    def _start_webcam(self):
        if not CV2_AVAILABLE:
            self._log_panel.log("OpenCV not found — cannot start webcam", "ERROR")
            return
        cam_idx = int(self._cam_combo.currentText())
        self._log_panel.log(f"Opening camera {cam_idx}…", "INFO")
        self._webcam_thread = WebcamThread(
            cam_index=cam_idx,
            frame_callback=self._on_raw_frame,
            error_callback=lambda e: self._signal_webcam_err.emit(e),
        )
        self._webcam_thread.start()
        self._webcam_active = True
        self._webcam_btn.setText("  🛑  STOP WEBCAM  ")
        self._webcam_btn.setStyleSheet(
            self._webcam_btn.styleSheet().replace(ACCENT_BLUE, FIRE_COLOR))
        self._set_status("WEBCAM ACTIVE", ACCENT_CYAN)
        self._header_pulse.set_color(ACCENT_CYAN)
    def _connect_rtsp(self):
        url = self._rtsp_input.text().strip()
        if not url:
            self._log_panel.log("Enter an RTSP URL first", "WARN")
            return
        if not CV2_AVAILABLE:
            self._log_panel.log("OpenCV not found — cannot connect to RTSP", "ERROR")
            return

        # Stop existing webcam/stream if active
        if self._webcam_active:
            self._stop_webcam()

        self._log_panel.log(f"Connecting to RTSP: {url}", "INFO")
        self._webcam_thread = WebcamThread(
            cam_index=url,          # WebcamThread accepts string URL too
            frame_callback=self._on_raw_frame,
            error_callback=lambda e: self._signal_webcam_err.emit(e),
        )
        self._webcam_thread.start()
        self._webcam_active = True
        self._webcam_btn.setText("  🛑  STOP STREAM  ")
        self._set_status(f"RTSP CONNECTED", ACCENT_CYAN)
        self._log_panel.log("RTSP stream started", "OK")
    def _stop_webcam(self):
        if self._webcam_thread:
            self._webcam_thread.stop()
            self._webcam_thread = None
        self._webcam_active = False
        self._webcam_btn.setText("  🎥  START WEBCAM  ")
        self._webcam_btn._apply_style(False)
        self._set_status("WEBCAM STOPPED", TEXT_DIM)
        self._header_pulse.set_color(ACCENT_CYAN)
        self._video_display.set_threat_glow(BORDER_COLOR, False)
        self._video_display._draw_idle_screen()
        self._log_panel.log("Webcam stopped", "INFO")

    def _on_raw_frame(self, frame):
        """Called from webcam thread — emit signal to main thread."""
        self._signal_frame.emit(frame)

    def _on_frame_received(self, frame):
        """Process frame on main thread."""
        self._fps_counter += 1

        # Display frame
        rgb = frame[..., ::-1].copy()
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img).scaled(
            self._video_display.width(),
            self._video_display.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._video_display.setPixmap(pixmap)

        # Inference (every other frame for performance)
        self._frame_skip = (self._frame_skip + 1) % 2
        if self._frame_skip == 0 and self._engine and self._engine.model_ready:
            try:
                result = self._engine.predict_frame(frame)
                self._signal_prediction.emit(result.to_dict())
            except Exception as e:
                logger.debug("Frame inference error: %s", e)

    # ── IMAGE UPLOAD ──────────────────────────────────────────────────────────

    def _load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", str(Path.home()),
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)"
        )
        if not path:
            return

        self._log_panel.log(f"Image loaded: {Path(path).name}", "INFO")
        pixmap = QPixmap(path).scaled(
            self._video_display.width(),
            self._video_display.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._video_display.setPixmap(pixmap)

        if self._engine and self._engine.model_ready:
            try:
                result = self._engine.predict_file(path)
                self._signal_prediction.emit(result.to_dict())
            except Exception as e:
                self._log_panel.log(f"Inference error: {e}", "ERROR")
        else:
            self._log_panel.log("No model loaded — cannot run inference", "WARN")

    # ── PREDICTION HANDLER ────────────────────────────────────────────────────

    def _on_prediction_received(self, result: dict):
        # Init signal
        if result.get("_init"):
            self._on_engine_initialized(
                result.get("device", "Unknown"),
                result.get("error")
            )
            return

        label      = result.get("label", "normal")
        confidence = result.get("confidence", 0.0)
        probs      = result.get("probabilities", {})
        latency    = result.get("latency_ms", 0.0)

        # Update stats
        self._stats["frames_analyzed"] += 1
        self._stats["total_latency"] += latency / 1000.0
        if label in ("fire", "accident"):
            self._stats["threats_detected"] += 1

        # Confidence bars
        self._bar_fire.set_value(probs.get("fire", 0.0))
        self._bar_accident.set_value(probs.get("accident", 0.0))
        self._bar_normal.set_value(probs.get("normal", 0.0))

        # Threat badge
        color, glow, status_text = STATUS_COLORS.get(label, STATUS_COLORS["idle"])
        self._threat_badge.set_status(status_text, color)
        self._conf_text.setText(f"CONFIDENCE: {confidence * 100:.1f}%")
        self._latency_text.setText(f"LATENCY: {latency:.1f} ms")

        # Video border glow
        is_threat = label in ("fire", "accident")
        self._video_display.set_threat_glow(color, is_threat)
        self._header_pulse.set_color(color if is_threat else ACCENT_CYAN)

        # Alert
        if is_threat and label != self._last_label:
            self._alert_system.trigger(label, confidence, self._sound_enabled)
            self._log_panel.log(f"⚠ {label.upper()} detected — {confidence*100:.1f}%", "WARN")

        self._last_label = label

    def _on_webcam_error(self, msg: str):
        self._log_panel.log(f"Webcam error: {msg}", "ERROR")
        self._stop_webcam()

    # ── MODEL CONTROL ─────────────────────────────────────────────────────────

    def _load_model_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Model File", str(ROOT / "models"),
            "PyTorch Model (*.pt *.pth)"
        )
        if not path:
            return
        try:
            if self._engine:
                self._engine.load_model(path)
            else:
                self._engine = InferenceEngine(path)
            self._model_status.setText(f"✓ {Path(path).name}")
            self._model_status.setStyleSheet(f"color: {NORMAL_COLOR}; background: transparent;")
            self._log_panel.log(f"Model loaded: {Path(path).name}", "OK")
            self._set_status("MODEL READY", NORMAL_COLOR)
        except Exception as e:
            self._log_panel.log(f"Load error: {e}", "ERROR")

    def _open_training_dialog(self):
        dataset_path = QFileDialog.getExistingDirectory(
            self, "Select Dataset Folder (with fire/, accident/, normal/ subfolders)",
            str(ROOT / "dataset")
        )
        if not dataset_path:
            return

        # Verify structure
        dataset = Path(dataset_path)
        missing = [c for c in ("fire", "accident", "normal") if not (dataset / c).is_dir()]
        if missing:
            self._log_panel.log(f"Missing subfolders: {missing}. Run dataset organizer.", "ERROR")
            return

        self._log_panel.log(f"Starting training on: {dataset_path}", "INFO")
        self._train_progress.setVisible(True)
        self._train_status_lbl.setVisible(True)
        self._train_btn.setEnabled(False)

        self._training_worker = TrainingWorker(
            dataset_path=dataset_path,
            epochs=15,
            lr=0.001,
        )
        self._training_worker.progress.connect(self._on_training_progress)
        self._training_worker.finished.connect(self._on_training_finished)
        self._training_worker.start()

    def _on_training_progress(self, pct: int, msg: str):
        self._train_progress.setValue(pct)
        self._train_status_lbl.setText(msg)
        self._log_panel.log(msg, "INFO")

    def _on_training_finished(self, success: bool, path_or_error: str):
        self._train_progress.setVisible(False)
        self._train_status_lbl.setVisible(False)
        self._train_btn.setEnabled(True)
        if success:
            self._log_panel.log(f"✓ Training complete: {path_or_error}", "OK")
            # Auto-load the trained model
            if self._engine:
                try:
                    self._engine.load_model(path_or_error)
                    self._model_status.setText(f"✓ Trained model loaded")
                    self._model_status.setStyleSheet(f"color: {NORMAL_COLOR}; background: transparent;")
                    self._set_status("MODEL READY", NORMAL_COLOR)
                except Exception as e:
                    self._log_panel.log(f"Auto-load failed: {e}", "WARN")
        else:
            self._log_panel.log(f"Training failed: {path_or_error}", "ERROR")

    # ── UTILITIES ─────────────────────────────────────────────────────────────

    def _toggle_sound(self):
        self._sound_enabled = not self._sound_enabled
        icon = "🔊" if self._sound_enabled else "🔇"
        self._sound_btn.setText(f"  {icon}  SOUND  ")
        self._log_panel.log(f"Sound {'enabled' if self._sound_enabled else 'muted'}", "INFO")

    def _set_status(self, text: str, color: str):
        self._status_label.setText(f"⬤  {text}")
        self._status_label.setStyleSheet(f"color: {color}; background: transparent; border: none;")

    def closeEvent(self, event):
        logger.info("Shutting down…")
        self._stop_webcam()
        if self._training_worker and self._training_worker.isRunning():
            self._training_worker.terminate()
        event.accept()


# ═══════════════════════════════════════════════════════════════════════════════
# SPLASH SCREEN
# ═══════════════════════════════════════════════════════════════════════════════

class SplashScreen(QWidget):
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setFixedSize(580, 320)
        self.setStyleSheet(f"background: {DARK_BG}; border-radius: 16px;")
        self._alpha = 0
        self._progress = 0
        self._angle = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 30)
        layout.setSpacing(16)

        title = QLabel("VIGIL-88 AI")
        title.setFont(QFont("Consolas", 28, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(f"color: {ACCENT_CYAN}; background: transparent;")
        layout.addWidget(title)

        subtitle = QLabel("INTELLIGENT THREAT DETECTION PLATFORM  ·  v4.0")
        subtitle.setFont(QFont("Consolas", 9))
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet(f"color: {TEXT_DIM}; background: transparent; letter-spacing: 4px;")
        layout.addWidget(subtitle)

        layout.addStretch()

        self._splash_bar = QProgressBar()
        self._splash_bar.setRange(0, 100)
        self._splash_bar.setValue(0)
        self._splash_bar.setFixedHeight(4)
        self._splash_bar.setStyleSheet(f"""
            QProgressBar {{
                background: {BORDER_COLOR};
                border: none;
                border-radius: 2px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {ACCENT_BLUE}, stop:1 {ACCENT_CYAN});
                border-radius: 2px;
            }}
        """)
        layout.addWidget(self._splash_bar)

        self._splash_label = QLabel("Initializing…")
        self._splash_label.setFont(QFont("Consolas", 9))
        self._splash_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._splash_label.setStyleSheet(f"color: {TEXT_DIM}; background: transparent;")
        layout.addWidget(self._splash_label)

        # Center on screen
        screen = QApplication.primaryScreen().geometry()
        self.move(
            (screen.width() - self.width()) // 2,
            (screen.height() - self.height()) // 2
        )

        # Animate
        self._timer = QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

        steps = [
            "Loading AI engine…",
            "Initializing webcam module…",
            "Loading alert system…",
            "Setting up UI…",
            "Ready.",
        ]
        self._steps = steps
        self._step_idx = 0

    def _tick(self):
        self._progress += 2
        self._splash_bar.setValue(min(100, self._progress))

        step_idx = min(len(self._steps) - 1, self._progress * len(self._steps) // 100)
        self._splash_label.setText(self._steps[step_idx])

        if self._progress >= 100:
            self._timer.stop()
            QTimer.singleShot(300, self.finished.emit)


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("VIGIL-88 AI")
    app.setOrganizationName("VIGIL-88Lab")

    # Global dark palette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window,          QColor(DARK_BG))
    palette.setColor(QPalette.ColorRole.WindowText,      QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Base,            QColor(CARD_BG))
    palette.setColor(QPalette.ColorRole.AlternateBase,   QColor(PANEL_BG))
    palette.setColor(QPalette.ColorRole.Text,            QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Button,          QColor(CARD_BG))
    palette.setColor(QPalette.ColorRole.ButtonText,      QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Highlight,       QColor(ACCENT_BLUE))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(TEXT_PRIMARY))
    app.setPalette(palette)

    # Splash
    splash = SplashScreen()
    splash.show()

    main_window = UrbanSafetyApp()

    def _show_main():
        splash.close()
        main_window.show()

    splash.finished.connect(_show_main)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
