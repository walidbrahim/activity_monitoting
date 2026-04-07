"""
apps.debug.debug_base
=====================
Shared scaffold for all per-module radar-engine debug GUIs.

Hardware wiring
---------------
Starts ``libs.controllers.radarController.RadarController`` as a daemon
process (identical to room_monitoring.py).  The controller pushes fully-
assembled (range_bins × antennas) complex frames into a
``multiprocessing.Queue``.  Each QTimer tick drains that queue and feeds
the most-recent frame directly to ``RadarEngine.process_frame()``.

If no frame is available on a given tick the tick is simply skipped —
the display remains frozen at the last good state.  There is no synthetic
fallback; these apps only operate on live radar data.
"""
from __future__ import annotations

import multiprocessing
import os
import queue as _queue
import sys
import time
from abc import abstractmethod

import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QMainWindow, QWidget, QLabel

# ── Project root ──────────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import load_profile, ConfigFactory, AppConfig              # noqa: E402
from radar_engine.orchestration.engine import RadarEngine              # noqa: E402
from radar_engine.core.models import EngineOutput                      # noqa: E402

# ── Colour palette ────────────────────────────────────────────────────────────

PALETTE = {
    "bg":      "#0F172A",
    "panel":   "#1E293B",
    "card":    "#243044",
    "border":  "#334155",
    "accent":  "#58a6ff",
    "accent2": "#8B5CF6",
    "ok":      "#10B981",
    "warn":    "#F59E0B",
    "alert":   "#EF4444",
    "text":    "#E2E8F0",
    "subtext": "#94A3B8",
    "cyan":    "#22D3EE",
    "orange":  "#FB923C",
    "violet":  "#A78BFA",
    "grid":    "#1E3A5F",
}

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {PALETTE['bg']};
    color: {PALETTE['text']};
    font-family: 'Inter', 'Segoe UI', 'Helvetica Neue', sans-serif;
    font-size: 12px;
}}
QGroupBox {{
    border: 1px solid {PALETTE['border']};
    border-radius: 8px;
    margin-top: 8px;
    padding: 8px;
    color: {PALETTE['subtext']};
    font-size: 11px;
    font-weight: bold;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
}}
QLabel {{ color: {PALETTE['text']}; }}
QLabel#subtext {{ color: {PALETTE['subtext']}; font-size: 11px; }}
QLabel#badge_ok    {{ background: {PALETTE['ok']};     color: #fff; border-radius: 6px; padding: 2px 8px; font-weight: bold; }}
QLabel#badge_warn  {{ background: {PALETTE['warn']};   color: #000; border-radius: 6px; padding: 2px 8px; font-weight: bold; }}
QLabel#badge_alert {{ background: {PALETTE['alert']};  color: #fff; border-radius: 6px; padding: 2px 8px; font-weight: bold; }}
QLabel#badge_cyan  {{ background: {PALETTE['cyan']};   color: #000; border-radius: 6px; padding: 2px 8px; font-weight: bold; }}
QLabel#badge_violet{{ background: {PALETTE['violet']}; color: #fff; border-radius: 6px; padding: 2px 8px; font-weight: bold; }}
QLabel#badge_dim   {{ background: {PALETTE['border']}; color: {PALETTE['subtext']}; border-radius: 6px; padding: 2px 8px; font-weight: bold; }}
QLabel#monospace   {{ font-family: 'JetBrains Mono', 'Courier New', monospace; font-size: 11px; color: {PALETTE['cyan']}; }}
QSplitter::handle  {{ background: {PALETTE['border']}; width: 1px; }}
QScrollBar:vertical {{ background: {PALETTE['bg']}; width: 6px; }}
QScrollBar::handle:vertical {{ background: {PALETTE['border']}; border-radius: 3px; }}
"""

# ── pyqtgraph global defaults ─────────────────────────────────────────────────
import pyqtgraph as pg  # noqa: E402

pg.setConfigOptions(antialias=True, background=PALETTE["bg"], foreground=PALETTE["text"])


def apply_plot_defaults(
    plot_item: pg.PlotItem,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    grid: bool = True,
) -> None:
    """Apply consistent dark-mode defaults to a PlotItem."""
    plot_item.getAxis("bottom").setTextPen(PALETTE["subtext"])
    plot_item.getAxis("left").setTextPen(PALETTE["subtext"])
    plot_item.getAxis("bottom").setPen(PALETTE["border"])
    plot_item.getAxis("left").setPen(PALETTE["border"])
    if title:
        plot_item.setTitle(title, color=PALETTE["text"], size="11pt")
    if xlabel:
        plot_item.setLabel("bottom", xlabel, color=PALETTE["subtext"])
    if ylabel:
        plot_item.setLabel("left", ylabel, color=PALETTE["subtext"])
    if grid:
        plot_item.showGrid(x=True, y=True, alpha=0.15)


# ── DebugBase ─────────────────────────────────────────────────────────────────

class DebugBase(QMainWindow):
    """Shared scaffolding for all per-module radar debug GUIs.

    Subclasses must implement:
        _build_ui(self, central: QWidget)  – populate the central widget
        render(self, output: EngineOutput) – called each frame
    """

    TITLE:    str = "Radar Debug"
    WINDOW_W: int = 1400
    WINDOW_H: int = 860

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.TITLE)
        self.resize(self.WINDOW_W, self.WINDOW_H)
        self.setStyleSheet(STYLESHEET)

        # ── Config ────────────────────────────────────────────────────────────
        self._app_cfg    = self._load_config()
        self._engine_cfg = ConfigFactory.engine_config(self._app_cfg)
        self._frame_rate = float(self._app_cfg.hardware.frame_rate)
        self._num_bins   = int(self._app_cfg.hardware.range_bins)
        self._num_ant    = int(self._app_cfg.hardware.antennas)

        # ── Engine ────────────────────────────────────────────────────────────
        self.engine = RadarEngine(cfg=self._engine_cfg, with_respiration=True)

        # ── Live hardware: spawn RadarController as a daemon process ──────────
        # Mirrors room_monitoring.py exactly.  Frames arrive as
        # np.ndarray(range_bins, antennas) complex on self._pt_fft_q.
        self._pt_fft_q   = multiprocessing.Queue()
        self._state_q    = multiprocessing.Queue()  # required by controller, unused here
        self._frame_idx  = 0
        self._hw_ok      = False  # True once the process is alive

        try:
            from libs.controllers.radarController import RadarController
            self._radar_proc = RadarController(
                state_q=self._state_q,
                pt_fft_q=self._pt_fft_q,
            )
            self._radar_proc.daemon = True
            self._radar_proc.start()
            self._hw_ok = True
            print("[DebugBase] RadarController process started — waiting for frames …")
        except Exception as exc:
            self._radar_proc = None
            print(f"[DebugBase] ERROR: could not start RadarController: {exc}")

        # ── Status bar ────────────────────────────────────────────────────────
        if self._hw_ok:
            status_text = "Frame: 0  |  📡 Connecting …"
        else:
            status_text = "⚠  Radar hardware not connected — check serial port"
        self._fps_label = QLabel(status_text)
        self._fps_label.setObjectName("subtext")
        self.statusBar().addPermanentWidget(self._fps_label)
        self.statusBar().setStyleSheet(
            f"background:{PALETTE['panel']}; color:{PALETTE['subtext']};"
        )

        # ── Build subclass UI ─────────────────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        self._build_ui(central)

        # ── Timer ─────────────────────────────────────────────────────────────
        interval_ms = max(10, int(1000 / self._frame_rate))
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        self._timer.stop()
        proc = getattr(self, "_radar_proc", None)
        if proc is not None and proc.is_alive():
            proc.terminate()
            proc.join(timeout=2.0)
        super().closeEvent(event)

    # ── Config ───────────────────────────────────────────────────────────────

    def _load_config(self) -> AppConfig:
        """Load profile stack from APP_PROFILE env var (mirrors main app)."""
        base     = os.path.join(_ROOT, "profiles", "base.yaml")
        selected = os.getenv("APP_PROFILE", "").strip()
        if selected and selected not in ("base", "base.yaml"):
            overlay = selected if selected.endswith(".yaml") else f"{selected}.yaml"
            if "/" not in overlay:
                overlay = os.path.join(_ROOT, "profiles", overlay)
            try:
                return load_profile(base, overlay)
            except FileNotFoundError:
                pass
        return load_profile(base)

    # ── Frame loop ────────────────────────────────────────────────────────────

    def _tick(self):
        """Drain all queued radar frames; feed the most-recent one to the engine.

        If the queue is empty this tick is skipped — the display freezes at
        the last good frame rather than advancing with stale or fake data.
        """
        if not self._hw_ok:
            return  # hardware never initialised — nothing to do

        # Drain the entire queue; keep only the newest frame
        latest_frame: np.ndarray | None = None
        while True:
            try:
                latest_frame = self._pt_fft_q.get_nowait()
            except _queue.Empty:
                break
            except Exception:
                break

        if latest_frame is None:
            return  # no frame this tick — skip

        output = self.engine.process_frame(latest_frame, timestamp=time.time())
        self._frame_idx += 1
        self._fps_label.setText(f"Frame: {self._frame_idx}  |  📡 Live")
        self.render(output)

    # ── Abstract interface ────────────────────────────────────────────────────

    def _build_ui(self, central: QWidget) -> None:
        """Override in subclass to build the widget tree into `central`.

        Called once during __init__ before the timer starts.
        """
        ...

    @abstractmethod
    def render(self, output: EngineOutput) -> None:
        """Called every frame with the latest EngineOutput. Must be overridden."""
        ...

    # ── Convenience helpers ───────────────────────────────────────────────────

    @staticmethod
    def make_plot_widget(
        title: str = "", xlabel: str = "", ylabel: str = "", grid: bool = True
    ) -> pg.PlotWidget:
        pw = pg.PlotWidget()
        pw.setBackground(PALETTE["panel"])
        apply_plot_defaults(pw.getPlotItem(), title, xlabel, ylabel, grid)
        return pw

    @staticmethod
    def styled_label(text: str, size: int = 12, bold: bool = False,
                     color: str | None = None) -> QLabel:
        lbl   = QLabel(text)
        style = f"font-size:{size}px;"
        if bold:
            style += "font-weight:bold;"
        if color:
            style += f"color:{color};"
        lbl.setStyleSheet(style)
        return lbl

    @staticmethod
    def badge_label(text: str, kind: str = "ok") -> QLabel:
        """Coloured pill badge. kind: ok | warn | alert | cyan | violet | dim"""
        lbl = QLabel(text)
        lbl.setObjectName(f"badge_{kind}")
        return lbl
