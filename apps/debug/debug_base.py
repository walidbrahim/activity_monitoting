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
from radar_engine.diagnostics import FrameRecorder                     # noqa: E402

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
QPushButton {{
    background-color: {PALETTE['card']};
    border: 1px solid {PALETTE['border']};
    border-radius: 6px;
    padding: 2px 10px;
    color: {PALETTE['text']};
}}
QPushButton:hover {{
    background-color: {PALETTE['border']};
}}
QPushButton:checked {{
    background-color: {PALETTE['alert']}40;
    border-color: {PALETTE['alert']};
    color: {PALETTE['alert']};
}}
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

        # ── Engine & Recorder ──────────────────────────────────────────────────
        self.engine   = RadarEngine(cfg=self._engine_cfg, with_respiration=True)
        self._recorder = FrameRecorder(capacity=10000)
        self._recording = False
        self._raw_buf   = []

        # ── Parameter Tuner ───────────────────────────────────────────────────
        self._tunable_params = []
        self._tuner_window = None

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
        self._fps_label.setMinimumWidth(200)
        self.statusBar().addWidget(self._fps_label)

        # Record button
        from PyQt6.QtWidgets import QPushButton
        self._record_btn = QPushButton("⏺  Record Session")
        self._record_btn.setCheckable(True)
        self._record_btn.clicked.connect(self._toggle_recording)
        # Tune button
        self._tune_btn = QPushButton("⚙️ Tune Parameters")
        self._tune_btn.clicked.connect(self._open_tuner)
        self.statusBar().addPermanentWidget(self._tune_btn)

        self.statusBar().setStyleSheet(
            f"background:{PALETTE['panel']}; color:{PALETTE['subtext']}; padding: 2px;"
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
        """Drain strictly *all* queued radar frames into the Engine immediately.
        Only the *latest* processed output forces a GUI render.
        This guarantees the RadarEngine strictly operates at 25 FPS even if
        the UI falls slightly behind.
        """
        if not self._hw_ok:
            return

        last_output = None
        frames_passed = 0
        while True:
            try:
                frame = self._pt_fft_q.get_nowait()
                last_output = self.engine.process_frame(frame, timestamp=time.time())
                self._frame_idx += 1
                frames_passed += 1
                
                if self._recording:
                    self._recorder.record(last_output)
                    self._raw_buf.append(np.array(frame, copy=True))
                    
            except _queue.Empty:
                break
            except Exception:
                break

        if last_output is None:
            return  # No frames arrived, so skip render

        self._frames_passed_since_render = frames_passed
        output = last_output
        
        base_status = f"Frame: {self._frame_idx}  |  📡 Live" + ("  🔴 RECORDING" if self._recording else "")
        diag_stats = []
        if output.activity:
            diag_stats.append(f"Occ: {output.activity.occupancy.name}")
        if output.tracked_target and output.tracked_target.valid:
            diag_stats.append(f"Track: Bin {output.tracked_target.bin_index}")
        else:
            diag_stats.append(f"Track: --")
            
        if output.vital_features:
            diag_stats.append(f"Vital: {output.vital_features.micro_state.name}")
        else:
            diag_stats.append(f"Vital: --")
            
        ghost = output.diagnostics.get("is_static_ghost", False)
        if ghost:
            diag_stats.append("👻 MASK DROPPED (Static)")
            
        full_status = base_status + "  |  " + "  |  ".join(diag_stats)
        self._fps_label.setText(full_status)

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

    def _toggle_recording(self, checked: bool) -> None:
        """Handle the Record / Stop transition and export data."""
        from pathlib import Path
        from PyQt6.QtWidgets import QApplication
        if checked:
            # Start
            self._recorder.reset()
            self._raw_buf = []
            self._recording = True
            self._record_btn.setText("⏹  Stop & Save")
            print("[DebugBase] Recording started...")
        else:
            # Stop & Save
            self._recording = False
            self._record_btn.setText("⏳ Saving...")
            self._record_btn.setEnabled(False)
            QApplication.processEvents()

            if self._recorder.buffered_frames > 0:
                ts = time.strftime("%Y%m%d_%H%M%S")
                profile = os.getenv("APP_PROFILE", "debug")
                folder  = Path(_ROOT) / "captures" / f"{profile}_{ts}"
                folder.mkdir(parents=True, exist_ok=True)
                
                csv_path = self._recorder.export_csv(folder / "frame_records.csv")
                json_path = self._recorder.export_json(folder / "frame_records.json")
                
                if self._raw_buf:
                    raw_path = folder / "frames.npz"
                    np.savez_compressed(raw_path, frames=np.stack(self._raw_buf, axis=0))
                
                print(f"[DebugBase] Session saved to {folder}")
                self.statusBar().showMessage(f"✅ Saved to: captures/{folder.name}", 5000)
            
            self._record_btn.setText("⏺  Record Session")
            self._record_btn.setEnabled(True)

    # ── Parameter Tuning ──────────────────────────────────────────────────────

    def add_tunable_param(self, pid: str, name: str, min_val, max_val, step, default, decimals: int = 2) -> None:
        """Register a parameter for real-time GUI tuning."""
        self._tunable_params.append({
            "id": pid, "name": name, "min": min_val, "max": max_val, 
            "step": step, "default": default, "decimals": decimals
        })

    def _open_tuner(self) -> None:
        """Launch the floating parameter tuning window."""
        if not self._tunable_params:
            print("[DebugBase] No tunable parameters registered for this app.")
            self.statusBar().showMessage("⚠ No tunables registered.", 3000)
            return
            
        from apps.debug.tuner_window import ParameterTunerWindow
        if not self._tuner_window:
            self._tuner_window = ParameterTunerWindow(self, self._tunable_params, self._apply_tuning_update, PALETTE)
        self._tuner_window.show()
        self._tuner_window.raise_()

    def _apply_tuning_update(self, updates: dict) -> None:
        """Apply new config and hard-reset the engine.

        Since sub-configs (like PreprocessingConfig) are frozen dataclasses,
        we must use dataclasses.replace to create updated instances.
        """
        from dataclasses import replace
        print(f"\n[DebugBase] Applying live parameter update: {updates}")

        # 1. Group updates by the sub-config title (e.g. 'preprocessing', 'detection')
        category_updates = {}
        for pid, val in updates.items():
            cat, key = pid.split(".", 1)
            if cat not in category_updates: category_updates[cat] = {}
            category_updates[cat][key] = val

            # Also sync the local 'default' state so the tuner UI re-opens correctly
            for p in self._tunable_params:
                if p["id"] == pid:
                    p["default"] = val
                    break

        # 2. Reconstruct the frozen sub-configs and inject them into the mutable EngineConfig
        for cat, fields in category_updates.items():
            old_sub = getattr(self._engine_cfg, cat)
            new_sub = replace(old_sub, **fields)
            setattr(self._engine_cfg, cat, new_sub)

        # 3. Hard reset RadarEngine with the new configuration
        from radar_engine.orchestration.engine import RadarEngine
        self.engine = RadarEngine(cfg=self._engine_cfg, with_respiration=True)
        
        # 3. Fast-forward the incoming queue to drop stale data
        drop_count = 0
        while not self._pt_fft_q.empty():
            try:
                self._pt_fft_q.get_nowait()
                drop_count += 1
            except:
                break
        
        self.statusBar().showMessage(f"♻️ Engine Config Updated. Dropped {drop_count} stale frames.", 4000)

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
