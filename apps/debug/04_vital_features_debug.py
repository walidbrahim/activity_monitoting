"""
apps/debug/04_vital_features_debug.py
======================================
Debug GUI for physiological feature extraction (Phase History, SNR, MicroState).

Layout:
  ┌──────────────────────────────┬──────────────────┐
  │  Unwrapped Phase History     │  MicroState &    │
  │  (Scrolling Heatmap)         │  SNR Badges      │
  ├──────────────────────────────┼──────────────────┤
  │  Power Spectral Density      │  Confidence      │
  │  (Welch PSD 0–4 Hz)          │  Gauges          │
  └──────────────────────────────┴──────────────────┘

Run:
    APP_PROFILE=lab python apps/debug/04_vital_features_debug.py
"""
from __future__ import annotations
import sys, os
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout,
    QGridLayout, QGroupBox, QLabel, QSizePolicy,
)
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QColor, QPen, QBrush, QFont, QTransform
import pyqtgraph as pg

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from apps.debug.debug_base import DebugBase, PALETTE, apply_plot_defaults  # noqa: E402
from radar_engine.core.models import EngineOutput                           # noqa: E402
from radar_engine.core.enums import MicroState                              # noqa: E402


# ── Quality Stats Widget ──────────────────────────────────────────────────────

class FeatureStats(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._snr = 0.0
        self._variance = 0.0
        self.setMinimumHeight(100)

    def update_data(self, snr: float, var: float):
        self._snr = snr; self._variance = var; self.update()

    def paintEvent(self, event):
        from PyQt6.QtGui import QPainter
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        
        # SNR Indicator
        p.setPen(QColor(PALETTE["subtext"]))
        p.setFont(QFont("Inter", 9))
        p.drawText(10, 20, "Signal-to-Noise Ratio (dB)")
        
        # Progress bar background
        p.setBrush(QBrush(QColor(PALETTE["border"])))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(QRectF(10, 30, w-20, 12), 6, 6)
        
        # Fill based on SNR (0..25 dB map)
        snr_frac = min(1.0, max(0.0, self._snr / 25.0))
        col = QColor(PALETTE["ok"] if snr_frac > 0.6 else PALETTE["warn"] if snr_frac > 0.3 else PALETTE["alert"])
        p.setBrush(QBrush(col))
        p.drawRoundedRect(QRectF(10, 30, (w-20)*snr_frac, 12), 6, 6)
        
        p.setPen(QColor("#FFFFFF"))
        p.setFont(QFont("Inter", 10, QFont.Weight.Bold))
        p.drawText(10, 30, w-20, 12, Qt.AlignmentFlag.AlignCenter, f"{self._snr:.1f} dB")
        p.end()


# ── Main ──────────────────────────────────────────────────────────────────────

class VitalFeaturesDebug(DebugBase):
    TITLE    = "🫀 Vital Feature Extraction — Debug"
    WINDOW_W = 1600
    WINDOW_H = 950

    _HIST_SECS = 30  # View window

    def _build_ui(self, central: QWidget) -> None:
        # ── Setup Parameter Tuner ─────────────────────────────────────────────
        self.add_tunable_param("vitals.motion_threshold", "Motion Threshold", 0.0, 100.0, 1.0, self._engine_cfg.vitals.motion_threshold, decimals=1)

        # ── UI Construction ───────────────────────────────────────────────────
        main_lay = QHBoxLayout(central)
        main_lay.setContentsMargins(10, 10, 10, 10)
        main_lay.setSpacing(12)

        # ── Left: Physiological Signal History ────────────────────────────────
        left = QVBoxLayout()
        
        # Phase Heatmap
        phase_grp = QGroupBox("Unwrapped Phase Time-History (Sinusoidal Waves = Breathing)")
        pl = QVBoxLayout(phase_grp)
        self._pw_phase = pg.PlotWidget()
        self._pw_phase.setBackground(PALETTE["panel"])
        apply_plot_defaults(self._pw_phase.getPlotItem(), xlabel="Time (s ago)", ylabel="Phase (rad)")
        
        self._phase_img = pg.ImageItem()
        self._phase_img.setColorMap(pg.colormap.get("CET-L4")) # Perceptually uniform blue-white-red
        tr = QTransform()
        tr.translate(-self._HIST_SECS, -np.pi)
        tr.scale(1.0/self._frame_rate, (2*np.pi)/50) # 50 rows for 2*pi range
        self._phase_img.setTransform(tr)
        self._pw_phase.addItem(self._phase_img)
        self._pw_phase.setYRange(-np.pi, np.pi)
        
        self._phase_buf = np.zeros((int(self._HIST_SECS * self._frame_rate), 50), dtype=np.float32)
        
        # Simple line plot overlay of the central phase value
        self._phase_curve = self._pw_phase.plot(pen=pg.mkPen("#FFFFFF", width=1.5))
        self._phase_vals  = np.zeros(len(self._phase_buf))
        self._time_axis   = np.linspace(-self._HIST_SECS, 0, len(self._phase_buf))
        
        pl.addWidget(self._pw_phase)
        left.addWidget(phase_grp, stretch=5)

        # Power Spectral Density (PSD)
        psd_grp = QGroupBox("Welch PSD (Signal Energy vs Frequency)")
        psl = QVBoxLayout(psd_grp)
        self._pw_psd = self.make_plot_widget(xlabel="Frequency (Hz)", ylabel="Power (dB/Hz)")
        self._pw_psd.setXRange(0, 4.0)
        self._pw_psd.setYRange(0, 100)
        self._psd_curve = self._pw_psd.plot(pen=pg.mkPen(PALETTE["cyan"], width=2), fillLevel=0, brush=PALETTE["cyan"]+"30")
        
        # Band markers
        self._pw_psd.addItem(pg.InfiniteLine(pos=0.1, angle=90, pen=pg.mkPen(PALETTE["text"]+"40"), label="BR Min"))
        self._pw_psd.addItem(pg.InfiniteLine(pos=0.5, angle=90, pen=pg.mkPen(PALETTE["text"]+"40"), label="BR Max"))
        
        psl.addWidget(self._pw_psd)
        left.addWidget(psd_grp, stretch=4)
        
        main_lay.addLayout(left, stretch=7)

        # ── Right: Status & Feature Badges ────────────────────────────────────
        right = QVBoxLayout()
        
        # MicroState Large Badge
        state_grp = QGroupBox("Real-time MicroState")
        sl = QVBoxLayout(state_grp)
        self._lbl_state = QLabel("IDLE")
        self._lbl_state.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_state.setFont(QFont("Inter", 24, QFont.Weight.Bold))
        self._lbl_state.setStyleSheet(f"color: {PALETTE['subtext']}; padding: 20px; background: {PALETTE['panel']}; border-radius: 12px;")
        sl.addWidget(self._lbl_state)
        right.addWidget(state_grp, stretch=2)

        # Feature stats (SNR/Var)
        stat_grp = QGroupBox("Signal Quality Metrics")
        stl = QVBoxLayout(stat_grp)
        self._feat_stats = FeatureStats()
        stl.addWidget(self._feat_stats)
        right.addWidget(stat_grp, stretch=2)

        # Diagnostics table
        diag_grp = QGroupBox("Feature Vector")
        dg = QGridLayout(diag_grp)
        self._l_v_psd = self._mk("Peak PSD");   self._l_v_conf = self._mk("IQ Complexity")
        self._l_v_bin = self._mk("Locked Bin"); self._l_v_snr  = self._mk("Vital SNR")
        self._l_v_var = self._mk("Phase Var")
        for i, (k,v) in enumerate([self._l_v_psd, self._l_v_conf, self._l_v_bin, self._l_v_snr, self._l_v_var]):
            dg.addWidget(k, i, 0); dg.addWidget(v, i, 1)
        right.addWidget(diag_grp, stretch=3)

        main_lay.addLayout(right, stretch=3)

    def _mk(self, txt):
        k = QLabel(txt+":"); k.setObjectName("subtext")
        v = QLabel("--");    v.setObjectName("monospace")
        return k, v

    def render(self, output: EngineOutput) -> None:
        vf = output.vital_features
        if not vf:
            # Clear plots if no features
            self._phase_buf *= 0.95
            self._phase_img.setImage(self._phase_buf)
            self._lbl_state.setText("NO TARGET")
            self._lbl_state.setStyleSheet(f"color: {PALETTE['subtext']}; padding: 20px; background: {PALETTE['panel']}; border-radius: 12px;")
            return

        # ── Update Phase History Heatmap ──────────────────────────────────────
        # Map current phase to a row in the heatmap [-pi...pi]
        ph = vf.unwrapped_phase_rad if hasattr(vf, "unwrapped_phase_rad") else 0.0
        row = np.zeros(50, dtype=np.float32)
        idx = int(np.clip(((ph % (2*np.pi)) / (2*np.pi)) * 50, 0, 49))
        row[idx] = 1.0
        
        self._phase_buf = np.roll(self._phase_buf, -1, axis=0)
        self._phase_buf[-1] = row
        self._phase_img.setImage(self._phase_buf, autoLevels=False)
        
        self._phase_vals = np.roll(self._phase_vals, -1)
        self._phase_vals[-1] = ph
        self._phase_curve.setData(self._time_axis, self._phase_vals)

        # ── Update PSD Plot ───────────────────────────────────────────────────
        if hasattr(vf, "psd_frequencies") and hasattr(vf, "psd_power"):
            self._psd_curve.setData(vf.psd_frequencies, 10 * np.log10(vf.psd_power + 1e-9) + 60) # Scaled dB

        # ── Update State Badge ────────────────────────────────────────────────
        state = vf.micro_state
        state_txt = state.name.replace("_", " ") if hasattr(state, "name") else str(state)
        self._lbl_state.setText(state_txt)
        if state == MicroState.LARGE_MOTION:
            col = PALETTE["alert"]
        elif state == MicroState.RESTING:
            col = PALETTE["ok"]
        else:
            col = PALETTE["cyan"]
        self._lbl_state.setStyleSheet(f"color: {col}; padding: 20px; background: {PALETTE['panel']}; border-radius: 12px; border: 2px solid {col}40;")

        # ── Update Stats & Labels ─────────────────────────────────────────────
        self._feat_stats.update_data(vf.snr_db, vf.phase_variance)
        
        self._l_v_psd[1].setText(f"{vf.peak_psd_power:.2e}")
        self._l_v_bin[1].setText(str(vf.bin_index))
        self._l_v_snr[1].setText(f"{vf.snr_db:.1f} dB")
        self._l_v_var[1].setText(f"{vf.phase_variance:.4f}")
        self._l_v_conf[1].setText(f"{vf.signal_complexity:.2f}" if hasattr(vf, "signal_complexity") else "--")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = VitalFeaturesDebug()
    win.show()
    sys.exit(app.exec())
