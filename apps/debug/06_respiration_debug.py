"""
apps/debug/06_respiration_debug.py
====================================
Debug GUI for Respiration Monitoring (Waveform, BPM, Spectrogram).

Layout:
  ┌──────────────────────────────┬──────────────────┐
  │  Breath Waveform             │  BPM Speedometer │
  │  (Filtered phase signal)     │  (0–40 BPM)      │
  ├──────────────────────────────┼──────────────────┤
  │  Respiration Spectrogram     │  Breath Status   │
  │  (Frequency vs Time)         │  (Inhale/Exhale) │
  └──────────────────────────────┴──────────────────┘

Run:
    APP_PROFILE=lab python apps/debug/06_respiration_debug.py
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


# ── BPM Speedometer (Arc) ─────────────────────────────────────────────────────

class BPMSpeedometer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.bpm = 0.0
        self.setMinimumSize(220, 220)

    def set_bpm(self, val: float):
        self.bpm = val; self.update()

    def paintEvent(self, event):
        from PyQt6.QtGui import QPainter
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        m = 20
        side = min(w, h) - 2*m
        rect = QRectF((w-side)/2, (h-side)/2, side, side)
        
        # Scale: 0 to 40 BPM
        frac = min(1.0, self.bpm / 40.0)
        
        # BG Arc
        p.setPen(QPen(QColor(PALETTE["border"]), 14, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawArc(rect, -30*16, 240*16)
        
        # Value Arc
        col = QColor(PALETTE["cyan"]) if 10 < self.bpm < 30 else QColor(PALETTE["warn"])
        p.setPen(QPen(col, 14, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawArc(rect, 210*16, -int(frac * 240 * 16))
        
        # Main BPM Text
        p.setPen(QColor(PALETTE["text"]))
        p.setFont(QFont("Inter", 32, QFont.Weight.Bold))
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter, f"{int(self.bpm)}")
        
        # Subtitle
        p.setFont(QFont("Inter", 10))
        p.setPen(QColor(PALETTE["subtext"]))
        sub_rect = QRectF(rect.x(), rect.y()+50, rect.width(), rect.height())
        p.drawText(sub_rect, Qt.AlignmentFlag.AlignCenter, "BREATHS / MIN")
        p.end()


# ── Breath Pulse Indicator ────────────────────────────────────────────────────

class BreathPulse(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pulse = 0.0 # 0..1
        self.setMinimumHeight(60)

    def set_pulse(self, p: float):
        self.pulse = p; self.update()

    def paintEvent(self, event):
        from PyQt6.QtGui import QPainter
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        
        # Pulsing circle
        size = 20 + 20 * self.pulse
        cx, cy = w/2, h/2
        col = QColor(PALETTE["ok"])
        col.setAlpha(int(100 + 155 * self.pulse))
        p.setBrush(QBrush(col))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(int(cx - size/2), int(cy - size/2), int(size), int(size))
        p.end()


# ── Main ──────────────────────────────────────────────────────────────────────

class RespirationDebug(DebugBase):
    TITLE    = "🌬️ Respiration Monitor — Debug"
    WINDOW_W = 1600
    WINDOW_H = 950

    _HIST_SECS = 45 # Longer window for slow breathing

    def _build_ui(self, central: QWidget) -> None:
        main_lay = QHBoxLayout(central)
        main_lay.setContentsMargins(10, 10, 10, 10)
        main_lay.setSpacing(12)

        # ── Left: Waveforms & Spectrogram ─────────────────────────────────────
        left = QVBoxLayout()
        
        # Waveform
        wave_grp = QGroupBox("Breath Waveform (Filtered Phase History)")
        wl = QVBoxLayout(wave_grp)
        self._pw_wave = pg.PlotWidget()
        self._pw_wave.setBackground(PALETTE["panel"])
        apply_plot_defaults(self._pw_wave.getPlotItem(), xlabel="Time (s ago)", ylabel="Disp. (mm)")
        self._pw_wave.setYRange(-0.5, 0.5)
        
        self._wave_buf = np.zeros(int(self._HIST_SECS * self._frame_rate))
        self._time_ax = np.linspace(-self._HIST_SECS, 0, len(self._wave_buf))
        self._wave_curve = self._pw_wave.plot(self._time_ax, self._wave_buf, pen=pg.mkPen(PALETTE["ok"], width=2))
        
        wl.addWidget(self._pw_wave)
        left.addWidget(wave_grp, stretch=4)

        # Spectrogram
        spec_grp = QGroupBox("Respiration Spectrogram (Y: Frequency 0–1 Hz / 0–60 BPM)")
        spl = QVBoxLayout(spec_grp)
        self._pw_spec = pg.PlotWidget()
        self._pw_spec.setBackground(PALETTE["panel"])
        apply_plot_defaults(self._pw_spec.getPlotItem(), xlabel="Time (s ago)", ylabel="BPM")
        self._pw_spec.setYRange(0, 45)
        
        self._spec_img = pg.ImageItem()
        self._spec_img.setColorMap(pg.colormap.get("viridis"))
        tr = QTransform()
        tr.translate(-self._HIST_SECS, 0)
        tr.scale(1.0/self._frame_rate, 45.0/100) # 100 rows for 45 BPM range
        self._spec_img.setTransform(tr)
        self._pw_spec.addItem(self._spec_img)
        
        self._spec_buf = np.zeros((len(self._wave_buf), 100), dtype=np.float32)
        
        # Smoothed line overlay on spectrogram
        self._spec_curve = self._pw_spec.plot(pen=pg.mkPen("#FFFFFF", width=2))
        self._spec_pts   = np.zeros(len(self._wave_buf))
        
        spl.addWidget(self._pw_spec)
        left.addWidget(spec_grp, stretch=5)

        main_lay.addLayout(left, stretch=7)

        # ── Right: Gauges & Diagnostics ──────────────────────────────────────
        right = QVBoxLayout()
        
        # Speedometer
        speed_grp = QGroupBox("Current Respiration Rate")
        sl = QVBoxLayout(speed_grp)
        self._gauge = BPMSpeedometer()
        sl.addWidget(self._gauge)
        right.addWidget(speed_grp, stretch=4)

        # Pulse indicator
        pulse_grp = QGroupBox("Breath Pulse")
        pl = QVBoxLayout(pulse_grp)
        self._pulse = BreathPulse()
        pl.addWidget(self._pulse)
        self._lbl_status = QLabel("WAITING...")
        self._lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_status.setStyleSheet("color: #94A3B8; font-weight: bold;")
        pl.addWidget(self._lbl_status)
        right.addWidget(pulse_grp, stretch=2)

        # Diagnostics
        diag_grp = QGroupBox("Detection Stats")
        dg = QGridLayout(diag_grp)
        self._l_v_bpm  = self._mk("Detected BPM");  self._l_v_conf = self._mk("Rate Confidence")
        self._l_v_pe   = self._mk("Peak Energy");   self._l_v_zrc  = self._mk("ZC/Peak Sync")
        for i, (k,v) in enumerate([self._l_v_bpm, self._l_v_conf, self._l_v_pe, self._l_v_zrc]):
            dg.addWidget(k, i, 0); dg.addWidget(v, i, 1)
        right.addWidget(diag_grp, stretch=3)

        main_lay.addLayout(right, stretch=3)

    def _mk(self, txt):
        k = QLabel(txt+":"); k.setObjectName("subtext")
        v = QLabel("--");    v.setObjectName("monospace")
        return k, v

    def render(self, output: EngineOutput) -> None:
        resp = output.respiration
        vf   = output.vital_features
        
        if not resp:
            self._wave_buf *= 0.98
            self._wave_curve.setData(self._time_ax, self._wave_buf)
            self._lbl_status.setText("SEARCHING...")
            self._pulse.set_pulse(0.0)
            return

        # ── Update Waveform ──
        sig = resp.breath_waveform if hasattr(resp, "breath_waveform") else 0.0
        self._wave_buf = np.roll(self._wave_buf, -1); self._wave_buf[-1] = sig
        self._wave_curve.setData(self._time_ax, self._wave_buf)

        # ── Update Spectrogram ──
        # Fill spectrogram column based on BPM (gaussian smear)
        bpm = resp.breathing_rate_bpm
        bpm_idx = int(np.clip(bpm / (45.0/100.0), 0, 99))
        
        spec_row = np.zeros(100, dtype=np.float32)
        spec_row[max(0, bpm_idx-1):min(100, bpm_idx+2)] = 0.8
        spec_row[bpm_idx] = 1.0
        
        self._spec_buf = np.roll(self._spec_buf, -1, axis=0)
        self._spec_buf[-1] = spec_row
        self._spec_img.setImage(self._spec_buf, autoLevels=False, levels=(0, 1))

        self._spec_pts = np.roll(self._spec_pts, -1); self._spec_pts[-1] = bpm
        self._spec_curve.setData(self._time_ax, self._spec_pts)

        # ── Update Gauges ──
        self._gauge.set_bpm(bpm)
        
        # Pulse animation (using sin wave approximation of breath phase if available)
        # Or simple peak detect. Here we'll use wave intensity.
        p_val = np.clip((sig + 0.5), 0, 1)
        self._pulse.set_pulse(float(p_val))
        self._lbl_status.setText("BREATHING DETECTED")

        # ── Update Stats ──
        self._l_v_bpm[1].setText(f"{bpm:.1f}")
        self._l_v_conf[1].setText(f"{resp.confidence:.2f}" if hasattr(resp, "confidence") else "--")
        self._l_v_pe[1].setText(f"{vf.peak_psd_power:.2e}" if vf else "--")
        self._l_v_zrc[1].setText("LOCKED" if resp.confidence > 0.5 else "ACQUIRING")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = RespirationDebug()
    win.show()
    sys.exit(app.exec())
