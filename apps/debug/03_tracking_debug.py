"""
apps/debug/03_tracking_debug.py
=================================
Debug GUI for TargetTracker.

Layout (redesign):
  ┌──────────────────────────────┬───────────────────────────────────┐
  │                              │  Track Confidence & Miss Status   │
  │  LARGE X-Y FLOOR PLAN        │  (Arc gauge + LED miss-bar)       │
  │  (smoothed, raw, trail)      ├───────────────────────────────────┤
  │                              │  Z-Height History Heatmap         │
  │                              │  (Y: Height, X: Time)             │
  ├──────────────────────────────┼───────────────────────────────────┤
  │  Motion Level Sparkline      │  Diagnostics & EMA Alpha Stats    │
  └──────────────────────────────┴───────────────────────────────────┘

Run:
    APP_PROFILE=lab python apps/debug/03_tracking_debug.py
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


# ── Confidence Arc Gauge ──────────────────────────────────────────────────────

class ConfidenceGauge(QWidget):
    def __init__(self, max_val: int, parent=None):
        super().__init__(parent)
        self.max_val = max(1, max_val)
        self.value   = 0
        self.setMinimumSize(160, 160)

    def set_value(self, v: int):
        self.value = v; self.update()

    def paintEvent(self, event):
        from PyQt6.QtGui import QPainter
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        m = 12
        side = min(w, h) - 2*m
        rect = QRectF((w-side)/2, (h-side)/2, side, side)
        frac = min(1.0, self.value / self.max_val)
        active = frac > 0

        # BG 
        pen = QPen(QColor(PALETTE["border"]), 12, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        p.setPen(pen); p.drawArc(rect, 30*16, 240*16)

        # Value
        col = QColor(PALETTE["ok"] if frac >= 1.0 else PALETTE["accent"])
        pen = QPen(col, 12, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        p.setPen(pen); p.drawArc(rect, 210*16, -int(frac * 240 * 16))

        # Text
        p.setPen(QColor(PALETTE["text"]))
        p.setFont(QFont("Inter", 20, QFont.Weight.Bold))
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter, str(self.value))
        
        p.setPen(QColor(PALETTE["subtext"]))
        p.setFont(QFont("Inter", 8))
        sub_rect = QRectF(rect.x(), rect.y()+40, rect.width(), rect.height())
        p.drawText(sub_rect, Qt.AlignmentFlag.AlignCenter, 
                  "CONFIRMED" if frac >= 1.0 else f"Goal: {self.max_val}")
        p.end()

# ── Miss-count LEDs ───────────────────────────────────────────────────────────

class MissLEDs(QWidget):
    def __init__(self, allowance: int, parent=None):
        super().__init__(parent)
        self.allowance = max(1, allowance)
        self.miss_count = 0
        self.setFixedHeight(45)

    def set_miss(self, m: int):
        self.miss_count = m; self.update()

    def paintEvent(self, event):
        from PyQt6.QtGui import QPainter
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        gap = 4
        led_w = (w - (self.allowance+1)*gap) / self.allowance
        for i in range(self.allowance):
            x = gap + i * (led_w + gap)
            active = i < self.miss_count
            color = QColor(PALETTE["alert"] if active else PALETTE["border"])
            p.setBrush(QBrush(color)); p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(QRectF(x, 10, led_w, h-20), 4, 4)
        p.end()


# ── Main ──────────────────────────────────────────────────────────────────────

class TrackingDebug(DebugBase):
    TITLE    = "📍 Target Tracker — Debug"
    WINDOW_W = 1600
    WINDOW_H = 950

    _HIST_SECS = 20
    _TRAIL_LEN = 100

    def _build_ui(self, central: QWidget) -> None:
        # ── Setup Parameter Tuner ─────────────────────────────────────────────
        self.add_tunable_param("tracking.max_miss_count", "Max Miss Count", 5, 500, 5, self._engine_cfg.tracking.max_miss_count, decimals=0)
        self.add_tunable_param("tracking.confidence_threshold", "Confidence Threshold", 1, 50, 1, self._engine_cfg.tracking.confidence_threshold, decimals=0)
        self.add_tunable_param("tracking.xyz_alpha", "XYZ Smooth Alpha", 0.01, 1.0, 0.01, self._engine_cfg.tracking.xyz_alpha, decimals=2)

        # ── UI Construction ───────────────────────────────────────────────────
        main_lay = QHBoxLayout(central)
        main_lay.setContentsMargins(10, 10, 10, 10)
        main_lay.setSpacing(12)

        # ── Left: Large Floor Plan ────────────────────────────────────────────
        left = QVBoxLayout()
        map_grp = QGroupBox("X-Y World Coordinates [ ○ smoothed │ ● raw │ ╌ trail ]")
        ml = QVBoxLayout(map_grp)
        self._pw_map = pg.PlotWidget()
        self._pw_map.setBackground(PALETTE["panel"])
        self._pw_map.setAspectLocked(True)
        apply_plot_defaults(self._pw_map.getPlotItem(), xlabel="X (metres)", ylabel="Y (metres)")
        
        # Room & Zones
        room = self._app_cfg.layout.get("Room", {"x":[0,2], "y":[0,3]})
        rx = room["x"]; ry = room["y"]
        self._pw_map.setXRange(rx[0]-0.5, rx[1]+0.5)
        self._pw_map.setYRange(ry[0]-0.5, ry[1]+0.5)
        
        room_poly = np.array([[rx[0],ry[0]],[rx[1],ry[0]],[rx[1],ry[1]],[rx[0],ry[1]],[rx[0],ry[0]]])
        self._pw_map.plot(room_poly[:,0], room_poly[:,1], pen=pg.mkPen("#CBD5E1", width=2))

        # Trail & Dots
        self._trail_curve = self._pw_map.plot(pen=pg.mkPen(PALETTE["text"]+"80", width=1.5, style=Qt.PenStyle.DashLine))
        self._trail_x, self._trail_y = [], []
        
        self._dot_raw = pg.ScatterPlotItem(symbol='o', size=12, brush=pg.mkBrush(PALETTE["subtext"]+"60"))
        self._dot_smooth = pg.ScatterPlotItem(symbol='o', size=24, brush=pg.mkBrush(PALETTE["cyan"]),
                                               pen=pg.mkPen("#FFFFFF", width=1.5))
        self._pw_map.addItem(self._dot_raw); self._pw_map.addItem(self._dot_smooth)

        # Jump reject message
        self._jump_txt = pg.TextItem("⚠️ JUMP REJECTED", color=PALETTE["alert"], anchor=(0.5,0.5))
        self._jump_txt.setFont(QFont("Inter", 14, QFont.Weight.Bold))
        self._jump_txt.setVisible(False)
        self._pw_map.addItem(self._jump_txt)
        self._jump_timer = 0

        ml.addWidget(self._pw_map)
        
        # Motion sparkline at bottom of left column
        motion_grp = QGroupBox(f"Motion Intensity (last {self._HIST_SECS}s)")
        mol = QVBoxLayout(motion_grp)
        self._pw_motion = self.make_plot_widget(xlabel="Time (s ago)", ylabel="Magnitude")
        self._pw_motion.setYRange(0, 1.0)
        self._motion_buf = np.zeros(int(self._HIST_SECS * self._frame_rate))
        self._motion_time = np.linspace(-self._HIST_SECS, 0, len(self._motion_buf))
        self._motion_curve = self._pw_motion.plot(self._motion_time, self._motion_buf, 
                                                   pen=pg.mkPen(PALETTE["accent2"], width=2), fillLevel=0, brush=PALETTE["accent2"]+"30")
        mol.addWidget(self._pw_motion)
        left.addWidget(map_grp, stretch=7)
        left.addWidget(motion_grp, stretch=2)
        main_lay.addLayout(left, stretch=6)

        # ── Right Col: Status & Z-History ─────────────────────────────────────
        right = QVBoxLayout()
        
        # Top status cluster
        status_row = QHBoxLayout()
        
        conf_grp = QGroupBox("Track Confidence")
        cl = QVBoxLayout(conf_grp)
        conf_thresh = getattr(self._app_cfg.tracking, "confidence_threshold", 4)
        self._conf_gauge = ConfidenceGauge(max_val=conf_thresh)
        cl.addWidget(self._conf_gauge, alignment=Qt.AlignmentFlag.AlignCenter)
        status_row.addWidget(conf_grp)

        miss_grp = QGroupBox("Persistence (Miss Count)")
        ml2 = QVBoxLayout(miss_grp)
        miss_allow = getattr(self._app_cfg.tracking, "miss_allowance", 20)
        self._miss_leds = MissLEDs(allowance=miss_allow)
        ml2.addWidget(self._miss_leds)
        self._lbl_miss = QLabel(f"0 / {miss_allow}")
        self._lbl_miss.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_miss.setObjectName("monospace")
        ml2.addWidget(self._lbl_miss)
        status_row.addWidget(miss_grp)
        
        right.addLayout(status_row, stretch=3)

        # Z-Height Heatmap (VERY useful for fall detection debug)
        z_grp = QGroupBox("Height Z-Timeline (Y: Height 0…2.5m)")
        zl = QVBoxLayout(z_grp)
        self._pw_z = pg.PlotWidget()
        self._pw_z.setBackground(PALETTE["panel"])
        apply_plot_defaults(self._pw_z.getPlotItem(), xlabel="Time (s ago)", ylabel="Height (m)")
        self._pw_z.setYRange(0, 2.5)
        
        self._z_img = pg.ImageItem()
        self._z_img.setColorMap(pg.colormap.get("magma"))
        tr = QTransform()
        tr.translate(-self._HIST_SECS, 0)
        tr.scale(1.0/self._frame_rate, 0.05) # 50 rows for 2.5m -> 0.05m per row
        self._z_img.setTransform(tr)
        self._pw_z.addItem(self._z_img)
        
        self._z_buf = np.zeros((int(self._HIST_SECS * self._frame_rate), 50), dtype=np.float32)
        
        # Smoothed line overlay on top of heatmap
        self._z_curve = self._pw_z.plot(pen=pg.mkPen("#FFFFFF", width=2))
        self._z_pts   = np.zeros(len(self._z_buf))
        
        zl.addWidget(self._pw_z)
        right.addWidget(z_grp, stretch=5)

        # Diag labels
        diag_grp = QGroupBox("Track Statistics")
        dg = QGridLayout(diag_grp)
        self._l_pos = self._mk("Coord XYZ"); self._l_range = self._mk("Slant Range")
        self._l_vz  = self._mk("Z-Speed");   self._l_alpha = self._mk("EMA Alpha")
        self._l_jump = self._mk("Jump count"); self._jump_counter = 0
        for i, (k,v) in enumerate([self._l_pos, self._l_range, self._l_vz, self._l_alpha, self._l_jump]):
            dg.addWidget(k, i, 0); dg.addWidget(v, i, 1)
        right.addWidget(diag_grp, stretch=2)

        main_lay.addLayout(right, stretch=4)

    def _mk(self, txt):
        k = QLabel(txt+":"); k.setObjectName("subtext")
        v = QLabel("--");    v.setObjectName("monospace")
        return k, v

    def render(self, output: EngineOutput) -> None:
        tr = output.tracked_target
        dg = output.diagnostics
        
        # ── Motion Update ─────────────────────────────────────────────────────
        mlvl = getattr(output.activity, "motion_level", 0.0) if output.activity else 0.0
        self._motion_buf = np.roll(self._motion_buf, -1)
        self._motion_buf[-1] = mlvl
        self._motion_curve.setData(self._motion_time, self._motion_buf)

        if tr and tr.valid:
            # ── Map & Trail ───────────────────────────────────────────────────
            self._trail_x.append(tr.smoothed_x_m); self._trail_y.append(tr.smoothed_y_m)
            if len(self._trail_x) > self._TRAIL_LEN:
                self._trail_x.pop(0); self._trail_y.pop(0)
            self._trail_curve.setData(self._trail_x, self._trail_y)
            self._dot_raw.setData([tr.x_m], [tr.y_m])
            self._dot_smooth.setData([tr.smoothed_x_m], [tr.smoothed_y_m])
            self._jump_txt.setPos(tr.smoothed_x_m, tr.smoothed_y_m + 0.3)

            # ── Height Update ─────────────────────────────────────────────────
            # Fill heatmap column based on Z (gaussian smear)
            z_idx = int(np.clip(tr.smoothed_z_m / 0.05, 0, 49))
            row = np.zeros(50, dtype=np.float32)
            row[max(0, z_idx-1):min(50, z_idx+2)] = 0.8 # smear
            row[z_idx] = 1.0
            
            self._z_buf = np.roll(self._z_buf, -1, axis=0)
            self._z_buf[-1] = row
            self._z_img.setImage(self._z_buf, autoLevels=False, levels=(0, 1))
            
            self._z_pts = np.roll(self._z_pts, -1)
            self._z_pts[-1] = tr.smoothed_z_m
            self._z_curve.setData(self._motion_time, self._z_pts)

            # ── Gauges & Labels ───────────────────────────────────────────────
            self._conf_gauge.set_value(tr.confidence)
            self._miss_leds.set_miss(tr.miss_count)
            self._lbl_miss.setText(f"{tr.miss_count} / {self._miss_leds.allowance}")
            
            self._l_pos[1].setText(f"({tr.smoothed_x_m:.2f}, {tr.smoothed_y_m:.2f}, {tr.smoothed_z_m:.2f})")
            self._l_range[1].setText(f"{np.sqrt(tr.x_m**2 + tr.y_m**2 + tr.z_m**2):.2f} m")
            self._l_vz[1].setText(f"{tr.v_z:+.3f} m/s")
            self._l_alpha[1].setText(f"{self._app_cfg.tracking.track_ema_alpha:.3f}")
        else:
            self._dot_raw.setData([], []); self._dot_smooth.setData([], [])
            self._conf_gauge.set_value(0); self._miss_leds.set_miss(0)
            # Fade height back to zero
            self._z_buf = np.roll(self._z_buf, -1, axis=0)
            self._z_buf[-1] = 0
            self._z_img.setImage(self._z_buf)

        # Jump reject flash
        if dg.get("jump_reject_dist_m"):
            self._jump_counter += 1
            self._jump_timer = 10
        
        self._l_jump[1].setText(str(self._jump_counter))
        if self._jump_timer > 0:
            self._jump_txt.setVisible(True); self._jump_timer -= 1
        else:
            self._jump_txt.setVisible(False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = TrackingDebug()
    win.show()
    sys.exit(app.exec())
