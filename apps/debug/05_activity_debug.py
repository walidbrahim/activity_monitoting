"""
apps/debug/05_activity_debug.py
================================
Debug GUI for Activity Pipeline (Occupancy, Posture, Fall detection).

Layout:
  ┌──────────────────────────────┬──────────────────┐
  │  Zone Occupancy Map          │  Current Posture │
  │  (Layout with active zones)  │  (Silhouette)    │
  ├──────────────────────────────┼──────────────────┤
  │  Activity State Timeline     │  Fall Detection  │
  │  (Last 5 minutes)            │  LED & Stats     │
  └──────────────────────────────┴──────────────────┘

Run:
    APP_PROFILE=lab python apps/debug/05_activity_debug.py
"""
from __future__ import annotations
import sys, os
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout,
    QGridLayout, QGroupBox, QLabel, QSizePolicy,
)
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QColor, QPen, QBrush, QFont, QPolygonF
import pyqtgraph as pg

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from apps.debug.debug_base import DebugBase, PALETTE, apply_plot_defaults  # noqa: E402
from radar_engine.core.models import EngineOutput                           # noqa: E402
from radar_engine.core.enums import PostureLabel, OccupancyState, FallState  # noqa: E402


# ── Posture Silhouette Widget ──────────────────────────────────────────────────

class PostureSilhouette(QWidget):
    """Draws a stylized silhouette based on the current posture label."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.posture = PostureLabel.UNKNOWN
        self.setMinimumSize(180, 180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_posture(self, p: PostureLabel):
        self.posture = p; self.update()

    def paintEvent(self, event):
        from PyQt6.QtGui import QPainter, QPainterPath
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        
        # Draw background card
        p.setBrush(QBrush(QColor(PALETTE["panel"])))
        p.setPen(QPen(QColor(PALETTE["border"]), 1))
        p.drawRoundedRect(QRectF(10, 10, w-20, h-20), 12, 12)
        
        # Center of the card
        cx, cy = w/2, h/2
        col = QColor(PALETTE["accent"])
        p.setPen(QPen(col, 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.setBrush(QBrush(col))

        if self.posture == PostureLabel.STANDING:
            # Standing silhouette
            p.drawEllipse(int(cx-8), int(cy-50), 16, 16) # Head
            p.drawLine(int(cx), int(cy-34), int(cx), int(cy+10)) # Torso
            p.drawLine(int(cx), int(cy-25), int(cx-20), int(cy-5)) # L-Arm
            p.drawLine(int(cx), int(cy-25), int(cx+20), int(cy-5)) # R-Arm
            p.drawLine(int(cx), int(cy+10), int(cx-15), int(cy+50)) # L-Leg
            p.drawLine(int(cx), int(cy+10), int(cx+15), int(cy+50)) # R-Leg
        
        elif self.posture == PostureLabel.SITTING:
            # Sitting silhouette
            p.drawEllipse(int(cx-8), int(cy-35), 16, 16) # Head
            p.drawLine(int(cx), int(cy-19), int(cx), int(cy+15)) # Torso
            p.drawLine(int(cx), int(cy+15), int(cx+30), int(cy+15)) # Thighs
            p.drawLine(int(cx+30), int(cy+15), int(cx+30), int(cy+45)) # Legs
            p.drawLine(int(cx), int(cy-10), int(cx+15), int(cy+5)) # L-Arm
            p.drawLine(int(cx), int(cy-10), int(cx-10), int(cy+5)) # R-Arm

        elif self.posture == PostureLabel.LYING:
            # Lying silhouette
            p.drawEllipse(int(cx-50), int(cy-20), 16, 16) # Head
            p.drawLine(int(cx-34), int(cy-12), int(cx+30), int(cy-12)) # Body
            p.drawLine(int(cx-10), int(cy-12), int(cx+5), int(cy+10)) # Arm
            p.drawLine(int(cx+30), int(cy-12), int(cx+50), int(cy-5)) # Legs
        
        else:
            # Unknown / Empty silhouette
            p.setPen(QPen(QColor(PALETTE["border"]), 2, Qt.PenStyle.DashLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(int(cx-8), int(cy-10), 16, 16)

        p.end()


# ── Main ──────────────────────────────────────────────────────────────────────

class ActivityDebug(DebugBase):
    TITLE    = "🧠 Activity Intelligence — Debug"
    WINDOW_W = 1600
    WINDOW_H = 950

    def _build_ui(self, central: QWidget) -> None:
        # ── Setup Parameter Tuner ─────────────────────────────────────────────
        self.add_tunable_param("activity.z_standing_threshold", "Standing Z Threshold", 0.5, 2.0, 0.1, self._engine_cfg.activity.z_standing_threshold, decimals=2)
        self.add_tunable_param("activity.z_lying_threshold", "Lying Z Threshold", 0.1, 1.5, 0.1, self._engine_cfg.activity.z_lying_threshold, decimals=2)
        self.add_tunable_param("activity.zone_debounce_frames", "Zone Debounce Frames", 1, 100, 1, self._engine_cfg.activity.zone_debounce_frames, decimals=0)

        # ── UI Construction ───────────────────────────────────────────────────
        grid = QGridLayout(central)
        grid.setContentsMargins(10, 10, 10, 10)
        grid.setSpacing(12)

        # ── Left: Zone Map ────────────────────────────────────────────────────
        map_grp = QGroupBox("Zone Occupancy Layout [ Green = Occupied │ Grey = Empty ]")
        ml = QVBoxLayout(map_grp)
        self._pw_map = pg.PlotWidget()
        self._pw_map.setBackground(PALETTE["panel"])
        self._pw_map.setAspectLocked(True)
        apply_plot_defaults(self._pw_map.getPlotItem(), xlabel="X (metres)", ylabel="Y (metres)")
        
        # Load zones and store items to update colors
        self._zone_items = {}
        room = self._app_cfg.layout.get("Room", {"x":[0,2], "y":[0,3]})
        rx = room["x"]; ry = room["y"]
        self._pw_map.setXRange(rx[0]-0.2, rx[1]+0.2); self._pw_map.setYRange(ry[0]-0.2, ry[1]+0.2)
        self._pw_map.plot([rx[0],rx[1],rx[1],rx[0],rx[0]], [ry[0],ry[0],ry[1],ry[1],ry[0]], pen=pg.mkPen("#CBD5E1", width=2))

        for zname, zval in self._app_cfg.layout.items():
            if not isinstance(zval, dict): continue
            if zval.get("type") in ("monitor", "ignore"):
                zx = zval.get("x"); zy = zval.get("y")
                rect = pg.QtWidgets.QGraphicsRectItem(zx[0], zy[0], zx[1]-zx[0], zy[1]-zy[0])
                rect.setBrush(QBrush(QColor(PALETTE["border"]+"40")))
                rect.setPen(QPen(QColor(PALETTE["border"]), 1))
                self._pw_map.addItem(rect)
                txt = pg.TextItem(zname, color=PALETTE["subtext"], anchor=(0,1))
                txt.setPos(zx[0], zy[1])
                self._pw_map.addItem(txt)
                self._zone_items[zname] = rect

        self._target_dot = pg.ScatterPlotItem(symbol='o', size=20, brush=pg.mkBrush(PALETTE["cyan"]))
        self._pw_map.addItem(self._target_dot)
        
        ml.addWidget(self._pw_map)
        grid.addWidget(map_grp, 0, 0, 2, 1)

        # ── Right Col: Posture Silhouette ─────────────────────────────────────
        posture_grp = QGroupBox("Current Posture Label")
        pl = QVBoxLayout(posture_grp)
        self._silhouette = PostureSilhouette()
        pl.addWidget(self._silhouette)
        self._lbl_posture = QLabel("UNKNOWN")
        self._lbl_posture.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_posture.setStyleSheet(f"color: {PALETTE['subtext']}; font-weight: bold; font-size: 14px;")
        pl.addWidget(self._lbl_posture)
        grid.addWidget(posture_grp, 0, 1)

        # ── Middle-Bottom: Fall Status ────────────────────────────────────────
        fall_grp = QGroupBox("Fall Detection Status")
        fl = QVBoxLayout(fall_grp)
        self._lbl_fall = QLabel("NO FALL DETECTED")
        self._lbl_fall.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_fall.setStyleSheet(f"background: {PALETTE['panel']}; color: {PALETTE['subtext']}; padding: 15px; font-weight: bold; font-size: 16px; border-radius: 8px;")
        fl.addWidget(self._lbl_fall)
        right_sub = QHBoxLayout()
        self._l_v_z_peak = self._mk("Peak Drop Velocity (m/s)")
        right_sub.addWidget(self._l_v_z_peak[0]); right_sub.addWidget(self._l_v_z_peak[1])
        fl.addLayout(right_sub)
        grid.addWidget(fall_grp, 1, 1)

        # ── Bottom: Timeline ──────────────────────────────────────────────────
        time_grp = QGroupBox("Activity Timeline (Occupancy History)")
        tl = QVBoxLayout(time_grp)
        self._pw_time = self.make_plot_widget(xlabel="Frames Ago", ylabel="State")
        self._pw_time.setYRange(0, 3)
        self._state_buf = np.zeros(200)
        self._state_curve = self._pw_time.plot(self._state_buf, pen=pg.mkPen(PALETTE["accent"], width=2), stepMode="center")
        tl.addWidget(self._pw_time)
        grid.addWidget(time_grp, 2, 0, 1, 2)

        grid.setRowStretch(0, 4); grid.setRowStretch(1, 2); grid.setRowStretch(2, 3)
        grid.setColumnStretch(0, 6); grid.setColumnStretch(1, 4)

    def _mk(self, txt):
        k = QLabel(txt+":"); k.setObjectName("subtext")
        v = QLabel("--");    v.setObjectName("monospace")
        return k, v

    def render(self, output: EngineOutput) -> None:
        act = output.activity
        tr  = output.tracked_target
        
        # ── Update Occupancy Map ──────────────────────────────────────────────
        for name, item in self._zone_items.items():
            occupied = False
            if act and act.current_zone == name: occupied = True
            color = PALETTE["ok"] if occupied else PALETTE["border"]
            item.setBrush(QBrush(QColor(color + "60")))
            item.setPen(QPen(QColor(color), 1.5))
        
        if tr and tr.valid:
            self._target_dot.setData([tr.smoothed_x_m], [tr.smoothed_y_m])
        else:
            self._target_dot.setData([], [])

        # ── Update Posture ────────────────────────────────────────────────────
        if act:
            self._silhouette.set_posture(act.posture)
            self._lbl_posture.setText(act.posture.name)
            
            # ── Update Fall Status ──
            if act.fall_state == FallState.FALL_DETECTED:
                self._lbl_fall.setText("⚠️ FALL DETECTED!")
                self._lbl_fall.setStyleSheet(f"background: {PALETTE['alert']}; color: white; padding: 15px; font-weight: bold; font-size: 16px; border-radius: 8px;")
            else:
                self._lbl_fall.setText("NO FALL")
                self._lbl_fall.setStyleSheet(f"background: {PALETTE['panel']}; color: {PALETTE['ok']}; padding: 15px; font-weight: bold; font-size: 16px; border-radius: 8px;")
            
            # ── Update Timeline ──
            occ_val = 1 if act.occupancy_state == OccupancyState.OCCUPIED else 0
            self._state_buf = np.roll(self._state_buf, -1)
            self._state_buf[-1] = occ_val
            self._state_curve.setData(self._state_buf)

            self._l_v_z_peak[1].setText(f"{tr.v_z:.2f}" if tr else "--")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ActivityDebug()
    win.show()
    sys.exit(app.exec())
