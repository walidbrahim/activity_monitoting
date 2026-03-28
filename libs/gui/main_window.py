import sys
import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QGridLayout, QLabel, QFrame, QApplication, QGraphicsRectItem, QTabWidget)
from PyQt6.QtCore import pyqtSlot, Qt, QTimer
from PyQt6.QtGui import QFont, QColor
import pyqtgraph as pg
from config import config

class CardWidget(QFrame):
    def __init__(self, title, lines):
        super().__init__()
        self.setObjectName("CardWidget")
        self.setStyleSheet(f"""
            QFrame#CardWidget {{
                background-color: {config.gui_theme.card_bg};
                border-radius: 10px;
                padding: 10px;
                border: 1px solid rgba(255, 255, 255, 0.08);
            }}
            QLabel {{
                background-color: transparent;
            }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        
        self.title_lbl = QLabel(title)
        self.title_lbl.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.title_lbl.setStyleSheet(f"color: {config.gui_theme.text};")
        layout.addWidget(self.title_lbl)
        
        self.value_labels = {}
        for line in lines:
            row = QHBoxLayout()
            lbl_key = QLabel(line)
            lbl_key.setStyleSheet(f"color: {config.gui_theme.subtext}; font-size: 15px;")
            lbl_val = QLabel("--")
            lbl_val.setStyleSheet(f"color: {config.gui_theme.text}; font-size: 15px; font-weight: bold;")
            lbl_val.setAlignment(Qt.AlignmentFlag.AlignRight)
            row.addWidget(lbl_key)
            row.addWidget(lbl_val)
            layout.addLayout(row)
            self.value_labels[line] = lbl_val

    def update_values(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.value_labels:
                self.value_labels[k].setText(str(v))

    def set_color(self, hex_color):
        self.setStyleSheet(f"""
            QFrame#CardWidget {{
                background-color: {hex_color};
                border-radius: 10px;
                padding: 10px;
                border: 1px solid rgba(255, 255, 255, 0.12);
            }}
            QLabel {{
                background-color: transparent;
            }}
        """)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Room Activity Monitor")
        self.resize(1200, 800)
        self.setStyleSheet(f"background-color: {config.gui_theme.fig_bg};")

        # History Buffers for plotting
        hist_len = int(config.respiration.resp_window_sec * config.radar.frame_rate)
        self.occ_hist = [0] * hist_len
        self.posture_hist = [0] * hist_len
        self.motion_hist = [0] * hist_len
        self.fall_hist = [0] * hist_len
        self.x_axis = np.linspace(-config.respiration.resp_window_sec, 0, hist_len)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 1. Main Body Layout (Left: Vitals Tabs, Right: Radar & Trends)
        body_layout = QHBoxLayout()
        
        # Left Column (Vitals Tabs)
        self.vitals_tabs = QTabWidget()
        self.vitals_tabs.setStyleSheet(f"""
            QTabWidget::pane {{ border: 1px solid {config.gui_theme.grid}; border-radius: 5px; }}
            QTabBar::tab {{ background: {config.gui_theme.panel_bg}; color: {config.gui_theme.subtext}; padding: 10px 20px; border-top-left-radius: 4px; border-top-right-radius: 4px; margin-right: 2px; }}
            QTabBar::tab:selected {{ background: {config.gui_theme.card_bg}; color: {config.gui_theme.text}; font-weight: bold; border-bottom: 2px solid {config.gui_theme.occupant}; }}
        """)

        # Tab 1: Respiration
        self.tab_resp = QWidget()
        resp_layout = QVBoxLayout(self.tab_resp)
        resp_layout.setContentsMargins(5, 5, 5, 5)

        self.resp_plot = self._create_plot("🫁 Live Breathing Signal", "", "")
        self.resp_plot.setYRange(-3.14, 3.14)
        self.curve_resp = self.resp_plot.plot(pen=pg.mkPen(color=config.gui_theme.occupant, width=2))
        self.curve_resp_zero = self.resp_plot.plot(pen=pg.mkPen(None))  # invisible baseline
        occ_c = QColor(config.gui_theme.occupant)
        self.fill_resp = pg.FillBetweenItem(self.curve_resp, self.curve_resp_zero, brush=pg.mkBrush(occ_c.red(), occ_c.green(), occ_c.blue(), 35))
        self.resp_plot.addItem(self.fill_resp)
        self.scatter_inhale = pg.ScatterPlotItem(size=10, brush=pg.mkBrush('green'), symbol='t')
        self.scatter_exhale = pg.ScatterPlotItem(size=10, brush=pg.mkBrush('red'), symbol='t1')
        self.resp_plot.addItem(self.scatter_inhale)
        self.resp_plot.addItem(self.scatter_exhale)
        resp_layout.addWidget(self.resp_plot, stretch=2)

        self.rr_plot = self._create_plot("📈 Respiration Rate (RR)", "Time (s)", "BPM")
        self.rr_plot.setYRange(0, 40)
        self.curve_rr = self.rr_plot.plot(pen=pg.mkPen(color=config.gui_theme.text, width=2))
        self.curve_rr_zero = self.rr_plot.plot(pen=pg.mkPen(None))
        txt_c = QColor(config.gui_theme.text)
        self.fill_rr = pg.FillBetweenItem(self.curve_rr, self.curve_rr_zero, brush=pg.mkBrush(txt_c.red(), txt_c.green(), txt_c.blue(), 25))
        self.rr_plot.addItem(self.fill_rr)
        resp_layout.addWidget(self.rr_plot, stretch=2)
        
        self.vitals_tabs.addTab(self.tab_resp, "🫁 Respiration")

        # Tab 2: Heart Rate (Placeholder)
        self.tab_hr = QWidget()
        hr_layout = QVBoxLayout(self.tab_hr)
        hr_layout.setContentsMargins(5, 5, 5, 5)
        
        self.hr_plot = self._create_plot("❤️ Live Heartbeat Signal", "", "")
        self.hr_plot.setYRange(-1, 1)
        self.curve_hr = self.hr_plot.plot(pen=pg.mkPen(color='#F43F5E', width=2))
        self.curve_hr_zero = self.hr_plot.plot(pen=pg.mkPen(None))
        self.fill_hr = pg.FillBetweenItem(self.curve_hr, self.curve_hr_zero, brush=pg.mkBrush(244, 63, 94, 35))
        self.hr_plot.addItem(self.fill_hr)
        hr_layout.addWidget(self.hr_plot, stretch=2)

        self.hr_rate_plot = self._create_plot("📈 Heart Rate (BPM)", "Time (s)", "BPM")
        self.hr_rate_plot.setYRange(40, 150)
        self.curve_hr_rate = self.hr_rate_plot.plot(pen=pg.mkPen(color=config.gui_theme.text, width=2))
        hr_layout.addWidget(self.hr_rate_plot, stretch=1)

        self.vitals_tabs.addTab(self.tab_hr, "❤️ Heart Rate")
        
        body_layout.addWidget(self.vitals_tabs, stretch=2)

        # Right Column (Radar Map & Trends)
        right_layout = QVBoxLayout()
        
        # Radar Layout (Top Right)
        self.radar_plot = self._create_plot("🎯 Region of Interest", "X (m)", "Y (m)")
        self.radar_plot.setAspectLocked(True)
        self._draw_static_environment()
        
        self.scatter_halo = pg.ScatterPlotItem(pxMode=False, pen=pg.mkPen(None))
        self.radar_plot.addItem(self.scatter_halo)
        self.scatter_occupant = pg.ScatterPlotItem(pxMode=False, pen=pg.mkPen(None))
        self.radar_plot.addItem(self.scatter_occupant)
        right_layout.addWidget(self.radar_plot, stretch=3)

        # Trends (Bottom Right)
        self.trend_plot = self._create_plot("📊 Confidence: Occupancy / Posture / Fall", "Time (s)", "Normalized")
        self.trend_plot.setYRange(0, 1.05)
        self.curve_occ = self.trend_plot.plot(pen=pg.mkPen(color='#38BDF8', width=2), name="Occupancy")
        self.curve_occ_zero = self.trend_plot.plot(pen=pg.mkPen(None))
        self.fill_occ = pg.FillBetweenItem(self.curve_occ, self.curve_occ_zero, brush=pg.mkBrush(56, 189, 248, 30))
        self.trend_plot.addItem(self.fill_occ)
        self.curve_post = self.trend_plot.plot(pen=pg.mkPen(color='#F59E0B', width=2), name="Posture")
        self.curve_mot = self.trend_plot.plot(pen=pg.mkPen(color='#22C55E', width=2, style=Qt.PenStyle.DashLine), name="Motion")
        self.curve_fall = self.trend_plot.plot(pen=pg.mkPen(color='#EF4444', width=2), name="Fall")
        right_layout.addWidget(self.trend_plot, stretch=1)

        body_layout.addLayout(right_layout, stretch=1)
        main_layout.addLayout(body_layout, stretch=4)

        # 2. Bottom Cards
        cards_layout = QHBoxLayout()
        self.occ_card = CardWidget("📍 Occupancy", ["Zone", "State", "Confidence", "Duration"])
        self.post_card = CardWidget("🧍 Posture & Motion", ["Posture", "Posture conf.", "Height (Range)", "Motion"])
        self.sys_card = CardWidget("⚙️ System", ["Radar", "Tracking", "Fall state", "Fall conf."])
        cards_layout.addWidget(self.occ_card)
        cards_layout.addWidget(self.post_card)
        cards_layout.addWidget(self.sys_card)
        main_layout.addLayout(cards_layout, stretch=1)

    def _create_plot(self, title, xlabel, ylabel):
        p = pg.PlotWidget()
        p.setBackground(config.gui_theme.panel_bg)
        p.setTitle(title, color=config.gui_theme.text, size="14pt", bold=True)
        p.setLabel('bottom', xlabel, color=config.gui_theme.text)
        p.setLabel('left', ylabel, color=config.gui_theme.text)
        p.showGrid(x=True, y=True, alpha=0.3)
        p.getAxis('bottom').setPen(config.gui_theme.grid)
        p.getAxis('left').setPen(config.gui_theme.grid)
        p.getAxis('bottom').setTextPen(config.gui_theme.subtext)
        p.getAxis('left').setTextPen(config.gui_theme.subtext)
        return p

    def _zone_color(self, zone_type):
        return getattr(config.gui_theme, zone_type, "#64748B")

    def _draw_static_environment(self):
        from PyQt6.QtGui import QPainterPath, QTransform
        
        radar_x, radar_y = 0, 0
        fov_deg = 120
        yaw_deg = 0
        
        # Determine room boundaries to set strictly zoomed axis limits
        room_x = [0, 5]
        room_y = [0, 5]

        # Draw boundaries
        for name, bounds in config.layout.items():
            btype = bounds.get("type")
            

            x_min, x_max = bounds["x"]
            y_min, y_max = bounds["y"]
            w, h = x_max - x_min, y_max - y_min
            
            if btype == "boundary":
                room_x = [x_min, x_max]
                room_y = [y_min, y_max]
            
            color = self._zone_color(btype) if btype != "boundary" else config.gui_theme.room_edge
            alpha = 50 if btype != "boundary" else 10
            
            rect = QGraphicsRectItem(x_min, y_min, w, h)
            
            if btype == "ignore":
                rect.setPen(pg.mkPen(color=color, width=1, style=Qt.PenStyle.DashLine))
                rect.setBrush(pg.mkBrush(QColor(color).getRgb()[:3] + (30,))) # Light red
            else:
                rect.setPen(pg.mkPen(color=color, width=2))
                rect.setBrush(pg.mkBrush(QColor(color).getRgb()[:3] + (alpha,)))
                
            self.radar_plot.addItem(rect)
            
            if not hasattr(self, 'zone_rects'):
                self.zone_rects = {}
            self.zone_rects[name] = {"item": rect, "color": color, "base_alpha": 30 if btype == "ignore" else alpha}
            
            text = pg.TextItem(name, anchor=(0.5, 0.5), color=config.gui_theme.text)
            text.setPos(x_min + w/2, y_min + h/2)
            self.radar_plot.addItem(text)
            
        # Draw Default Radar Point & FOV (Using fallback chain)
        d_zone = getattr(config.app, "default_radar_pose", "Room")
        d_pose = config.layout.get(d_zone, {}).get("radar_pose", None)
        room_pose = config.layout.get("Room", {}).get("radar_pose", {})
        
        radar_x = d_pose.get("x") if d_pose else room_pose.get("x", 1.22)
        radar_y = d_pose.get("y") if d_pose else room_pose.get("y", 3.27)
        fov_deg = d_pose.get("fov_deg", 120) if d_pose else room_pose.get("fov_deg", 120)
        yaw_deg = d_pose.get("yaw_deg", 180) if d_pose else room_pose.get("yaw_deg", 180)
        
        self.static_radar_scatter = pg.ScatterPlotItem(x=[radar_x], y=[radar_y], size=15, symbol='t', brush=pg.mkBrush(config.gui_theme.radar))
        self.radar_plot.addItem(self.static_radar_scatter)
        self.static_radar_text = pg.TextItem("Radar", anchor=(0.5, 1.5), color=config.gui_theme.text)
        self.static_radar_text.setPos(radar_x, radar_y)
        self.radar_plot.addItem(self.static_radar_text)
        
        path = QPainterPath()
        path.moveTo(0, 0)
        path.arcTo(-5, -5, 10, 10, yaw_deg - 90 - fov_deg/2, fov_deg)
        path.closeSubpath()
        
        self.static_fov_item = pg.QtWidgets.QGraphicsPathItem(path)
        self.static_fov_item.setPos(radar_x, radar_y)
        self.static_fov_item.setPen(pg.mkPen(color=config.gui_theme.fov, style=Qt.PenStyle.DashLine))
        self.static_fov_item.setBrush(pg.mkBrush(QColor(config.gui_theme.fov).getRgb()[:3] + (15,)))
        self.radar_plot.addItem(self.static_fov_item)
            
        # Lock radar plot strictly to room boundaries (exact fit)
        self.radar_plot.setXRange(room_x[0], room_x[1], padding=0)
        self.radar_plot.setYRange(room_y[0], room_y[1], padding=0)

    def _normalize_motion(self, motion_str):
        s = (motion_str or "").lower()
        if "still" in s: return 0.10
        if "breath" in s: return 0.35
        if "rest" in s: return 0.25
        if "move" in s: return 0.75
        if "active" in s: return 0.90
        return 0.20

    @pyqtSlot(float, float, float, float)
    def update_radar_fov(self, x, y, yaw_deg, fov_deg=120):
        if hasattr(self, 'static_radar_scatter'):
            self.static_radar_scatter.setData(x=[x], y=[y])
        if hasattr(self, 'static_radar_text'):
            self.static_radar_text.setPos(x, y)
        if hasattr(self, 'static_fov_item'):
            from PyQt6.QtGui import QPainterPath
            path = QPainterPath()
            path.moveTo(0, 0)
            path.arcTo(-5, -5, 10, 10, yaw_deg - 90 - fov_deg/2, fov_deg)
            path.closeSubpath()
            self.static_fov_item.setPath(path)
            self.static_fov_item.setPos(x, y)

    @pyqtSlot(dict, dict)
    def update_dashboard(self, occ_dict, resp_dict):
        # Update Cards
        zone = occ_dict.get("zone", "--")
        
        # Highlight Logic — Dynamic Color Temperature
        if hasattr(self, 'zone_rects'):
            occ_conf = occ_dict.get('occ_confidence', 0) / 100.0
            for name, data in self.zone_rects.items():
                rect = data["item"]
                color = data["color"]
                base_alpha = data["base_alpha"]
                
                if name in zone:
                    # Shift hue based on confidence: blue → amber → green
                    if occ_conf > 0.7:
                        highlight_color = QColor("#22C55E")  # Vibrant green
                    elif occ_conf > 0.4:
                        highlight_color = QColor("#F59E0B")  # Warm amber
                    else:
                        highlight_color = QColor(color)       # Base zone color
                    rect.setBrush(pg.mkBrush(highlight_color.red(), highlight_color.green(), highlight_color.blue(), min(255, base_alpha + 100)))
                else:
                    rect.setBrush(pg.mkBrush(QColor(color).getRgb()[:3] + (base_alpha,)))
        
        state = occ_dict.get("status", "Waiting")
        mot_str = occ_dict.get("motion_str", "--")
        
        self.occ_card.update_values(
            Zone=zone,
            State=state[:20] + "..." if len(state)>20 else state,
            Confidence=f"{int(occ_dict.get('occ_confidence', 0))}%",
            Duration=occ_dict.get("duration_str", "--")
        )

        z = occ_dict.get("Z")
        r = occ_dict.get("Range", 0.0)
        height_range_str = f"{z:.2f} m ({r:.2f} m)" if z else "--"
        self.post_card.update_values(
            Posture=occ_dict.get("posture", "--"),
            **{"Posture conf.": f"{int(occ_dict.get('posture_confidence', 0))}%"},
            **{"Height (Range)": height_range_str},
            Motion=mot_str
        )

        fc = occ_dict.get('fall_confidence', 0)
        self.sys_card.update_values(
            Radar="Online",
            Tracking="Good" if occ_dict.get('occ_confidence', 0) > 50 else "Weak",
            **{"Fall state": "Fall Detected!" if fc > 80 else "Normal"},
            **{"Fall conf.": f"{int(fc)}%"}
        )

        # Update Colors
        if "apnea" in state.lower() or fc > 80:
            self.occ_card.set_color(config.gui_theme.card_alert)
        elif "floor" in zone.lower() or "bounds" in zone.lower():
            self.occ_card.set_color(config.gui_theme.card_warn)
        elif "occupied" in state.lower():
            self.occ_card.set_color(config.gui_theme.card_ok)
        else:
            self.occ_card.set_color(config.gui_theme.card_bg)

        # Update Trends
        self.occ_hist.pop(0)
        self.occ_hist.append(np.clip(occ_dict.get('occ_confidence', 0)/100.0, 0, 1))
        self.curve_occ.setData(self.x_axis, self.occ_hist)
        self.curve_occ_zero.setData(self.x_axis, np.zeros(len(self.occ_hist)))

        self.posture_hist.pop(0)
        self.posture_hist.append(np.clip(occ_dict.get('posture_confidence', 0)/100.0, 0, 1))
        self.curve_post.setData(self.x_axis, self.posture_hist)

        self.motion_hist.pop(0)
        self.motion_hist.append(self._normalize_motion(mot_str))
        self.curve_mot.setData(self.x_axis, self.motion_hist)

        self.fall_hist.pop(0)
        self.fall_hist.append(np.clip(fc/100.0, 0, 1))
        self.curve_fall.setData(self.x_axis, self.fall_hist)

        # Update Radar Scatter
        if occ_dict.get("X") is not None and occ_dict.get("status") != "No Occupant":
            x, y = occ_dict["X"], occ_dict["Y"]
            occ_norm = np.clip(occ_dict.get('occ_confidence', 0) / 100.0, 0, 1)
            
            # Matplotlib radii translated to pyqtgraph diameters
            dot_diam = (0.045 + 0.03 * occ_norm) * 2
            halo_diam = (0.12 + 0.18 * occ_norm) * 2
            alpha = int((0.12 + 0.18 * occ_norm) * 255)
            
            base_color = QColor(config.gui_theme.occupant)
            if fc > 80:
                base_color = QColor("#EF4444")  # Fall alert color
            
            halo_color = QColor(base_color.red(), base_color.green(), base_color.blue(), alpha)
            
            self.scatter_halo.setData(x=[x], y=[y], size=halo_diam, brush=pg.mkBrush(halo_color))
            self.scatter_occupant.setData(x=[x], y=[y], size=dot_diam, brush=pg.mkBrush(base_color))
        else:
            self.scatter_occupant.setData([], [])
            self.scatter_halo.setData([], [])

        # Update Respiratory
        if resp_dict and resp_dict.get('confidence', 0) > 0:
            sig = resp_dict.get('live_signal', [])
            x_sig = self.x_axis[-len(sig):]
            self.curve_resp.setData(x_sig, sig)
            self.curve_resp_zero.setData(x_sig, np.zeros(len(sig)))
            
            rr_hist = resp_dict.get('rr_history', [])
            x_rr = self.x_axis[-len(rr_hist):]
            self.curve_rr.setData(x_rr, rr_hist)
            self.curve_rr_zero.setData(x_rr, np.zeros(len(rr_hist)))
            
            # Scatters
            inhs = resp_dict.get('inhales', [])
            exhs = resp_dict.get('exhales', [])
            if len(inhs) > 0 and len(sig) > 0:
                self.scatter_inhale.setData(self.x_axis[inhs], sig[inhs])
            else:
                self.scatter_inhale.setData([], [])
                
            if len(exhs) > 0 and len(sig) > 0:
                self.scatter_exhale.setData(self.x_axis[exhs], sig[exhs])
            else:
                self.scatter_exhale.setData([], [])
        else:
            self.curve_resp.setData([], [])
            self.curve_resp_zero.setData([], [])
            self.curve_rr.setData([], [])
            self.curve_rr_zero.setData([], [])
            self.scatter_inhale.setData([], [])
            self.scatter_exhale.setData([], [])
