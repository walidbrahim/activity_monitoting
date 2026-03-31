import sys
import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QGridLayout, QLabel, QFrame, QApplication, QGraphicsRectItem, QTabWidget)
from PyQt6.QtCore import pyqtSlot, Qt, QTimer
from PyQt6.QtGui import QFont, QColor
import pyqtgraph as pg
from config import config
from libs.gui.posture_widget import PostureCard, RadarPostureItem, _classify_motion

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
        self.title_lbl.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        self.title_lbl.setStyleSheet(f"color: {config.gui_theme.text};")
        layout.addWidget(self.title_lbl)
        
        self.value_labels = {}
        for line in lines:
            row = QHBoxLayout()
            lbl_key = QLabel(line)
            lbl_key.setStyleSheet(f"color: {config.gui_theme.subtext}; font-size: 20px;")
            lbl_val = QLabel("--")
            lbl_val.setStyleSheet(f"color: {config.gui_theme.text}; font-size: 20px; font-weight: bold;")
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
        self.occ_hist       = [0]   * hist_len
        self.posture_hist   = [0]   * hist_len
        self.motion_hist    = [0]   * hist_len
        self.fall_hist      = [0]   * hist_len
        self.mag_hist       = [0]   * hist_len
        self.thresh_hist    = [0]   * hist_len
        self.height_hist    = [0.0] * hist_len  # real-time Z (height) history
        self.posture_num_hist = [0] * hist_len  # posture encoded as integer for timeline plot
        self.x_axis = np.linspace(-config.respiration.resp_window_sec, 0, hist_len)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Two-column layout: each column manages its own vertical split independently.
        # Left col:  vitals tabs (top)          + occ/posture cards (bottom)
        # Right col: radar map  (top, 3/4 tall) + analytics/confidence plots (bottom, 1/4)
        columns_layout = QHBoxLayout()
        columns_layout.setSpacing(10)
        columns_layout.setContentsMargins(0, 0, 0, 0)

        left_col = QVBoxLayout()
        left_col.setSpacing(8)

        # Left Column — Vitals Tabs
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

        self.resp_plot = self._create_plot("🫁 Live Breathing Signal", "Time (s)", "Displacement (mm)")
        self._resp_window = config.respiration.resp_window_sec
        self.resp_plot.setXRange(-self._resp_window, 0, padding=0)
        self.resp_plot.setYRange(-25, 25)
        self.resp_plot.enableAutoRange(axis='x', enable=False)
        self.resp_plot.enableAutoRange(axis='y', enable=False)
        self.curve_resp = self.resp_plot.plot(pen=pg.mkPen(color=config.gui_theme.occupant, width=2))
        self.curve_resp_zero = self.resp_plot.plot(pen=pg.mkPen(None))  # invisible baseline
        occ_c = QColor(config.gui_theme.occupant)
        self.fill_resp = pg.FillBetweenItem(self.curve_resp, self.curve_resp_zero, brush=pg.mkBrush(occ_c.red(), occ_c.green(), occ_c.blue(), 35))
        self.resp_plot.addItem(self.fill_resp)
        self.scatter_inhale = pg.ScatterPlotItem(size=10, brush=pg.mkBrush('green'), symbol='t')
        self.scatter_exhale = pg.ScatterPlotItem(size=10, brush=pg.mkBrush('red'), symbol='t1')
        self.resp_plot.addItem(self.scatter_inhale)
        self.resp_plot.addItem(self.scatter_exhale)

        # Annotations on respiration plot
        self._resp_annotations = {}
        ann_style_white = {'color': config.gui_theme.text, 'size': '11pt'}
        ann_style_red = {'color': '#EF4444', 'size': '11pt', 'bold': True}
        ann_style_green = {'color': '#22C55E', 'size': '11pt'}

        self._resp_ann_depth = pg.TextItem("", anchor=(0, 0), color=config.gui_theme.text)
        self._resp_ann_depth.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.resp_plot.addItem(self._resp_ann_depth)

        self._resp_ann_apnea = pg.TextItem("", anchor=(1, 0), color='#EF4444')
        self._resp_ann_apnea.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.resp_plot.addItem(self._resp_ann_apnea)

        self._resp_ann_cycles = pg.TextItem("", anchor=(0, 1), color=config.gui_theme.subtext)
        self._resp_ann_cycles.setFont(QFont("Arial", 10))
        self.resp_plot.addItem(self._resp_ann_cycles)

        self._resp_ann_conf = pg.TextItem("", anchor=(1, 1), color=config.gui_theme.subtext)
        self._resp_ann_conf.setFont(QFont("Arial", 10))
        self.resp_plot.addItem(self._resp_ann_conf)

        # Apnea red-zone overlay regions
        self._apnea_regions = []

        resp_layout.addWidget(self.resp_plot, stretch=2)

        self.rr_plot = self._create_plot("📈 Respiration Rate (RR)", "Time (s)", "BPM")
        self.rr_plot.setXRange(-self._resp_window, 0, padding=0)
        self.rr_plot.setYRange(0, 40)
        self.rr_plot.enableAutoRange(axis='x', enable=False)
        self.curve_rr = self.rr_plot.plot(pen=pg.mkPen(color=config.gui_theme.text, width=2))
        self.curve_rr_zero = self.rr_plot.plot(pen=pg.mkPen(None))
        txt_c = QColor(config.gui_theme.text)
        self.fill_rr = pg.FillBetweenItem(self.curve_rr, self.curve_rr_zero, brush=pg.mkBrush(txt_c.red(), txt_c.green(), txt_c.blue(), 25))
        self.rr_plot.addItem(self.fill_rr)

        # Annotations on RR plot
        self._rr_ann_current = pg.TextItem("", anchor=(1, 0), color=config.gui_theme.text)
        self._rr_ann_current.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.rr_plot.addItem(self._rr_ann_current)

        self._rr_ann_brv = pg.TextItem("", anchor=(0, 0), color=config.gui_theme.subtext)
        self._rr_ann_brv.setFont(QFont("Arial", 10))
        self.rr_plot.addItem(self._rr_ann_brv)

        self._rr_ann_cycle = pg.TextItem("", anchor=(0.5, 0), color=config.gui_theme.subtext)
        self._rr_ann_cycle.setFont(QFont("Arial", 10))
        self.rr_plot.addItem(self._rr_ann_cycle)

        resp_layout.addWidget(self.rr_plot, stretch=2)
        
        self.vitals_tabs.addTab(self.tab_resp, "🫁 Respiratory Information")

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

        self.vitals_tabs.addTab(self.tab_hr, "❤️ Cardiac Information")
        left_col.addWidget(self.vitals_tabs, stretch=4)

        # Right Column (Radar Map + Analytics directly below)
        right_layout = QVBoxLayout()
        
        # Radar Layout (Top Right)
        self.radar_plot = self._create_plot("🎯 Region of Interest", "X (m)", "Y (m)")
        self.radar_plot.setAspectLocked(True)
        self._draw_static_environment()
        
        self.scatter_halo = pg.ScatterPlotItem(pxMode=False, pen=pg.mkPen(None))
        self.radar_plot.addItem(self.scatter_halo)
        self.scatter_occupant = pg.ScatterPlotItem(pxMode=False, pen=pg.mkPen(None))
        self.radar_plot.addItem(self.scatter_occupant)
        
        self.radar_posture_item = RadarPostureItem()
        self.radar_posture_item.hide()
        self.radar_plot.addItem(self.radar_posture_item)
        
        right_layout.addWidget(self.radar_plot, stretch=3)

        # Analytics Tabs (Bottom Right)
        self.analytics_tabs = QTabWidget()
        self.analytics_tabs.setStyleSheet(f"""
            QTabWidget::pane {{ border: 1px solid {config.gui_theme.grid}; border-radius: 5px; }}
            QTabBar::tab {{ background: {config.gui_theme.panel_bg}; color: {config.gui_theme.subtext}; padding: 8px 15px; border-top-left-radius: 4px; border-top-right-radius: 4px; margin-right: 2px; }}
            QTabBar::tab:selected {{ background: {config.gui_theme.card_bg}; color: {config.gui_theme.text}; font-weight: bold; border-bottom: 2px solid {config.gui_theme.occupant}; }}
        """)

        # Tab 1: Confidence
        self.tab_conf = QWidget()
        conf_layout = QVBoxLayout(self.tab_conf)
        conf_layout.setContentsMargins(5, 5, 5, 5)

        self.trend_plot = self._create_plot("📊 Confidence: Occupancy / Posture / Fall", "Time (s)", "Normalized")
        self.trend_plot.setYRange(0, 1.05)
        self.curve_occ = self.trend_plot.plot(pen=pg.mkPen(color='#38BDF8', width=2), name="Occupancy")
        self.curve_occ_zero = self.trend_plot.plot(pen=pg.mkPen(None))
        self.fill_occ = pg.FillBetweenItem(self.curve_occ, self.curve_occ_zero, brush=pg.mkBrush(56, 189, 248, 30))
        self.trend_plot.addItem(self.fill_occ)
        self.curve_post = self.trend_plot.plot(pen=pg.mkPen(color='#F59E0B', width=2), name="Posture")
        self.curve_mot = self.trend_plot.plot(pen=pg.mkPen(color='#22C55E', width=2, style=Qt.PenStyle.DashLine), name="Motion")
        self.curve_fall = self.trend_plot.plot(pen=pg.mkPen(color='#EF4444', width=2), name="Fall")

        # Legend
        conf_legend = self.trend_plot.addLegend(
            offset=(10, 10),
            labelTextColor=config.gui_theme.text,
            brush=pg.mkBrush(config.gui_theme.card_bg + "CC"),
            pen=pg.mkPen(config.gui_theme.grid),
        )
        conf_legend.addItem(self.curve_occ,  "Occupancy")
        conf_legend.addItem(self.curve_post, "Posture")
        conf_legend.addItem(self.curve_mot,  "Motion")
        conf_legend.addItem(self.curve_fall, "Fall")

        conf_layout.addWidget(self.trend_plot)
        self.analytics_tabs.addTab(self.tab_conf, "📊 Confidence")


        # Tab 2: Target Power
        self.tab_power = QWidget()
        power_layout = QVBoxLayout(self.tab_power)
        power_layout.setContentsMargins(0, 0, 0, 0)
        
        self.mag_plot = self._create_plot("⚡ Dynamic Target Search Power", "Time (s)", "Magnitude")
        self.mag_plot.enableAutoRange(axis='y', enable=True)
        self.curve_mag = self.mag_plot.plot(pen=pg.mkPen(color='#38BDF8', width=2), name="Target Mag")
        self.curve_mag_fill = self.mag_plot.plot(pen=pg.mkPen(None))
        self.fill_mag = pg.FillBetweenItem(self.curve_mag, self.curve_mag_fill, brush=pg.mkBrush(56, 189, 248, 25))
        self.mag_plot.addItem(self.fill_mag)
        
        self.curve_thresh = self.mag_plot.plot(pen=pg.mkPen(color='#EF4444', width=2, style=Qt.PenStyle.DashLine), name="Threshold")
        
        self._mag_ann_bin = pg.TextItem("", anchor=(0, 1), color=config.gui_theme.text)
        self._mag_ann_bin.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self._mag_ann_bin.setPos(-config.respiration.resp_window_sec, 200) # dynamic later but initial
        self.mag_plot.addItem(self._mag_ann_bin)
        
        power_layout.addWidget(self.mag_plot)
        self.analytics_tabs.addTab(self.tab_power, "⚡ Target Power")

        # Tab 3: Height
        self.tab_height  = QWidget()
        height_tab_layout = QVBoxLayout(self.tab_height)
        height_tab_layout.setContentsMargins(5, 5, 5, 5)

        self.height_plot = self._create_plot("📏 Estimated Height (Z)", "Time (s)", "Height (m)")
        self.height_plot.setXRange(-config.respiration.resp_window_sec, 0, padding=0)
        self.height_plot.setYRange(0, 2.2)
        self.height_plot.enableAutoRange(axis='x', enable=False)
        self.height_plot.enableAutoRange(axis='y', enable=False)
        self.curve_height = self.height_plot.plot(
            pen=pg.mkPen(color=config.gui_theme.occupant, width=2), name="Z")

        # Threshold lines
        sit_thresh   = config.posture.sitting_threshold
        stand_thresh = config.posture.standing_threshold
        self._height_line_sit = pg.InfiniteLine(
            pos=sit_thresh, angle=0,
            pen=pg.mkPen(color="#F59E0B", width=1, style=Qt.PenStyle.DashLine),
            label=f"Sitting (≥{sit_thresh:.1f} m)",
            labelOpts={"color": "#F59E0B", "position": 0.96})
        self.height_plot.addItem(self._height_line_sit)

        self._height_line_stand = pg.InfiniteLine(
            pos=stand_thresh, angle=0,
            pen=pg.mkPen(color="#22C55E", width=1, style=Qt.PenStyle.DashLine),
            label=f"Standing (≥{stand_thresh:.1f} m)",
            labelOpts={"color": "#22C55E", "position": 0.96})
        self.height_plot.addItem(self._height_line_stand)

        height_tab_layout.addWidget(self.height_plot)
        self.analytics_tabs.addTab(self.tab_height, "📏 Height")

        # Tab 4: Posture Timeline
        # Encodes posture strings as integers: Fallen=-1, Unknown=0, Lying=1, Sitting=2, Standing=3, Walking=4
        _POSTURE_CODES = {"Fallen": -1, "Unknown": 0, "Lying Down": 1, "Sitting": 2, "Standing": 3, "Walking": 4}
        _POSTURE_COLORS = {
            "Fallen":     "#EF4444",
            "Unknown":    "#6B7280",
            "Lying Down": "#6366F1",
            "Sitting":    "#14B8A6",
            "Standing":   "#22C55E",
            "Walking":    "#14B8A6",  # Teal, matching the map animation
        }
        self._posture_codes = _POSTURE_CODES   # keep ref for update_dashboard

        self.tab_posture_tl  = QWidget()
        posture_tl_layout = QVBoxLayout(self.tab_posture_tl)
        posture_tl_layout.setContentsMargins(5, 5, 5, 5)

        self.posture_tl_plot = self._create_plot("🧐 Posture Timeline", "Time (s)", "")
        self.posture_tl_plot.setXRange(-config.respiration.resp_window_sec, 0, padding=0)
        self.posture_tl_plot.setYRange(-1.7, 4.7)
        self.posture_tl_plot.enableAutoRange(axis='x', enable=False)
        self.posture_tl_plot.enableAutoRange(axis='y', enable=False)

        # Custom integer → label Y-axis ticks
        ax_left = self.posture_tl_plot.getAxis('left')
        ax_left.setTicks([[
            (-1, "Fallen"), (0, "Unknown"), (1, "Lying"), (2, "Sitting"), (3, "Standing"), (4, "Walking")
        ]])

        # Colored reference lines for each posture level
        for name, code in _POSTURE_CODES.items():
            color = _POSTURE_COLORS[name]
            ref_line = pg.InfiniteLine(
                pos=code, angle=0,
                pen=pg.mkPen(color=color, width=1, style=Qt.PenStyle.DotLine))
            ref_line.setZValue(-10)
            self.posture_tl_plot.addItem(ref_line)

        # Step-function line showing posture over time
        self.curve_posture_tl = self.posture_tl_plot.plot(
            pen=pg.mkPen(color=config.gui_theme.occupant, width=2.5),
            name="Posture")
        # Current posture label annotation (top-right)
        self._posture_tl_ann = pg.TextItem("", anchor=(1, 0), color=config.gui_theme.text)
        self._posture_tl_ann.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.posture_tl_plot.addItem(self._posture_tl_ann)

        posture_tl_layout.addWidget(self.posture_tl_plot)
        self.analytics_tabs.addTab(self.tab_posture_tl, "🧐 Posture")

        # Add analytics DIRECTLY below radar — 40% of right column height (stretch 3:2)
        right_layout.addWidget(self.analytics_tabs, stretch=2)

        # Assemble two-column layout and add to main
        columns_layout.addLayout(left_col,    stretch=2)
        columns_layout.addLayout(right_layout, stretch=1)
        main_layout.addLayout(columns_layout, stretch=5)

        # 2. Bottom Cards — in the left column, directly below vitals tabs
        left_cards_widget = QWidget()
        left_cards_layout = QHBoxLayout(left_cards_widget)
        left_cards_layout.setContentsMargins(0, 0, 0, 0)
        left_cards_layout.setSpacing(10)

        self.occ_card  = CardWidget("📍 Occupancy", ["Zone", "State", "Confidence", "Duration", "Target Power"])
        self.post_card = PostureCard()
        left_cards_layout.addWidget(self.occ_card,  stretch=1)
        left_cards_layout.addWidget(self.post_card, stretch=2)

        left_col.addWidget(left_cards_widget, stretch=1)
        # 3. Breathing Info Bar
        self.resp_info_bar = QLabel("🫁 Waiting for breathing data...")
        self.resp_info_bar.setStyleSheet(f"""
            color: {config.gui_theme.text};
            background-color: {config.gui_theme.card_bg};
            padding: 8px 15px;
            border-radius: 6px;
            font-size: 20px;
            border: 1px solid rgba(255, 255, 255, 0.08);
        """)
        self.resp_info_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.resp_info_bar)

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
            
            if "Room" not in name:
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
        """Trend plot value: Walking=0.9, Moving=0.8, Resting=0.1, Unknown=0.2."""
        cls, _ = _classify_motion(motion_str)
        if cls == "Walking":  return 0.9
        if cls == "Resting":  return 0.1
        if cls == "Moving":   return 0.8
        return 0.2

    @staticmethod
    def _classify_motion_display(mot_str: str, posture: str,
                                 status: str, occ_conf: float) -> str:
        """
        Context-aware three-class motion label for display only.
        Priority: Walk > Move > Static > '--'

        Walk (highest priority):
          motion_str == 'Walking'

        Move:
          motion_str in active labels
          OR status contains 'Moving'
          OR posture == 'Standing'
          OR posture == 'Sitting' AND occupancy is valid (>0)

        Static (only when clearly resting in bed-like state):
          occupancy is valid
          AND posture == 'Lying Down'
          AND (motion_str is resting OR status is monitoring/apnea)
        """
        _MOVE_LABELS    = {"Major Movement", "Restless/Shifting",
                           "Postural Shift", "Restless/Fidgeting"}
        _STATIC_STATUS  = {"Still / Monitoring...", "Possible Apnea"}
        occ_valid = occ_conf > 0

        # 1. Walking
        if mot_str == "Walking":
            return "Walk"

        # 2. Move
        if (mot_str in _MOVE_LABELS
                or "Moving" in (status or "")
                or posture == "Standing"
                or (posture == "Sitting" and occ_valid)):
            return "Move"

        # 3. Static — only when truly resting in monitored bed-like state
        resting_mot = (mot_str or "").lower()
        is_rest_str = any(kw in resting_mot
                          for kw in ("breath", "rest", "still", "static"))
        in_still_status = (status or "") in _STATIC_STATUS
        if (occ_valid
                and posture == "Lying Down"
                and (is_rest_str or in_still_status)):
            return "Static"

        return "--"

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
        # print("zone: ", zone)
        
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
        mot_str     = occ_dict.get("motion_str", "--")
        posture_str = occ_dict.get("posture", "--")
        occ_conf    = occ_dict.get('occ_confidence', 0)
        # Context-aware three-class display label (pipeline unchanged)
        mot_display = self._classify_motion_display(
            mot_str, posture_str, state, occ_conf)

        self.occ_card.update_values(
            Zone=zone,
            State=state[:20] + "..." if len(state)>20 else state,
            Confidence=f"{int(occ_conf)}%",
            Duration=occ_dict.get("duration_str", "--"),
            **{"Target Power": f"{occ_dict.get('dynamic_mag', 0):.1f} / {occ_dict.get('detection_threshold', 150.0):.1f}"}
        )

        z = occ_dict.get("Z")
        r = occ_dict.get("Range", 0.0)
        height_range_str = f"{z:.2f} m ({r:.2f} m)" if z else "--"
        posture_conf = occ_dict.get('posture_confidence', 0)
        self.post_card.update_values(
            Posture=posture_str,
            **{"Posture conf.": f"{int(posture_conf)}%"},
            **{"Height (Range)": height_range_str},
            Motion=mot_display
        )
        # Pass mot_display as the motion class so the widget animation matches the label
        self.post_card.set_posture_state(posture_str, posture_conf, mot_str)

        fc = occ_dict.get('fall_confidence', 0)

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

        # Update Height Trend
        z_val = occ_dict.get("Z") or 0.0
        self.height_hist.pop(0)
        self.height_hist.append(float(z_val))
        self.curve_height.setData(self.x_axis, self.height_hist)

        # Update Posture Timeline
        # For the timeline specifically, inject 'Walking' as the top-level state
        # instead of the physical base posture ('Standing') to show locomotion segments.
        tl_posture_str = "Walking" if mot_str == "Walking" else posture_str
        posture_code = self._posture_codes.get(tl_posture_str, 0)
        self.posture_num_hist.pop(0)
        self.posture_num_hist.append(posture_code)
        self.curve_posture_tl.setData(self.x_axis, self.posture_num_hist)
        # Update annotation
        tl_colors = {"Fallen": "#EF4444", "Unknown": "#6B7280",
                     "Lying Down": "#6366F1", "Sitting": "#14B8A6", 
                     "Standing": "#22C55E", "Walking": "#14B8A6"}
        self._posture_tl_ann.setText(tl_posture_str or "Unknown")
        self._posture_tl_ann.setColor(QColor(tl_colors.get(tl_posture_str, config.gui_theme.subtext)))
        self._posture_tl_ann.setPos(0, 4.7)

        # Update Target Power Trends
        self.mag_hist.pop(0)
        self.mag_hist.append(occ_dict.get('dynamic_mag', 0.0))
        self.curve_mag.setData(self.x_axis, self.mag_hist)
        self.curve_mag_fill.setData(self.x_axis, np.zeros(len(self.mag_hist)))
        
        thresh_val = occ_dict.get('detection_threshold', 150.0)
        self.thresh_hist.pop(0)
        self.thresh_hist.append(thresh_val)
        self.curve_thresh.setData(self.x_axis, self.thresh_hist)

        sel_bin = occ_dict.get('final_bin', 0)
        self._mag_ann_bin.setText(f"Active Bin: {sel_bin}")

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
            
            # Posture Graphic: show in monitored zones OR when walking in transit
            zone = occ_dict.get("zone", "")
            base_zone = zone.split(" - ")[0]
            is_monitored = (base_zone in config.layout and
                            config.layout[base_zone].get("type") == "monitor")
            is_walking   = (mot_str == "Walking")

            if is_monitored or is_walking:
                self.scatter_occupant.setData([], [])  # Hide plain dot
                self.radar_posture_item.setPos(x, y)
                self.radar_posture_item.set_state(
                    occ_dict.get("posture", "Unknown"),
                    occ_dict.get('posture_confidence', 0),
                    mot_str)
                self.radar_posture_item.show()
            else:
                # Non-monitored, non-walking: standard confidence dot
                self.radar_posture_item.hide()
                self.scatter_occupant.setData(x=[x], y=[y], size=dot_diam, brush=pg.mkBrush(base_color))

                
        else:
            self.scatter_occupant.setData([], [])
            self.scatter_halo.setData([], [])
            self.radar_posture_item.hide()

        # Update Respiratory
        if resp_dict and (resp_dict.get('confidence', 0) > 0 or resp_dict.get('is_calibrating', False)):
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
                self.scatter_inhale.setData(x_sig[inhs], sig[inhs])
            else:
                self.scatter_inhale.setData([], [])

            if len(exhs) > 0 and len(sig) > 0:
                self.scatter_exhale.setData(x_sig[exhs], sig[exhs])
            else:
                self.scatter_exhale.setData([], [])

            # --- Apnea Red-Zone Highlighting ---
            for region in self._apnea_regions:
                self.resp_plot.removeItem(region)
            self._apnea_regions.clear()

            for (start, end) in resp_dict.get('apnea_segments', []):
                if 0 <= start < len(x_sig) and 0 < end <= len(x_sig):
                    t_start = x_sig[start]
                    t_end = x_sig[min(end, len(x_sig) - 1)]
                    region = pg.LinearRegionItem(
                        values=[t_start, t_end],
                        brush=pg.mkBrush(239, 68, 68, 40),  # #EF4444 at 40 alpha
                        movable=False
                    )
                    region.setZValue(-10)
                    self.resp_plot.addItem(region)
                    self._apnea_regions.append(region)

            # --- Annotations on Respiration Plot (fixed coordinates) ---
            rx_min = -self._resp_window
            rx_max = 0
            ry_min = -20
            ry_max = 20

            # Top-left: Depth
            depth = resp_dict.get('depth', 'unknown')
            depth_colors = {'normal': '#22C55E', 'deep': '#38BDF8', 'shallow': '#F59E0B', 'apnea': '#EF4444'}
            depth_color = depth_colors.get(depth, config.gui_theme.subtext)
            self._resp_ann_depth.setText(f"Depth: {depth.capitalize()}")
            self._resp_ann_depth.setColor(QColor(depth_color))
            self._resp_ann_depth.setPos(rx_min, ry_max)

            # Top-right: Apnea count
            apnea_count = resp_dict.get('apnea_count', 0)
            apnea_text = f"Apnea: {apnea_count}"
            if resp_dict.get('apnea_active', False):
                dur = resp_dict.get('apnea_duration', 0)
                apnea_text += f" (active {dur:.1f}s)"
            self._resp_ann_apnea.setText(apnea_text)
            self._resp_ann_apnea.setPos(rx_max, ry_max)

            # Bottom-left: Cycle count
            cycle_count = resp_dict.get('cycle_count', 0)
            self._resp_ann_cycles.setText(f"Cycles: {cycle_count}")
            self._resp_ann_cycles.setPos(rx_min, ry_min)

            # Top-right: Confidence / Calibration Override
            is_calib = resp_dict.get('is_calibrating', False)
            if is_calib:
                self._resp_ann_conf.setText("Calibrating Threshold...")
                self._resp_ann_conf.setColor(QColor('#F59E0B'))
                self._resp_ann_apnea.setText("")
            else:
                conf = resp_dict.get('confidence', 0)
                conf_color = '#22C55E' if conf > 60 else ('#F59E0B' if conf > 30 else '#EF4444')
                self._resp_ann_conf.setText(f"Confidence: {conf:.0f}%")
                self._resp_ann_conf.setColor(QColor(conf_color))
                
            self._resp_ann_conf.setPos(rx_max, ry_min)

            # --- Annotations on RR Plot (fixed coordinates) ---
            rrx_min = -self._resp_window
            rrx_max = 0
            rry_max = 40

            # Top-right: Current RR
            rr_val = resp_dict.get('rr_current', 0)
            if rr_val > 0 and not is_calib:
                rr_color = '#22C55E' if 6 <= rr_val <= 30 else '#F59E0B'
                self._rr_ann_current.setText(f"RR: {rr_val:.1f} bpm")
                self._rr_ann_current.setColor(QColor(rr_color))
            else:
                self._rr_ann_current.setText("RR: --")
                self._rr_ann_current.setColor(QColor(config.gui_theme.subtext))
            self._rr_ann_current.setPos(rrx_max, rry_max)

            # Top-left: BRV
            brv = resp_dict.get('brv_value', 0)
            brv_text = f"BRV: {brv:.3f}s" if (brv > 0 and not is_calib) else "BRV: --"
            self._rr_ann_brv.setText(brv_text)
            self._rr_ann_brv.setPos(rrx_min, rry_max)

            # Top-center: Last cycle
            last_cyc = resp_dict.get('last_cycle_duration', 0)
            cyc_text = f"Last cycle: {last_cyc:.2f}s" if (last_cyc > 0 and not is_calib) else "Last cycle: --"
            self._rr_ann_cycle.setText(cyc_text)
            self._rr_ann_cycle.setPos((rrx_min + rrx_max) / 2, rry_max)

            # Update breathing info bar
            sig = resp_dict.get('live_signal', np.array([]))
            locked_bin = resp_dict.get('locked_bin', 0)
            bin_dist = (locked_bin or 0) * config.radar.range_resolution
            if is_calib:
                self.resp_info_bar.setText(f"🎯 Bin: {locked_bin} ({bin_dist:.2f} m)   |   ⏳ Calibrating Apnea Threshold for 40 seconds...")
            else:
                parts = [
                    f"🎯 Bin: {locked_bin} ({bin_dist:.2f} m)",
                    f"📊 Confidence: {resp_dict.get('confidence', 0):.0f}%",
                    f"💨 RR: {rr_val:.1f} bpm" if rr_val > 0 else "💨 RR: --",
                    f"🔄 Cycles: {resp_dict.get('cycle_count', 0)}",
                    f"🚫 Apnea: {resp_dict.get('apnea_count', 0)}",
                    f"📏 Depth: {resp_dict.get('depth', 'unknown')}",
                    f"📐 Amplitude: {np.ptp(sig):.2f} mm" if len(sig) > 0 else "📐 Amplitude: --",
                ]
                self.resp_info_bar.setText("   |   ".join(parts))

        else:
            self.curve_resp.setData([], [])
            self.curve_resp_zero.setData([], [])
            self.curve_rr.setData([], [])
            self.curve_rr_zero.setData([], [])
            self.scatter_inhale.setData([], [])
            self.scatter_exhale.setData([], [])

            # Clear apnea regions
            for region in self._apnea_regions:
                self.resp_plot.removeItem(region)
            self._apnea_regions.clear()

            # Clear annotations
            self._resp_ann_depth.setText("")
            self._resp_ann_apnea.setText("")
            self._resp_ann_cycles.setText("")
            self._resp_ann_conf.setText("")
            self._rr_ann_current.setText("")
            self._rr_ann_brv.setText("")
            self._rr_ann_cycle.setText("")
            self.resp_info_bar.setText("🫁 Waiting for breathing data...")
