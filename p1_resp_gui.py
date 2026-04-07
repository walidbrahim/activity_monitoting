"""
test_respiration_real.py
─────────────────────────
High-precision, multi-sensor respiratory monitoring test stack.

Motion uses radar_engine output directly (motion_score + motion_str).
Confidence: motion_factor × posture_factor displayed as a live badge.
"""
import sys, os, multiprocessing, queue
import numpy as np
from scipy import signal

from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTabWidget
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont, QColor, QPixmap, QImage
import pyqtgraph as pg

sys.path.append(os.getcwd())

from libs.controllers.radarController import RadarController
# from libs.controllers.vernier_belt_controller import VernierBeltControllerThread
# from libs.controllers.witmotion_controller import WitMotionControllerThread
from apps.bed_monitor.controller import BedMonitorController
from libs.gui.posture_widget import RadarPostureItem
from config import load_profile, ConfigFactory


# ══════════════════════════════════════════════════════════════════════════════
#  Confidence
# ══════════════════════════════════════════════════════════════════════════════
POSTURE_FACTORS = {
    "Lying Down": 1.0,
    "Fallen":     1.0,   # treat == lie
    "Unknown":    0.8,
    "Sitting":    0.5,
    "Standing":   0.2,
}

def compute_confidence(norm_score: float, posture_str: str) -> float:
    """Returns 0–100 % signal quality.

    Motion factor uses engine motion_score [0, 1] with a dead-band:
      - score ≤ LOWER_BAND (0.20)  → factor = 1.0
      - score in LOWER_BAND..UPPER_BAND (0.85) → linear decay 1.0 → 0.0
      - score ≥ UPPER_BAND         → factor = 0.0
    """
    LOWER_BAND = 0.20
    UPPER_BAND = 0.85
    if norm_score <= LOWER_BAND:
        motion_factor = 1.0
    elif norm_score >= UPPER_BAND:
        motion_factor = 0.0
    else:
        motion_factor = 1.0 - (norm_score - LOWER_BAND) / (UPPER_BAND - LOWER_BAND)
    posture_factor = POSTURE_FACTORS.get(posture_str, 0.8)
    return motion_factor * posture_factor * 100.0


def confidence_color(conf: float) -> str:
    if conf >= 70: return '#10B981'   # green
    if conf >= 40: return '#F59E0B'   # amber
    return '#EF4444'                  # red


def confidence_symbol(conf: float) -> str:
    if conf >= 70: return '✓'
    if conf >= 40: return '~'
    return '⚠'


# ══════════════════════════════════════════════════════════════════════════════
#  Main Application
# ══════════════════════════════════════════════════════════════════════════════
class RespirationMonitorApp(QMainWindow):

    def __init__(self):
        super().__init__()
        print("Initializing Optimized Respiration Monitor ...")

        self.state_q  = multiprocessing.Queue()
        self.pt_fft_q = multiprocessing.Queue()

        # ── Configuration (Layered Architecture) ──────────────────────────────
        base_profile = "profiles/base.yaml"
        selected = os.getenv("APP_PROFILE", "home").strip()
        if selected in ("", "base", "base.yaml", base_profile):
            profile_stack = [base_profile]
        else:
            overlay = selected if selected.endswith(".yaml") else f"{selected}.yaml"
            if "/" not in overlay:
                overlay = f"profiles/{overlay}"
            profile_stack = [base_profile, overlay]

        self.app_cfg = load_profile(*profile_stack)
        self.eng_cfg = ConfigFactory.engine_config(self.app_cfg)
        print(f"Successfully loaded configuration stack: {profile_stack}")

        self.fps               = float(self.eng_cfg.hardware.frame_rate)
        self.window_sec        = float(self.eng_cfg.respiration.window_sec)
        self.radar_frame_count = int(self.window_sec * self.fps)

        # ── Radar ─────────────────────────────────────────────────────────────
        self.radar_process = RadarController(state_q=self.state_q, pt_fft_q=self.pt_fft_q)
        self.radar_process.start()

        # ── Vernier Belt ──────────────────────────────────────────────────────
        self.vernier_belt_realtime_q = multiprocessing.Queue()
        self.belt_thread = None
        if self.app_cfg.vernier.enabled:
            from libs.controllers.vernier_belt_controller import VernierBeltControllerThread
            self.belt_thread = VernierBeltControllerThread(
                vernier_belt_realtime_q=self.vernier_belt_realtime_q,
                vernier_belt_connection_q=multiprocessing.Queue(),
                start_vernier_belt_q=multiprocessing.Queue(),
                sensors=self.app_cfg.vernier.sensors,
                period=int(1000/self.app_cfg.vernier.rate_hz)
            )
            self.belt_thread.start()
            self.belt_thread.start_vernier_belt_q.put(True)

        # ── WitMotion IMUs ────────────────────────────────────────────────────
        self.imu_threads, self.imu_queues = [], []
        for imu_cfg in [self.app_cfg.witmotion1, self.app_cfg.witmotion2]:
            if imu_cfg.enabled:
                from libs.controllers.witmotion_controller import WitMotionControllerThread
                q, cmd_q = multiprocessing.Queue(maxsize=1000), multiprocessing.Queue()
                t = WitMotionControllerThread(
                    witmotion_mac=imu_cfg.mac,
                    witmotion_realtime_q=q,
                    start_witmotion_q=cmd_q,
                    location=imu_cfg.location
                )
                t.start(); cmd_q.put(True)
                self.imu_threads.append(t)
                self.imu_queues.append(q)

        # ── Background Process Thread (New Architecture) ──────────────────────
        self.processor = BedMonitorController(
            pt_fft_q=self.pt_fft_q,
            vernier_belt_realtime_q=self.vernier_belt_realtime_q,
            imu_queues=self.imu_queues,
            cfg=self.eng_cfg,
            app_cfg=self.app_cfg,
            belt_window_sec=self.window_sec,
            belt_rate_hz=getattr(self.app_cfg.vernier, 'rate_hz', 10.0),
            recording_cfg=self.app_cfg.recording.model_dump(),
            db_cfg=self.app_cfg.database.model_dump(),
        )
        self.processor.data_ready.connect(self.on_data_ready)

        # Apply configured default radar pose at startup so localization uses
        # the intended coordinate transform from the first frame.
        default_zone = getattr(self.app_cfg.app, "default_radar_pose", "Room")
        default_pose = self.app_cfg.layout.get(default_zone, {}).get("radar_pose")
        if isinstance(default_pose, dict):
            self.processor.update_radar_pose(default_pose)

        # ── Engine motion-score decision boundary ─────────────────────────────
        rest_max = float(self.eng_cfg.activity.motion.rest_max)
        restless_max = max(float(self.eng_cfg.activity.motion.restless_max), 1e-9)
        self.motion_moving_threshold = min(1.0, max(0.0, rest_max / restless_max))

        # ── Display Buffers ───────────────────────────────────────────────────
        N = self.radar_frame_count
        self.radar_display_buffer = np.zeros(N)
        self.rr_history           = np.zeros(N)
        self.conf_history         = np.zeros(N)   # 0–100
        self.height_history       = np.zeros(N)   # Z height
        self.motion_norm_hist     = np.zeros(N)   # engine motion_score [0, 1]
        self.bin_history          = np.zeros(N)
        self.posture_history      = np.zeros(N)

        belt_n = int(self.window_sec * self.app_cfg.vernier.rate_hz)
        self.belt_display_buffer = np.zeros(belt_n)

        self.imu_buffers     = []
        self.imu_offset_emas = []
        self.imu_scale_emas  = []
        for imu_cfg in [self.app_cfg.witmotion1, self.app_cfg.witmotion2]:
            if imu_cfg.enabled:
                self.imu_buffers.append(np.zeros(int(self.window_sec * imu_cfg.rate_hz)))
                self.imu_offset_emas.append(None)
                self.imu_scale_emas.append(None)

        # ── Scale lock (respiration signal) ───────────────────────────────────
        self.radar_scale_locked = False
        self.radar_scale_value  = 1.0
        self.radar_scale_frames = 0
        self.SCALE_LOCK_FRAMES  = int(self.fps * 15)   # lock after 15 s
        self.scale_ema_alpha    = 0.02

        # ── Session / status state ────────────────────────────────────────────
        self._in_bed          = False
        self._last_zone       = "No Occupant Detected"
        self._last_rr         = 0.0
        self._last_conf       = 0.0
        self._last_motion_str = "Unknown"
        self._last_is_moving  = None
        self._last_bin        = 0
        self._last_posture    = "Unknown"

        self.init_ui()

        # ── Camera ───────────────────────────────────────────────────────────
        self._cv2 = None
        self._camera_cap = None
        try:
            import cv2 as _cv2_mod
            self._cv2 = _cv2_mod
            cap = _cv2_mod.VideoCapture(0)
            if cap.isOpened():
                self._camera_cap = cap
            else:
                cap.release()
        except ImportError:
            pass

        # ── State for info panel ─────────────────────────────────────────────
        self._apnea_active  = False
        self._last_depth    = "--"
        self._last_cycles   = 0

        # ── 6. Final UI Setup
        self._update_status_bar()
        
        # Start Engine
        self.processor.start()
        
        # UI Refresh Timer (only for sensor queues not handled by RadarEngine)
        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self.poll_sensor_queues)
        self.ui_timer.start(40) # 25 fps UI feel

    # ═══════════════════════════════════════════════════════════════════════════
    def init_ui(self):
        self.setWindowTitle("P1: Realtime Spatial and Behavior-Aware On-Bed Respiration Monitoring")
        self.resize(1700, 960)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)   # 2-column root
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        def mkplot(title, ylabel="Value"):
            p = pg.PlotWidget()
            p.setBackground('#0F172A')
            p.showGrid(x=True, y=True, alpha=0.1)
            p.setTitle(f"<span style='font-size:13pt; color:#CBD5E1'>{title}</span>")
            label_font = QFont("Arial", 12)
            p.setLabel('left', ylabel)
            p.getAxis('left').setTickFont(label_font)
            p.getAxis('bottom').setTickFont(label_font)
            p.getAxis('left').setStyle(tickTextWidth=55)
            p.getAxis('left').label.setFont(QFont("Arial", 12))
            p.getAxis('bottom').label.setFont(QFont("Arial", 12))
            return p

        # ╔═════════════════════════════════════╗
        # ║  LEFT COLUMN — 3 main signal plots    ║
        # ╚═════════════════════════════════════╝
        left_widget = QWidget()
        left_widget.setStyleSheet("background: #0B1120;")
        layout = QVBoxLayout(left_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        main_layout.addWidget(left_widget, stretch=3)

        # ── Plot 1: Breathing Signal ──────────────────────────────────────────
        self.plot_comparison = mkplot(
            "Respiration Signal — Radar (+ References)", ylabel="Amplitude")
        layout.addWidget(self.plot_comparison, stretch=1)
        self.plot_comparison.setYRange(-10, 10)  # bandpass phase displacement (degrees)


        # Secondary ViewBox for Vernier Belt (Right Y-Axis)
        self.belt_view = pg.ViewBox()
        self.plot_comparison.showAxis('right')
        self.plot_comparison.scene().addItem(self.belt_view)
        self.plot_comparison.getAxis('right').linkToView(self.belt_view)
        self.belt_view.setXLink(self.plot_comparison)
        self.plot_comparison.getAxis('right').setLabel('Belt Force', color='#FACC15')

        def update_views():
            self.belt_view.setGeometry(self.plot_comparison.getViewBox().sceneBoundingRect())
        self.plot_comparison.getViewBox().sigResized.connect(update_views)

        self.legend = self.plot_comparison.addLegend(offset=(10, 10))
        self.legend.setLabelTextColor("#CBD5E1")

        self.curve_radar = self.plot_comparison.plot(
            pen=pg.mkPen(QColor(34, 211, 238, 200), width=2), name="Radar")
        self.curve_belt = None
        if self.app_cfg.vernier.enabled:
            # Curve added to secondary ViewBox
            self.curve_belt = pg.PlotCurveItem(
                pen=pg.mkPen('#FACC15', width=1.2), name="Belt")
            self.belt_view.addItem(self.curve_belt)
            self.legend.addItem(self.curve_belt, "Belt")

        imu_colors = ['#A855F7', '#F472B6']
        self.curve_imus = []
        for i, t in enumerate(self.imu_threads):
            crv = self.plot_comparison.plot(
                    pen=pg.mkPen(imu_colors[i], width=1.2), name=f"IMU ({t.location})")
            self.curve_imus.append(crv)

        self.scatter_inhales = pg.ScatterPlotItem(size=9, pen=None, brush='#3B82F6')
        self.scatter_exhales = pg.ScatterPlotItem(size=9, pen=None, brush='#EF4444')
        self.plot_comparison.addItem(self.scatter_inhales)
        self.plot_comparison.addItem(self.scatter_exhales)

        # Apnea overlay regions on signal plot (dynamically updated each frame)
        self._apnea_regions = []

        # Annotations — top-right (anchor bottom-right so rows don't clip top)
        def mktxt(color, font_size, bold=False):
            t = pg.TextItem("", color=color, anchor=(1, 0))
            t.setFont(QFont("Arial", font_size, QFont.Weight.Bold if bold else QFont.Weight.Normal))
            self.plot_comparison.addItem(t)
            return t

        self.ann_quality = mktxt('#10B981', 18, bold=True)    # confidence badge
        self.ann_apnea   = mktxt('#FB923C', 16)
        self.ann_cycles  = mktxt('#94A3B8', 15)
        self.ann_brv     = mktxt('#64748B', 14)

        # Place anchor at (1,0) = top-right of each text item; x=-0.3 = near right edge
        self.ann_quality.setPos(-0.3,  9.4)
        self.ann_apnea.setPos(  -0.3,  6.4)
        self.ann_cycles.setPos( -0.3,   3.4)
        self.ann_brv.setPos(    -0.3,   0.4)

        # ── Calibration Status Badge ─────────────────────────────────────
        self.ann_status = pg.TextItem("CALIBRATING", color='#FB923C', anchor=(0, 0))
        self.ann_status.setFont(QFont("Arial", 22, QFont.Weight.Bold))
        self.ann_status.setPos(-28, 25.2)   # Top-left of the plot
        self.plot_comparison.addItem(self.ann_status)

        # Bed / motion overlays
        self.warning_text = pg.TextItem("SUBJECT NOT IN BED", color='#EF4444', anchor=(0.5, 0.5))
        self.warning_text.setFont(QFont("Arial", 40, QFont.Weight.Bold))
        self.plot_comparison.addItem(self.warning_text)
        self.warning_text.setPos(-15, 0)
        self.warning_text.hide()

        # ── Plot 2: RR Trend with confidence band ────────────────────────────
        self.plot_rr = mkplot("Respiration Rate Trend", ylabel="BPM")
        layout.addWidget(self.plot_rr, stretch=1)
        self.plot_rr.setYRange(0, 40)

        # Confidence shading band: fill between rr_upper and rr_lower
        self.curve_rr_upper = self.plot_rr.plot(pen=None)
        self.curve_rr_lower = self.plot_rr.plot(pen=None)
        self.fill_rr = pg.FillBetweenItem(
            self.curve_rr_upper, self.curve_rr_lower,
            brush=pg.mkBrush(248, 67, 94, 40)    # semi-transparent pink
        )
        self.plot_rr.addItem(self.fill_rr)
        # Main RR curve (on top of band)
        self.curve_rr = self.plot_rr.plot(pen=pg.mkPen('#F43F5E', width=2))

        self.ann_rr = pg.TextItem("RR: -- BPM", color='#F43F5E', anchor=(1, 0))
        self.ann_rr.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.ann_rr.setPos(-29, 37)
        self.plot_rr.addItem(self.ann_rr)

        # ── Plot 3: Motion State (normalized score + threshold line) ─────────
        self.plot_motion = mkplot(
            "Motion Score (engine output 0–1)", ylabel="Motion Score")
        layout.addWidget(self.plot_motion, stretch=1)
        # Filled area above threshold becomes red, below is purple
        self.curve_motion_raw = self.plot_motion.plot(
            pen=pg.mkPen('#A855F7', width=2),
            fillLevel=0, brush=pg.mkBrush(168, 85, 247, 50)
        )
        self.line_motion_thresh = pg.InfiniteLine(
            pos=self.motion_moving_threshold, angle=0,
            pen=pg.mkPen('#F59E0B', width=1.5, style=Qt.PenStyle.DashLine),
            label="Moving Threshold"
        )
        self.plot_motion.addItem(self.line_motion_thresh)
        self.motion_state_text = pg.TextItem("Waiting for target...", color='#94A3B8', anchor=(0, 0))
        self.motion_state_text.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.motion_state_text.setPos(-29, 2)
        self.plot_motion.addItem(self.motion_state_text)
        
        # ── Plot 4: Posture History ──────────────────────────────────────────
        self.plot_posture_hist = mkplot("Posture History Trend", ylabel="Posture")
        layout.addWidget(self.plot_posture_hist, stretch=1)
        self.curve_posture_hist = self.plot_posture_hist.plot(pen=pg.mkPen('#38BDF8', width=2))
        self.plot_posture_hist.setYRange(0, 4)
        self.plot_posture_hist.getAxis('left').setTicks(
            [[(1, 'Stand'), (2, 'Sit'), (3, 'Lie')]])

        # ╔═════════════════════════════════════╗
        # ║  RIGHT COLUMN                        ║
        # ║  Camera feed (top)                   ║
        # ║  Animated posture icon               ║
        # ║  Live metrics panel                  ║
        # ║  Tabbed secondary plots (bottom)     ║
        # ╚═════════════════════════════════════╝
        right_widget = QWidget()
        right_widget.setStyleSheet("background: #0B1120;")
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)
        main_layout.addWidget(right_widget, stretch=1)

        # ── Camera view ───────────────────────────────────────────────────────
        self.camera_label = QLabel("⚠  No Camera")
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setStyleSheet(
            "background:#0A0F1E; color:#475569; font-size:14px;"
            "border:1px solid #1E293B; border-radius:4px;")
        self.camera_label.setMinimumHeight(230)
        right_layout.addWidget(self.camera_label, stretch=4)

        # ── Animated posture icon (no axes) ───────────────────────────────────
        self.plot_posture_icon = pg.PlotWidget()
        self.plot_posture_icon.setBackground('#0F172A')
        self.plot_posture_icon.hideAxis('left')
        self.plot_posture_icon.hideAxis('bottom')
        self.plot_posture_icon.setYRange(0, 6)
        self.plot_posture_icon.setXRange(0, 4)
        self.plot_posture_icon.setFixedHeight(160)
        self.plot_posture_icon.setTitle(
            "<span style='font-size:11pt; color:#94A3B8'>Current Posture</span>")
        self.posture_icon = RadarPostureItem()
        self.plot_posture_icon.addItem(self.posture_icon)
        self.posture_icon.setPos(2, 3)
        right_layout.addWidget(self.plot_posture_icon)

        # ── Live metrics info panel ────────────────────────────────────────────
        self.info_label = QLabel()
        self.info_label.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.info_label.setStyleSheet(
            "background:#0F172A; color:#94A3B8; padding:10px;"
            "border:1px solid #1E293B; border-radius:4px;")
        self.info_label.setWordWrap(True)
        self.info_label.setTextFormat(Qt.TextFormat.RichText)
        right_layout.addWidget(self.info_label, stretch=2)

        # ── Tabbed secondary plots ─────────────────────────────────────────────
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet(
            "QTabBar::tab { font-size: 12px; padding: 5px 12px; color: #CBD5E1; }")
        right_layout.addWidget(self.tab_widget, stretch=4)

        self.plot_bin = mkplot("Locked Range Bin", ylabel="Bin Index")
        self.curve_bin = self.plot_bin.plot(pen=pg.mkPen('#10B981', width=1.5))
        self.plot_bin.setYRange(0, self.eng_cfg.hardware.num_range_bins)
        self.tab_widget.addTab(self.plot_bin, "Range Tracking")

        self.plot_height = mkplot("Height History (Z)", ylabel="Height (m)")
        self.plot_height_legend = self.plot_height.addLegend(offset=(10, 10))
        self.plot_height_legend.setLabelTextColor("#CBD5E1")
        self.plot_height.setYRange(0, 2.5)
        
        self.curve_height   = self.plot_height.plot(
            pen=pg.mkPen('#10B981', width=2), name="Track Z")
        
        # Posture thresholds
        self.line_sit = pg.InfiniteLine(
            pos=self.eng_cfg.activity.posture.sitting_threshold_m, angle=0,
            pen=pg.mkPen('#F59E0B', width=1.5, style=Qt.PenStyle.DashLine),
            label=f"Sit ({self.eng_cfg.activity.posture.sitting_threshold_m}m)")
        self.line_stand = pg.InfiniteLine(
            pos=self.eng_cfg.activity.posture.standing_threshold_m, angle=0,
            pen=pg.mkPen('#34D399', width=1.5, style=Qt.PenStyle.DashLine),
            label=f"Stand ({self.eng_cfg.activity.posture.standing_threshold_m}m)")
        self.plot_height.addItem(self.line_sit)
        self.plot_height.addItem(self.line_stand)
        
        self.tab_widget.addTab(self.plot_height, "Height History")
        self.tab_widget.setCurrentIndex(1)

        self.plot_deriv = mkplot(
            "Phase Velocity Derivative (Apnea Detection)", ylabel="Norm. Velocity")
        self.plot_deriv.setYRange(0, 1.1)
        self.curve_deriv = self.plot_deriv.plot(
            pen=pg.mkPen('#34D399', width=1.5),
            fillLevel=0, brush=pg.mkBrush(52, 211, 153, 40))
        self.line_apnea_thresh = pg.InfiniteLine(
            pos=0.2, angle=0,
            pen=pg.mkPen('#EF4444', width=1.5, style=Qt.PenStyle.DashLine),
            label="Apnea Threshold")
        self.plot_deriv.addItem(self.line_apnea_thresh)
        self.tab_widget.addTab(self.plot_deriv, "Derivative / Apnea")
        self._apnea_regions_deriv = []

        # Status bar styling removed (redundant)

    # ══════════════════════════════════════════════════════════════════════
    def poll_sensor_queues(self):
        """Drains non-radar sensors (IMUs, Belt) that are not managed by the BedMonitorController."""
        self._update_camera()
        
        # ── IMUs ──────────────────────────────────────────────────────────────
        # Skip polling if the processor is currently using these queues for auto-alignment
        if not getattr(self.processor, 'is_aligning', False):
            for i, q in enumerate(self.imu_queues):
                samples = []
                while not q.empty():
                    try:
                        samples.append(q.get_nowait())
                    except queue.Empty: break
                if samples:
                        z = [s[2] for s in samples]
                        med = np.median(z); p2p = max(z) - min(z)
                        if self.imu_offset_emas[i] is None:
                            self.imu_offset_emas[i] = med
                            self.imu_scale_emas[i]  = max(0.1, p2p/2)
                        else:
                            α = 0.05
                            self.imu_offset_emas[i] = (1-α)*self.imu_offset_emas[i] + α*med
                            if p2p > 0.05:
                                self.imu_scale_emas[i] = (1-α)*self.imu_scale_emas[i] + α*(p2p/2)
                        nz  = (np.array(z) - self.imu_offset_emas[i]) / max(0.01, self.imu_scale_emas[i])
                        buf = self.imu_buffers[i]
                        if len(nz) > len(buf): nz = nz[-len(buf):]
                        self.imu_buffers[i] = np.roll(buf, -len(nz)); self.imu_buffers[i][-len(nz):] = nz

        self._update_status_bar()

    def on_data_ready(self, out, resp_out, frames):
        """Event-driven update slot called whenever the RadarEngine finishes a frame batch."""
        zone         = out.get('zone', "No Occupant Detected")
        is_valid     = out.get('is_valid', False)
        self._last_zone = zone

        # ── Bed-zone gate ─────────────────────────────────────────────────────
        spectral_hist = out.get('spectral_history')
        pipeline_ready = out.get('final_bin') is not None and spectral_hist is not None
        is_in_bed = is_valid and ("Bed" in zone) and pipeline_ready

        if not is_in_bed:
            if self._in_bed:
                self._in_bed = False
                self._reset_all()
            self.warning_text.show()
            self._update_status_bar()
            self.update_plots(None, None)
            return

        if not self._in_bed:
            self._in_bed = True
            self._reset_all()
        self.warning_text.hide()

        # ── Motion from radar_engine (no GUI-side rescoring) ─────────────────
        locked_bin     = out.get('final_bin')
        motion_score   = float(out.get('motion_score', out.get('motion_level', 0.0)) or 0.0)
        motion_score   = min(1.0, max(0.0, motion_score))
        posture_str    = out.get('posture', "Unknown")
        motion_label   = str(out.get('motion_str', "Unknown"))
        if posture_str == "Fallen":
            posture_str = "Lying Down"

        label_low = motion_label.lower()
        is_moving = (
            motion_score >= self.motion_moving_threshold
            or ("walking" in label_low)
            or ("shifting" in label_low)
            or ("movement" in label_low)
        )
        motion_state = "Moving" if is_moving else "Still"
        self._last_motion_str = f"{motion_state} ({motion_label})"
        self._last_is_moving  = is_moving

        self._last_bin        = int(locked_bin) if locked_bin is not None else 0
        self._last_posture    = posture_str

        # ── Confidence ────────────────────────────────────────────────────────
        conf = compute_confidence(motion_score, posture_str)
        self._last_conf = conf

        # ── Histories ─────────────────────────────────────────────────────────
        self.motion_norm_hist = np.roll(self.motion_norm_hist, -frames)
        self.motion_norm_hist[-frames:] = motion_score

        self.bin_history = np.roll(self.bin_history, -frames)
        self.bin_history[-frames:] = self._last_bin

        p_map = {"Standing": 1, "Sitting": 2, "Lying Down": 3}
        self.posture_history = np.roll(self.posture_history, -frames)
        self.posture_history[-frames:] = p_map.get(posture_str, 0)

        if resp_out and 'live_signal' in resp_out:
            sig  = resp_out['live_signal']
            self.radar_display_buffer = sig
            self._last_rr  = resp_out.get('rr_current', 0.0)
            self._apnea_active = bool(resp_out.get('apnea_active', False))
            self._last_depth   = resp_out.get('depth', '--')
            self._last_cycles  = resp_out.get('cycle_count', 0)
            belt_hist = resp_out.get('belt_history')
            if belt_hist is not None:
                self.belt_display_buffer = np.asarray(belt_hist).copy()

        # RR history
        rr_val = resp_out.get('rr_current', 0.0) if resp_out else 0.0
        self.rr_history = np.roll(self.rr_history, -frames)
        self.rr_history[-frames:] = rr_val

        self.conf_history = np.roll(self.conf_history, -frames)
        self.conf_history[-frames:] = conf

        self.height_history = np.roll(self.height_history, -frames)
        self.height_history[-frames:] = out.get('Z', 0.0) if out.get('Z') is not None else 0.0

        self._update_status_bar()
        self.update_plots(resp_out, out)

    # ═════════════════════════════════════════════════════════════════════════
    def update_plots(self, resp_out, out_dict):
        t_r = np.linspace(-self.window_sec, 0, self.radar_frame_count)

        # ── 1. Signal Plot ────────────────────────────────────────────────────
        # Radar curve opacity scales with confidence
        alpha = max(60, int(255 * self._last_conf / 100)) if self._in_bed else 80
        self.curve_radar.setPen(pg.mkPen(QColor(34, 211, 238, alpha), width=2))
        self.curve_radar.setData(t_r, self.radar_display_buffer)

        if self.belt_thread and self.curve_belt:
            tb = np.linspace(-self.window_sec, 0, len(self.belt_display_buffer))
            self.curve_belt.setData(tb, self.belt_display_buffer)
            # Fixed Y-range for Belt (dynamic rescaling disabled per user request)
            self.belt_view.setYRange(-15, 15)
        for i, crv in enumerate(self.curve_imus):
            ti = np.linspace(-self.window_sec, 0, len(self.imu_buffers[i]))
            crv.setData(ti, self.imu_buffers[i])

        # Confidence badge
        if self._in_bed:
            c   = self._last_conf
            sym = confidence_symbol(c)
            col = confidence_color(c)
            self.ann_quality.setColor(col)
            self.ann_quality.setText(f"{sym} Signal Quality: {c:.0f}%")

        # Update Calibration Status Badge
        is_calib  = resp_out.get('is_calibrating', False) if resp_out else True
        is_active = self._in_bed
        if is_calib:
            self.ann_status.setText("CALIBRATING...")
            self.ann_status.setColor('#FB923C')
            self.ann_status.show()
        elif is_active:
            self.ann_status.setText("LIVE")
            self.ann_status.setColor('#10B981')
            self.ann_status.show()
        else:
            self.ann_status.hide()

        # Respiration annotations + scatters
        if resp_out:
            rr        = resp_out.get('rr_current', 0.0)
            apnea_cnt = resp_out.get('apnea_count', 0)
            apnea_act = resp_out.get('apnea_active', False)
            cycles    = resp_out.get('cycle_count', 0)
            depth     = resp_out.get('depth', '--')
            brv       = resp_out.get('brv_value', 0.0)
            last_c    = resp_out.get('last_cycle_duration', 0.0)
            apnea_segs = resp_out.get('apnea_segments', [])

            apnea_col = '#EF4444' if apnea_act else '#FB923C'
            self.ann_apnea.setColor(apnea_col)
            self.ann_apnea.setText(
                f"⚠ APNEA ({apnea_cnt} events)" if apnea_act else f"Apnea events: {apnea_cnt}")
            self.ann_cycles.setText(f"Cycles: {cycles}  |  Depth: {depth}")
            self.ann_brv.setText(f"BRV: {brv:.3f} s  |  Last Cycle: {last_c:.1f} s")

            # ── Apnea segment overlays on signal plot ─────────────────────────
            # Remove old regions
            for r in self._apnea_regions:
                self.plot_comparison.removeItem(r)
            self._apnea_regions.clear()
            for (s_idx, e_idx) in apnea_segs:
                # Convert buffer indices to time values
                t_s = t_r[min(s_idx, len(t_r)-1)]
                t_e = t_r[min(e_idx, len(t_r)-1)]
                if t_e > t_s:
                    region = pg.LinearRegionItem(
                        values=(t_s, t_e),
                        brush=pg.mkBrush(239, 68, 68, 50),
                        pen=pg.mkPen(None),
                        movable=False
                    )
                    self.plot_comparison.addItem(region)
                    self._apnea_regions.append(region)

            # Scatter markers — only show when confidence ≥ 40 %
            if self._last_conf >= 40:
                in_idx = [i for i in resp_out.get('inhales', [])
                          if 0 <= i < len(self.radar_display_buffer)]
                ex_idx = [i for i in resp_out.get('exhales', [])
                          if 0 <= i < len(self.radar_display_buffer)]
                if in_idx:
                    self.scatter_inhales.setData(
                        x=t_r[in_idx], y=self.radar_display_buffer[in_idx])
                else:
                    self.scatter_inhales.clear()
                if ex_idx:
                    self.scatter_exhales.setData(
                        x=t_r[ex_idx], y=self.radar_display_buffer[ex_idx])
                else:
                    self.scatter_exhales.clear()
            else:
                self.scatter_inhales.clear(); self.scatter_exhales.clear()
        else:
            # Clear apnea regions when no resp data
            for r in self._apnea_regions:
                self.plot_comparison.removeItem(r)
            self._apnea_regions.clear()
            self.scatter_inhales.clear(); self.scatter_exhales.clear()

        # ── 2. RR Trend with confidence band ─────────────────────────────────
        rr_main = self.rr_history
        # Uncertainty band width: 0 BPM when conf=100%, 8 BPM when conf=0%
        uncertainty = (1.0 - self.conf_history / 100.0) * 8.0
        rr_upper = rr_main + uncertainty
        rr_lower = np.maximum(0, rr_main - uncertainty)

        self.curve_rr_upper.setData(t_r, rr_upper)
        self.curve_rr_lower.setData(t_r, rr_lower)

        # Update RR curves with median filtering for visual smoothness
        if len(rr_main) > 10:
            # Apply 11-sample median filter (~0.4s) to the trend line
            smoothed_rr = signal.medfilt(rr_main, kernel_size=11)
            self.curve_rr.setData(t_r, smoothed_rr)
        else:
            self.curve_rr.setData(t_r, rr_main)

        # Update Height History plot
        self.curve_height.setData(t_r, self.height_history)

        if resp_out:
            self.ann_rr.setText(f"RR: {resp_out.get('rr_current', 0.0):.1f} BPM")

        # ── 3. Motion Score ───────────────────────────────────────────────────
        self.curve_motion_raw.setData(t_r, self.motion_norm_hist)
        # Update state label
        col_map = {True: '#F59E0B', False: '#10B981', None: '#94A3B8'}
        col = col_map.get(self._last_is_moving, '#94A3B8')
        self.motion_state_text.setColor(col)
        self.motion_state_text.setText(self._last_motion_str)
        # Adjust Y range to always show threshold
        top = max(1.0, float(np.max(self.motion_norm_hist)) * 1.15)
        self.plot_motion.setYRange(0, top, padding=0)

        # ── 4. Bin + Posture ──────────────────────────────────────────────────
        self.curve_bin.setData(t_r, self.bin_history)
        self.curve_posture_hist.setData(t_r, self.posture_history)

        # ── 5. Derivative tab ─────────────────────────────────────────────────
        deriv_signal = resp_out.get('derivative_signal', np.zeros(self.radar_frame_count)) if resp_out else np.zeros(self.radar_frame_count)
        if len(deriv_signal) == len(t_r):
            self.curve_deriv.setData(t_r, deriv_signal)

        # Update apnea threshold line from live pipeline value
        thresh = resp_out.get('apnea_threshold', 0.2) if resp_out else 0.2
        self.line_apnea_thresh.setValue(thresh)

        # Apnea overlay on derivative plot
        apnea_segs_for_deriv = resp_out.get('apnea_segments', []) if resp_out else []
        for r in self._apnea_regions_deriv:
            self.plot_deriv.removeItem(r)
        self._apnea_regions_deriv.clear()
        for (s_idx, e_idx) in apnea_segs_for_deriv:
            t_s = t_r[min(s_idx, len(t_r)-1)]
            t_e = t_r[min(e_idx, len(t_r)-1)]
            if t_e > t_s:
                region = pg.LinearRegionItem(
                    values=(t_s, t_e),
                    brush=pg.mkBrush(239, 68, 68, 60),
                    pen=pg.mkPen(None),
                    movable=False
                )
                self.plot_deriv.addItem(region)
                self._apnea_regions_deriv.append(region)
        self.plot_deriv.setXRange(-self.window_sec, 0, padding=0)

        # ── X-range lock ──────────────────────────────────────────────────────
        for p in [self.plot_comparison, self.plot_rr, self.plot_motion,
                  self.plot_bin, self.plot_posture_hist]:
            p.setXRange(-self.window_sec, 0, padding=0)

        # ── Posture Icon ──────────────────────────────────────────────────────
        if out_dict:
            self.posture_icon.set_state(
                out_dict.get('posture', "Unknown"),
                out_dict.get('posture_confidence', 0),
                out_dict.get('motion_str', "Resting")
            )

    def _reset_all(self):
        """Reset rolling buffers and transient UI state when target leaves bed."""
        self.radar_display_buffer.fill(0.0)
        self.rr_history.fill(0.0)
        self.conf_history.fill(0.0)
        self.height_history.fill(0.0)
        self.motion_norm_hist.fill(0.0)
        self.bin_history.fill(0.0)
        self.posture_history.fill(0.0)

        if hasattr(self, "belt_display_buffer"):
            self.belt_display_buffer.fill(0.0)

        self._last_rr = 0.0
        self._last_conf = 0.0
        self._last_motion_str = "Unknown"
        self._last_is_moving = None
        self._last_bin = 0
        self._last_posture = "Unknown"
        self._apnea_active = False
        self._last_depth = "--"
        self._last_cycles = 0

        self.scatter_inhales.clear()
        self.scatter_exhales.clear()
        for r in self._apnea_regions:
            self.plot_comparison.removeItem(r)
        self._apnea_regions.clear()
        for r in self._apnea_regions_deriv:
            self.plot_deriv.removeItem(r)
        self._apnea_regions_deriv.clear()

    def _update_camera(self):
        """Grab a camera frame (if available) and paint it into camera_label."""
        if self._camera_cap is None:
            return
        ok, frame = self._camera_cap.read()
        if not ok or frame is None:
            return
        if self._cv2 is not None:
            frame = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        img = QImage(frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(
            self.camera_label.width(),
            self.camera_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.camera_label.setPixmap(pix)

    def _update_status_bar(self):
        """Update right-side info panel text."""
        in_bed = "Yes" if self._in_bed else "No"
        rr_text = f"{self._last_rr:.1f} BPM" if self._last_rr > 0 else "--"
        conf_text = f"{self._last_conf:.0f}%"
        apnea_text = "Active" if self._apnea_active else "No"
        self.info_label.setText(
            "<b>Live Status</b><br/>"
            f"Zone: <b>{self._last_zone}</b><br/>"
            f"In Bed: <b>{in_bed}</b><br/>"
            f"Posture: <b>{self._last_posture}</b><br/>"
            f"Motion: <b>{self._last_motion_str}</b><br/>"
            f"RR: <b>{rr_text}</b><br/>"
            f"Signal Quality: <b>{conf_text}</b><br/>"
            f"Apnea: <b>{apnea_text}</b><br/>"
            f"Breath Depth: <b>{self._last_depth}</b><br/>"
            f"Cycles: <b>{self._last_cycles}</b>"
        )

    # ═════════════════════════════════════════════════════════════════════════
    def closeEvent(self, event):
        print("Cleaning up threads...")
        if hasattr(self, "processor") and self.processor is not None:
            self.processor.stop()
        self.radar_process.terminate()
        for t in self.imu_threads:
            t.stop()
            t.join(timeout=2.0)
        if self.belt_thread:
            self.belt_thread.stop()
            self.belt_thread.join(timeout=2.0)
        if self._camera_cap is not None:
            self._camera_cap.release()
        super().closeEvent(event)


def run_app():
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    app.setApplicationName("Respiration Monitor Test Stack")
    window = RespirationMonitorApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run_app()
