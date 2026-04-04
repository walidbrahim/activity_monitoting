"""
test_respiration_real.py
──────────────────────────
High-precision, multi-sensor respiratory monitoring test stack.

Motion estimation: A+B hybrid (spectral magnitude change + phase-velocity IQR)
with stability-gated rolling-percentile auto-calibration.
Confidence: motion_factor × posture_factor displayed as a live badge.
"""
import sys, os, multiprocessing, queue
import numpy as np
from scipy import signal
from collections import deque

from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTabWidget
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont, QColor, QPixmap, QImage
import pyqtgraph as pg

sys.path.append(os.getcwd())

from libs.controllers.radarController import RadarController
from libs.controllers.vernier_belt_controller import VernierBeltControllerThread
from libs.controllers.witmotion_controller import WitMotionControllerThread
from libs.pipelines.activityPipeline import ActivityPipeline
from libs.pipelines.respirationPipeline import RespiratoryPipelineV2
from libs.gui.posture_widget import RadarPostureItem
from config import config


# ══════════════════════════════════════════════════════════════════════════════
#  Motion Scorer — A+B Hybrid with Stability-Gated Rolling Calibration
# ══════════════════════════════════════════════════════════════════════════════
class MotionScorer:
    """
    Score A (always available):
        max per-bin std of |spectral_history| over the last SCORE_WIN_SEC.
        Captures power spread across range bins — movement illuminates many bins.

    Score B (after bin lock):
        IQR of |Δphase| at the locked bin over the last SCORE_WIN_SEC.
        Captures irregular phase jumps; breathing is periodic and bounded.

    Auto-calibration:
        Frames are admitted to the calibration pool only when the coarse
        ActivityPipeline motion_level < restless_max AND posture_confidence > 60 %.
        This prevents entry-phase motion from corrupting the baseline.

        Threshold = 10th-percentile(pool) × FACTOR
        → naturally tracks the "quiet breathing" floor, not the entry burst.
    """

    SCORE_WIN_SEC    = 1.0          # look-back window for A and B
    MIN_POOL_SAMPLES = 30           # frames required before threshold is valid
    THRESHOLD_FACTOR = 6.0          # moving ≡ 6 × quiet baseline (was 3.0)
    MIN_THRESHOLD    = 0.5          # absolute floor (radar noise floor)
    BLEND_FRAMES     = 5            # A→B transition blend length

    def __init__(self, fps: float, restless_max: float):
        self.fps          = fps
        self.restless_max = restless_max
        self._pool              = deque(maxlen=int(fps * 30))  # 30-s calibration pool
        self._threshold         = None
        self._is_moving         = None   # None = "Calibrating"
        self._blend_counter     = 0
        self._prev_bin_locked   = False
        self._last_raw_score    = 0.0
        self._last_norm_score   = 0.0

    def reset(self):
        self._pool.clear()
        self._threshold       = None
        self._is_moving       = None
        self._blend_counter   = 0
        self._prev_bin_locked = False
        self._last_raw_score  = 0.0
        self._last_norm_score = 0.0

    # ── Scorers ──────────────────────────────────────────────────────────────
    def _score_A(self, spectral_history: np.ndarray) -> float:
        """Max per-bin std of magnitude over last SCORE_WIN_SEC frames."""
        n = max(4, int(self.fps * self.SCORE_WIN_SEC))
        recent = spectral_history[:, -n:]           # (bins, n)
        per_bin_std = np.std(np.abs(recent), axis=1)
        return float(np.max(per_bin_std))

    def _score_B(self, spectral_history: np.ndarray, locked_bin: int) -> float:
        """IQR of absolute phase velocity at locked_bin over last SCORE_WIN_SEC."""
        n = max(5, int(self.fps * self.SCORE_WIN_SEC))
        raw_phase  = np.angle(spectral_history[locked_bin, -n:])   # radians
        unwrapped  = np.unwrap(raw_phase) * (180.0 / np.pi)        # deg
        diff_phase = np.diff(unwrapped)
        if len(diff_phase) < 4:
            return 0.0
        q75, q25 = np.percentile(np.abs(diff_phase), [75, 25])
        return float(q75 - q25)

    # ── Main update ───────────────────────────────────────────────────────────
    def update(self, spectral_history: np.ndarray, locked_bin,
               coarse_motion_level: float, posture_conf: float,
               posture_str: str):
        """
        Returns
        -------
        raw_score    : float  — physical score (arbitrary units)
        norm_score   : float  — score / threshold  (1.0 = boundary)
        is_moving    : bool|None  — None while calibrating
        state_str    : str
        """
        bin_locked = locked_bin is not None

        # ── Compute raw score (A or B with blend) ────────────────────────────
        sA = self._score_A(spectral_history)

        if bin_locked:
            sB = self._score_B(spectral_history, locked_bin)
            if not self._prev_bin_locked:
                # Just acquired lock — start blend
                self._blend_counter = self.BLEND_FRAMES
            if self._blend_counter > 0:
                alpha = 1.0 - self._blend_counter / self.BLEND_FRAMES   # 0→1
                raw = (1 - alpha) * sA + alpha * sB
                self._blend_counter -= 1
            else:
                raw = sB
        else:
            raw = sA

        self._prev_bin_locked = bin_locked
        self._last_raw_score  = raw

        # ── Stability gate for calibration pool ─────────────────────────────
        is_stable = (
            coarse_motion_level < self.restless_max
            and posture_conf > 60.0
            and posture_str not in ("Unknown",)
        )
        if is_stable:
            self._pool.append(raw)

        # ── Compute / update threshold ───────────────────────────────────────
        if len(self._pool) >= self.MIN_POOL_SAMPLES:
            baseline = float(np.percentile(list(self._pool), 10))
            self._threshold = max(self.MIN_THRESHOLD, baseline * self.THRESHOLD_FACTOR)

        # ── Binary decision ──────────────────────────────────────────────────
        if self._threshold is None:
            self._is_moving = None
            norm = self._last_norm_score  # keep last value for display
            state_str = f"Calibrating ({len(self._pool)}/{self.MIN_POOL_SAMPLES})"
        else:
            norm = raw / self._threshold
            self._is_moving = norm > 1.0
            state_str = "Moving" if self._is_moving else "Still"

        self._last_norm_score = norm
        return raw, norm, self._is_moving, state_str

    @property
    def is_calibrated(self) -> bool:
        return self._threshold is not None

    @property
    def is_moving(self):           # convenience
        return self._is_moving


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

    Motion factor uses a dead-band:
      - norm_score ≤ LOWER_BAND (0.7)  → factor = 1.0  (no penalty)
      - norm_score   LOWER_BAND..UPPER_BAND → linear decay 1.0 → 0.0
      - norm_score ≥ UPPER_BAND (2.0)  → factor = 0.0
    This prevents small breathing fluctuations from degrading confidence.
    """
    LOWER_BAND = 0.7   # below this: perfect confidence
    UPPER_BAND = 2.0   # above this: zero confidence
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
        print("Initializing Optimized Respiration Monitor (PyQt6)...")

        self.state_q  = multiprocessing.Queue()
        self.pt_fft_q = multiprocessing.Queue()

        self.fps              = config.radar.frame_rate
        self.window_sec       = config.respiration.resp_window_sec
        self.radar_frame_count = int(self.window_sec * self.fps)

        # ── Radar ─────────────────────────────────────────────────────────────
        self.radar_process = RadarController(state_q=self.state_q, pt_fft_q=self.pt_fft_q)
        self.radar_process.start()

        # ── Vernier Belt ──────────────────────────────────────────────────────
        self.vernier_belt_realtime_q = queue.Queue()
        self.belt_thread = None
        if config.vernier.enabled:
            self.belt_thread = VernierBeltControllerThread(
                vernier_belt_realtime_q=self.vernier_belt_realtime_q,
                vernier_belt_connection_q=queue.Queue(),
                start_vernier_belt_q=queue.Queue()
            )
            self.belt_thread.start()
            self.belt_thread.start_vernier_belt_q.put(True)

        # ── WitMotion IMUs ────────────────────────────────────────────────────
        self.imu_threads, self.imu_queues = [], []
        for imu_cfg in [config.witmotion1, config.witmotion2]:
            if imu_cfg.enabled:
                q, cmd_q = queue.Queue(maxsize=1000), queue.Queue()
                t = WitMotionControllerThread(
                    witmotion_mac=imu_cfg.mac,
                    witmotion_realtime_q=q,
                    start_witmotion_q=cmd_q,
                    location=imu_cfg.location
                )
                t.start(); cmd_q.put(True)
                self.imu_threads.append(t)
                self.imu_queues.append(q)

        # ── Pipelines ─────────────────────────────────────────────────────────
        self.act_pipeline  = ActivityPipeline(config.radar.range_idx_num, config.radar.range_resolution)
        self.resp_pipeline = RespiratoryPipelineV2()

        # ── Motion Scorer ─────────────────────────────────────────────────────
        self.motion_scorer = MotionScorer(self.fps, config.motion.restless_max)

        # ── Display Buffers ───────────────────────────────────────────────────
        N = self.radar_frame_count
        self.radar_display_buffer = np.zeros(N)
        self.rr_history           = np.zeros(N)
        self.conf_history         = np.zeros(N)   # 0–100
        self.height_history       = np.zeros(N)   # Z height
        self.motion_norm_hist     = np.zeros(N)   # normalized score (threshold=1.0)
        self.bin_history          = np.zeros(N)
        self.posture_history      = np.zeros(N)

        belt_n = int(self.window_sec * config.vernier.rate_hz)
        self.belt_display_buffer = np.zeros(belt_n)

        self.imu_buffers     = []
        self.imu_offset_emas = []
        self.imu_scale_emas  = []
        for imu_cfg in [config.witmotion1, config.witmotion2]:
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
        self._last_motion_str = "Calibrating"
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

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_loop)
        self.timer.start(33)

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
        if config.vernier.enabled:
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
            "Motion Score (normalized — threshold = 1.0)", ylabel="Norm. Score")
        layout.addWidget(self.plot_motion, stretch=1)
        # Filled area above threshold becomes red, below is purple
        self.curve_motion_raw = self.plot_motion.plot(
            pen=pg.mkPen('#A855F7', width=2),
            fillLevel=0, brush=pg.mkBrush(168, 85, 247, 50)
        )
        self.line_motion_thresh = pg.InfiniteLine(
            pos=1.0, angle=0,
            pen=pg.mkPen('#F59E0B', width=1.5, style=Qt.PenStyle.DashLine),
            label="Moving Threshold"
        )
        self.plot_motion.addItem(self.line_motion_thresh)
        self.motion_state_text = pg.TextItem("Calibrating...", color='#94A3B8', anchor=(0, 0))
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
        self.plot_bin.setYRange(0, config.radar.range_idx_num)
        self.tab_widget.addTab(self.plot_bin, "Range Tracking")

        self.plot_height = mkplot("Height History (Z)", ylabel="Height (m)")
        self.plot_height_legend = self.plot_height.addLegend(offset=(10, 10))
        self.plot_height_legend.setLabelTextColor("#CBD5E1")
        self.plot_height.setYRange(0, 2.5)
        
        self.curve_height   = self.plot_height.plot(
            pen=pg.mkPen('#10B981', width=2), name="Track Z")
        
        # Posture thresholds
        self.line_sit = pg.InfiniteLine(
            pos=config.posture.sitting_threshold, angle=0,
            pen=pg.mkPen('#F59E0B', width=1.5, style=Qt.PenStyle.DashLine),
            label=f"Sit ({config.posture.sitting_threshold}m)")
        self.line_stand = pg.InfiniteLine(
            pos=config.posture.standing_threshold, angle=0,
            pen=pg.mkPen('#34D399', width=1.5, style=Qt.PenStyle.DashLine),
            label=f"Stand ({config.posture.standing_threshold}m)")
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



    # ═════════════════════════════════════════════════════════════════════════
    #  Helpers
    # ═════════════════════════════════════════════════════════════════════════
    def _reset_all(self):
        N = self.radar_frame_count
        self.radar_display_buffer.fill(0)
        self.belt_display_buffer.fill(0)
        for b in self.imu_buffers: b.fill(0)
        self.rr_history.fill(0)
        self.conf_history.fill(0)
        self.height_history.fill(0)
        self.motion_norm_hist.fill(0)
        self.bin_history.fill(0)
        self.posture_history.fill(0)

        self.radar_scale_locked = False
        self.radar_scale_value  = 1.0
        self.radar_scale_frames = 0
        self.motion_scorer.reset()
        self.resp_pipeline._reset_state()

        self.scatter_inhales.clear()
        self.scatter_exhales.clear()
        for r in self._apnea_regions:
            self.plot_comparison.removeItem(r)
        self._apnea_regions.clear()
        for r in self._apnea_regions_deriv:
            self.plot_deriv.removeItem(r)
        self._apnea_regions_deriv.clear()
        self.ann_rr.setText("RR: -- BPM")
        self.ann_quality.setText("")
        self.ann_apnea.setText("")
        self.ann_cycles.setText("")
        self.ann_brv.setText("")
        self.motion_state_text.setText("Calibrating...")

    def _update_info_label(self):
        """Rich-text metrics panel in the right column."""
        apnea_txt = (
            "<span style='color:#EF4444'>⚠ APNEA</span>"
            if getattr(self, '_apnea_active', False) else
            "<span style='color:#475569'>—</span>"
        )
        rr_col   = '#F43F5E' if self._last_rr > 0 else '#475569'
        q        = getattr(self, '_last_conf', 0.0)
        q_col    = '#10B981' if q >= 70 else ('#F59E0B' if q >= 40 else '#EF4444')
        depth    = getattr(self, '_last_depth',  '--')
        cycles   = getattr(self, '_last_cycles', 0)
        html = (
            "<div style='font-family:Arial; font-size:13px; line-height:2.0;'>"
            f"<b style='color:#38BDF8'>Zone</b>: <span style='color:#E2E8F0'>{self._last_zone}</span><br>"
            f"<b style='color:{rr_col}'>RR</b>: <span style='color:#E2E8F0'>{self._last_rr:.1f} BPM</span><br>"
            f"<b style='color:{q_col}'>Quality</b>: <span style='color:#E2E8F0'>{q:.0f}%</span><br>"
            f"<b style='color:#A855F7'>Motion</b>: <span style='color:#E2E8F0'>{self._last_motion_str}</span><br>"
            f"<b style='color:#CBD5E1'>Posture</b>: <span style='color:#E2E8F0'>{self._last_posture}</span><br>"
            f"<b style='color:#94A3B8'>Depth</b>: <span style='color:#E2E8F0'>{depth}</span> "
            f"&nbsp;&nbsp;<b style='color:#94A3B8'>Cycles</b>: <span style='color:#E2E8F0'>{cycles}</span><br>"
            f"<b style='color:#FB923C'>Apnea</b>: {apnea_txt}"
            "</div>"
        )
        self.info_label.setText(html)

    def _update_status_bar(self):
        # self.statusBar().showMessage(...) removed (redundant)
        self._update_info_label()

    def _update_camera(self):
        """Grab one frame from the webcam and push it to camera_label."""
        if self._camera_cap is None or self._cv2 is None:
            return
        ret, frame = self._camera_cap.read()
        if not ret:
            return
        # Convert BGR→RGB and copy the bytes so the QImage owns its data
        frame_rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        img = QImage(frame_rgb.tobytes(), w, h, ch * w,
                     QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(
            self.camera_label.width(), self.camera_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        self.camera_label.setPixmap(pix)

    # ═════════════════════════════════════════════════════════════════════════
    #  Main Timer Loop
    # ═════════════════════════════════════════════════════════════════════════
    def update_loop(self):
        self._update_camera()
        # ── IMUs (always stream) ──────────────────────────────────────────────
        for i, q in enumerate(self.imu_queues):
            samples = []
            while not q.empty():
                try: samples.append(q.get_nowait())
                except queue.Empty: break
            if samples:
                z   = [s[2] for s in samples]
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

        # ── Belt (always stream) ──────────────────────────────────────────────
        if self.belt_thread:
            chunk = []
            while not self.vernier_belt_realtime_q.empty():
                try: chunk.append(self.vernier_belt_realtime_q.get_nowait())
                except queue.Empty: break
            if chunk:
                b = np.array(chunk)
                self.belt_display_buffer = np.roll(self.belt_display_buffer, -len(b))
                # Stable rolling mean removal for raw-ish force waveform
                alpha = 0.05
                target_mean = np.mean(b)
                if not hasattr(self, '_belt_ema_mean'): self._belt_ema_mean = target_mean
                self._belt_ema_mean = (1 - alpha) * self._belt_ema_mean + alpha * target_mean
                self.belt_display_buffer[-len(b):] = (b - self._belt_ema_mean)

        # ── Radar frames ──────────────────────────────────────────────────────
        frames = 0
        while True:
            try: self.act_pipeline.process_frame(self.pt_fft_q.get_nowait()); frames += 1
            except queue.Empty: break
        if frames == 0:
            self._update_status_bar(); return

        out = self.act_pipeline.output_dict
        zone         = out.get('zone', "No Occupant Detected")
        is_valid     = out.get('is_valid', False)
        self._last_zone = zone

        # ── Bed-zone gate ─────────────────────────────────────────────────────
        pipeline_ready = 'final_bin' in out and 'spectral_history' in out
        is_in_bed = is_valid and ("Bed" in zone) and pipeline_ready

        if not is_in_bed:
            if self._in_bed:
                self._in_bed = False; self._reset_all()
            self.warning_text.show()
            self._update_status_bar()
            self.update_plots(None, None); return

        if not self._in_bed:
            self._in_bed = True; self._reset_all()
        self.warning_text.hide()

        # ── Motion scoring (A+B hybrid) ───────────────────────────────────────
        locked_bin     = out.get('final_bin')
        spectral_hist  = out['spectral_history']
        coarse_motion  = self.act_pipeline.motion_level
        posture_conf   = out.get('posture_confidence', 0.0)
        posture_str    = out.get('posture', "Unknown")
        if posture_str == "Fallen": posture_str = "Lying Down"

        raw_score, norm_score, is_moving, motion_str = self.motion_scorer.update(
            spectral_hist, locked_bin, coarse_motion, posture_conf, posture_str)

        self._last_motion_str = motion_str
        self._last_bin        = int(locked_bin) if locked_bin is not None else 0
        self._last_posture    = posture_str

        # ── Confidence ────────────────────────────────────────────────────────
        conf = compute_confidence(norm_score, posture_str)
        self._last_conf = conf

        # ── Histories ─────────────────────────────────────────────────────────
        self.motion_norm_hist = np.roll(self.motion_norm_hist, -frames)
        self.motion_norm_hist[-frames:] = norm_score

        self.bin_history = np.roll(self.bin_history, -frames)
        self.bin_history[-frames:] = self._last_bin

        p_map = {"Standing": 1, "Sitting": 2, "Lying Down": 3}
        self.posture_history = np.roll(self.posture_history, -frames)
        self.posture_history[-frames:] = p_map.get(posture_str, 0)

        # ── Respiration pipeline ──────────────────────────────────────────────
        resp_out = self.resp_pipeline.process(out, frames=frames)

        if resp_out and 'live_signal' in resp_out:
            sig  = resp_out['live_signal']
            # No scaling needed — bandpass signal is already in physical units (degrees of phase)
            self.radar_display_buffer = sig
            self._last_rr  = resp_out.get('rr_current', 0.0)
            self._apnea_active = bool(resp_out.get('apnea_active', False))
            self._last_depth   = resp_out.get('depth', '--')
            self._last_cycles  = resp_out.get('cycle_count', 0)

        # RR history (update only when confident enough → Option B)
        # Always update buffer but mask low-confidence frames with NaN for display
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
        if self._in_bed and self.motion_scorer.is_calibrated:
            c   = self._last_conf
            sym = confidence_symbol(c)
            col = confidence_color(c)
            self.ann_quality.setColor(col)
            self.ann_quality.setText(f"{sym} Signal Quality: {c:.0f}%")
        elif self._in_bed:
            self.ann_quality.setColor('#94A3B8')
            self.ann_quality.setText("⏳ Calibrating motion...")

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
        col = col_map.get(self.motion_scorer.is_moving, '#94A3B8')
        self.motion_state_text.setColor(col)
        self.motion_state_text.setText(self._last_motion_str)
        # Adjust Y range to always show threshold
        top = max(2.0, float(np.max(self.motion_norm_hist)) * 1.15)
        self.plot_motion.setYRange(0, top, padding=0)

        # ── 4. Bin + Posture ──────────────────────────────────────────────────
        self.curve_bin.setData(t_r, self.bin_history)
        self.curve_posture_hist.setData(t_r, self.posture_history)

        # ── 5. Derivative tab ─────────────────────────────────────────────────
        deriv_signal = resp_out.get('derivative_signal', np.zeros(self.radar_frame_count)) if resp_out else np.zeros(self.radar_frame_count)
        if len(deriv_signal) == len(t_r):
            self.curve_deriv.setData(t_r, deriv_signal)

        # Update apnea threshold line from live pipeline value
        thresh = getattr(self.resp_pipeline, 'apnea_threshold', 0.2)
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

    # ═════════════════════════════════════════════════════════════════════════
    def closeEvent(self, event):
        print("Cleaning up threads...")
        self.radar_process.terminate()
        for t in self.imu_threads: t.stop()
        if self.belt_thread: self.belt_thread.stop()
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
