"""
apps/debug/01_preprocessor_debug.py
=====================================
Debug GUI for RadarFramePreprocessor + ClutterMap.

Layout (after redesign):
  │  [ Raw = clutter-free + clutter_map_L1 │ Dynamic = clutter-free ] │
  ├──────────────────────────────────┬──────────────────┬─────────────────────┤
  │  Clutter Map L1-norm             │ α-activity (EMA) │ Dynamic Result      │
  ├───────────────────────────────────────────────────────────────────────────┤
  │  THREE HEATMAPS (Side-by-Side):                                           │
  │  1. Clutter Map (Plasma)       - What is the background?                  │
  │  2. α-Activity (Inferno)       - Where is adaptation frozen?              │
  │  3. Dynamic Result (Viridis)   - What is left after subtraction?          │
  ├───────────────────────────────────────────────────────────────────────────┤
  │  SCR rolling time-series | warmup ring | diagnostics strip                │
  └───────────────────────────────────────────────────────────────────────────┘

Run:
    APP_PROFILE=lab python apps/debug/01_preprocessor_debug.py
"""
from __future__ import annotations
import sys, os, math
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout,
    QGridLayout, QGroupBox, QLabel, QSizePolicy, QTabWidget
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QTransform
import pyqtgraph as pg

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from apps.debug.debug_base import DebugBase, PALETTE, apply_plot_defaults  # noqa: E402
from radar_engine.core.models import EngineOutput                           # noqa: E402


# ── Circular progress widget (warmup ring) ────────────────────────────────────

class WarmupRing(QWidget):
    def __init__(self, total: int, parent=None):
        super().__init__(parent)
        self.total   = max(1, total)
        self.current = 0
        self.active  = True
        self.setMinimumSize(110, 110)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def set_value(self, remaining: int | None, active: bool):
        self.active  = active
        self.current = self.total - remaining if remaining is not None else self.total
        self.update()

    def paintEvent(self, event):
        from PyQt6.QtGui import QPainter, QPen, QColor, QFont
        from PyQt6.QtCore import QRectF
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        m    = 10
        rect = QRectF(m, m, w - 2*m, h - 2*m)
        pen  = QPen(QColor(PALETTE["border"]), 9, Qt.PenStyle.SolidLine,
                    Qt.PenCapStyle.RoundCap)
        p.setPen(pen); p.drawArc(rect, 0, 360*16)
        frac  = min(1.0, self.current / self.total)
        color = QColor(PALETTE["ok"]) if frac >= 1.0 else QColor(PALETTE["accent"])
        pen   = QPen(color, 9, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        p.setPen(pen); p.drawArc(rect, 90*16, -int(frac * 360 * 16))
        p.setPen(QColor(PALETTE["text"]))
        p.setFont(QFont("Inter", 12, QFont.Weight.Bold))
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                   "✓" if not self.active else str(self.total - self.current))
        sub = QRectF(m, m + 28, w - 2*m, h - 2*m)
        p.setFont(QFont("Inter", 7)); p.setPen(QColor(PALETTE["subtext"]))
        p.drawText(sub, Qt.AlignmentFlag.AlignCenter,
                   "frames left" if self.active else "Calibrated")
        p.end()


# ── Helper: a PlotWidget with an ImageItem properly scaled to real units ──────

def _make_heatmap_widget(
    title: str, xlabel: str, ylabel: str,
    cmap: str, n_rows: int, n_cols: int,
    x_scale: float, x_offset: float,
    y_scale: float, y_offset: float = 0.0,
    levels: tuple = (0, 1),
) -> tuple:
    """Return (PlotWidget, ImageItem, np.ndarray buffer).

    x_scale / y_scale map pixel → physical unit (seconds / metres).
    """
    pw      = pg.PlotWidget()
    pw.setBackground(PALETTE["panel"])
    apply_plot_defaults(pw.getPlotItem(), xlabel=xlabel, ylabel=ylabel)
    pw.getPlotItem().setTitle(title, color=PALETTE["subtext"], size="14pt")

    img_item = pg.ImageItem()
    img_item.setColorMap(pg.colormap.get(cmap))

    tr = QTransform()
    tr.translate(x_offset, y_offset)
    tr.scale(x_scale, y_scale)
    img_item.setTransform(tr)

    pw.addItem(img_item)

    # Colorbar
    cbar = pg.ColorBarItem(
        values=levels,
        colorMap=pg.colormap.get(cmap),
    )
    cbar.setImageItem(img_item, insert_in=pw.getPlotItem())

    buf = np.zeros((n_rows, n_cols), dtype=np.float32)
    return pw, img_item, buf


# ── Main debug window ─────────────────────────────────────────────────────────

class PreprocessorDebug(DebugBase):
    TITLE    = "🔬 Preprocessor & Clutter Map — Debug"
    WINDOW_W = 1700
    WINDOW_H = 1000

    _HIST_SECS = 20   # time depth of the heatmaps

    def _build_ui(self, central: QWidget) -> None:
        # ── Setup Parameter Tuner ─────────────────────────────────────────────
        self.add_tunable_param("preprocessing.clutter_ema_alpha", "Clutter EMA Alpha", 0.001, 1.0, 0.001, self._engine_cfg.preprocessing.clutter_ema_alpha, decimals=3)
        self.add_tunable_param("preprocessing.static_clutter_margin", "Clutter Margin", 0.0, 5000.0, 10.0, self._engine_cfg.preprocessing.static_clutter_margin, decimals=1)
        self.add_tunable_param("preprocessing.warmup_frames", "Warmup Frames", 10, 750, 1, self._engine_cfg.preprocessing.warmup_frames, decimals=0)
        
        self.add_tunable_param("preprocessing.features_clutter_removal", "ENABLE Clutter Removal", 0, 0, 0, self._engine_cfg.preprocessing.features_clutter_removal)
        self.add_tunable_param("preprocessing.features_target_protection", "ENABLE Target Protection", 0, 0, 0, self._engine_cfg.preprocessing.features_target_protection)

        # ── UI Construction ───────────────────────────────────────────────────
        outer = QVBoxLayout(central)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        range_res = self._app_cfg.hardware.range_resolution   # m / bin
        n         = self._num_bins
        dt        = 1.0 / self._frame_rate
        hist_rows = int(self._HIST_SECS * self._frame_rate)
        self._hist_rows   = hist_rows
        self._range_res   = range_res
        max_range_m       = n * range_res

        # ── Row 0: magnitude profile ──────────────────────────────────────────
        profile_grp = QGroupBox(
            "Magnitude Profile — grey fill = Raw signal (clutter + target) │ "
            "blue fill = Dynamic (after clutter subtraction, clutter-free)"
        )
        pg_lay = QVBoxLayout(profile_grp)
        self._pw_profile = self.make_plot_widget(
            ylabel="Magnitude", xlabel=f"Range Bin  (1 bin = {range_res:.3f} m)"
        )
        # Dual bin+metre x-axis ticks
        ax_b = self._pw_profile.getPlotItem().getAxis("bottom")
        tick_bins  = list(range(0, n + 1, 5))
        ax_b.setTicks([[(b, f"{b}\n{b*range_res:.2f}m") for b in tick_bins]])

        self._raw_curve = self._pw_profile.plot(
            pen=pg.mkPen(PALETTE["subtext"], width=1.5), name="Raw")
        self._raw_zero  = self._pw_profile.plot(np.zeros(n), pen=None)
        self._fill_raw  = pg.FillBetweenItem(
            self._raw_curve, self._raw_zero,
            brush=pg.mkBrush(PALETTE["subtext"] + "30"))
        self._pw_profile.addItem(self._fill_raw)

        self._dyn_curve = self._pw_profile.plot(
            pen=pg.mkPen(PALETTE["accent"], width=2.5), name="Dynamic (clutter-free)")
        self._dyn_zero  = self._pw_profile.plot(np.zeros(n), pen=None)
        self._fill_dyn  = pg.FillBetweenItem(
            self._dyn_curve, self._dyn_zero,
            brush=pg.mkBrush(PALETTE["accent"] + "30"))
        self._pw_profile.addItem(self._fill_dyn)

        leg = self._pw_profile.addLegend(offset=(10, 10))
        leg.addItem(self._raw_curve, "Raw (clutter incl.)")
        leg.addItem(self._dyn_curve, "Dynamic (clutter-free)")
        pg_lay.addWidget(self._pw_profile)
        outer.addWidget(profile_grp, stretch=4)

        # ── Row 1: THREE heatmaps side by side ────────────────────────────────
        maps_row = QHBoxLayout()
        min_r = self._app_cfg.detection.min_search_range_m

        # 1. Clutter map L1-norm heatmap (Plasma)
        clutter_grp = QGroupBox("1. Clutter Map (Static Background)")
        cl_lay = QVBoxLayout(clutter_grp)
        (self._pw_clutter,
         self._clutter_img,
         self._clutter_buf) = _make_heatmap_widget(
            title="Learned Clutter [Plasma]",
            xlabel="Time (s ago)",
            ylabel="Range (m)",
            cmap="plasma",
            n_rows=hist_rows,
            n_cols=n,
            x_scale=dt,
            x_offset=-self._HIST_SECS,
            y_scale=range_res,
            levels=(0, 4000),
        )
        self._pw_clutter.addItem(pg.InfiniteLine(
            pos=min_r, angle=0,
            pen=pg.mkPen(PALETTE["warn"], width=1, style=Qt.PenStyle.DashLine)))
        cl_lay.addWidget(self._pw_clutter)
        maps_row.addWidget(clutter_grp, stretch=1)

        # 2. α-activity heatmap (Inferno)
        alpha_grp = QGroupBox("2. α-Activity (Adaptation Rate)")
        al_lay = QVBoxLayout(alpha_grp)
        (self._pw_alpha,
         self._alpha_img,
         self._alpha_buf) = _make_heatmap_widget(
            title="Frozen[Black] vs Adaptive[Bright] [Inferno]",
            xlabel="Time (s ago)",
            ylabel="Range (m)",
            cmap="inferno",
            n_rows=hist_rows,
            n_cols=n,
            x_scale=dt,
            x_offset=-self._HIST_SECS,
            y_scale=range_res,
            levels=(0, 1),
        )
        al_lay.addWidget(self._pw_alpha)
        maps_row.addWidget(alpha_grp, stretch=1)

        # 3. Dynamic result heatmap (Viridis)
        dyn_grp = QGroupBox("3. Dynamic Result (After Subtraction)")
        dy_lay = QVBoxLayout(dyn_grp)
        (self._pw_dyn_map,
         self._dyn_map_img,
         self._dyn_map_buf) = _make_heatmap_widget(
            title="Final Moving Targets [Viridis]",
            xlabel="Time (s ago)",
            ylabel="Range (m)",
            cmap="viridis",
            n_rows=hist_rows,
            n_cols=n,
            x_scale=dt,
            x_offset=-self._HIST_SECS,
            y_scale=range_res,
            levels=(0, 3000),
        )
        # also show min_range here for reference
        self._pw_dyn_map.addItem(pg.InfiniteLine(
            pos=min_r, angle=0,
            pen=pg.mkPen(PALETTE["subtext"], width=1, style=Qt.PenStyle.DotLine)))
        dy_lay.addWidget(self._pw_dyn_map)
        maps_row.addWidget(dyn_grp, stretch=1)

        outer.addLayout(maps_row, stretch=7)

        # ── Row 2: SCR chart + warmup ring + diagnostics ──────────────────────
        bottom_row = QHBoxLayout()

        # Tabs for I/Q Trace and SCR
        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(f"QTabBar::tab {{ padding: 8px 16px; font-weight: bold; background: {PALETTE['bg']}; color: {PALETTE['text']}; border: 1px solid {PALETTE['panel']}; }} QTabBar::tab:selected {{ background: {PALETTE['panel']}; color: {PALETTE['cyan']}; border-bottom: 2px solid {PALETTE['cyan']}; }}")
        
        # Tab 1: I/Q Data Trace
        iq_tab = QWidget()
        iq_lay = QHBoxLayout(iq_tab)
        iq_lay.setContentsMargins(0, 0, 0, 0)

        self._pw_iq = self.make_plot_widget(ylabel="Imaginary (Q)", xlabel="Real (I)")
        self._pw_iq.setTitle("Protected Bin Phase Rotation")
        self._pw_iq.setAspectLocked(True) # Force 1:1 aspect ratio for true phase circle
        self._pw_iq.showGrid(x=True, y=True, alpha=0.3)
        self._iq_scatter = pg.ScatterPlotItem(size=6, pen=pg.mkPen(None), brush=pg.mkBrush(PALETTE["cyan"]+'cc'))
        self._pw_iq.addItem(self._iq_scatter)
        self._iq_line = self._pw_iq.plot(pen=pg.mkPen(PALETTE["cyan"] + '44', width=1)) # Faint connecting line
        self._pw_iq.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(PALETTE["subtext"]+'88', style=Qt.PenStyle.DashLine)))
        self._pw_iq.addItem(pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen(PALETTE["subtext"]+'88', style=Qt.PenStyle.DashLine)))
        iq_lay.addWidget(self._pw_iq, stretch=1)

        self._pw_iq_time = self.make_plot_widget(ylabel="Amplitude", xlabel="Time (s)")
        self._pw_iq_time.setTitle("I and Q Time Evolution")
        self._pw_iq_time.addLegend(offset=(10, 10))
        self._pw_iq_time.showGrid(x=True, y=True, alpha=0.3)
        self._pw_iq_time.setXRange(-20, 0, padding=0)
        self._pw_iq_time.enableAutoRange(axis='x', enable=False)
        self._iq_time_i = self._pw_iq_time.plot(pen=pg.mkPen(PALETTE["cyan"], width=2), name="I (Real)")
        self._iq_time_q = self._pw_iq_time.plot(pen=pg.mkPen(PALETTE["orange"], width=2), name="Q (Imaginary)")
        iq_lay.addWidget(self._pw_iq_time, stretch=2)

        vitals_grp = QGroupBox("Vitals Assessment")
        v_lay = QVBoxLayout(vitals_grp)
        self._lbl_vitals = QLabel("Waiting...")
        self._lbl_vitals.setStyleSheet(f"font-size: 10pt; font-family: monospace; font-weight: bold; color: {PALETTE['accent']};")
        v_lay.addWidget(self._lbl_vitals, alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        v_lay.addStretch(1)
        iq_lay.addWidget(vitals_grp, stretch=1)

        self._tabs.addTab(iq_tab, "  I/Q Engine Trace  ")

        # Tab 2: SCR chart
        self._pw_scr    = self.make_plot_widget(ylabel="SCR (dB)", xlabel="Time (s)")
        self._pw_scr.setTitle(f"Signal-to-Clutter Ratio (rolling {self._HIST_SECS} s)")
        n_scr           = int(self._HIST_SECS * self._frame_rate)
        self._scr_buf   = np.zeros(n_scr)
        self._scr_time  = np.linspace(-self._HIST_SECS, 0, n_scr)
        self._scr_curve = self._pw_scr.plot(
            self._scr_time, self._scr_buf,
            pen=pg.mkPen(PALETTE["cyan"], width=2))
        ref_line = pg.InfiniteLine(
            pos=0, angle=0,
            pen=pg.mkPen(PALETTE["subtext"], width=1, style=Qt.PenStyle.DashLine),
            label="0 dB", labelOpts={"color": PALETTE["subtext"], "position": 0.95})
        self._pw_scr.addItem(ref_line)
        self._tabs.addTab(self._pw_scr, "  SCR History  ")
        
        bottom_row.addWidget(self._tabs, stretch=4)

        # Warmup ring
        wu_grp  = QGroupBox("Warmup / Calibration")
        wu_lay  = QVBoxLayout(wu_grp)
        wu_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        warmup_total = int(self._app_cfg.preprocessing.warmup_frames)
        self._warmup_ring = WarmupRing(total=warmup_total)
        wu_lay.addWidget(self._warmup_ring, alignment=Qt.AlignmentFlag.AlignCenter)
        sub_col = PALETTE["subtext"]
        alpha_cfg_val = self._app_cfg.preprocessing.clutter_ema_alpha
        wu_lay.addWidget(QLabel(
            f"<small style='color:{sub_col}'>"
            f"α warmup = 0.30<br>"
            f"α global = {alpha_cfg_val:.4f}<br>"
            f"α protected ≈ 0.001</small>"
        ), alignment=Qt.AlignmentFlag.AlignCenter)
        bottom_row.addWidget(wu_grp, stretch=1)

        # Diagnostics
        diag_grp  = QGroupBox("Diagnostics")
        diag_lay  = QGridLayout(diag_grp)
        self._lbl_frame    = self._mk("Frame")
        self._lbl_warmup   = self._mk("Warmup")
        self._lbl_alpha    = self._mk("Global α (EMA)")
        self._lbl_ghost    = self._mk("Ghost Trap MASK")
        self._lbl_raw_peak = self._mk("Raw Peak Mag")
        self._lbl_dyn_peak = self._mk("Dyn Peak Mag")
        self._lbl_scr      = self._mk("Current SCR")
        self._lbl_peak_bin = self._mk("Peak Bin")
        self._lbl_peak_m   = self._mk("Peak Range (m)")
        for i, (k, v) in enumerate([
            self._lbl_frame, self._lbl_warmup, self._lbl_alpha, self._lbl_ghost,
            self._lbl_raw_peak, self._lbl_dyn_peak, self._lbl_scr,
            self._lbl_peak_bin, self._lbl_peak_m,
        ]):
            diag_lay.addWidget(k, i // 2, (i % 2) * 2)
            diag_lay.addWidget(v, i // 2, (i % 2) * 2 + 1)
        bottom_row.addWidget(diag_grp, stretch=2)

        outer.addLayout(bottom_row, stretch=2)

    def _mk(self, label: str):
        k = QLabel(label + ":")
        k.setStyleSheet(f"color:{PALETTE['subtext']};font-size:14px;")
        v = QLabel("--")
        v.setStyleSheet(
            f"font-family:'Courier New',monospace;color:{PALETTE['cyan']};font-size:16px;font-weight:bold;")
        return k, v

    # ── render ────────────────────────────────────────────────────────────────

    def render(self, output: EngineOutput) -> None:
        diag = output.diagnostics
        pp   = self.engine.preprocessor
        n    = self._num_bins

        # ── Cutter map L1 per bin (sum over antennas of |complex|) ────────────
        clutter_l1 = np.sum(np.abs(pp.clutter_map._map), axis=1).astype(np.float32)

        # ── Dynamic magnitude (from candidates) ───────────────────────────────
        dyn_mag = np.zeros(n, dtype=np.float32)
        if output.candidates:
            for c in output.candidates:
                if 0 <= c.bin_index < n:
                    dyn_mag[c.bin_index] = max(dyn_mag[c.bin_index], c.magnitude)

        raw_mag = dyn_mag + clutter_l1   # approximate raw signal

        # ── Profile curves ────────────────────────────────────────────────────
        bins = np.arange(n)
        self._raw_curve.setData(bins, raw_mag)
        self._dyn_curve.setData(bins, dyn_mag)

        # ── Dynamic Limits (Auto-scaling) ─────────────────────────────────────
        if not hasattr(self, "_clutter_max_limit"): self._clutter_max_limit = 1000.0
        if not hasattr(self, "_dyn_max_limit"): self._dyn_max_limit = 1000.0
        
        c_max = float(clutter_l1.max())
        d_max = float(dyn_mag.max())
        
        # Grow fast, decay slow, but maintain minimum floors so we don't zoom into empty-room noise
        self._clutter_max_limit = max(c_max, self._clutter_max_limit * 0.99, 1000.0)
        self._dyn_max_limit = max(d_max, self._dyn_max_limit * 0.99, 500.0)

        # Determine how many temporal frames to shift by matching the engine logic
        k = getattr(self, "_frames_passed_since_render", 1)
        k = max(1, min(k, self._clutter_buf.shape[0]))  # Ensure safe clamping

        # ── Update buffers & heatmaps ──────────────────────────────────────────
        self._clutter_buf = np.roll(self._clutter_buf, -k, axis=0)
        self._clutter_buf[-k:] = clutter_l1
        self._clutter_img.setImage(self._clutter_buf, autoLevels=False, levels=(0, max(self._clutter_max_limit, 10.0)))

        # ── Dynamic heatmap ──
        self._dyn_map_buf = np.roll(self._dyn_map_buf, -k, axis=0)
        self._dyn_map_buf[-k:] = dyn_mag
        self._dyn_map_img.setImage(self._dyn_map_buf, autoLevels=False,
                                   levels=(0, max(self._dyn_max_limit, 10.0)))

        # ── α-activity heatmap ──
        real_alpha = diag.get("clutter_alpha")
        if real_alpha is not None:
            # Scale it relative to the global alpha so the visual makes sense 
            # (Global = Bright, Protected = Black)
            global_val = self._engine_cfg.preprocessing.clutter_ema_alpha
            alpha_viz = (real_alpha / max(global_val, 1e-6)).astype(np.float32)
            alpha_viz = np.clip(alpha_viz, 0.0, 1.0)
        else:
            alpha_viz = np.zeros(n, dtype=np.float32)

        self._alpha_buf = np.roll(self._alpha_buf, -k, axis=0)
        self._alpha_buf[-k:] = alpha_viz
        self._alpha_img.setImage(self._alpha_buf, autoLevels=False)

        # ── I/Q Data Trace ────────────────────────────────────────────────────
        hist = output.spectral_history
        if hist is not None and output.tracked_target and output.tracked_target.valid:
            try:
                t_bin = output.tracked_target.bin_index
                iq_buf = hist.get_bin_history(t_bin)
                if iq_buf is not None and len(iq_buf) > 0:
                    # Clamp to last 20 seconds
                    _IQ_SECS = 20
                    max_iq_frames = int(_IQ_SECS * self._frame_rate)
                    # iq_buf shape is (num_antennas, num_frames)
                    iq_c = iq_buf[0, -max_iq_frames:]  # Antenna 0, last 20s
                    iq_c = iq_c - np.mean(iq_c)  # Zero-mean to center orbital
                    r, i = np.real(iq_c), np.imag(iq_c)

                    # Parametric phase-plane (I vs Q)
                    self._iq_scatter.setData(r, i)
                    self._iq_line.setData(r, i)

                    # scale tightly 1:1
                    max_amp = max(np.max(np.abs(r)), np.max(np.abs(i))) * 1.2
                    if np.isnan(max_amp) or max_amp < 1e-3: 
                        max_amp = 1e-3
                    self._pw_iq.setXRange(-max_amp, max_amp, padding=0)
                    self._pw_iq.setYRange(-max_amp, max_amp, padding=0)

                    # Time-domain I and Q (x-axis: last 20 sec window)
                    n_frames = len(iq_c)
                    time_idx = np.linspace(-n_frames / self._frame_rate, 0, n_frames)
                    if time_idx[0] < -20:
                        # Safety to ensure we don't draw outside -20
                        time_idx = np.clip(time_idx, -20.0, 0.0)
                    self._iq_time_i.setData(time_idx, r)
                    self._iq_time_q.setData(time_idx, i)
            except Exception as e:
                pass
        else:
            self._iq_scatter.setData([], [])
            self._iq_line.setData([], [])
            self._iq_time_i.setData([], [])
            self._iq_time_q.setData([], [])

        # Vitals display
        v_text = "No track\n(Standing By)"
        if output.tracked_target and output.tracked_target.valid:
            vital = output.diagnostics.get("tracked_vital")
            if vital:
                v_text = (
                    f"Micro-State:  {vital.micro_state.name}\n"
                    f"Aliveness:    {vital.aliveness_score*100:4.1f}%\n"
                    f"Phase Var:    {vital.phase_variance:4.2f} rad\n"
                    f"Phase PtP:    {vital.phase_ptp:4.1f}°\n"
                    f"Displace:     {vital.displacement_mm:4.1f}mm\n"
                    f"Vital Mult:   {vital.vital_multiplier:4.2f}x"
                )
            else:
                v_text = "No track\n(Dropped Peak)"
        self._lbl_vitals.setText(v_text)

        # ── SCR ───────────────────────────────────────────────────────────────
        dyn_peak     = float(dyn_mag.max())
        clutter_peak = float(clutter_l1.max())
        if clutter_peak > 0 and dyn_peak > 0:
            scr_db = 20 * math.log10(dyn_peak / max(clutter_peak, 1e-9))
        else:
            scr_db = 0.0
        self._scr_buf = np.roll(self._scr_buf, -1)
        self._scr_buf[-1] = scr_db
        self._scr_curve.setData(self._scr_time, self._scr_buf)

        # ── Warmup ────────────────────────────────────────────────────────────
        warmup_active    = diag.get("warmup_active", False)
        warmup_remaining = diag.get("warmup_remaining", 0)
        self._warmup_ring.set_value(
            warmup_remaining if warmup_active else None,
            active=warmup_active,
        )

        # ── Diagnostics ───────────────────────────────────────────────────────
        peak_bin = int(np.argmax(dyn_mag)) if dyn_peak > 0 else 0
        self._lbl_frame[1].setText(str(output.frame_index))
        self._lbl_warmup[1].setText(
            f"ACTIVE ({warmup_remaining} left)" if warmup_active else "Complete ✓")
        
        is_ghost = diag.get("is_static_ghost", False)
        
        # Check if alpha actually contains a protection dip
        is_protected = False
        if real_alpha is not None and real_alpha.min() < (self._app_cfg.preprocessing.clutter_ema_alpha * 0.1):
            is_protected = True
            
        if is_ghost:
            self._lbl_ghost[1].setText("DROPPED (Dead/Ghost)")
            self._lbl_ghost[1].setStyleSheet(f"font-family:'Courier New',monospace;color:{PALETTE['warn']};font-size:16px;font-weight:bold;")
        elif is_protected:
            self._lbl_ghost[1].setText("ACTIVE (Protected)")
            self._lbl_ghost[1].setStyleSheet(f"font-family:'Courier New',monospace;color:{PALETTE['ok']};font-size:16px;font-weight:bold;")
        else:
            self._lbl_ghost[1].setText("STANDBY (Empty)")
            self._lbl_ghost[1].setStyleSheet(f"font-family:'Courier New',monospace;color:{PALETTE['subtext']};font-size:16px;")
            
        alpha_cfg = self._app_cfg.preprocessing.clutter_ema_alpha
        self._lbl_alpha[1].setText(f"{alpha_cfg:.4f}")
        self._lbl_raw_peak[1].setText(f"{float(raw_mag.max()):.1f}")
        self._lbl_dyn_peak[1].setText(f"{dyn_peak:.1f}")
        self._lbl_scr[1].setText(f"{scr_db:+.1f} dB")
        self._lbl_peak_bin[1].setText(str(peak_bin))
        self._lbl_peak_m[1].setText(f"{peak_bin * self._range_res:.2f} m")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = PreprocessorDebug()
    win.show()
    sys.exit(app.exec())
