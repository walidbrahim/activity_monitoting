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
    QGridLayout, QGroupBox, QLabel, QSizePolicy,
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
    pw.getPlotItem().setTitle(title, color=PALETTE["subtext"], size="9pt")

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

        outer.addLayout(maps_row, stretch=5)

        # ── Row 2: SCR chart + warmup ring + diagnostics ──────────────────────
        bottom_row = QHBoxLayout()

        scr_grp  = QGroupBox(f"Signal-to-Clutter Ratio — SCR (rolling {self._HIST_SECS} s)")
        scr_lay  = QVBoxLayout(scr_grp)
        self._pw_scr    = self.make_plot_widget(ylabel="SCR (dB)", xlabel="Time (s)")
        n_scr           = int(self._HIST_SECS * self._frame_rate)
        self._scr_buf   = np.zeros(n_scr)
        self._scr_time  = np.linspace(-self._HIST_SECS, 0, n_scr)
        self._scr_curve = self._pw_scr.plot(
            self._scr_time, self._scr_buf,
            pen=pg.mkPen(PALETTE["cyan"], width=2))
        # 0 dB reference
        ref_line = pg.InfiniteLine(
            pos=0, angle=0,
            pen=pg.mkPen(PALETTE["subtext"], width=1, style=Qt.PenStyle.DashLine),
            label="0 dB", labelOpts={"color": PALETTE["subtext"], "position": 0.95})
        self._pw_scr.addItem(ref_line)
        scr_lay.addWidget(self._pw_scr)
        bottom_row.addWidget(scr_grp, stretch=5)

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
        self._lbl_raw_peak = self._mk("Raw Peak Mag")
        self._lbl_dyn_peak = self._mk("Dyn Peak Mag")
        self._lbl_scr      = self._mk("Current SCR")
        self._lbl_peak_bin = self._mk("Peak Bin")
        self._lbl_peak_m   = self._mk("Peak Range (m)")
        for i, (k, v) in enumerate([
            self._lbl_frame, self._lbl_warmup, self._lbl_alpha,
            self._lbl_raw_peak, self._lbl_dyn_peak, self._lbl_scr,
            self._lbl_peak_bin, self._lbl_peak_m,
        ]):
            diag_lay.addWidget(k, i, 0); diag_lay.addWidget(v, i, 1)
        bottom_row.addWidget(diag_grp, stretch=2)

        outer.addLayout(bottom_row, stretch=2)

    def _mk(self, label: str):
        k = QLabel(label + ":")
        k.setStyleSheet(f"color:{PALETTE['subtext']};font-size:11px;")
        v = QLabel("--")
        v.setStyleSheet(
            f"font-family:'Courier New',monospace;color:{PALETTE['cyan']};font-size:11px;")
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

        # ── Update buffers & heatmaps ──────────────────────────────────────────
        self._clutter_buf = np.roll(self._clutter_buf, -1, axis=0)
        self._clutter_buf[-1] = clutter_l1
        self._clutter_img.setImage(self._clutter_buf, autoLevels=False)

        # ── Dynamic heatmap ──
        self._dyn_map_buf = np.roll(self._dyn_map_buf, -1, axis=0)
        self._dyn_map_buf[-1] = dyn_mag
        self._dyn_map_img.setImage(self._dyn_map_buf, autoLevels=False,
                                   levels=(0, max(float(dyn_mag.max()), 1000)))

        # ── α-activity heatmap ──
        if not hasattr(self, "_prev_clutter"):
            self._prev_clutter = clutter_l1.copy()
        delta = np.abs(clutter_l1 - self._prev_clutter)
        self._prev_clutter = clutter_l1.copy()
        if not hasattr(self, "_delta_max"):
            self._delta_max = 1.0
        self._delta_max = max(self._delta_max * 0.99, float(delta.max()), 1.0)
        alpha_proxy = (delta / self._delta_max).astype(np.float32)

        self._alpha_buf = np.roll(self._alpha_buf, -1, axis=0)
        self._alpha_buf[-1] = alpha_proxy
        self._alpha_img.setImage(self._alpha_buf, autoLevels=False)

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
