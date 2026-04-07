"""
apps/debug/02_detection_debug.py
=================================
Debug GUI for TargetDetector + CFAR.

Panels:
  ┌─────────────────────────────────────┬────────────────────────────┐
  │  Dynamic profile + CFAR envelope    │  X-Y candidate scatter     │
  │  green fill = above threshold       │  (colour by zone / valid)  │
  │  stem markers at detected peaks     │                            │
  ├─────────────────────────────────────┼────────────────────────────┤
  │  Candidate magnitude history raster │  Reject breakdown donut    │
  │  (range metres × time seconds)      │  + top-5 magnitude bars    │
  └─────────────────────────────────────┴────────────────────────────┘

Run:
    APP_PROFILE=lab python apps/debug/02_detection_debug.py
"""
from __future__ import annotations
import sys, os
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout,
    QGridLayout, QGroupBox, QLabel,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPen, QBrush
import pyqtgraph as pg

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from apps.debug.debug_base import DebugBase, PALETTE, apply_plot_defaults  # noqa: E402
from radar_engine.core.models import EngineOutput                           # noqa: E402
from radar_engine.detection.cfar import compute_cfar_threshold              # noqa: E402



# ── Main ──────────────────────────────────────────────────────────────────────

def _reject_badge(count: int, label: str, colour: str) -> QLabel:
    """Coloured count badge for the diagnostics strip."""
    lbl = QLabel(f"{count}  {label}")
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lbl.setStyleSheet(
        f"background:{colour}22;border:1px solid {colour}80;"
        f"border-radius:6px;padding:4px 10px;"
        f"color:{colour};font-size:13px;font-weight:bold;"
    )
    return lbl

class DetectionDebug(DebugBase):
    TITLE    = "🎯 Detection & CFAR — Debug"
    WINDOW_W = 1700
    WINDOW_H = 1000

    _HIST_SECS = 20      # history raster depth in seconds

    def _build_ui(self, central: QWidget) -> None:
        grid = QGridLayout(central)
        grid.setContentsMargins(10, 10, 10, 10)
        grid.setSpacing(10)

        range_res   = self._app_cfg.hardware.range_resolution  # m/bin
        min_range_m = self._app_cfg.detection.min_search_range_m
        min_bin     = int(min_range_m / range_res)
        n           = self._num_bins
        self._range_res   = range_res
        self._min_range_m = min_range_m

        # ── Row 0 left: CFAR profile plot ─────────────────────────────────────
        profile_grp = QGroupBox(
            "Dynamic Magnitude Profile + CFAR Threshold  "
            "[ — signal │ ╌ CFAR │ ▪ min-range gate │ ● detected peaks ]"
        )
        pl = QVBoxLayout(profile_grp)
        self._pw_profile = self.make_plot_widget(
            xlabel=f"Range Bin  (1 bin = {range_res:.2f} m)",
            ylabel="Magnitude"
        )
        # Range-in-metres secondary tick labels via custom axis
        ax = self._pw_profile.getPlotItem().getAxis("bottom")
        tick_bins  = list(range(0, n + 1, 5))
        tick_pairs = [(b, f"{b}\n({b * range_res:.1f}m)") for b in tick_bins]
        ax.setTicks([tick_pairs])

        # Curves
        self._dyn_curve  = self._pw_profile.plot(
            pen=pg.mkPen(PALETTE["text"], width=2), name="Dynamic mag")
        self._cfar_curve = self._pw_profile.plot(
            pen=pg.mkPen(PALETTE["warn"], width=1.5,
                         style=Qt.PenStyle.DashLine), name="CFAR threshold")

        # Green fill: area where magnitude exceeds CFAR (detection zone)
        self._fill_detect = pg.FillBetweenItem(
            self._dyn_curve, self._cfar_curve,
            brush=pg.mkBrush(PALETTE["ok"] + "40"),   # translucent green
        )
        self._pw_profile.addItem(self._fill_detect)

        # Vertical gate line for min_search_range
        self._gate_line = pg.InfiniteLine(
            pos=min_bin, angle=90,
            pen=pg.mkPen(PALETTE["subtext"], width=1.5, style=Qt.PenStyle.DotLine),
            label=f"min range\n{min_range_m:.2f} m",
            labelOpts={"color": PALETTE["subtext"], "position": 0.85},
        )
        self._pw_profile.addItem(self._gate_line)

        # Stem scatter for detected peaks
        self._peak_scatter = pg.ScatterPlotItem(
            symbol='o', size=10, brush=pg.mkBrush(PALETTE["cyan"]),
            pen=pg.mkPen(PALETTE["cyan"], width=1),
        )
        self._pw_profile.addItem(self._peak_scatter)

        # Rejected peaks (below CFAR)
        self._reject_scatter = pg.ScatterPlotItem(
            symbol='x', size=9,
            pen=pg.mkPen(PALETTE["alert"], width=2),
        )
        self._pw_profile.addItem(self._reject_scatter)

        leg = self._pw_profile.addLegend(offset=(10, 10))
        leg.addItem(self._dyn_curve,     "Dynamic mag")
        leg.addItem(self._cfar_curve,    "CFAR threshold")
        leg.addItem(self._peak_scatter,  "Detected peaks")
        leg.addItem(self._reject_scatter,"Rejected (below CFAR)")

        pl.addWidget(self._pw_profile)
        grid.addWidget(profile_grp, 0, 0, 1, 2)

        # ── Row 0 right: X-Y scatter ──────────────────────────────────────────
        scatter_grp = QGroupBox(
            "Candidate X-Y Map  [ ● valid (size ∝ magnitude) │ ✕ rejected ]"
        )
        sl = QVBoxLayout(scatter_grp)
        self._pw_scatter = pg.PlotWidget()
        self._pw_scatter.setBackground(PALETTE["panel"])
        self._pw_scatter.setAspectLocked(True)
        apply_plot_defaults(self._pw_scatter.getPlotItem(),
                            xlabel="X (m)", ylabel="Y (m)")
        room = self._app_cfg.layout.get("Room", {})
        rx = room.get("x", [0, 2]); ry = room.get("y", [0, 3])
        self._pw_scatter.setXRange(rx[0] - 0.3, rx[1] + 0.3)
        self._pw_scatter.setYRange(ry[0] - 0.3, ry[1] + 0.3)
        room_poly = np.array([
            [rx[0], ry[0]], [rx[1], ry[0]],
            [rx[1], ry[1]], [rx[0], ry[1]], [rx[0], ry[0]]
        ])
        self._pw_scatter.plot(room_poly[:, 0], room_poly[:, 1],
                              pen=pg.mkPen("#CBD5E1", width=2))
        # Zone overlays
        for zname, zval in self._app_cfg.layout.items():
            if not isinstance(zval, dict):
                continue
            ztype = zval.get("type", "")
            if ztype == "monitor":
                zx = zval.get("x", [0, 1]); zy = zval.get("y", [0, 1])
                ri = pg.QtWidgets.QGraphicsRectItem(zx[0], zy[0],
                                                    zx[1]-zx[0], zy[1]-zy[0])
                ri.setBrush(QBrush(QColor(PALETTE["ok"] + "20")))
                ri.setPen(QPen(QColor(PALETTE["ok"] + "60"), 1))
                self._pw_scatter.addItem(ri)
                lbl = pg.TextItem(zname, color=PALETTE["subtext"], anchor=(0, 1))
                lbl.setPos(zx[0], zy[1])
                self._pw_scatter.addItem(lbl)

        self._scatter_valid   = pg.ScatterPlotItem(
            symbol='o', brush=pg.mkBrush(PALETTE["ok"]))
        self._scatter_invalid = pg.ScatterPlotItem(
            symbol='x', size=10, pen=pg.mkPen(PALETTE["alert"], width=2))
        self._pw_scatter.addItem(self._scatter_valid)
        self._pw_scatter.addItem(self._scatter_invalid)
        sl.addWidget(self._pw_scatter)
        grid.addWidget(scatter_grp, 0, 2, 2, 1)

        # ── Row 1 left: history raster with proper axes ───────────────────────
        hist_rows = int(self._HIST_SECS * self._frame_rate)
        self._hist_rows = hist_rows
        self._hist_data = np.zeros((hist_rows, n), dtype=np.float32)

        hist_grp = QGroupBox(
            f"Magnitude History — Y: Range (m), X: Time (last {self._HIST_SECS}s, newest at right)"
        )
        hl = QVBoxLayout(hist_grp)

        # Use PlotWidget + ImageItem to get real axes
        self._pw_hist = pg.PlotWidget()
        self._pw_hist.setBackground(PALETTE["panel"])
        apply_plot_defaults(self._pw_hist.getPlotItem(),
                            xlabel="Time (s ago)", ylabel="Range (m)")

        self._hist_img_item = pg.ImageItem()
        self._hist_img_item.setColorMap(pg.colormap.get("viridis"))
        self._pw_hist.addItem(self._hist_img_item)

        # Anchor: image spans [–hist_secs → 0] on X, [0 → n*range_res] on Y
        # ImageItem transform: each pixel → (dt_per_row, range_res)
        dt = 1.0 / self._frame_rate
        from PyQt6.QtGui import QTransform
        tr = QTransform()
        tr.translate(-self._HIST_SECS, 0.0)       # X origin = –hist_secs
        tr.scale(dt, range_res)                    # each column = dt s, each row = range_res m
        self._hist_img_item.setTransform(tr)

        # Colourbar
        cbar = pg.ColorBarItem(
            values=(0, 3000),
            colorMap=pg.colormap.get("viridis"),
            label="Magnitude",
        )
        cbar.setImageItem(self._hist_img_item, insert_in=self._pw_hist.getPlotItem())

        # Min-range horizontal line on history plot
        self._hist_range_line = pg.InfiniteLine(
            pos=min_range_m, angle=0,
            pen=pg.mkPen(PALETTE["subtext"], width=1, style=Qt.PenStyle.DotLine),
            label=f"min range {min_range_m:.2f} m",
            labelOpts={"color": PALETTE["subtext"], "position": 0.05},
        )
        self._pw_hist.addItem(self._hist_range_line)

        hl.addWidget(self._pw_hist)
        grid.addWidget(hist_grp, 1, 0)

        # ── Row 1 centre: top-5 bar ───────────────────────────────────────────
        bar_grp = QGroupBox("Top-5 Detected Candidates (by magnitude)")
        bl = QVBoxLayout(bar_grp)
        self._pw_bar = self.make_plot_widget(
            xlabel="Candidate rank", ylabel="Magnitude")
        self._bar_ax = self._pw_bar.getPlotItem().getAxis("bottom")
        self._bar_item = pg.BarGraphItem(
            x=np.arange(5), height=np.zeros(5),
            width=0.6, brush=pg.mkBrush(PALETTE["accent"]))
        self._pw_bar.addItem(self._bar_item)
        bl.addWidget(self._pw_bar)
        grid.addWidget(bar_grp, 1, 1)

        # ── Diagnostics strip ─────────────────────────────────────────────────
        diag_grp = QGroupBox("Detection Diagnostics")
        dg = QHBoxLayout(diag_grp)

        # Reject count badges (updated each frame)
        self._badge_valid    = QLabel("0  ✓ valid")
        self._badge_cfar     = QLabel("0  ✗ below CFAR")
        self._badge_zone     = QLabel("0  ✗ out of zone")
        for lbl, col in [
            (self._badge_valid, PALETTE["ok"]),
            (self._badge_cfar,  PALETTE["warn"]),
            (self._badge_zone,  PALETTE["alert"]),
        ]:
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet(
                f"background:{col}22;border:1px solid {col}80;"
                f"border-radius:6px;padding:4px 12px;"
                f"color:{col};font-size:13px;font-weight:bold;"
            )
            dg.addWidget(lbl)

        dg.addSpacing(20)

        # Key metric labels
        self._lbl_peaks     = self._mk("Total")
        self._lbl_cfar_pass = self._mk("CFAR pass")
        self._lbl_best_rng  = self._mk("Best range")
        self._lbl_best_mag  = self._mk("Best magnitude")
        self._lbl_best_zone = self._mk("Best zone")
        self._lbl_cfar_thr  = self._mk("CFAR @ best bin")
        for k, v in [self._lbl_peaks, self._lbl_cfar_pass,
                     self._lbl_best_rng, self._lbl_best_mag,
                     self._lbl_best_zone, self._lbl_cfar_thr]:
            col_box = QVBoxLayout()
            col_box.addWidget(k, alignment=Qt.AlignmentFlag.AlignCenter)
            col_box.addWidget(v, alignment=Qt.AlignmentFlag.AlignCenter)
            dg.addLayout(col_box)

        grid.addWidget(diag_grp, 2, 0, 1, 3)

        grid.setRowStretch(0, 5)
        grid.setRowStretch(1, 5)
        grid.setRowStretch(2, 1)
        grid.setColumnStretch(0, 3)
        grid.setColumnStretch(1, 2)
        grid.setColumnStretch(2, 3)

    def _mk(self, label: str):
        k = QLabel(label + ":")
        k.setStyleSheet(f"color:{PALETTE['subtext']};font-size:11px;")
        v = QLabel("--")
        v.setStyleSheet(f"color:{PALETTE['cyan']};font-family:monospace;font-size:11px;")
        return k, v

    # ── render ────────────────────────────────────────────────────────────────

    def render(self, output: EngineOutput) -> None:
        cands = output.candidates or []
        n     = self._num_bins

        # ── Reconstruct full per-bin arrays from candidates ───────────────────
        dyn_mag  = np.zeros(n, dtype=np.float32)
        cfar_thr = np.zeros(n, dtype=np.float32)
        for c in cands:
            if 0 <= c.bin_index < n:
                dyn_mag[c.bin_index]  = max(dyn_mag[c.bin_index], c.magnitude)
                cfar_thr[c.bin_index] = c.cfar_threshold

        # For bins with no candidate, interpolate CFAR from neighbours
        # (so the dashed line is continuous rather than zero everywhere)
        known = cfar_thr > 0
        if known.any():
            bins_all = np.arange(n)
            bins_known = bins_all[known]
            vals_known = cfar_thr[known]
            cfar_full  = np.interp(bins_all, bins_known, vals_known)
        else:
            cfar_full = cfar_thr.copy()

        # ── Profile curves ────────────────────────────────────────────────────
        bin_axis = np.arange(n)
        self._dyn_curve.setData(bin_axis, dyn_mag)
        self._cfar_curve.setData(bin_axis, cfar_full)

        # ── Peak scatter markers ──────────────────────────────────────────────
        valid_cands   = [c for c in cands if c.valid]
        invalid_cands = [c for c in cands if not c.valid
                         and c.reject_reason == "below_cfar"]

        if valid_cands:
            self._peak_scatter.setData(
                [c.bin_index for c in valid_cands],
                [c.magnitude for c in valid_cands],
                size=[min(18, max(8, int(c.magnitude / 150 + 6))) for c in valid_cands],
            )
        else:
            self._peak_scatter.setData([], [])

        if invalid_cands:
            self._reject_scatter.setData(
                [c.bin_index for c in invalid_cands],
                [c.magnitude for c in invalid_cands],
            )
        else:
            self._reject_scatter.setData([], [])

        # ── History raster ────────────────────────────────────────────────────
        # Raster is (time_rows × range_bins); newest frame → last row
        # Transpose so ImageItem X = time, Y = range
        self._hist_data = np.roll(self._hist_data, -1, axis=0)
        self._hist_data[-1] = dyn_mag
        peak_val = float(dyn_mag.max())
        # _hist_data shape: (time_rows, range_bins)
        # ImageItem first axis = X (time), second axis = Y (range) — no transpose needed
        self._hist_img_item.setImage(
            self._hist_data,
            autoLevels=False,
            levels=(0, max(peak_val, 200)),
        )

        # ── X-Y scatter ───────────────────────────────────────────────────────
        valid_pts = [(c.x_m, c.y_m) for c in valid_cands]
        sizes     = [min(22, max(7, int(c.magnitude / 200 + 7))) for c in valid_cands]
        invalid_pts = [(c.x_m, c.y_m) for c in cands if not c.valid]

        self._scatter_valid.setData(
            [p[0] for p in valid_pts],
            [p[1] for p in valid_pts],
            size=sizes,
        )
        self._scatter_invalid.setData(
            [p[0] for p in invalid_pts],
            [p[1] for p in invalid_pts],
        )

        # ── Reject counts ─────────────────────────────────────────────────────
        counts = {"valid": 0, "below_cfar": 0, "out_of_zone": 0}
        for c in cands:
            if c.valid:
                counts["valid"] += 1
            elif c.reject_reason == "below_cfar":
                counts["below_cfar"] += 1
            elif c.reject_reason in ("out_of_bounds", "out_of_zone", "ignored_zone"):
                counts["out_of_zone"] += 1

        self._badge_valid.setText(f"{counts['valid']}  ✓ valid")
        self._badge_cfar.setText(f"{counts['below_cfar']}  ✗ below CFAR")
        self._badge_zone.setText(f"{counts['out_of_zone']}  ✗ out of zone")

        # ── Top-5 bar with range labels ───────────────────────────────────────
        top5 = sorted(valid_cands, key=lambda c: c.magnitude, reverse=True)[:5]
        heights = [c.magnitude for c in top5] + [0] * (5 - len(top5))
        self._bar_item.setOpts(height=heights)
        tick_labels = [(i, f"#{i+1}\n{top5[i].range_m:.2f}m" if i < len(top5) else "")
                       for i in range(5)]
        self._bar_ax.setTicks([tick_labels])

        # ── Diagnostics ───────────────────────────────────────────────────────
        best = top5[0] if top5 else None
        below = counts["below_cfar"]
        cfar_pass = len(cands) - below

        self._lbl_peaks[1].setText(str(len(cands)))
        self._lbl_cfar_pass[1].setText(str(cfar_pass))
        self._lbl_best_rng[1].setText(f"{best.range_m:.2f} m" if best else "--")
        self._lbl_best_mag[1].setText(f"{best.magnitude:.1f}" if best else "--")
        self._lbl_best_zone[1].setText(best.zone if best else "--")
        self._lbl_cfar_thr[1].setText(f"{best.cfar_threshold:.1f}" if best else "--")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = DetectionDebug()
    win.show()
    sys.exit(app.exec())
