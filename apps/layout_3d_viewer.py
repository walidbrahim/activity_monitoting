"""Standalone 3D layout viewer for detection candidate validation.

Shows:
- Room/layout zones as wireframe boxes.
- All detection candidates (valid/rejected with different colors).
- Selected candidate (best valid candidate).
- Tracked smoothed point from the engine output.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import math
import queue
import time
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget

# Ensure project root is importable when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep matplotlib cache writable inside workspace environments.
MPL_CACHE = Path(".mplcache")
MPL_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE.resolve()))

import matplotlib
matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from config import ConfigFactory, load_profile
from libs.controllers.radarController import RadarController
from radar_engine.config.engine import EngineConfig
from radar_engine.core.models import EngineOutput
from radar_engine.orchestration.pipelines import BedMonitoringEngine


def _profile_stack() -> list[str]:
    base_profile = "profiles/base.yaml"
    selected = os.getenv("APP_PROFILE", "base").strip()
    if selected in ("", "base", "base.yaml", base_profile):
        return [base_profile]
    overlay = selected if selected.endswith(".yaml") else f"{selected}.yaml"
    if "/" not in overlay:
        overlay = f"profiles/{overlay}"
    return [base_profile, overlay]


def _box_edges(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float):
    p000 = (x0, y0, z0)
    p001 = (x0, y0, z1)
    p010 = (x0, y1, z0)
    p011 = (x0, y1, z1)
    p100 = (x1, y0, z0)
    p101 = (x1, y0, z1)
    p110 = (x1, y1, z0)
    p111 = (x1, y1, z1)
    return [
        (p000, p100), (p000, p010), (p000, p001),
        (p111, p101), (p111, p110), (p111, p011),
        (p001, p101), (p001, p011),
        (p010, p110), (p010, p011),
        (p100, p110), (p100, p101),
    ]


class Layout3DViewer(QMainWindow):
    def __init__(self, app_cfg):
        super().__init__()
        self._cfg = app_cfg
        self._render_hz = float(os.getenv("VIEWER_RENDER_HZ", "5.0"))
        self._label_top_k = int(os.getenv("VIEWER_LABEL_TOP_K", "4"))
        self._last_render_ts = 0.0
        self.setWindowTitle("Radar 3D Layout & Candidate Viewer")
        self.resize(1180, 780)

        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        self._status = QLabel("Waiting for radar frames...", self)
        self._status.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self._status)

        self._fig = Figure(figsize=(12, 7))
        self._canvas = FigureCanvas(self._fig)
        
        # Split layout: 3 quarters for 3D view, 1 quarter for Power bar chart
        gs = self._fig.add_gridspec(1, 4)
        self._ax = self._fig.add_subplot(gs[0, :3], projection="3d")
        self._ax_bar = self._fig.add_subplot(gs[0, 3])
        
        layout.addWidget(self._canvas)

        self._draw_static_scene()

    def _draw_static_scene(self) -> None:
        ax = self._ax
        ax.clear()

        room = self._cfg.layout.get("Room", {})
        room_x = room.get("x", [0.0, 2.0])
        room_y = room.get("y", [0.0, 3.0])
        room_z = room.get("z", [0.0, 2.7])
        ax.set_xlim(float(room_x[0]), float(room_x[1]))
        ax.set_ylim(float(room_y[0]), float(room_y[1]))
        ax.set_zlim(float(room_z[0]), float(room_z[1]))
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("Layout + Candidates + Selected Dot")

        for zone_name, zone in self._cfg.layout.items():
            if not isinstance(zone, dict):
                continue
            x = zone.get("x")
            y = zone.get("y")
            z = zone.get("z")
            if not (isinstance(x, list) and isinstance(y, list) and isinstance(z, list)):
                continue

            zone_type = str(zone.get("type", "")).lower()
            color = {
                "boundary": "#7f8c8d",
                "monitor": "#1f77b4",
                "ignore": "#d62728",
            }.get(zone_type, "#95a5a6")

            for a, b in _box_edges(float(x[0]), float(x[1]), float(y[0]), float(y[1]), float(z[0]), float(z[1])):
                ax.plot(
                    [a[0], b[0]],
                    [a[1], b[1]],
                    [a[2], b[2]],
                    color=color,
                    linewidth=1.5 if zone_type == "monitor" else 1.0,
                    alpha=0.85,
                )

            cx = 0.5 * (float(x[0]) + float(x[1]))
            cy = 0.5 * (float(y[0]) + float(y[1]))
            cz = float(z[1])
            ax.text(cx, cy, cz, zone_name, color=color, fontsize=9)

        # Radar pose marker + a short yaw direction line
        pose_zone = getattr(self._cfg.app, "default_radar_pose", "Room")
        radar_pose = self._cfg.layout.get(pose_zone, {}).get("radar_pose") or self._cfg.layout.get("Room", {}).get("radar_pose")
        if isinstance(radar_pose, dict):
            rx = float(radar_pose.get("x", 0.0))
            ry = float(radar_pose.get("y", 0.0))
            rz = float(radar_pose.get("z", 1.0))
            yaw = float(radar_pose.get("yaw_deg", 0.0))
            pitch = float(radar_pose.get("pitch_deg", 0.0))
            yaw_rad = math.radians(yaw)
            pitch_rad = math.radians(pitch)

            import numpy as np

            # Engine-identical rotation matrices
            Ry = np.array([
                [ math.cos(yaw_rad), -math.sin(yaw_rad), 0],
                [ math.sin(yaw_rad),  math.cos(yaw_rad), 0],
                [                 0,                  0, 1],
            ])
            Rp = np.array([
                [1,                   0,                    0],
                [0, math.cos(pitch_rad), -math.sin(pitch_rad)],
                [0, math.sin(pitch_rad),  math.cos(pitch_rad)],
            ])
            R_mat = np.dot(Ry, Rp)

            # Central direction line (length 0.4m)
            fw = np.dot(R_mat, np.array([0.0, 0.4, 0.0]))
            ax.scatter([rx], [ry], [rz], c="#f39c12", s=60, marker="^", label="Radar")
            ax.plot([rx, rx + fw[0]], [ry, ry + fw[1]], [rz, rz + fw[2]], c="#f39c12", linewidth=2.0)

            # ── Draw 3D FoV Pyramid ──
            az_fov = float(radar_pose.get("fov_deg", 120.0))
            el_fov = float(radar_pose.get("elev_fov_deg", 45.0))
            r_len = 2.5 # Project 2.5 meters out

            h_az = math.radians(az_fov / 2.0)
            h_el = math.radians(el_fov / 2.0)

            angles = [
                ( h_az,  h_el),
                ( h_az, -h_el),
                (-h_az, -h_el),
                (-h_az,  h_el),
            ]

            base_pts = []
            for (az, el) in angles:
                # Local coordinate projection identical to localization.py
                Pr = np.array([
                    r_len * math.sin(az) * math.cos(el),
                    r_len * math.cos(az) * math.cos(el),
                    r_len * math.sin(el)
                ])
                Pb = np.dot(R_mat, Pr)
                base_pts.append((rx + Pb[0], ry + Pb[1], rz + Pb[2]))

            # Ray lines from Radar origin to base corners
            for p in base_pts:
                ax.plot([rx, p[0]], [ry, p[1]], [rz, p[2]], c="#f39c12", linewidth=1.0, alpha=0.3, linestyle="--")
            
            # Connect the base corners to form the outer rectangular face
            for i in range(4):
                pa, pb = base_pts[i], base_pts[(i+1)%4]
                ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]], c="#f39c12", linewidth=1.0, alpha=0.3, linestyle="-")

        ax.legend(loc="upper left")
        self._canvas.draw_idle()

    def on_data(self, output: EngineOutput, frames: int) -> None:
        candidates = output.candidates or []
        valid = [c for c in candidates if c.valid]
        rejected = [c for c in candidates if not c.valid]
        selected = max(valid, key=lambda c: c.magnitude) if valid else None
        tracked = output.tracked_target

        activity = output.activity
        zone = activity.zone if activity else "Unknown"
        occupancy = activity.occupancy.value if activity else "unknown"
        motion = float(activity.motion_score) if activity else 0.0
        selected_mag = float(selected.magnitude) if selected is not None else 0.0
        self._status.setText(
            f"Occupancy: {occupancy} | Zone: {zone} | Motion score: {motion:.2f} | "
            f"Cands: {len(candidates)} | Best mag: {selected_mag:.1f} | Frame batch: {frames}"
        )

        # Throttle expensive 3D redraw to keep UI responsive.
        now = time.monotonic()
        if now - self._last_render_ts < (1.0 / max(self._render_hz, 0.5)):
            return
        self._last_render_ts = now

        self._draw_static_scene()
        ax = self._ax

        if rejected:
            ax.scatter(
                [float(c.x_m) for c in rejected],
                [float(c.y_m) for c in rejected],
                [float(c.z_m) for c in rejected],
                c="#95a5a6",
                s=18,
                alpha=0.40,
                marker="o",
                label="Rejected candidates",
            )

        if valid:
            ax.scatter(
                [float(c.x_m) for c in valid],
                [float(c.y_m) for c in valid],
                [float(c.z_m) for c in valid],
                c="#2ecc71",
                s=28,
                alpha=0.9,
                marker="o",
                label="Valid candidates",
            )

        # Show coordinate labels only for top-K magnitudes to prevent UI lag.
        top_valid = sorted(valid, key=lambda c: c.magnitude, reverse=True)[: self._label_top_k]
        top_rejected = sorted(rejected, key=lambda c: c.magnitude, reverse=True)[: self._label_top_k]

        for c in top_rejected:
            ax.text(
                float(c.x_m),
                float(c.y_m),
                float(c.z_m) + 0.02,
                f"({c.x_m:.2f}, {c.y_m:.2f}, {c.z_m:.2f})",
                color="#7f8c8d",
                fontsize=7,
            )

        for c in top_valid:
            ax.text(
                float(c.x_m),
                float(c.y_m),
                float(c.z_m) + 0.02,
                f"({c.x_m:.2f}, {c.y_m:.2f}, {c.z_m:.2f})",
                color="#1f7a4c",
                fontsize=8,
            )

        if selected is not None:
            ax.scatter(
                [float(selected.x_m)],
                [float(selected.y_m)],
                [float(selected.z_m)],
                c="#f1c40f",
                s=110,
                marker="*",
                edgecolors="#7f6a00",
                linewidths=0.9,
                label="Selected candidate",
            )
            ax.text(
                float(selected.x_m),
                float(selected.y_m),
                float(selected.z_m) + 0.05,
                f"Selected ({selected.x_m:.2f}, {selected.y_m:.2f}, {selected.z_m:.2f})",
                color="#7f6a00",
                fontsize=9,
                fontweight="bold",
            )

        if tracked is not None and tracked.valid:
            ax.scatter(
                [float(tracked.smoothed_x_m)],
                [float(tracked.smoothed_y_m)],
                [float(tracked.smoothed_z_m)],
                c="#00bfff",
                s=90,
                marker="D",
                edgecolors="#005577",
                linewidths=0.8,
                label="Tracked point",
            )

        # ── Draw dynamic CFAR vs Power bar chart ──
        self._ax_bar.clear()
        best_overall = max(candidates, key=lambda c: c.magnitude, default=None)
        if best_overall:
            labels = ["Peak SNR", "CFAR Thresh"]
            mag = float(best_overall.magnitude)
            thr = float(best_overall.cfar_threshold)
            colors = ["#2ecc71" if best_overall.valid else "#e74c3c", "#95a5a6"]
            bars = self._ax_bar.bar(labels, [mag, thr], color=colors, width=0.6)
            self._ax_bar.set_title(f"Loudest Peak (Bin {best_overall.bin_index})\n{best_overall.range_m:.2f}m Away", fontsize=10)
            self._ax_bar.set_ylim(0, max(max(mag, thr) * 1.25, 100))
            
            # Label bars with integers
            for b in bars:
                h = b.get_height()
                self._ax_bar.text(b.get_x() + b.get_width() / 2, h + (max(mag, thr)*0.02), f"{int(h)}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        else:
            self._ax_bar.set_title("No Radar Echoes", fontsize=10)
            self._ax_bar.set_ylim(0, 100)
            self._ax_bar.set_xticks([])

        ax.legend(loc="upper left")
        self._canvas.draw_idle()


class EngineTapThread(QThread):
    """Minimal thread: radar queue -> BedMonitoringEngine -> typed output signal."""

    data_ready = pyqtSignal(object, int)  # (EngineOutput, frames_processed)

    def __init__(self, pt_fft_q, cfg: EngineConfig, default_pose: dict | None = None, parent=None):
        super().__init__(parent)
        self._pt_fft_q = pt_fft_q
        self._engine = BedMonitoringEngine(cfg=cfg)
        self._running = True
        if isinstance(default_pose, dict):
            self._engine.update_radar_pose(default_pose)
            self._engine.reset()

    def run(self) -> None:
        while self._running:
            try:
                fft_frame = self._pt_fft_q.get(timeout=0.1)
                output = self._engine.process_frame(fft_frame)
                frames_processed = 1

                while not self._pt_fft_q.empty():
                    try:
                        fft_frame = self._pt_fft_q.get_nowait()
                        output = self._engine.process_frame(fft_frame)
                        frames_processed += 1
                    except queue.Empty:
                        break

                self.data_ready.emit(output, frames_processed)
            except queue.Empty:
                continue

    def stop(self) -> None:
        self._running = False
        self.wait()


def run_app() -> None:
    print("Starting 3D layout candidate viewer ...")
    app = QApplication(sys.argv)

    profile_stack = _profile_stack()
    app_cfg = load_profile(*profile_stack)
    eng_cfg = ConfigFactory.engine_config(app_cfg)
    print(f"Successfully loaded configuration stack: {profile_stack}")

    state_q = multiprocessing.Queue()
    pt_fft_q = multiprocessing.Queue()

    radar_process = RadarController(state_q=state_q, pt_fft_q=pt_fft_q)
    radar_process.start()

    default_zone = getattr(app_cfg.app, "default_radar_pose", "Room")
    default_pose = app_cfg.layout.get(default_zone, {}).get("radar_pose")
    engine_tap = EngineTapThread(
        pt_fft_q=pt_fft_q,
        cfg=eng_cfg,
        default_pose=default_pose if isinstance(default_pose, dict) else None,
    )

    window = Layout3DViewer(app_cfg)
    engine_tap.data_ready.connect(window.on_data)
    window.show()
    engine_tap.start()

    exit_code = app.exec()

    print("Cleaning up processes ...")
    engine_tap.stop()
    if radar_process.is_alive():
        radar_process.terminate()
        radar_process.join()

    sys.exit(exit_code)


if __name__ == "__main__":
    run_app()
