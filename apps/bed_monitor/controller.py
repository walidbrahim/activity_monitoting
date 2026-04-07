"""
apps.bed_monitor.controller
=============================
BedMonitorController — application facade for bed monitoring.

Wraps ``BedMonitoringEngine`` (the new typed engine) and emits the same
``data_ready(dict, dict)`` Qt signal that the existing GUI connects to.

Thread safety: ``_pipeline_lock`` guards all engine calls; Qt signals are
emitted outside the lock.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from radar_engine.config.engine import EngineConfig
from radar_engine.orchestration.pipelines import BedMonitoringEngine
from radar_engine.core.models import EngineOutput
from radar_engine.diagnostics import FrameRecorder, EventLog
from apps.common.display_policy import DisplayPolicy
from apps.bed_monitor.session_db import DbConfig, SessionDbLogger

logger = logging.getLogger(__name__)


class BedMonitorController(QThread):
    """Background processing thread backed by RadarEngine.

    Args:
        pt_fft_q:                      Queue of raw FFT frames from RadarController.
        vernier_belt_realtime_q:       Optional realtime Vernier belt data queue.
        vernier_belt_connection_q:     Optional Vernier belt connection status queue.
        parent:                        Qt parent object.
    """

    # Emits (activity_dict, respiratory_dict, frames_processed)
    data_ready = pyqtSignal(dict, dict, int)

    def __init__(
        self,
        pt_fft_q,
        vernier_belt_realtime_q=None,
        vernier_belt_connection_q=None,
        imu_queues: list = None,
        parent=None,
        cfg: EngineConfig | None = None,
        app_cfg: any = None, # AppConfig from pydantic
        belt_window_sec: float = 30.0,
        belt_rate_hz:    float = 10.0,
        recording_cfg: dict | None = None,
        db_cfg: dict | None = None,
    ) -> None:
        super().__init__(parent)
        self.pt_fft_q                  = pt_fft_q
        self.vernier_belt_realtime_q   = vernier_belt_realtime_q
        self.vernier_belt_connection_q = vernier_belt_connection_q
        self.imu_queues                = imu_queues or []
        self.app_cfg                   = app_cfg
        self.running                   = True
        self.is_aligning               = False
        self._pipeline_lock            = threading.Lock()
        self._cfg                      = cfg

        # New engine + display policy
        self._engine  = BedMonitoringEngine(cfg=cfg)
        self._policy  = DisplayPolicy()

        # Vernier Belt history — window sized from explicit constructor args
        belt_window_frames   = int(belt_window_sec * belt_rate_hz)
        self._belt_history   = np.zeros(belt_window_frames)
        self._belt_total     = 0
        self._belt_connected = False

        # Session recording (optional)
        self._record_cfg = recording_cfg or {}
        self._record_enabled = bool(self._record_cfg.get("enabled", False))
        self._save_raw_frames = bool(self._record_cfg.get("save_raw_frames", True))
        self._save_frame_records = bool(self._record_cfg.get("save_frame_records", True))
        self._save_event_log = bool(self._record_cfg.get("save_event_log", True))
        self._record_output_dir = Path(self._record_cfg.get("output_dir", "captures"))
        self._record_session_name = str(self._record_cfg.get("session_name", "")).strip()
        self._record_started_at = time.time()
        self._raw_frames: list[np.ndarray] = []
        self._frame_recorder = FrameRecorder(
            capacity=int(self._record_cfg.get("recorder_capacity", 5000))
        )
        self._event_log = EventLog(
            capacity=int(self._record_cfg.get("event_capacity", 2000))
        )
        if self._record_enabled:
            self._record_output_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Session recording enabled. Output dir: %s", self._record_output_dir)

        # Session DB logging (optional, SQLite)
        self._db_cfg = db_cfg or {}
        monitor_zones = {
            name for name, zone_cfg in (cfg.layout if cfg else {}).items()
            if isinstance(zone_cfg, dict) and str(zone_cfg.get("type", "")).lower() == "monitor"
        }
        self._session_db = SessionDbLogger(
            DbConfig(
                enabled=bool(self._db_cfg.get("enabled", True)),
                path=str(self._db_cfg.get("path", "data/monitoring.sqlite3")),
                rr_bucket_sec=int(self._db_cfg.get("rr_bucket_sec", 10)),
                session_end_grace_sec=int(self._db_cfg.get("session_end_grace_sec", 10)),
                subject_id=str(self._db_cfg.get("subject_id", "subject_001")),
                device_id=str(self._db_cfg.get("device_id", "radar_001")),
            ),
            monitor_zone_names=monitor_zones,
            fps=(cfg.hardware.frame_rate if cfg else 25.0),
        )

    # ── Qt thread entry point ─────────────────────────────────────────────────

    def run(self) -> None:
        print("\n🚀 BedMonitorController: Processing thread started.")
        
        # ── 0. Initial IMU Auto-Alignment ─────────────────────────────────────
        auto_align_req = False
        if self.app_cfg and hasattr(self.app_cfg, "app"):
            auto_align_req = getattr(self.app_cfg.app, "imu_auto_align_enabled", False)
        
        print(f"📊 Auto-Alignment Configuration: {auto_align_req}")

        if auto_align_req:
            self._perform_startup_alignment()

        print("📡 Entering main radar processing loop...")
        while self.running:
            try:
                # 1. Vernier belt data (non-blocking drain)
                self._drain_belt_queues()

                # 1. Block on FFT queue (timeout allows clean shutdown)
                fft_frame = self.pt_fft_q.get(timeout=0.1)

                with self._pipeline_lock:
                    # 2. Process primary frame
                    output = self._engine.process_frame(fft_frame)
                    self._record_step(fft_frame, output)
                    frames_processed = 1

                    # 3. Drain any extra queued frames (catch-up mode)
                    while not self.pt_fft_q.empty():
                        try:
                            fft_frame = self.pt_fft_q.get_nowait()
                            output    = self._engine.process_frame(fft_frame)
                            self._record_step(fft_frame, output)
                            frames_processed += 1
                        except queue.Empty:
                            break

                # 4. Convert typed output → legacy dicts for GUI
                act_dict, resp_dict = self._to_legacy_dicts(output, frames_processed)

                # 5. Inject belt data
                if resp_dict:
                    resp_dict["belt_history"]       = self._belt_history.copy()
                    resp_dict["belt_samples_total"] = self._belt_total
                    resp_dict["belt_connected"]     = self._belt_connected

                self.data_ready.emit(act_dict, resp_dict, frames_processed)

            except queue.Empty:
                # No frame arrived — publish empty-room state
                with self._pipeline_lock:
                    empty = self._engine_empty_dict()
                self.data_ready.emit(empty, {}, 1)

    # ── Radar pose update (called from GUI thread) ─────────────────────────────

    def update_radar_pose(self, pose_dict: dict) -> None:
        """Handle a radar pose change issued by the xArm controller."""
        if not pose_dict:
            return
        with self._pipeline_lock:
            # Apply new geometry, then clear stale temporal state.
            self._engine.update_radar_pose(pose_dict)
            self._engine.reset()
            logger.info("RadarEngine reset due to pose update: %s", pose_dict)

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def stop(self) -> None:
        self.running = False
        self.wait()
        self._session_db.shutdown()
        self._export_recording()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _perform_startup_alignment(self) -> None:
        """Collects 1 second of IMU data and updates the radar pose."""
        self.is_aligning = True
        idx = int(getattr(self.app_cfg.app, "radar_imu_index", 0))
        if idx >= len(self.imu_queues):
            print(f"Auto-alignment stalled: IMU index {idx} not provided")
            self.is_aligning = False
            return

        q = self.imu_queues[idx]
        yaw_offset = float(getattr(self.app_cfg.app, "imu_yaw_offset", 180.0))
        pitch_mult = float(getattr(self.app_cfg.app, "radar_pitch_multiplier", -1.0))

        print("Starting Radar Auto-Alignment... waiting for IMU samples.")
        samples = []
        timeout = time.time() + 30.0 # max 30s wait for connection

        last_log = time.time()
        while len(samples) < 25 and time.time() < timeout and self.running:
            try:
                data = q.get(timeout=0.1)
                # Store: a_x, a_z, angl_x
                samples.append((data[0], data[2], data[6]))
                if len(samples) % 5 == 0:
                    print(f"Alignment Progress: {len(samples)}/25 samples collected...")
            except queue.Empty:
                if time.time() - last_log > 2.0:
                    print(f"Auto-Alignment still waiting for packets (current: {len(samples)})...")
                    last_log = time.time()
                continue

        if len(samples) < 10:
            print(f"Auto-alignment failed: insufficient IMU samples ({len(samples)})")
            self.is_aligning = False
            return

        import math
        avg_a_x = np.mean([s[0] for s in samples])
        avg_a_z = np.mean([s[1] for s in samples])
        avg_yaw = np.mean([s[2] for s in samples])

        raw_pitch = math.degrees(math.atan2(avg_a_z, -avg_a_x))
        pitch_offset = float(getattr(self.app_cfg.app, "imu_pitch_offset", 0.0))
        new_pitch = (raw_pitch + pitch_offset) * pitch_mult
        
        new_yaw   = (avg_yaw + yaw_offset) % 360

        print(f"✅ Auto-Alignment Complete: IMU(P:{raw_pitch:.1f}, Y:{avg_yaw:.1f}) -> Radar(P:{new_pitch:.1f}, Y:{new_yaw:.1f})")
        self.is_aligning = False

        # Update the engine pose (maintaining XYZ coordinates from config)
        default_zone = getattr(self.app_cfg.app, "default_radar_pose", "Bed")
        current_pose = self.app_cfg.layout.get(default_zone, {}).get("radar_pose", {})

        new_pose = {
            "x": float(current_pose.get("x", 0.0)),
            "y": float(current_pose.get("y", 0.0)),
            "z": float(current_pose.get("z", 0.0)),
            "pitch_deg": float(new_pitch),
            "yaw_deg": float(new_yaw),
            "fov_deg": float(current_pose.get("fov_deg", 120.0))
        }
        self.update_radar_pose(new_pose)

    def _drain_belt_queues(self) -> None:
        """Non-blocking drain of Vernier belt queues."""
        if self.vernier_belt_realtime_q:
            new_data: list[float] = []
            while not self.vernier_belt_realtime_q.empty():
                try:
                    new_data.append(self.vernier_belt_realtime_q.get_nowait())
                except queue.Empty:
                    break
            if new_data:
                self._belt_history = np.roll(self._belt_history, -len(new_data))
                self._belt_history[-len(new_data):] = new_data
                self._belt_total += len(new_data)

        if self.vernier_belt_connection_q:
            while not self.vernier_belt_connection_q.empty():
                try:
                    self._belt_connected = bool(self.vernier_belt_connection_q.get_nowait())
                except queue.Empty:
                    break

    def _to_legacy_dicts(
        self,
        output:           EngineOutput,
        frames_processed: int = 1,
    ) -> tuple[dict, dict]:
        """Convert a typed EngineOutput into legacy (act_dict, resp_dict) tuples.

        The act_dict keys and value types match ``ActivityPipeline.output_dict``
        exactly so all existing GUI display code works without modification.

        This is the permanent app-facade translation layer.
        """
        # ── Activity dict ──────────────────────────────────────────────────────
        display  = self._policy.render(output)
        tracked  = output.tracked_target
        activity = output.activity
        all_cands = output.candidates or []
        valid_cands = [c for c in all_cands if c.valid]
        best_cand = max(valid_cands, key=lambda c: c.magnitude) if valid_cands else None
        sh = getattr(self._engine.preprocessor, "spectral_history", None)
        cand_payload = [
            {
                "bin": int(c.bin_index),
                "x": float(c.x_m),
                "y": float(c.y_m),
                "z": float(c.z_m),
                "mag": float(c.magnitude),
                "valid": bool(c.valid),
                "zone": str(c.zone),
                "reject_reason": c.reject_reason,
            }
            for c in all_cands
        ]

        act_dict: dict = {
            # Geometry
            "X": tracked.smoothed_x_m if tracked else 0.0,
            "Y": tracked.smoothed_y_m if tracked else 0.0,
            "Z": tracked.smoothed_z_m if tracked else 0.0,
            "posture_z": tracked.smoothed_z_m if tracked else 0.0,

            # Detection
            "final_bin":           tracked.bin_index if tracked else None,
            "dynamic_mag":         float(best_cand.magnitude) if best_cand else 0.0,
            "detection_threshold": float(self._engine._cfg.detection.detection_threshold),

            # Display strings (from DisplayPolicy)
            "status":              display["status"],
            "zone":                display["zone"],
            "posture":             display["posture"],
            "motion_str":          display["motion_str"],
            "occ_confidence":      display["occ_confidence"],
            "posture_confidence":  display["posture_confidence"],
            "fall_confidence":     display["fall_confidence"],
            "duration_str":        display["duration_str"],
            "micro_state":         display["micro_state"],
            "spectral_history":    sh.get_ordered_2d() if sh else None,
            "raw_spectral_cube":   sh.get_ordered_cube() if sh else None,

            # Validity
            "is_valid": output.has_target,

            # Primary app-facing motion output. Keep motion_level as a
            # compatibility alias for older GUI code paths.
            "motion_score": activity.motion_score if activity else 0.0,
            "motion_level": activity.motion_score if activity else 0.0,

            # Detection diagnostics for 3D viewer.
            "candidates": cand_payload,
            "selected_candidate": (
                {
                    "bin": int(best_cand.bin_index),
                    "x": float(best_cand.x_m),
                    "y": float(best_cand.y_m),
                    "z": float(best_cand.z_m),
                    "mag": float(best_cand.magnitude),
                    "zone": str(best_cand.zone),
                }
                if best_cand is not None
                else None
            ),
        }

        # ── Respiration dict ───────────────────────────────────────────────
        resp_dict: dict = {}
        rm = output.respiration_metrics
        rs = output.respiration_signal

        if rs is not None and rm is not None:
            eng_cfg = self._engine._cfg
            window  = int(
                eng_cfg.respiration.window_sec * eng_cfg.hardware.frame_rate
            )
            resp_dict = {
                "live_signal":        rs.live_signal,
                "derivative_signal":  rs.derivative_signal,
                "locked_bin":         rs.locked_bin,
                "confidence":         90.0 if not (rs.quality.reason == "calibrating") else 0.0,
                "is_calibrating":     rs.quality.reason == "calibrating",
                "rr_current":         rm.rr_bpm or 0.0,
                "cycle_count":        rm.cycle_count,
                "apnea_count":        rm.apnea_event_count,
                "depth":              rm.breath_depth,
                "inhales":            rm.inhale_indices,
                "exhales":            rm.exhale_indices,
                "rr_history":         rm.rr_history,
                "apnea_active":       rm.apnea_active,
                "apnea_segments":     rm.apnea_segments,
                "apnea_duration":     rm.apnea_duration_s,
                "last_cycle_duration":rm.cycle_duration_s,
                "brv_value":          rm.brv_value,
                "apnea_threshold":    getattr(self._engine.resp_extractor, "apnea_threshold", 0.2),
                "Motion_State_bin":   display["micro_state"],
            }

        return act_dict, resp_dict

    def _engine_empty_dict(self) -> dict:
        """Empty-room activity dict."""
        return {
            "X": 0.0, "Y": 0.0, "Z": 0.0, "posture_z": 0.0,
            "final_bin": None,
            "dynamic_mag": 0.0,
            "detection_threshold": float(self._engine._cfg.detection.detection_threshold),
            "status": "No Occupant", "zone": "No Occupant Detected",
            "posture": "Unknown", "motion_str": "Unknown",
            "occ_confidence": 0.0, "posture_confidence": 0.0,
            "fall_confidence": 0.0, "duration_str": "--",
            "micro_state": "STABLE",
            "spectral_history": None, "raw_spectral_cube": None,
            "is_valid": False,
            "motion_score": 0.0, "motion_level": 0.0,
            "candidates": [], "selected_candidate": None,
        }

    def _record_step(self, raw_frame: np.ndarray, output: EngineOutput) -> None:
        """Capture optional raw/diagnostic artifacts for this processed frame."""
        self._session_db.ingest(output, frames=1)

        if not self._record_enabled:
            return

        if self._save_raw_frames:
            self._raw_frames.append(np.array(raw_frame, copy=True))
        if self._save_frame_records:
            self._frame_recorder.record(output)
        if self._save_event_log:
            self._event_log.update(output)

    def _export_recording(self) -> None:
        """Persist capture artifacts to disk at session end."""
        if not self._record_enabled:
            return

        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(self._record_started_at))
        session = self._record_session_name or f"session_{ts}"
        out_dir = self._record_output_dir / session
        out_dir.mkdir(parents=True, exist_ok=True)

        if self._save_raw_frames and self._raw_frames:
            frames_np = np.stack(self._raw_frames, axis=0)
            np.savez_compressed(out_dir / "frames.npz", frames=frames_np)

        if self._save_frame_records:
            self._frame_recorder.export_csv(out_dir / "frame_records.csv")
            self._frame_recorder.export_json(out_dir / "frame_records.json")

        if self._save_event_log:
            self._event_log.export_json(out_dir / "events.json")
            self._event_log.export_text(out_dir / "events.txt")

        logger.info("Session artifacts exported to %s", out_dir)
