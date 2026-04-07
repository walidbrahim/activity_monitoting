"""
radar_engine.orchestration.engine
=====================================
RadarEngine — the main entry point for all radar processing.

Orchestrates the module chain:
    RadarFramePreprocessor
        → TargetDetector
        → TargetTracker
        → ActivityInferencer
        → RespirationExtractor      (optional)
        → RespirationAnalyzer       (optional)

After each frame it packages a fully typed EngineOutput and updates
the preprocessor's masking state from the tracker's result (so that
the next frame's clutter map uses the correct spatial masking).

Design notes:
  - The engine is fully GUI-agnostic.
  - Configuration is injected once at construction time through
    ``EngineConfig``; modules can be swapped via constructor kwargs
    for testing.
  - ``with_respiration=False`` produces an ActivityOnly engine (useful
    for transit-zone or presence-only applications).
"""

from __future__ import annotations

import logging
import time
from typing import Sequence

import numpy as np

from radar_engine.config.engine import EngineConfig
from radar_engine.core.base import RadarModule, NullModule
from radar_engine.core.context import RadarContext
from radar_engine.core.enums import OccupancyState
from radar_engine.core.models import EngineOutput
from radar_engine.preprocessing.preprocessor import RadarFramePreprocessor
from radar_engine.detection.target_detector import TargetDetector
from radar_engine.detection.zoning import ZoneEvaluator
from radar_engine.tracking.target_tracker import TargetTracker
from radar_engine.activity.inferencer import ActivityInferencer
from radar_engine.respiration.extractor import RespirationExtractor
from radar_engine.respiration.analyzer import RespirationAnalyzer

logger = logging.getLogger(__name__)


def _build_rotation_matrix(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Build a 3×3 world-frame rotation matrix from radar pose (yaw + pitch).

    Replicates the rotation matrix construction used in the original
    ActivityPipeline.__init__() so that coordinate transforms are identical.
    """
    yaw   = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)

    Ry = np.array([
        [ np.cos(yaw), -np.sin(yaw), 0],
        [ np.sin(yaw),  np.cos(yaw), 0],
        [           0,            0, 1],
    ])
    Rp = np.array([
        [1,             0,              0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch),  np.cos(pitch)],
    ])
    return np.dot(Ry, Rp)


class RadarEngine:
    """Full-stack radar processing engine.

    Constructs all modules from a typed ``EngineConfig`` and processes one
    frame at a time, returning a fully typed ``EngineOutput``.

    Args:
        cfg:              Engine configuration. When None, safe built-in
                          defaults are used for local/testing scenarios.
        with_respiration: Include RespirationExtractor + Analyzer when True.
        preprocessor:     Override the default RadarFramePreprocessor.
        detector:         Override the default TargetDetector.
        tracker:          Override the default TargetTracker.
        inferencer:       Override the default ActivityInferencer.
        resp_extractor:   Override the default RespirationExtractor.
        resp_analyzer:    Override the default RespirationAnalyzer.
    """

    def __init__(
        self,
        cfg:               EngineConfig | None = None,
        with_respiration:  bool                = True,
        preprocessor:      RadarModule | None  = None,
        detector:          RadarModule | None  = None,
        tracker:           RadarModule | None  = None,
        inferencer:        RadarModule | None  = None,
        resp_extractor:    RadarModule | None  = None,
        resp_analyzer:     RadarModule | None  = None,
    ) -> None:
        # ── Require explicit config ───────────────────────────────────────────────
        if cfg is None:
            from radar_engine.config.hardware import RadarHardwareConfig
            cfg = EngineConfig(
                hardware=RadarHardwareConfig(
                    num_range_bins=35, num_antennas=8,
                    frame_rate=25.0, range_resolution=0.15,
                )
            )
            logger.warning(
                "RadarEngine: no EngineConfig supplied; using built-in defaults. "
                "Pass an explicit EngineConfig for production use."
            )

        self._cfg = cfg
        hw  = cfg.hardware
        pre = cfg.preprocessing
        det = cfg.detection
        trk = cfg.tracking
        act = cfg.activity
        res = cfg.respiration

        # ── Radar geometry (from layout.radar_pose) ───────────────────────────
        self._T = np.array([0.0, 0.0, 0.0])
        self._R = np.eye(3)
        for zone_cfg in cfg.layout.values():
            rp = zone_cfg.get("radar_pose") if isinstance(zone_cfg, dict) else None
            if rp:
                self._T = np.array([rp.get("x", 0.0), rp.get("y", 0.0), rp.get("z", 0.0)])
                self._R = _build_rotation_matrix(rp.get("yaw_deg", 0), rp.get("pitch_deg", 0))
                break

        # ── Zone evaluator (layout-injected, no global config) ────────────────
        self._zone_evaluator = ZoneEvaluator(cfg.layout)
        self._monitor_zone_names = {
            name
            for name, zone_cfg in cfg.layout.items()
            if isinstance(zone_cfg, dict) and str(zone_cfg.get("type", "")).lower() == "monitor"
        }

        # ── Feature flags shim (duck-typed from PreprocessingConfig) ──────────
        class _FeatureFlags:
            def __init__(self, p):
                self.clutter_removal      = p.features_clutter_removal
                self.vital_analysis       = p.features_vital_analysis
                self.temporal_persistence = p.features_temporal_persistence
                self.adaptive_smoothing   = p.features_adaptive_smoothing
                self.fall_posture         = p.features_fall_posture
                self.apnea_state          = p.features_apnea_state
                self.tethering            = False
        features = _FeatureFlags(pre)

        # ── Spectral history window ───────────────────────────────────────────
        spectral_frames = int(res.window_sec * hw.frame_rate)

        # ── Module construction ───────────────────────────────────────────────
        self.preprocessor: RadarFramePreprocessor = preprocessor or RadarFramePreprocessor(
            num_bins             = hw.num_range_bins,
            num_antennas         = hw.num_antennas,
            spectral_frames      = spectral_frames,
            alpha                = pre.clutter_ema_alpha,
            warmup_frames        = pre.warmup_frames,
            features             = features,
            confidence_threshold = trk.confidence_threshold,
        )

        self.detector: TargetDetector = detector or TargetDetector(
            range_resolution = hw.range_resolution,
            frame_rate       = hw.frame_rate,
            R                = self._R,
            T                = self._T,
            features         = features,
            zone_evaluator   = self._zone_evaluator,
        )

        self.tracker: TargetTracker = tracker or TargetTracker(cfg=trk)

        self.inferencer: ActivityInferencer = inferencer or ActivityInferencer(cfg=act)

        if with_respiration:
            self.resp_extractor: RadarModule = resp_extractor or RespirationExtractor(
                cfg=res, frame_rate=hw.frame_rate
            )
            self.resp_analyzer: RadarModule = resp_analyzer or RespirationAnalyzer(
                cfg=res, frame_rate=hw.frame_rate
            )
        else:
            self.resp_extractor = NullModule("resp_extractor_disabled")
            self.resp_analyzer  = NullModule("resp_analyzer_disabled")

        self._with_respiration = with_respiration

        # Frame counter
        self._frame_index: int = 0


    # ── Public API ─────────────────────────────────────────────────────────────

    def process_frame(
        self,
        raw_frame: np.ndarray,
        timestamp: float | None = None,
        frames:    int          = 1,
    ) -> EngineOutput:
        """Process one radar frame and return a typed EngineOutput.

        Args:
            raw_frame: FFT frame from the radar; shape (range_bins, antennas).
            timestamp: Unix timestamp; defaults to time.time() if None.
            frames:    Number of radar frames represented by this call (used
                       for respiration buffer advancement).

        Returns:
            Fully typed EngineOutput regardless of tracking / respiration state.
        """
        if timestamp is None:
            timestamp = time.time()
        self._frame_index += 1

        # ── Build context ──────────────────────────────────────────────────────
        ctx = RadarContext(
            frame_index = self._frame_index,
            timestamp   = timestamp,
            raw_frame   = raw_frame,
        )
        ctx.diagnostics["frames_in_tick"] = frames

        # ── 1. Preprocessing ───────────────────────────────────────────────────
        ctx = self.preprocessor.process(ctx)

        if ctx.preprocessed and ctx.preprocessed.warmup_active:
            # Still warming up — return an empty output with diagnostics
            return EngineOutput(
                timestamp   = timestamp,
                frame_index = self._frame_index,
                diagnostics = dict(ctx.diagnostics),
            )

        # Expose SpectralHistory to downstream modules via typed context field
        ctx.spectral_history = self.preprocessor.spectral_history

        # ── 2. Detection ───────────────────────────────────────────────────────
        ctx = self.detector.process(ctx)

        # ── 3. Tracking ────────────────────────────────────────────────────────
        ctx = self.tracker.process(ctx)

        # Expose motion_level to the inferencer via typed context field
        ctx.motion_level = self.tracker.motion_level

        # ── 4. Activity inference ──────────────────────────────────────────────
        ctx = self.inferencer.process(ctx)

        # ── 5. Update clutter masking state for next frame ─────────────────────
        tracked = ctx.tracked_target
        self.preprocessor.update_masking_state(
            is_occupied      = self.inferencer.is_occupied,
            last_target_bin  = tracked.bin_index if (tracked and tracked.valid) else None,
            track_confidence = tracked.confidence if tracked else 0,
        )

        # ── 6. Respiration ─────────────────────────────────────────────────────
        if self._with_respiration and tracked and tracked.valid:
            # Respiration is allowed only in configured monitor zones (e.g., Bed).
            zone = ctx.activity.zone if ctx.activity and ctx.activity.valid else ""
            base_zone = zone.split(" - ", 1)[0] if zone else ""
            in_monitor_zone = base_zone in self._monitor_zone_names

            if in_monitor_zone:
                # Expose confirmed bin to the extractor via typed context field
                ctx.target_bin = tracked.bin_index
                ctx = self.resp_extractor.process(ctx)
                ctx = self.resp_analyzer.process(ctx)
                ctx.diagnostics["respiration_eligible"] = True
            else:
                # Leaving monitor zone should clear respiration state so the next
                # re-entry starts cleanly from acquisition/calibration.
                self.resp_extractor.reset()
                self.resp_analyzer.reset()
                ctx.diagnostics["respiration_eligible"] = False
                ctx.diagnostics["respiration_skip_reason"] = "non_monitor_zone"

        # ── 7. Package output ──────────────────────────────────────────────────
        return EngineOutput(
            timestamp           = timestamp,
            frame_index         = self._frame_index,
            candidates          = ctx.candidates or [],
            tracked_target      = ctx.tracked_target,
            vital_features      = ctx.vital_features,
            activity            = ctx.activity,
            respiration_signal  = ctx.respiration_signal,
            respiration_metrics = ctx.respiration_metrics,
            diagnostics         = {
                k: v for k, v in ctx.diagnostics.items()
                if not k.startswith("_")   # strip internal private keys
            },
        )

    def reset(self) -> None:
        """Reset all modules — call on radar pose change or hard restart."""
        self.preprocessor.reset()
        self.detector.reset()
        self.tracker.reset()
        self.inferencer.reset()
        self.resp_extractor.reset()
        self.resp_analyzer.reset()
        self._frame_index = 0
        logger.info("RadarEngine reset.")

    def update_radar_pose(self, pose_dict: dict) -> None:
        """Update runtime radar pose used by localization.

        This updates the detector's world transform immediately so subsequent
        frames use the new robot/radar geometry.
        """
        if not pose_dict:
            return

        self._T = np.array([
            pose_dict.get("x", float(self._T[0])),
            pose_dict.get("y", float(self._T[1])),
            pose_dict.get("z", float(self._T[2])),
        ])
        self._R = _build_rotation_matrix(
            pose_dict.get("yaw_deg", 0.0),
            pose_dict.get("pitch_deg", 0.0),
        )

        # Keep detector transform in sync with the engine pose.
        self.detector.R = self._R
        self.detector.T = self._T

        logger.info(
            "RadarEngine pose updated: x=%.3f y=%.3f z=%.3f yaw=%.1f pitch=%.1f",
            self._T[0], self._T[1], self._T[2],
            pose_dict.get("yaw_deg", 0.0),
            pose_dict.get("pitch_deg", 0.0),
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def frame_index(self) -> int:
        return self._frame_index
