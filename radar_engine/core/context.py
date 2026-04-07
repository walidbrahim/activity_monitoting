"""
radar_engine.core.context
=========================
RadarContext — the shared pipeline state bag passed between all RadarModule
instances in a single processing chain.

Design rationale (refactor.md F6):
  Each module reads what it needs from the context, writes its outputs back
  into it, and returns the updated context. This makes modules composable and
  testable: any module can be unit-tested by constructing a RadarContext with
  only the fields it reads.

Ownership contract:
  - ``raw_frame`` and ``timestamp`` / ``frame_index`` are set by the caller
    (RadarEngine) and are read-only for all modules.
  - All other fields start as None and are populated progressively as modules
    run in order.
  - A module must not depend on a field that is populated by a later module
    (i.e. the pipeline order defines a strict data-flow dependency).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from radar_engine.core.models import (
    TargetCandidate,
    VitalFeatures,
    TrackedTarget,
    ActivityState,
    RespirationSignal,
    RespirationMetrics,
)


@dataclass
class PreprocessedFrame:
    """Intermediate products from RadarFramePreprocessor.

    Attributes:
        corrected_data:       Antenna-corrected complex frame; shape (bins, antennas).
        dynamic_data:         Clutter-subtracted complex frame; shape (bins, antennas).
        dynamic_mag_profile:  Per-bin magnitude after clutter subtraction; shape (bins,).
        raw_mag_profile:      Per-bin magnitude before clutter subtraction; shape (bins,).
        warmup_active:        True when the frame was produced during the warmup window;
                              downstream modules should skip processing when True.
    """
    corrected_data:      np.ndarray
    dynamic_data:        np.ndarray
    dynamic_mag_profile: np.ndarray
    raw_mag_profile:     np.ndarray
    warmup_active:       bool = False


@dataclass
class RadarContext:
    """Shared pipeline state bag for one radar frame.

    All fields beyond ``raw_frame``, ``timestamp``, and ``frame_index`` are
    optional (None) until the responsible module populates them.

    Attributes:
        frame_index:        Cumulative frame counter since engine start.
        timestamp:          Unix timestamp of this frame [seconds].
        raw_frame:          Raw FFT frame from the radar; shape (bins, antennas).

        preprocessed:       Output of RadarFramePreprocessor.
        candidates:         Output of TargetDetector — scored TargetCandidate list.
        vital_features:     Output of VitalFeatureExtractor — per best-candidate VitalFeatures.
        all_vital_features: Per-candidate VitalFeatures map (bin_index → VitalFeatures).
        tracked_target:     Output of TargetTracker — current TrackedTarget.
        activity:           Output of ActivityInferencer — current ActivityState.
        respiration_signal: Output of RespirationExtractor.
        respiration_metrics: Output of RespirationAnalyzer.

        diagnostics:        Arbitrary key-value store for per-frame debug records.
                            Modules may write structured entries here without coupling
                            to a specific logging implementation.
    """
    # ── Required inputs (set by RadarEngine before calling modules) ─────────
    frame_index: int
    timestamp:   float
    raw_frame:   np.ndarray

    # ── Progressive outputs (populated by modules in pipeline order) ────────
    preprocessed:       PreprocessedFrame | None              = None
    candidates:         list[TargetCandidate] | None          = None
    vital_features:     VitalFeatures | None                  = None
    all_vital_features: dict[int, VitalFeatures] | None       = None
    tracked_target:     TrackedTarget | None                  = None
    activity:           ActivityState | None                  = None
    respiration_signal: RespirationSignal | None              = None
    respiration_metrics: RespirationMetrics | None            = None

    # ── Inter-module typed handoffs (explicit contracts, not hidden in diagnostics)
    spectral_history:  Any | None = None   # SpectralHistory from preprocessor → detector, extractor
    motion_level:      float      = 0.0    # Tracker motion EMA → activity inferencer
    target_bin:        int | None = None   # Confirmed range bin → respiration extractor

    # ── Debug ────────────────────────────────────────────────────────────────
    diagnostics: dict[str, Any] = field(default_factory=dict)

    # ── Convenience properties ───────────────────────────────────────────────

    @property
    def is_warmed_up(self) -> bool:
        """True when the preprocessor has passed its warmup gate."""
        return (
            self.preprocessed is not None
            and not self.preprocessed.warmup_active
        )

    @property
    def has_track(self) -> bool:
        """True when the tracker reports a confirmed, valid target."""
        return self.tracked_target is not None and self.tracked_target.valid

    @property
    def best_candidate_bin(self) -> int | None:
        """Range bin of the highest-scored valid candidate, or None."""
        if not self.candidates:
            return None
        valid = [c for c in self.candidates if c.valid]
        if not valid:
            return None
        return max(valid, key=lambda c: c.magnitude).bin_index
