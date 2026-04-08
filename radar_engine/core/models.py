"""
radar_engine.core.models
========================
Stable typed data contracts for all engine inputs and outputs.

Design rules (from refactor.md F5, Rules 2 & 4):
  - Pipelines exchange these objects instead of loosely-typed dicts.
  - Every major output carries an explicit validity state and reason code.
  - Models are pure data — no processing logic lives here.
  - All fields have documented semantics and units where applicable.

Dataclasses use ``field(default_factory=...)`` for mutable defaults so that
instances never accidentally share state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from radar_engine.core.enums import (
    ValidityState,
    PostureLabel,
    MotionLabel,
    MicroState,
    RespirationState,
    OccupancyState,
    FallState,
)


# ---------------------------------------------------------------------------
# Quality / Validity wrapper
# ---------------------------------------------------------------------------

@dataclass
class QualityFlag:
    """Reusable validity wrapper attached to engine outputs.

    Attributes:
        state:  Coarse validity category (VALID / DEGRADED / INVALID).
        score:  Continuous quality indicator in [0, 1]; 1.0 is best.
        reason: Optional machine-readable reason code when state != VALID.
                Examples: "high_motion", "low_snr", "bin_drift", "not_lying".
    """
    state:  ValidityState
    score:  float                = 1.0
    reason: str | None           = None

    @classmethod
    def valid(cls, score: float = 1.0) -> "QualityFlag":
        """Convenience constructor for a valid flag."""
        return cls(state=ValidityState.VALID, score=score)

    @classmethod
    def degraded(cls, score: float = 0.5, reason: str | None = None) -> "QualityFlag":
        """Convenience constructor for a degraded flag."""
        return cls(state=ValidityState.DEGRADED, score=score, reason=reason)

    @classmethod
    def invalid(cls, reason: str | None = None) -> "QualityFlag":
        """Convenience constructor for an invalid flag."""
        return cls(state=ValidityState.INVALID, score=0.0, reason=reason)

    @property
    def is_valid(self) -> bool:
        return self.state == ValidityState.VALID

    @property
    def is_usable(self) -> bool:
        """True when the output is either VALID or DEGRADED (i.e. not totally unusable)."""
        return self.state != ValidityState.INVALID


# ---------------------------------------------------------------------------
# Detection layer outputs
# ---------------------------------------------------------------------------

@dataclass
class TargetCandidate:
    """One peak candidate produced by TargetDetector for a single radar frame.

    Attributes:
        bin_index:      Range bin index of the peak in the dynamic magnitude profile.
        range_m:        Estimated slant range to the candidate [metres].
        x_m:            World-frame X coordinate [metres].
        y_m:            World-frame Y coordinate [metres].
        z_m:            World-frame Z (height) coordinate [metres].
        magnitude:      Dynamic magnitude of the candidate after clutter suppression.
        azimuth_rad:    Estimated azimuth angle [radians].
        elevation_rad:  Estimated elevation angle [radians].
        zone:           Zone label from the spatial layout (e.g. "Bed - Center").
        valid:          False when the candidate was rejected before scoring.
        reject_reason:  Machine-readable rejection cause when valid is False.
                        Examples: "below_cfar", "out_of_bounds", "ignored_zone",
                        "jump_reject".
    """
    bin_index:     int
    range_m:       float
    x_m:           float
    y_m:           float
    z_m:           float
    magnitude:     float
    azimuth_rad:   float
    elevation_rad: float
    zone:          str
    valid:         bool
    cfar_threshold:float = 0.0
    reject_reason: str | None = None


# ---------------------------------------------------------------------------
# Activity feature layer outputs
# ---------------------------------------------------------------------------

@dataclass
class VitalFeatures:
    """Per-candidate phase-history features from VitalFeatureExtractor.

    These features are computed from the beamformed spectral history window
    for a specific candidate bin. They inform both tracking (candidate scoring)
    and activity inference (micro-state labeling).

    Attributes:
        phase_ptp:            Peak-to-peak span of the unwrapped phase signal [rad].
        displacement_mm:      Estimated chest displacement amplitude [mm].
        phase_variance:       Variance of the first-difference of the unwrapped phase.
        phase_drift:          Absolute phase drift over the history window [rad].
        spectral_prominence:  Ratio of respiratory-band peak to local noise floor.
        autocorr_quality:     Normalised autocorrelation peak in the respiratory lag range.
        spectral_entropy:     Normalised spectral entropy of the evaluation band [0, 1];
                              lower is more periodic (better).
        aliveness_score:      Composite aliveness indicator [0, 1]; 1.0 = clearly alive.
        micro_state:          Phase-history classification label (MicroState enum).
        vital_multiplier:     Candidate scoring weight derived from aliveness/micro_state
                              [0.01 – 1.0]; used by TargetDetector's candidate ranker.
    """
    phase_ptp:           float
    displacement_mm:     float
    phase_variance:      float
    phase_drift:         float
    spectral_prominence: float
    autocorr_quality:    float
    spectral_entropy:    float
    aliveness_score:     float
    micro_state:         MicroState
    vital_multiplier:    float = 1.0


# ---------------------------------------------------------------------------
# Tracking layer outputs
# ---------------------------------------------------------------------------

@dataclass
class TrackedTarget:
    """The single confirmed target maintained by TargetTracker across frames.

    Attributes:
        bin_index:      Last-valid range bin of the confirmed target.
        x_m:            Raw (unsmoothed) X from the latest confirmed candidate [m].
        y_m:            Raw (unsmoothed) Y [m].
        z_m:            Raw (unsmoothed) Z [m].
        smoothed_x_m:   EMA-smoothed track X [m].
        smoothed_y_m:   EMA-smoothed track Y [m].
        smoothed_z_m:   EMA-smoothed track Z [m].
        confidence:     Discrete confidence counter (0 → confidence_threshold).
        miss_count:     Consecutive frames without a valid detection.
        v_z:            Estimated vertical velocity [m/s]; used for fall detection.
        valid:          False when the tracker has no confirmed target.
        validity_reason: Reason code when valid is False.
                        Examples: "no_target", "miss_exceeded", "jump_reject",
                        "ghost_zone", "warmup".
    """
    bin_index:      int
    x_m:            float
    y_m:            float
    z_m:            float
    smoothed_x_m:   float
    smoothed_y_m:   float
    smoothed_z_m:   float
    confidence:     int
    miss_count:     int
    v_z:            float = 0.0
    valid:          bool  = True
    validity_reason: str | None = None

    @classmethod
    def invalid(cls, reason: str) -> "TrackedTarget":
        """Return a sentinel TrackedTarget that marks tracking failure."""
        return cls(
            bin_index=0, x_m=0.0, y_m=0.0, z_m=0.0,
            smoothed_x_m=0.0, smoothed_y_m=0.0, smoothed_z_m=0.0,
            confidence=0, miss_count=0, v_z=0.0,
            valid=False, validity_reason=reason
        )


# ---------------------------------------------------------------------------
# Activity inference layer outputs
# ---------------------------------------------------------------------------

@dataclass
class ActivityState:
    """Activity state produced by ActivityInferencer for one radar frame.

    This object carries the engine's estimation output only. The controller
    layer decides how to gate, display, or alert on these values.

    Attributes:
        occupancy:          Coarse occupancy classification (OccupancyState enum).
        occupancy_confidence: Composite confidence index [0, 100].
        zone:               Full zone label including sub-zone suffix
                            (e.g. "Bed - Center", "Floor / Transit").
        subzone:            Sub-zone label only (e.g. "Center", "Head Edge"); None if
                            not applicable.
        posture:            Estimated body posture (PostureLabel enum).
        posture_confidence: Posture confidence [0, 100].
        motion:             Motion classification (MotionLabel enum).
        motion_score:       Stable motion score [0, 1] derived from the classified
                            motion state; intended as the primary app-facing motion output.
        motion_level:       Continuous tracker-level motion energy [0, ∞]; retained
                            for engine internals and debugging, not as the primary output.
        micro_state:        Best-candidate micro-state label (MicroState enum).
        is_walking:         True when sustained XY displacement detected in transit zone.
        fall_state:         Fall detection state (FallState enum).
        fall_confidence:    Fall confidence [0, 100].
        duration_str:       Time spent in current zone, formatted as "Xm Ys".
                            Empty string when no zone timer is running.
        occupied_reflection: EMA of the raw reflection magnitude from the target bin;
                             used for continuity gating in monitored zones.
        apnea_frame_count:  Frames since last active motion in a monitored zone;
                            used for apnea-candidate state transitions.
        valid:              False when no confirmed target exists.
        validity_reason:    Reason code when valid is False.
    """
    occupancy:            OccupancyState
    occupancy_confidence: float
    zone:                 str
    subzone:              str | None
    posture:              PostureLabel
    posture_confidence:   float
    motion:               MotionLabel
    motion_score:         float
    motion_level:         float
    micro_state:          MicroState
    is_walking:           bool
    fall_state:           FallState
    fall_confidence:      float
    duration_str:         str
    occupied_reflection:  float | None
    apnea_frame_count:    int
    valid:                bool
    validity_reason:      str | None = None

    @classmethod
    def empty(cls) -> "ActivityState":
        """Return a sentinel ActivityState representing an empty room."""
        return cls(
            occupancy=OccupancyState.EMPTY,
            occupancy_confidence=0.0,
            zone="No Occupant Detected",
            subzone=None,
            posture=PostureLabel.UNKNOWN,
            posture_confidence=0.0,
            motion=MotionLabel.UNKNOWN,
            motion_score=0.0,
            motion_level=0.0,
            micro_state=MicroState.STABLE,
            is_walking=False,
            fall_state=FallState.NONE,
            fall_confidence=0.0,
            duration_str="",
            occupied_reflection=None,
            apnea_frame_count=0,
            valid=False,
            validity_reason="no_target",
        )


# ---------------------------------------------------------------------------
# Respiration extraction layer outputs
# ---------------------------------------------------------------------------

@dataclass
class RespirationSignal:
    """Extracted respiratory waveform and extraction quality.

    Produced by RespirationExtractor. Does not contain RR or apnea — those
    are computed by RespirationAnalyzer from this object.

    Attributes:
        state:            Current state-machine label of the extractor.
        locked_bin:       Range bin the extractor is currently locked onto;
                          None when in OFF/ACQUIRE state.
        live_signal:      Time-ordered respiratory displacement waveform [mm],
                          length = window_frames. Zeroed when extractor is inactive.
        phase_signal:     Raw unwrapped phase used as the signal source [rad];
                          length = window_frames. May be None before first lock.
        derivative_signal: Normalised absolute derivative of the live signal [0, 1];
                           used for apnea detection.
        apnea_threshold:  Current dynamic apnea threshold used by analyzer logic.
        threshold_calibrated: True once threshold calibration window is complete.
        quality:          Extraction quality flag encapsulating SQI and gate weights.
        frames_since_present: Frames elapsed since the target entered eligibility.
    """
    state:                RespirationState
    locked_bin:           int | None
    live_signal:          np.ndarray          # shape (window_frames,)
    phase_signal:         np.ndarray | None
    derivative_signal:    np.ndarray          # shape (window_frames,)
    quality:              QualityFlag
    apnea_threshold:      float               = 0.20
    threshold_calibrated: bool                = False
    frames_since_present: int = 0

    @classmethod
    def inactive(cls, window_frames: int) -> "RespirationSignal":
        """Return a zero-filled inactive signal for OFF/SUSPEND states."""
        return cls(
            state=RespirationState.OFF,
            locked_bin=None,
            live_signal=np.zeros(window_frames),
            phase_signal=None,
            derivative_signal=np.zeros(window_frames),
            apnea_threshold=0.20,
            threshold_calibrated=False,
            quality=QualityFlag.invalid("extractor_inactive"),
            frames_since_present=0,
        )


# ---------------------------------------------------------------------------
# Respiration analysis layer outputs
# ---------------------------------------------------------------------------

@dataclass
class RespirationMetrics:
    """Physiological metrics produced by RespirationAnalyzer.

    Attributes:
        rr_bpm:           Estimated respiratory rate [breaths/min]; None when invalid.
        rr_quality:       RR validity flag with reason code.
        cycle_duration_s: Duration of the last detected breath cycle [seconds].
        cycle_count:      Total breath cycles counted since tracking started.
        inhale_indices:   Buffer indices of inhalation onset markers (troughs).
        exhale_indices:   Buffer indices of exhalation onset markers (peaks).
        breath_depth:     Qualitative depth label: "shallow", "normal", "deep",
                          "apnea", or "--".
        brv_value:        Breath-rate variability: std of last N cycle durations [s].
        apnea_active:     True when an apnea episode is currently active.
        apnea_quality:    Apnea validity flag; INVALID when calibration window not done.
        apnea_duration_s: Duration of the current apnea episode [seconds].
        apnea_event_count: Total distinct apnea events since tracking started.
        rr_history:       Rolling window of per-frame RR estimates; shape (window_frames,).
        apnea_trace:      Rolling boolean apnea mask; shape (window_frames,).
        apnea_segments:   List of (start, end) buffer-index pairs for current apnea spans.
    """
    rr_bpm:            float | None
    rr_quality:        QualityFlag
    cycle_duration_s:  float
    cycle_count:       int
    inhale_indices:    list[int]             = field(default_factory=list)
    exhale_indices:    list[int]             = field(default_factory=list)
    breath_depth:      str                   = "--"
    brv_value:         float                 = 0.0
    apnea_active:      bool                  = False
    apnea_quality:     QualityFlag           = field(default_factory=lambda: QualityFlag.invalid("not_calibrated"))
    apnea_duration_s:  float                 = 0.0
    apnea_event_count: int                   = 0
    rr_history:        np.ndarray            = field(default_factory=lambda: np.zeros(0))
    apnea_trace:       np.ndarray            = field(default_factory=lambda: np.zeros(0, dtype=bool))
    apnea_segments:    list[tuple[int, int]] = field(default_factory=list)

    @classmethod
    def empty(cls, window_frames: int = 0) -> "RespirationMetrics":
        """Return a zeroed metrics object for inactive states."""
        return cls(
            rr_bpm=None,
            rr_quality=QualityFlag.invalid("extractor_inactive"),
            cycle_duration_s=0.0,
            cycle_count=0,
            breath_depth="--",
            brv_value=0.0,
            rr_history=np.zeros(window_frames),
            apnea_trace=np.zeros(window_frames, dtype=bool),
        )


# ---------------------------------------------------------------------------
# Engine output (top-level contract)
# ---------------------------------------------------------------------------

@dataclass
class EngineOutput:
    """Unified output object returned by RadarEngine.process_frame().

    This is the stable API surface that all application layers consume.
    Every field is either a typed model or None when the corresponding
    stage was skipped or produced no result.

    Design principle (refactor.md F7):
        The engine outputs raw estimations and quality flags.
        The controller layer decides what to display or suppress.

    Attributes:
        timestamp:          Unix timestamp of the processed frame [seconds].
        frame_index:        Cumulative frame count since engine start.
        candidates:         All scored TargetCandidate objects from this frame
                            (both valid and rejected, for diagnostics).
        tracked_target:     The current confirmed target state; None when no track.
        vital_features:     VitalFeatures for the best candidate; None when not computed.
        activity:           ActivityState output; None when no track.
        respiration_signal: Extracted respiratory waveform and quality.
        respiration_metrics: Physiological analysis results.
        diagnostics:        Arbitrary per-frame debug fields (not for production display).
    """
    timestamp:            float
    frame_index:          int
    candidates:           list[TargetCandidate]      = field(default_factory=list)
    tracked_target:       TrackedTarget | None        = None
    vital_features:       VitalFeatures | None        = None
    activity:             ActivityState | None        = None
    respiration_signal:   RespirationSignal | None   = None
    respiration_metrics:  RespirationMetrics | None  = None
    diagnostics:          dict[str, Any]             = field(default_factory=dict)
    spectral_history:     Any                        = None  # Exposed for raw debug viz

    @property
    def has_target(self) -> bool:
        return self.tracked_target is not None and self.tracked_target.valid

    @property
    def has_respiration(self) -> bool:
        return (
            self.respiration_signal is not None
            and self.respiration_signal.quality.is_usable
        )
