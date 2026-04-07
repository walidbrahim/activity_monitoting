"""
radar_engine.activity.inferencer
====================================
ActivityInferencer — combines zone debounce, posture estimation, motion
classification, occupancy state machine, and fall detection into a single
typed ActivityState output.

Extracted from ActivityPipeline._step4_activity_inference() and
ActivityPipeline._step5_alert_logic() (engine-side components only).

Step 5 boundary split (refactor.md Phase 2 decision):
    Engine → emits raw scores + enum labels (this module).
    Controller → assembles status strings and display gating (Phase 5).

No display strings are produced here. The controller layer maps
OccupancyState / PostureLabel / FallState to user-facing text.
"""

from __future__ import annotations

import logging
import time
from collections import deque

import numpy as np

from radar_engine.core.base import RadarModule
from radar_engine.core.context import RadarContext
from radar_engine.core.enums import (
    MicroState, MotionLabel, OccupancyState, PostureLabel, FallState
)
from radar_engine.core.models import ActivityState, TrackedTarget
from radar_engine.config.activity import ActivityConfig

logger = logging.getLogger(__name__)


class ActivityInferencer(RadarModule):
    """Produces a typed ActivityState from each confirmed TrackedTarget.

    Internals (all mirrors of the original ActivityPipeline fields):

    Zone debounce:
        _zone_history, _current_stable_zone, _frames_to_confirm_zone,
        _subzone_history, _stable_subzone_label.

    Posture:
        _stable_posture — hysteresis label (changes only after
        posture_confirm_frames consecutive agreement frames).

    Occupancy / presence:
        is_occupied, entry_frames, _apnea_frames,
        _occupied_reflection (EMA of target-bin raw magnitude).

    Fall detection:
        _fall_persist_frames, _zone_last, _zone_transition_cooldown.

    Zone timer:
        _zone_entry_time.
    """

    # Fixed structural constants (not tunable per-deployment)
    _POSTURE_CONFIRM: int   = 8     # consecutive frames to confirm a posture change
    _APNEA_FRAMES:    int   = 150   # rest frames before apnea candidacy
    _ZONE_CONFIRM:    int   = 5     # zone debounce window
    _REFL_ON_EMA:     float = 0.10  # EMA alpha for reflection continuity

    def __init__(self, cfg: ActivityConfig | None = None) -> None:
        """
        Args:
            cfg: ActivityConfig dataclass.  When None, dataclass defaults are
                 used — useful for unit tests that don't need a YAML file.
        """
        if cfg is None:
            cfg = ActivityConfig()

        p = cfg.posture
        m = cfg.motion

        # ── Posture thresholds ─────────────────────────────────────────────────
        self._SITTING_THRESH   = p.sitting_threshold_m
        self._STANDING_THRESH  = p.standing_threshold_m

        # ── Motion thresholds ─────────────────────────────────────────────────
        self._REST_MAX         = m.rest_max
        self._RESTLESS_MAX     = m.restless_max

        # ── Zone / occupancy parameters ───────────────────────────────────────
        self._CONF_THRESHOLD   = 3
        self._FRAMES_TO_OCCUPY = int(cfg.entry_hold_seconds * 25)  # engine overrides if needed
        self._MISS_ALLOWANCE   = 5
        self._REFL_DIP_FRAMES  = cfg.reflection_dip_tolerance

        # ── Fall detection ────────────────────────────────────────────────────
        self._FALL_THRESHOLD   = p.fall_threshold_m
        self._FALL_VZ_THRESH   = p.fall_velocity_threshold
        self._FALL_PERSIST     = 45   # frames (~1.8 s at 25 fps)
        self._FALL_ENABLE      = p.fall_detection_enable
        self._ZONE_COOLDOWN    = p.fall_cooldown_frames

        # ── Posture Z proxy ───────────────────────────────────────────────────
        self._POSTURE_Z_NB_M   = p.posture_z_neighborhood_m
        self._POSTURE_Z_BIAS   = p.posture_z_bias

        # ── Mutable state ─────────────────────────────────────────────────────
        # Zone debounce
        self._zone_history:         deque  = deque(maxlen=self._ZONE_CONFIRM)
        self._current_stable_zone:  str    = "No Occupant Detected"
        self._subzone_history:      deque  = deque(maxlen=self._ZONE_CONFIRM)
        self._stable_subzone_label: str | None = None

        # Posture hysteresis
        self._posture_buf:    deque        = deque(maxlen=self._POSTURE_CONFIRM)
        self._stable_posture: PostureLabel = PostureLabel.UNKNOWN

        # Occupancy
        self.is_occupied:       bool  = False
        self.entry_frames:      int   = 0
        self._apnea_frames:     int   = 0
        self._zone_entry_time:  float = 0.0
        self._occupied_refl:    float | None = None
        self._refl_dip_frames:  int   = 0

        # Fall
        self._fall_persist_frames:      int   = 0
        self._zone_last:                str   = ""
        self._zone_transition_cooldown: int   = 0


    # ── RadarModule interface ──────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all persistent state."""
        self._zone_history.clear()
        self._current_stable_zone  = "No Occupant Detected"
        self._subzone_history.clear()
        self._stable_subzone_label = None
        self._posture_buf.clear()
        self._stable_posture       = PostureLabel.UNKNOWN
        self.is_occupied           = False
        self.entry_frames          = 0
        self._apnea_frames         = 0
        self._zone_entry_time      = 0.0
        self._occupied_refl        = None
        self._refl_dip_frames      = 0
        self._fall_persist_frames  = 0
        self._zone_last            = ""
        self._zone_transition_cooldown = 0

    def process(self, context: RadarContext) -> RadarContext:
        """Run activity inference for one confirmed-target frame.

        Reads:
            context.tracked_target   — must be valid (is_confirmed).
            context.candidates       — for posture Z-proxy.
            context.all_vital_features — for micro_state + raw magnitude.
            context.preprocessed     — for raw mag profile (occupied_reflection).

        Writes:
            context.activity — typed ActivityState.
        """
        target = context.tracked_target

        if target is None or not target.valid:
            # No confirmed target — update occupancy state machine and return empty
            self._update_occupancy_empty()
            context.activity = ActivityState.empty()
            return context

        # ── Step 4: Activity inference ────────────────────────────────────────
        context.activity = self._infer(context, target)
        return context

    # ── Private: main inference ───────────────────────────────────────────────

    def _infer(self, context: RadarContext, target: TrackedTarget) -> ActivityState:
        pre    = context.preprocessed
        cands  = context.candidates or []
        vitals = context.all_vital_features or {}

        x = target.smoothed_x_m
        y = target.smoothed_y_m
        z = target.smoothed_z_m

        # ── Zone (raw from best candidate, debounced) ─────────────────────────
        valid_cands = [c for c in cands if c.valid]
        
        if valid_cands:
            best_cand  = max(valid_cands, key=lambda c: c.magnitude)
            raw_zone   = best_cand.zone
            # Subzone = suffix after " - " e.g. "Bed - Center" → "Center"
            if " - " in raw_zone:
                raw_subzone = raw_zone.split(" - ", 1)[1]
            else:
                raw_subzone = None
        else:
            raw_zone    = "No Occupant Detected"
            raw_subzone = None

        stable_zone    = self._debounce_zone(raw_zone)
        stable_subzone = self._debounce_subzone(raw_subzone)

        # ── Posture proxy Z ───────────────────────────────────────────────────
        posture_z = self._compute_posture_z(valid_cands, x, y, z)

        # ── Posture classification ────────────────────────────────────────────
        posture, posture_conf = self._classify_posture(posture_z, stable_zone)

        # ── Micro-state ───────────────────────────────────────────────────────
        best_vf   = context.vital_features
        micro_str = best_vf.micro_state if best_vf else MicroState.STABLE

        # ── Motion level (from tracker via typed context field) ─────────────
        motion_level = context.motion_level

        # ── Motion classification ─────────────────────────────────────────────
        motion_label, motion_score, is_walking = self._classify_motion(
            motion_level, micro_str, stable_zone, posture)

        # ── Occupied reflection EMA (for continuity gating) ──────────────────
        occ_refl = self._update_occ_reflection(
            context, target.bin_index, stable_zone)

        # ── Occupancy state machine ───────────────────────────────────────────
        occupancy, occ_conf = self._update_occupancy(
            target, stable_zone, motion_label, occ_refl)

        # ── Apnea frame counter ───────────────────────────────────────────────
        if motion_label == MotionLabel.RESTING:
            self._apnea_frames += 1
        else:
            self._apnea_frames = 0

        if self._apnea_frames >= self._APNEA_FRAMES:
            if occupancy == OccupancyState.OCCUPIED:
                occupancy = OccupancyState.APNEA_CANDIDATE

        # ── Fall detection ────────────────────────────────────────────────────
        fall_state, fall_conf = self._detect_fall(
            target, stable_zone, posture, motion_label)

        # ── Zone timer ────────────────────────────────────────────────────────
        dur_str = self._zone_duration(stable_zone)

        return ActivityState(
            occupancy            = occupancy,
            occupancy_confidence = occ_conf,
            zone                 = stable_zone,
            subzone              = stable_subzone,
            posture              = posture,
            posture_confidence   = posture_conf,
            motion               = motion_label,
            motion_score         = motion_score,
            motion_level         = motion_level,
            micro_state          = micro_str,
            is_walking           = is_walking,
            fall_state           = fall_state,
            fall_confidence      = fall_conf,
            duration_str         = dur_str,
            occupied_reflection  = occ_refl,
            apnea_frame_count    = self._apnea_frames,
            valid                = True,
        )

    # ── Zone debounce ─────────────────────────────────────────────────────────

    def _debounce_zone(self, raw_zone: str) -> str:
        self._zone_history.append(raw_zone)
        if (len(self._zone_history) == self._ZONE_CONFIRM and
                len(set(self._zone_history)) == 1):
            if raw_zone != self._current_stable_zone:
                self._current_stable_zone = raw_zone
                self._zone_entry_time     = time.time()
        return self._current_stable_zone

    def _debounce_subzone(self, raw_subzone: str | None) -> str | None:
        self._subzone_history.append(raw_subzone)
        if (len(self._subzone_history) == self._ZONE_CONFIRM and
                len(set(self._subzone_history)) == 1):
            self._stable_subzone_label = raw_subzone
        return self._stable_subzone_label

    # ── Posture ───────────────────────────────────────────────────────────────

    def _compute_posture_z(
        self,
        valid_cands: list,
        x: float, y: float, z: float,
    ) -> float:
        """Strategy-A + B posture Z proxy.

        Finds the highest-Z valid candidate within posture_z_neighborhood_m
        of the tracked XY position. Falls back to tracked Z if none found.
        """
        if not valid_cands:
            return z

        neighborhood = self._POSTURE_Z_NB_M
        nearby = [
            c for c in valid_cands
            if np.sqrt((c.x_m - x) ** 2 + (c.y_m - y) ** 2) <= neighborhood
        ]
        if not nearby:
            return z

        posture_z = max(c.z_m for c in nearby) + self._POSTURE_Z_BIAS
        return float(posture_z)

    def _classify_posture(
        self,
        posture_z: float,
        zone: str,
    ) -> tuple[PostureLabel, float]:
        """Height-threshold posture with hysteresis confirm buffer."""
        if posture_z >= self._STANDING_THRESH:
            raw_posture = PostureLabel.STANDING
            conf        = min(100.0, 60.0 + (posture_z - self._STANDING_THRESH) * 80.0)
        elif posture_z >= self._SITTING_THRESH:
            raw_posture = PostureLabel.SITTING
            conf        = 70.0
        else:
            raw_posture = PostureLabel.LYING_DOWN
            conf        = min(100.0, 75.0 + (self._SITTING_THRESH - posture_z) * 80.0)

        self._posture_buf.append(raw_posture)
        if (len(self._posture_buf) == self._POSTURE_CONFIRM and
                len(set(self._posture_buf)) == 1):
            self._stable_posture = raw_posture

        return self._stable_posture, float(conf)

    # ── Motion ────────────────────────────────────────────────────────────────

    def _classify_motion(
        self,
        motion_level: float,
        micro_state:  MicroState,
        zone:         str,
        posture:      PostureLabel,
    ) -> tuple[MotionLabel, float, bool]:
        """Compute motion_score [0, 1] and walking flag.

        ``motion_score`` is the primary app-facing output, derived directly from
        the tracker's ``motion_level`` EMA (normalised to ``restless_max``).
        ``MotionLabel`` is kept minimal (RESTING / SHIFTING / WALKING) only
        because the occupancy state machine needs a coarse label to determine
        MONITORING vs OCCUPIED transitions.
        Walking detection is fully preserved.
        """
        # ── Walking detection (unchanged from legacy) ──────────────────────
        if micro_state == MicroState.MACRO_PHASE and motion_level > self._RESTLESS_MAX:
            if "Transit" in zone or "Floor" in zone:
                return MotionLabel.WALKING, 0.90, True

        # ── motion_score: normalised continuous output ─────────────────────
        denom       = max(self._RESTLESS_MAX, 1e-9)
        motion_score = float(min(1.0, motion_level / denom))

        # ── Coarse label: only RESTING vs SHIFTING needed downstream ──────
        if motion_score < (self._REST_MAX / denom):
            label = MotionLabel.RESTING
        else:
            label = MotionLabel.SHIFTING

        return label, motion_score, False

    # ── Occupied reflection ───────────────────────────────────────────────────

    def _update_occ_reflection(
        self,
        context:  RadarContext,
        bin_idx:  int,
        zone:     str,
    ) -> float | None:
        """EMA of the raw-magnitude at the target bin (reflection continuity)."""
        if context.preprocessed is None:
            return self._occupied_refl
        if "Bed" not in zone and "Monitor" not in zone:
            return self._occupied_refl

        raw_mag   = float(context.preprocessed.raw_mag_profile[bin_idx])
        if self._occupied_refl is None:
            self._occupied_refl = raw_mag
        else:
            alpha = self._REFL_ON_EMA
            self._occupied_refl = alpha * raw_mag + (1 - alpha) * self._occupied_refl
        return self._occupied_refl

    # ── Occupancy state machine ───────────────────────────────────────────────

    def _update_occupancy(
        self,
        target:       TrackedTarget,
        zone:         str,
        motion:       MotionLabel,
        occ_refl:     float | None,
    ) -> tuple[OccupancyState, float]:
        if not self.is_occupied:
            self.entry_frames += 1
            if self.entry_frames >= self._FRAMES_TO_OCCUPY:
                self.is_occupied  = True
                self.entry_frames = 0
                logger.info("Occupancy CONFIRMED in zone: %s", zone)
            else:
                frac = self.entry_frames / self._FRAMES_TO_OCCUPY
                return OccupancyState.ENTERING, frac * 50.0

        # Confirmed occupied
        occ_conf = min(100.0, 60.0 + float(target.confidence) * 10.0)

        if motion == MotionLabel.RESTING:
            return OccupancyState.MONITORING, occ_conf
        return OccupancyState.OCCUPIED, occ_conf

    def _update_occupancy_empty(self) -> None:
        """Called when no confirmed target; resets entry state."""
        if self.is_occupied:
            self.is_occupied  = False
            self._apnea_frames = 0
            logger.info("Occupancy cleared (target lost).")
        self.entry_frames = 0

    # ── Fall detection ────────────────────────────────────────────────────────

    def _detect_fall(
        self,
        target:  TrackedTarget,
        zone:    str,
        posture: PostureLabel,
        motion:  MotionLabel,
    ) -> tuple[FallState, float]:
        if not self._FALL_ENABLE:
            return FallState.NONE, 0.0

        # Zone transition cooldown
        if zone != self._zone_last:
            self._zone_transition_cooldown = self._ZONE_COOLDOWN
            self._zone_last = zone
        if self._zone_transition_cooldown > 0:
            self._zone_transition_cooldown -= 1
            return FallState.COOLDOWN, 0.0

        # Rapid downward velocity + low posture → fall candidate
        fall_conf = 0.0
        if target.v_z <= self._FALL_VZ_THRESH:
            fall_conf += 50.0
        if posture == PostureLabel.LYING_DOWN:
            fall_conf += 30.0
        if motion in (MotionLabel.SHIFTING, MotionLabel.WALKING):
            fall_conf += 20.0

        if fall_conf >= self._FALL_THRESHOLD * 100.0:
            self._fall_persist_frames = self._FALL_PERSIST
        elif self._fall_persist_frames > 0:
            self._fall_persist_frames -= 1
            fall_conf = max(fall_conf, 60.0)  # sustain after event

        if self._fall_persist_frames > 0:
            return FallState.DETECTED, fall_conf
        if fall_conf > 0:
            return FallState.CANDIDATE, fall_conf

        return FallState.NONE, 0.0

    # ── Zone timer ────────────────────────────────────────────────────────────

    def _zone_duration(self, zone: str) -> str:
        if not self._zone_entry_time:
            return ""
        if "No Occupant" in zone:
            return ""
        elapsed = time.time() - self._zone_entry_time
        m = int(elapsed // 60)
        s = int(elapsed % 60)
        return f"{m}m {s:02d}s"
