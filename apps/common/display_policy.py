"""
apps.common.display_policy
============================
DisplayPolicy — maps typed engine enums and scores to human-readable
display strings for the GUI layer.

This is the **controller-side** "Step 5" complement to ActivityInferencer.
The engine outputs raw enums (OccupancyState, FallState, PostureLabel …).
This module is the single place where those enums become the status strings,
posture labels, and motion descriptions the GUI has always consumed.

Rules (refactor.md Step 5 boundary):
  - NO processing logic lives here — only enum → string lookups and
    simple threshold comparisons for display gating.
  - All strings are identical to those produced by the legacy
    ActivityPipeline._step5_alert_logic() so the existing GUI widgets
    need zero changes during the migration.
"""

from __future__ import annotations

from radar_engine.core.enums import (
    FallState, MotionLabel, OccupancyState, PostureLabel
)
from radar_engine.core.models import ActivityState, EngineOutput


# ---------------------------------------------------------------------------
# Status string map  (mirrors _step5_alert_logic output strings exactly)
# ---------------------------------------------------------------------------

_OCCUPANCY_TO_STATUS: dict[OccupancyState, str] = {
    OccupancyState.EMPTY:            "No Occupant",
    OccupancyState.ENTERING:         "Occupant Entering...",
    OccupancyState.OCCUPIED:         "Occupied (Breathing/Moving)",
    OccupancyState.MONITORING:       "Still / Monitoring...",
    OccupancyState.APNEA_CANDIDATE:  "Possible Apnea",
    OccupancyState.PRESENCE:         "Room Presence",
}

_POSTURE_TO_STR: dict[PostureLabel, str] = {
    PostureLabel.LYING_DOWN: "Lying Down",
    PostureLabel.SITTING:    "Sitting",
    PostureLabel.STANDING:   "Standing",
    PostureLabel.UNKNOWN:    "Unknown",
}

_MOTION_TO_STR: dict[MotionLabel, str] = {
    MotionLabel.RESTING:        "Resting/Breathing",
    MotionLabel.FIDGETING:      "Restless/Fidgeting",
    MotionLabel.SHIFTING:       "Restless/Shifting",
    MotionLabel.POSTURAL_SHIFT: "Postural Shift",
    MotionLabel.WALKING:        "Walking",
    MotionLabel.MAJOR_MOVEMENT: "Major Movement",
    MotionLabel.STATIC:         "Static",
    MotionLabel.UNKNOWN:        "Unknown",
}


class DisplayPolicy:
    """Converts typed engine output into GUI-consumable string fields.

    Stateless transformation layer — instantiate once per application and
    call ``render()`` each frame.

    Fall override:
        If FallState.DETECTED with confidence > 60 the status string becomes
        "CRITICAL: Fall Detected!" and posture becomes "Fallen", exactly
        matching the legacy _step5_alert_logic behaviour.

    Floor / Transit zone override:
        When the zone contains "Floor / Transit", the word "Occupied" in the
        status is replaced with "In the Room" (same as legacy logic).
    """

    def render(self, output: EngineOutput) -> dict:
        """Map a typed EngineOutput to a GUI-compatible display dict.

        The returned dict has exactly the keys the existing GUI expects so
        the legacy widget code requires no changes.

        Args:
            output: Typed EngineOutput from RadarEngine.process_frame().

        Returns:
            dict with string fields for the GUI display layer.
        """
        activity = output.activity

        if activity is None or not activity.valid:
            return self._empty_display()

        # ── Status string ──────────────────────────────────────────────────
        status = _OCCUPANCY_TO_STATUS.get(
            activity.occupancy, "Occupied (Breathing/Moving)"
        )

        # Floor / Transit label override (matches legacy verbatim)
        if "Floor" in activity.zone or "Transit" in activity.zone:
            status = status.replace("Occupied", "In the Room")

        # Fall state override
        posture_str = _POSTURE_TO_STR.get(activity.posture, "Unknown")
        fall_conf   = activity.fall_confidence

        if activity.fall_state == FallState.DETECTED and fall_conf > 60.0:
            status      = "CRITICAL: Fall Detected!"
            posture_str = "Fallen"
        elif activity.fall_state == FallState.CANDIDATE and fall_conf > 40.0:
            status = "Fall Candidate — Monitoring..."

        # Weak-signal state (occupied_reflection dip gating)
        # The engine doesn't do reflection-ratio gating yet (deferred to Phase 5+).
        # When the occupied_reflection field is eventually populated the controller
        # can add the "Occupied / weak signal" state here without touching the engine.

        # ── Motion string ──────────────────────────────────────────────────
        motion_str = _MOTION_TO_STR.get(activity.motion, "Unknown")

        # ── Walking override for posture confidence ────────────────────────
        posture_confidence = activity.posture_confidence
        if activity.is_walking:
            posture_confidence = 95.0  # mirrors config.motion.walk_posture_conf

        return {
            "status":             status,
            "zone":               activity.zone,
            "posture":            posture_str,
            "motion_str":         motion_str,
            "occ_confidence":     activity.occupancy_confidence,
            "posture_confidence": posture_confidence,
            "fall_confidence":    activity.fall_confidence,
            "duration_str":       activity.duration_str,
            "micro_state":        activity.micro_state.value if activity.micro_state else "STABLE",
            "is_walking":         activity.is_walking,
        }

    # ── Sentinel ───────────────────────────────────────────────────────────

    @staticmethod
    def _empty_display() -> dict:
        return {
            "status":             "No Occupant",
            "zone":               "No Occupant Detected",
            "posture":            "Unknown",
            "motion_str":         "Unknown",
            "occ_confidence":     0.0,
            "posture_confidence": 0.0,
            "fall_confidence":    0.0,
            "duration_str":       "--",
            "micro_state":        "STABLE",
            "is_walking":         False,
        }
