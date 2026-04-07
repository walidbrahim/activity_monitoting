"""
radar_engine.core.enums
=======================
All semantic label enumerations used across the radar processing engine.

Design rules (from refactor.md F10, Rule 4):
  - Every major engine output carries an explicit validity state.
  - Enums replace raw strings so comparisons are type-safe and IDE-friendly.
  - No GUI-specific strings belong here; display text is the controller's job.
"""

from enum import Enum, auto


# ---------------------------------------------------------------------------
# Validity / Quality
# ---------------------------------------------------------------------------

class ValidityState(Enum):
    """Communicates the usability of an engine output to the application layer.

    VALID    — output is reliable and can be used directly.
    DEGRADED — output is available but its confidence is reduced; the
               application layer should decide whether to display it.
    INVALID  — output cannot be trusted; the application layer should omit or
               mask it.
    """
    VALID    = "valid"
    DEGRADED = "degraded"
    INVALID  = "invalid"


# ---------------------------------------------------------------------------
# Posture
# ---------------------------------------------------------------------------

class PostureLabel(Enum):
    """Coarse body-posture classification produced by ActivityInferencer.

    The engine emits one of these values; the controller decides whether to
    display it and how to word it.
    """
    LYING_DOWN = "lying_down"
    SITTING    = "sitting"
    STANDING   = "standing"
    FALLEN     = "fallen"
    UNKNOWN    = "unknown"


# ---------------------------------------------------------------------------
# Motion
# ---------------------------------------------------------------------------

class MotionLabel(Enum):
    """Fine-grained motion classification produced by ActivityInferencer.

    Values are ordered roughly from least to most energetic.
    The controller maps these to user-facing strings (e.g. "Resting/Breathing").
    """
    RESTING        = "resting"        # negligible motion, breathing only
    FIDGETING      = "fidgeting"      # low-level micro-motion (MICRO_PHASE)
    SHIFTING       = "shifting"       # moderate positional drift
    POSTURAL_SHIFT = "postural_shift" # deliberate posture change (MACRO_PHASE in monitor zone)
    MAJOR_MOVEMENT = "major_movement" # high motion_level
    WALKING        = "walking"        # sustained XY displacement in transit zone
    STATIC         = "static"         # apnea-like stillness (very low motion_level)
    UNKNOWN        = "unknown"


# ---------------------------------------------------------------------------
# Micro-state (per-candidate phase analysis)
# ---------------------------------------------------------------------------

class MicroState(Enum):
    """Per-candidate phase-history classification from VitalFeatureExtractor.

    This is an internal engine concept. The controller may use it to refine
    display decisions but should not expose raw values to the user.
    """
    ALIVE            = "ALIVE"
    WEAK_VITAL       = "WEAK_VITAL"
    DEAD_SPACE       = "DEAD_SPACE"
    STATIC_GHOST     = "STATIC_GHOST"
    MECHANICAL_ROTOR = "MECHANICAL_ROTOR"
    MACRO_PHASE      = "MACRO_PHASE"
    MICRO_PHASE      = "MICRO_PHASE"
    STABLE           = "STABLE"       # default / not yet computed


# ---------------------------------------------------------------------------
# Respiration pipeline state
# ---------------------------------------------------------------------------

class RespirationState(Enum):
    """State-machine labels for the RespirationExtractor.

    OFF     — no eligible target; pipeline is idle.
    ACQUIRE — target is eligible but the acquisition window has not elapsed.
    TRACK   — actively extracting and analysing respiration.
    HOLD    — brief eligibility interruption (e.g. postural shift); pipeline
              holds its last state.
    SUSPEND — target is definitely ineligible (walking, major movement, etc.);
              pipeline resets on re-entry.
    """
    OFF     = "OFF"
    ACQUIRE = "ACQUIRE"
    TRACK   = "TRACK"
    HOLD    = "HOLD"
    SUSPEND = "SUSPEND"


# ---------------------------------------------------------------------------
# Zone type (mirrors config.layout[zone]["type"])
# ---------------------------------------------------------------------------

class ZoneType(Enum):
    """Spatial zone classification from the layout configuration.

    MONITOR  — primary observation zone (bed, chair, etc.); full respiration
               pipeline is eligible here.
    IGNORE   — interference / exclusion zone; detections here are rejected.
    ROOM     — global boundary; detections outside are rejected as ghosts.
    TRANSIT  — fallback floor/transit zone; walking is detected here.
    UNKNOWN  — zone not found in layout config.
    """
    MONITOR = "monitor"
    IGNORE  = "ignore"
    ROOM    = "room"
    TRANSIT = "transit"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Occupancy state
# ---------------------------------------------------------------------------

class OccupancyState(Enum):
    """Coarse occupancy label output by the presence/occupancy sub-module.

    The controller maps these to user-facing status text.
    """
    EMPTY       = "empty"        # no confirmed target
    ENTERING    = "entering"     # target detected but entry debounce not yet passed
    OCCUPIED    = "occupied"     # confirmed, active target
    MONITORING  = "monitoring"   # occupied, still/breathing (no active motion)
    APNEA_CANDIDATE = "apnea_candidate"  # occupied, static beyond apnea_frames threshold
    PRESENCE    = "presence"     # confirmed presence in non-monitored zone


# ---------------------------------------------------------------------------
# Fall state
# ---------------------------------------------------------------------------

class FallState(Enum):
    """Fall detection state produced by the fall sub-module.

    NONE      — no fall evidence.
    CANDIDATE — fall scoring active but confidence < threshold.
    DETECTED  — fall confidence above display threshold.
    COOLDOWN  — post-zone-transition suppression window is active.
    """
    NONE      = "none"
    CANDIDATE = "candidate"
    DETECTED  = "detected"
    COOLDOWN  = "cooldown"
