"""radar_engine.config.vitals — Aliveness scoring and micro-state parameters."""
from dataclasses import dataclass


@dataclass(frozen=True)
class VitalsConfig:
    """Configuration for VitalFeatureExtractor micro-state classification.

    Legacy source: tuning.* aliveness / ghost / displacement keys.
    """
    # ── Aliveness composite score ──────────────────────────────────────────────
    aliveness_threshold:        float = 0.40   # WAS: tuning.aliveness_threshold
    # Weight input to aliveness: displacement must be in this range to score well.
    breathing_displacement_min: float = 0.5    # mm   WAS: tuning.breathing_displacement_min
    breathing_displacement_max: float = 12.0   # mm   WAS: tuning.breathing_displacement_max

    # ── SNR / ghost rejection ──────────────────────────────────────────────────
    min_person_snr:             float = 80.0   # WAS: tuning.min_person_snr
    ghost_phase_threshold:      float = 0.0005 # WAS: tuning.ghost_phase_threshold

    # ── MACRO_PHASE / MICRO_PHASE classification ───────────────────────────────
    macro_displacement_mm:      float = 15.0   # WAS: tuning.macro_displacement_mm

    # ── Position stability (STATIC_GHOST gate) ────────────────────────────────
    position_stability_window:  int   = 75     # WAS: tuning.position_stability_window
    max_clutter_position_var:   float = 0.08   # WAS: tuning.max_clutter_position_var
