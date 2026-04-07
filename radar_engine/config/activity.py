"""radar_engine.config.activity — Posture, motion, and occupancy parameters."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PostureCfg:
    """Posture state-machine thresholds and fall detection parameters.

    All heights are in metres (Z coordinate in world frame).

    Legacy source: top-level posture.* + selected tuning.* keys.
    """
    # ── Posture thresholds ─────────────────────────────────────────────────────
    sitting_threshold_m:      float = 0.45   # WAS: posture.sitting_threshold
    standing_threshold_m:     float = 0.95   # WAS: posture.standing_threshold
    hysteresis_m:             float = 0.05   # WAS: tuning.posture_hysteresis_m

    # ── Floor / Transit bias ───────────────────────────────────────────────────
    # Extra margin to exit-Standing / enter-Sitting when in transit zone.
    # Z_b is noisier in open space so we bias strongly toward Standing.
    transit_standing_bias_m:  float = 0.08   # WAS: tuning.transit_standing_bias_m

    # ── Posture Z proxy ────────────────────────────────────────────────────────
    posture_z_neighborhood_m: float = 0.30   # WAS: tuning.posture_z_neighborhood_m
    posture_z_bias:           float = 0.0    # WAS: tuning.posture_z_bias
    posture_transit_z_bonus:  float = 0.15   # WAS: tuning.posture_transit_z_bonus

    # ── Fall detection ─────────────────────────────────────────────────────────
    fall_detection_enable:    bool  = False   # WAS: posture.fall_detection_enable
    fall_threshold_m:         float = 0.30    # WAS: posture.fall_threshold
    fall_velocity_threshold:  float = -1.2    # WAS: posture.fall_velocity_threshold
    fall_cooldown_frames:     int   = 25      # WAS: tuning.fall_cooldown_frames


@dataclass(frozen=True)
class MotionCfg:
    """Motion classification thresholds.

    Legacy source: top-level motion.* keys.
    """
    rest_max:            float = 0.10    # WAS: motion.rest_max
    restless_max:        float = 0.30    # WAS: motion.restless_max
    walk_window_frames:  int   = 15      # WAS: motion.walk_window_frames
    walk_displacement_m: float = 0.30    # WAS: motion.walk_displacement_m
    walk_posture_conf:   float = 90.0    # WAS: motion.walk_posture_conf


@dataclass(frozen=True)
class ActivityConfig:
    """Top-level activity inference configuration.

    Aggregates posture and motion sub-configs, plus zone debouncing parameters.

    Legacy source: pipeline.frame_to_confirm_zone + tuning.entry_hold_seconds
                   + posture.* + motion.*
    """
    zone_debounce_frames:  int      = 50    # WAS: pipeline.frame_to_confirm_zone
    entry_hold_seconds:    float    = 3.0   # WAS: tuning.entry_hold_seconds

    # Display stabilisation
    subzone_debounce_frames: int    = 15    # WAS: tuning.subzone_debounce_frames
    bin_vote_window:          int   = 7     # WAS: tuning.bin_vote_window

    # Reflection continuity (occupied-zone signal floor)
    continuity_ratio:          float = 0.60  # WAS: tuning.continuity_ratio
    reflection_dip_tolerance:  int   = 5     # WAS: tuning.reflection_dip_tolerance

    posture: PostureCfg = field(default_factory=PostureCfg)
    motion:  MotionCfg  = field(default_factory=MotionCfg)
