"""radar_engine.config.tracking — Target persistence and smoothing parameters."""
from dataclasses import dataclass


@dataclass(frozen=True)
class TrackingConfig:
    """Configuration for the TargetTracker pipeline stage.

    Args:
        track_ema_alpha:          EMA coefficient for smoothing tracked XYZ coordinates.
                                  Lower = smoother but more latency.
                                  Legacy key: pipeline.track_alpha
        confidence_threshold:     Frame count required for the track confidence to be
                                  considered \"confirmed\". (Not the same as buffer_size.)
                                  Legacy key: pipeline.buffer_size  (repurposed)
        miss_allowance:           Consecutive frames without a valid candidate before
                                  the track is dropped.
                                  Legacy key: pipeline.miss_allowance
        jump_reject_distance_m:   Candidate-to-track distance (m) beyond which a new
                                  candidate is rejected as a spatial jump / ghost.
                                  Legacy key: tuning.jump_reject_distance
        coord_buffer_size:        Sliding window length for coordinate median smoothing.
                                  Legacy key: pipeline.buffer_size
        frames_to_occupy:         Consecutive confirmed frames before track is declared
                                  OCCUPIED (entry debounce).
                                  Derived from tuning.entry_hold_seconds if not set.
    """
    track_ema_alpha:        float = 0.05
    confidence_threshold:   int   = 3
    miss_allowance:         int   = 25
    jump_reject_distance_m: float = 1.0
    coord_buffer_size:      int   = 50
    frames_to_occupy:       int   = 75    # ~3 s at 25 fps
