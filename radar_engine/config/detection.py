"""radar_engine.config.detection — CFAR and candidate selection parameters."""
from dataclasses import dataclass


@dataclass(frozen=True)
class DetectionConfig:
    """Configuration for the TargetDetector pipeline stage.

    Args:
        use_cfar:              Apply CA-CFAR thresholding (vs static threshold only).
        detection_threshold:   Minimum dynamic-profile magnitude for a bin to be
                               considered a candidate.
        num_candidates:        Maximum number of range-bin peaks to evaluate per frame.
                               YAML key: detection.num_candidates
                               Legacy key: tuning.num_candidates
        min_search_range_m:    Minimum range (metres) below which peaks are rejected
                               (guards against near-field artefacts).
                               Legacy key: tuning.min_search_range
        z_clip_min:            Minimum Z estimate (m) — clips extremely negative Z.
                               Legacy key: tuning.z_clip_min
        z_clip_max:            Maximum Z estimate (m) — clips ceiling returns.
                               Legacy key: tuning.z_clip_max
    """
    use_cfar:            bool  = True
    detection_threshold: float = 150.0
    num_candidates:      int   = 15
    min_search_range_m:  float = 0.30
    z_clip_min:          float = 0.05
    z_clip_max:          float = 1.80
