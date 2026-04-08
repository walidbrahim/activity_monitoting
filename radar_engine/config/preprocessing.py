"""radar_engine.config.preprocessing — Clutter suppression and warmup parameters."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuration for the RadarFramePreprocessor pipeline stage.

    Args:
        clutter_ema_alpha:     EMA learning rate for the clutter map.
                               Lower = slower adaptation (more aggressive suppression).
                               YAML key: preprocessing.clutter_ema_alpha
                               Legacy key: pipeline.alpha
        static_clutter_margin: Additive margin on top of the clutter map used to
                               compute the static threshold before CFAR.
                               YAML key: preprocessing.static_clutter_margin
                               Legacy key: pipeline.static_margin
        warmup_frames:         Number of frames the preprocessor runs in warmup
                               mode (building the initial clutter map) before
                               passing data downstream.
                               YAML key: preprocessing.warmup_frames
                               Legacy key: radar.ti_1dfft_queue_len
        features_clutter_removal:   Enable/disable clutter suppression step.
        features_vital_analysis:    Enable/disable vital-feature extraction.
        features_temporal_persistence: Enable/disable temporal track persistence.
        features_adaptive_smoothing:   Enable/disable adaptive EMA smoothing.
    """
    clutter_ema_alpha:     float = 0.05
    static_clutter_margin: float = 200.0
    warmup_frames:         int   = 25

    # Feature flags (mirrors legacy pipeline.features.*)
    features_clutter_removal:      bool = True
    features_vital_analysis:       bool = True
    features_temporal_persistence: bool = True
    features_adaptive_smoothing:   bool = True
    features_fall_posture:         bool = True
    features_apnea_state:          bool = True
    features_target_protection:    bool = True
