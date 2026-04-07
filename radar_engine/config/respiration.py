"""radar_engine.config.respiration — Respiratory signal chain parameters."""
from dataclasses import dataclass


@dataclass(frozen=True)
class RespirationConfig:
    """Configuration for RespirationExtractor and RespirationAnalyzer.

    Legacy source: respiration.* + selected tuning.* keys.
    """
    # ── Signal window ──────────────────────────────────────────────────────────
    window_sec:          float = 30.0   # WAS: respiration.resp_window_sec
    # Frame count = window_sec × frame_rate (computed by engine at construction)

    # ── Low-pass filter (Butterworth, applied to Δφ signal) ───────────────────
    lowpass_cutoff_hz:   float = 0.5    # WAS: respiration.resp_lowpass_cutoff
    lowpass_order:       int   = 4      # WAS: respiration.resp_lowpass_order

    # ── Bin locking ───────────────────────────────────────────────────────────
    bin_stability_sec:   float = 2.0    # WAS: respiration.bin_stability_sec

    # ── Apnea detection ───────────────────────────────────────────────────────
    apnea_hold_window_sec: float = 5.0  # WAS: respiration.apnea_hold_window_sec
    apnea_merge_gap_sec:   float = 0.5  # WAS: respiration.apnea_merge_gap_sec

    # ── RR analysis ───────────────────────────────────────────────────────────
    brv_history_size:      int   = 20   # WAS: respiration.brv_history_size
    cycle_tracker_history: int   = 5    # WAS: respiration.cycle_tracker_history

    # ── Occupied-zone reflection continuity (shared with activity) ────────────
    # NOTE: also present in ActivityConfig — RespirationExtractor checks the
    # occupancy signal quality before locking onto a bin.
    resp_threshold:        float = 0.1  # WAS: respiration.resp_threshold
