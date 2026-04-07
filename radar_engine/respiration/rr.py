"""
radar_engine.respiration.rr
=============================
BreathCycleTracker — frame-based breath-cycle timing and RR estimation.

Extracted from respirationPipeline.py (class BreathCycleTracker, lines 112–175)
without any behavioural changes.

Uses exhale-peak frame indices to compute per-cycle durations and derives:
    - rolling average RR (BPM)
    - breathing-rate variability (BRV, SDNN analogue)

Owned by: RespirationAnalyzer (one instance per pipeline lifetime).
"""

from __future__ import annotations

import numpy as np


class BreathCycleTracker:
    """Tracks breath cycles using exhale-peak global frame indices.

    Each exhale peak is converted to an absolute global frame index.
    Consecutive peak pairs yield a cycle duration; only durations in the
    physiologically plausible range (0.5 – 6.0 s, i.e. 10 – 120 BPM)
    are accepted.

    State:
        count:                  Total accepted breath cycles since reset.
        cycle_durations:        List of accepted cycle durations [seconds].
        _last_peak_global_frame: Global frame index of the last accepted exhale.
    """

    def __init__(self, history_size: int = 5, fps: float = 25.0) -> None:
        """
        Args:
            history_size: Number of most-recent cycles to average for RR.
            fps:          Frame rate [Hz].
        """
        self.history_size = history_size
        self.fps          = fps
        self.count:                   int          = 0
        self.cycle_durations:         list[float]  = []
        self._last_peak_global_frame: int | None   = None

    def update(
        self,
        peak_indices:    list[int],
        global_frame_idx: int,
        buffer_len:       int,
    ) -> None:
        """Process a list of exhale-peak buffer indices for one frame.

        Args:
            peak_indices:     Sorted list of exhale peak indices relative to
                              the current display_signal buffer.
            global_frame_idx: Cumulative frame count since tracking started.
            buffer_len:       Length of the display_signal buffer.
        """
        if not peak_indices:
            return

        global_offset = global_frame_idx - buffer_len

        for idx in peak_indices:
            global_frame = global_offset + idx

            if self._last_peak_global_frame is None:
                self._last_peak_global_frame = global_frame
                continue

            duration_sec = (global_frame - self._last_peak_global_frame) / self.fps

            if 0.5 < duration_sec < 6.0:
                # Valid breath cycle
                self.cycle_durations.append(duration_sec)
                self.count += 1
                self._last_peak_global_frame = global_frame
            elif duration_sec >= 6.0:
                # Too long (likely apnea gap): reset anchor without counting
                self._last_peak_global_frame = global_frame
            # < 0.5 s: noise — ignore, do not update anchor

    # ── Derived metrics ───────────────────────────────────────────────────────

    @property
    def last_duration(self) -> float:
        """Duration of the most recent accepted breath cycle [seconds]."""
        return self.cycle_durations[-1] if self.cycle_durations else 0.0

    def get_rr_avg(self) -> float:
        """Rolling average respiratory rate [BPM] over the last history_size cycles."""
        if not self.cycle_durations:
            return 0.0
        last_n  = self.cycle_durations[-self.history_size:]
        avg_dur = float(np.mean(last_n))
        return 60.0 / avg_dur if avg_dur > 0 else 0.0

    def get_brv(self, n: int = 20) -> float:
        """Breathing-rate variability: std of the last n cycle durations [seconds]."""
        if len(self.cycle_durations) < 5:
            return 0.0
        return float(np.std(self.cycle_durations[-n:]))

    def reset(self) -> None:
        """Clear all cycle history."""
        self.count                    = 0
        self.cycle_durations          = []
        self._last_peak_global_frame  = None
