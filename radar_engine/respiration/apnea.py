"""
radar_engine.respiration.apnea
================================
ApneaTracker — cross-frame apnea event deduplication and counting.

Extracted from respirationPipeline.py (class ApneaTracker, lines 60–106)
without any behavioural changes.

Tracks apnea episodes using global frame indices to merge overlapping or
adjacent segments produced each frame and prevent double-counting.

Owned by: RespirationAnalyzer (one instance per pipeline lifetime).
"""

from __future__ import annotations


class ApneaTracker:
    """Stateful apnea event deduplicator using global frame-index tracking.

    Each call to ``update()`` receives the current frame's apnea segments
    expressed as buffer-relative (start, end) index pairs. The tracker
    converts them to global frame indices and merges any overlapping or
    near-adjacent events (within 0.5 s tolerance) so that a rolling apnea
    region does not increment the count on every frame.

    State:
        count:           Total distinct apnea events since reset.
        durations_sec:   Per-event duration [seconds].
        _global_events:  Internal list of [global_start, global_end] pairs.
    """

    def __init__(self) -> None:
        self.count:           int         = 0
        self.durations_sec:   list[float] = []
        self._global_events:  list[list[int]] = []  # [[g_start, g_end], ...]

    def update(
        self,
        segments:           list[tuple[int, int]],
        fps:                float,
        current_buffer_len: int,
        global_frame_idx:   int,
    ) -> None:
        """Merge new apnea segments into the global event registry.

        Args:
            segments:           List of (start_idx, end_idx) relative to the
                                current live_signal buffer (buffer-relative).
            fps:                Frame rate [Hz]; used for duration conversion and
                                overlap tolerance calculation.
            current_buffer_len: Length of the live_signal buffer (= window_frames).
            global_frame_idx:   Cumulative frame count since tracking started.
        """
        global_offset = global_frame_idx - current_buffer_len
        tolerance     = int(0.5 * fps)  # merge events within 0.5 s gap

        for (s, e) in segments:
            if e <= 0 or s >= current_buffer_len:
                continue  # segment fully outside the current buffer

            g_start = global_offset + s
            g_end   = global_offset + e

            matched = False
            if self._global_events:
                last_ev = self._global_events[-1]
                if g_start <= (last_ev[1] + tolerance):
                    # Extend the last event if this segment expands it
                    matched = True
                    if g_end > last_ev[1]:
                        last_ev[1] = g_end
                        self.durations_sec[-1] = (last_ev[1] - last_ev[0]) / fps

            if not matched:
                self._global_events.append([g_start, g_end])
                self.count += 1
                self.durations_sec.append((g_end - g_start) / fps)

    def reset(self) -> None:
        """Clear all event history."""
        self.count        = 0
        self.durations_sec.clear()
        self._global_events.clear()
