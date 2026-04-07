"""
radar_engine.respiration.analyzer
=====================================
RespirationAnalyzer — physiological metrics from a RespirationSignal.

Extracted from RespiratoryPipelineV2.process() (analysis path only):
  - Apnea detection state machine (step: Detect Apnea, lines 697–723).
  - ApneaTracker segment deduplication.
  - Breath-cycle / RR computation with anchor-based dropout (lines 727–743).
  - Depth estimation from recent peak-trough amplitude (lines 751–767).
  - BRV calculation (lines 769–772).

All thresholds, timing constants, and state logic are preserved exactly.

Ownership: RespirationAnalyzer consumes RespirationSignal and writes
RespirationMetrics into RadarContext.
"""

from __future__ import annotations

import logging

import numpy as np

from radar_engine.core.base import RadarModule
from radar_engine.core.context import RadarContext
from radar_engine.core.models import QualityFlag, RespirationMetrics
from radar_engine.respiration.apnea import ApneaTracker
from radar_engine.respiration.peaks import detect_respiratory_peaks
from radar_engine.respiration.rr import BreathCycleTracker
from radar_engine.config.respiration import RespirationConfig

logger = logging.getLogger(__name__)


class RespirationAnalyzer(RadarModule):
    """Produces physiological metrics from the conditioned respiration signal.

    This module consumes ``context.respiration_signal`` (written by
    RespirationExtractor) and computes:
        - Per-frame apnea detection (5 s window, dynamic threshold).
        - ApneaTracker cross-frame deduplication and count.
        - Breath-cycle RR with anchor-based dropout on long silences.
        - BRV (breathing-rate variability, SDNN).
        - Depth classification from peak-trough amplitude ratio.

    State:
        apnea_tracker:   ApneaTracker (persistent cross-frame deduplication).
        cycle_tracker:   BreathCycleTracker (persistent cycle timing).
        apnea_trace:     Rolling boolean buffer (window_frames,).
        rr_history_buf:  Rolling RR buffer (window_frames,).
        apnea_active:    Last-frame apnea flag.
        live_apnea_frames: Frames elapsed since apnea went active.
    """

    def __init__(
        self,
        cfg:        RespirationConfig | None = None,
        frame_rate: float                    = 25.0,
    ) -> None:
        """
        Args:
            cfg:        RespirationConfig dataclass.  None → dataclass defaults.
            frame_rate: Radar frame rate [Hz].
        """
        if cfg is None:
            cfg = RespirationConfig()

        self.fps           = frame_rate
        self.window_frames = int(cfg.window_sec * self.fps)

        # Stateful trackers (verbatim from V2.__init__)
        self.apnea_tracker = ApneaTracker()
        self.cycle_tracker = BreathCycleTracker(
            history_size = cfg.cycle_tracker_history,
            fps          = self.fps,
        )

        # Rolling buffers
        self.apnea_trace:    np.ndarray = np.zeros(self.window_frames, dtype=bool)
        self.rr_history_buf: np.ndarray = np.zeros(self.window_frames)

        # Apnea state
        self.apnea_active:      bool = False
        self.live_apnea_frames: int  = 0

        # Cycle-RR anchor for dropout handling
        self._last_peak_global: int | None = None

        # Internal frame counter (mirrors extractor's _global_frame_idx)
        self._global_frame_idx: int = 0

    # ── RadarModule interface ──────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all analysis state."""
        self.apnea_tracker.reset()
        self.cycle_tracker.reset()
        self.apnea_trace        = np.zeros(self.window_frames, dtype=bool)
        self.rr_history_buf     = np.zeros(self.window_frames)
        self.apnea_active       = False
        self.live_apnea_frames  = 0
        self._last_peak_global  = None
        self._global_frame_idx  = 0

    def process(self, context: RadarContext) -> RadarContext:
        """Run physiological analysis for one frame batch.

        Reads:
            context.respiration_signal              — from RespirationExtractor.
            context.diagnostics["frames_in_tick"]   — frame batch size.
            context.respiration_signal.apnea_threshold — extractor threshold.
            context.respiration_signal.threshold_calibrated — extractor flag.
            context.respiration_signal.derivative_signal — normalised deriv buffer.
            context.respiration_signal.live_signal — frozen display signal.

        Writes:
            context.respiration_metrics — RespirationMetrics.
        """
        resp_sig   = context.respiration_signal
        frames     = int(context.diagnostics.get("frames_in_tick", 1))

        if resp_sig is None or not resp_sig.quality.is_usable:
            context.respiration_metrics = RespirationMetrics.empty(self.window_frames)
            return context

        self._global_frame_idx += frames

        # Pull pre-computed signal state from typed RespirationSignal.
        scale_derivative     = resp_sig.derivative_signal
        display_signal       = resp_sig.live_signal
        apnea_threshold      = float(resp_sig.apnea_threshold)
        threshold_calibrated = bool(resp_sig.threshold_calibrated)

        # ── Apnea detection ───────────────────────────────────────────────────
        # V2 algorithm: 5 s window, P95 of derivative ≤ threshold → apnea
        apnea_len = int(5.0 * self.fps)

        if len(scale_derivative) > apnea_len and threshold_calibrated:
            recent_deriv = scale_derivative[-apnea_len:]
            curr_apnea   = bool(np.percentile(recent_deriv, 95) <= apnea_threshold)

            if curr_apnea and not self.apnea_active:
                self.apnea_active = True
                self.apnea_trace[-apnea_len:] = True
                self.live_apnea_frames = apnea_len
            elif not curr_apnea:
                self.apnea_active = False

        self.apnea_trace = np.roll(self.apnea_trace, -frames)
        self.apnea_trace[-frames:] = self.apnea_active

        if self.apnea_active:
            self.live_apnea_frames += frames
        else:
            self.live_apnea_frames = 0

        # ── Apnea segment array from trace ────────────────────────────────────
        trace_view = self.apnea_trace[-len(scale_derivative):].astype(int)
        diffs      = np.diff(np.concatenate(([0], trace_view, [0])))
        starts     = np.where(diffs ==  1)[0]
        ends       = np.where(diffs == -1)[0]
        apnea_segments = list(zip(starts.tolist(), ends.tolist()))

        self.apnea_tracker.update(
            apnea_segments, self.fps, len(scale_derivative), self._global_frame_idx
        )

        # ── Peak detection + RR ───────────────────────────────────────────────
        troughs, peaks = detect_respiratory_peaks(display_signal, self.fps)

        if peaks:
            self.cycle_tracker.update(peaks, self._global_frame_idx, len(display_signal))
            self._last_peak_global = (
                self._global_frame_idx - len(display_signal) + peaks[-1]
            )

        # Anchor-based RR dropout (V2 logic: degrade RR when last peak is stale)
        base_rr = self.cycle_tracker.get_rr_avg()
        if self._last_peak_global is not None and base_rr > 0:
            time_since_last = (self._global_frame_idx - self._last_peak_global) / self.fps
            avg_dur = 60.0 / base_rr
            if time_since_last > avg_dur:
                current_rr = 60.0 / time_since_last
            else:
                current_rr = base_rr
        else:
            current_rr = base_rr

        # Zero RR when apnea and rate is implausibly low (< 6 BPM)
        if self.apnea_active and current_rr < 6.0:
            current_rr = 0.0

        self.rr_history_buf = np.roll(self.rr_history_buf, -frames)
        self.rr_history_buf[-frames:] = current_rr

        # ── Depth estimation ──────────────────────────────────────────────────
        depth_str = "--"
        if not self.apnea_active and peaks and troughs:
            recent_cutoff  = max(0, len(display_signal) - int(10 * self.fps))
            recent_peaks   = [p for p in peaks   if p >= recent_cutoff]
            recent_troughs = [t for t in troughs if t >= recent_cutoff]

            if recent_peaks and recent_troughs:
                lp = recent_peaks[-1]
                lt = recent_troughs[-1]
                if lp < len(display_signal) and lt < len(display_signal):
                    depth_val = abs(display_signal[lp] - display_signal[lt])
                    if depth_val < 5.0:
                        depth_str = "shallow"
                    elif depth_val > 15.0:
                        depth_str = "deep"
                    else:
                        depth_str = "normal"

        # ── BRV ───────────────────────────────────────────────────────────────
        brv_val = 0.0
        if len(self.cycle_tracker.cycle_durations) > 1:
            brv_val = float(np.std(self.cycle_tracker.cycle_durations))

        # ── RR quality flag ───────────────────────────────────────────────────
        if not threshold_calibrated:
            rr_quality = QualityFlag.invalid("calibrating")
        elif current_rr <= 0:
            rr_quality = QualityFlag.invalid("no_rr")
        elif self.apnea_active:
            rr_quality = QualityFlag.degraded(score=0.3, reason="apnea_active")
        else:
            rr_quality = QualityFlag.valid(score=0.9)

        apnea_quality = (
            QualityFlag.invalid("not_calibrated")
            if not threshold_calibrated
            else QualityFlag.valid()
        )

        context.respiration_metrics = RespirationMetrics(
            rr_bpm            = current_rr if current_rr > 0 else None,
            rr_quality        = rr_quality,
            cycle_duration_s  = self.cycle_tracker.last_duration,
            cycle_count       = self.cycle_tracker.count,
            inhale_indices    = troughs,
            exhale_indices    = peaks,
            breath_depth      = depth_str,
            brv_value         = brv_val,
            apnea_active      = self.apnea_active,
            apnea_quality     = apnea_quality,
            apnea_duration_s  = self.live_apnea_frames / self.fps,
            apnea_event_count = self.apnea_tracker.count,
            rr_history        = np.copy(self.rr_history_buf),
            apnea_trace       = np.copy(self.apnea_trace),
            apnea_segments    = apnea_segments,
        )

        return context
