"""
radar_engine.tracking.target_tracker
=======================================
TargetTracker — single-target confirmation, miss-allowance, and coordinate
smoothing pipeline stage.

Extracted from ActivityPipeline._step3_tracking() without any numerical changes.
All thresholds, EMA alphas, buffer sizes, and tethering logic are preserved
exactly as they appear in the original implementation.

State:
    track_confidence: Confidence counter (0 → confidence_threshold).
    miss_count:       Consecutive frames with no valid detection.
    coord_buffer:     Sliding window of raw XYZ observations for median filtering.
    track_x/y/z:      Current smoothed track position.
    motion_level:     EMA of inter-frame displacement magnitude.
    z_history:        Per-frame Z observations for vertical velocity estimation.
    _xy_track_hist:   Per-frame XY history for walking detection (not used yet
                      by the tracker — exposed for ActivityInferencer).
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np

from radar_engine.core.base import RadarModule
from radar_engine.core.context import RadarContext
from radar_engine.core.models import TargetCandidate, TrackedTarget
from radar_engine.config.tracking import TrackingConfig

logger = logging.getLogger(__name__)


class TargetTracker(RadarModule):
    """Single-target confidence tracker with miss-allowance and EMA smoothing.

    Mirrors the complete _step3_tracking state machine including:
    - Confidence accumulation (increment on hit, handle miss / jump rejection).
    - Coord buffer median filter (5-sample) for stable XY readout.
    - Adaptive EMA for smooth display track (fast alpha near high-confidence).
    - Vertical velocity (v_z) for fall detection downstream.
    - Motion level (EMA of inter-frame displacement) for motion classification.
    """

    def __init__(self, cfg: TrackingConfig | None = None) -> None:
        """
        Args:
            cfg: TrackingConfig dataclass.  When omitted, dataclass defaults
                 are used — useful for tests that don't need a YAML file.
        """
        if cfg is None:
            cfg = TrackingConfig()

        self.confidence_threshold = cfg.confidence_threshold
        self.miss_allowance       = cfg.miss_allowance
        self.ema_alpha_fast       = 0.4    # adaptive fast-converge alpha
        self.ema_alpha_slow       = cfg.track_ema_alpha
        self.max_jump_m           = cfg.jump_reject_distance_m
        self.coord_buffer_size    = cfg.coord_buffer_size
        self.motion_ema_alpha     = 0.15   # motion-level EMA (exposed via constant)
        self.z_history_len        = 10     # vertical velocity window

        # Per-track state
        self.track_confidence:  int          = 0
        self.miss_count:        int          = 0
        self.track_x:           float | None = None
        self.track_y:           float | None = None
        self.track_z:           float | None = None
        self.motion_level:      float        = 0.0
        self.last_target_bin:   int | None   = None
        self.last_accepted_bin: int | None   = None

        self.coord_buffer:  deque = deque(maxlen=self.coord_buffer_size)
        self.z_history:     deque = deque(maxlen=self.z_history_len)
        self._xy_track_hist:deque = deque(maxlen=30)  # 30-frame XY history

    # ── RadarModule interface ──────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all tracking state."""
        self.track_confidence  = 0
        self.miss_count        = 0
        self.track_x           = None
        self.track_y           = None
        self.track_z           = None
        self.motion_level      = 0.0
        self.last_target_bin   = None
        self.last_accepted_bin = None
        self.coord_buffer.clear()
        self.z_history.clear()
        self._xy_track_hist.clear()

    def process(self, context: RadarContext) -> RadarContext:
        """Run one tracking step.

        Reads:
            context.candidates — list[TargetCandidate]

        Writes:
            context.tracked_target — TrackedTarget (valid=True only when
                                     confidence >= threshold and miss_count==0)
        """
        if context.candidates is None:
            context.tracked_target = TrackedTarget.invalid("no_candidates")
            return context

        valid_cands = [c for c in context.candidates if c.valid]

        # ── 1. Select best candidate ──────────────────────────────────────────
        best: TargetCandidate | None = None
        if valid_cands:
            best = max(valid_cands, key=lambda c: c.magnitude)

        # ── 2. Jump rejection ─────────────────────────────────────────────────
        jump_rejected = False
        if best is not None and self.track_x is not None:
            dist = np.sqrt(
                (best.x_m - self.track_x) ** 2 +
                (best.y_m - self.track_y) ** 2 +
                (best.z_m - self.track_z) ** 2
            )
            if dist > self.max_jump_m and self.track_confidence >= self.confidence_threshold:
                jump_rejected = True
                context.diagnostics["jump_reject_dist_m"] = dist
                logger.debug("Jump rejected: dist=%.2f m > %.2f m", dist, self.max_jump_m)
                best = None

        # ── 3. Miss / hit state machine ───────────────────────────────────────
        if best is None:
            self.miss_count += 1
            if self.miss_count > self.miss_allowance:
                # Track lost
                self.track_confidence  = 0
                self.miss_count        = 0
                self.track_x           = None
                self.track_y           = None
                self.track_z           = None
                self.coord_buffer.clear()
                self.last_target_bin   = None
                context.tracked_target = TrackedTarget.invalid(
                    "jump_reject" if jump_rejected else "miss_exceeded"
                )
                return context
        else:
            self.miss_count = 0
            if self.track_confidence < self.confidence_threshold:
                self.track_confidence += 1
            self.last_target_bin = best.bin_index

        # ── 4. Coordinate initialisation or EMA update ────────────────────────
        if best is not None:
            if self.track_x is None:
                # First detection — initialise from raw candidate
                self.track_x = best.x_m
                self.track_y = best.y_m
                self.track_z = best.z_m
            else:
                # Adaptive EMA: faster when confidence is low (still converging)
                alpha = (self.ema_alpha_fast
                         if self.track_confidence < self.confidence_threshold
                         else self.ema_alpha_slow)
                self.track_x = alpha * best.x_m + (1 - alpha) * self.track_x
                self.track_y = alpha * best.y_m + (1 - alpha) * self.track_y
                self.track_z = alpha * best.z_m + (1 - alpha) * self.track_z

            # Coord buffer for median-filtered readout
            self.coord_buffer.append((best.x_m, best.y_m, best.z_m))

        if self.track_x is None:
            context.tracked_target = TrackedTarget.invalid("no_initial_position")
            return context

        # ── 5. Median-filtered stable position ────────────────────────────────
        if self.coord_buffer:
            buf_arr  = np.array(self.coord_buffer)
            stable_x = float(np.median(buf_arr[:, 0]))
            stable_y = float(np.median(buf_arr[:, 1]))
            stable_z = float(np.median(buf_arr[:, 2]))
        else:
            stable_x, stable_y, stable_z = self.track_x, self.track_y, self.track_z

        # ── 6. Motion level (EMA of inter-frame displacement) ─────────────────
        if self._xy_track_hist:
            prev_x, prev_y = self._xy_track_hist[-1]
            disp = np.sqrt((stable_x - prev_x) ** 2 + (stable_y - prev_y) ** 2)
            self.motion_level = (self.motion_ema_alpha * disp +
                                 (1 - self.motion_ema_alpha) * self.motion_level)
        self._xy_track_hist.append((stable_x, stable_y))

        # ── 7. Vertical velocity for fall detection ───────────────────────────
        self.z_history.append(stable_z)
        if len(self.z_history) >= 2:
            v_z = float(self.z_history[-1] - self.z_history[-2])
        else:
            v_z = 0.0

        # ── 8. Assemble TrackedTarget ─────────────────────────────────────────
        # Keep a confirmed track valid during short miss bursts to avoid
        # occupancy flicker between "No Occupant" and "Occupied".
        is_confirmed = (
            self.track_confidence >= self.confidence_threshold
            and self.miss_count <= self.miss_allowance
        )
        if is_confirmed and self.miss_count > 0:
            validity_reason = "intermittent_miss_hold"
        else:
            validity_reason = None if is_confirmed else "insufficient_confidence"

        context.tracked_target = TrackedTarget(
            bin_index      = self.last_target_bin or 0,
            x_m            = best.x_m if best else self.track_x,
            y_m            = best.y_m if best else self.track_y,
            z_m            = best.z_m if best else self.track_z,
            smoothed_x_m   = stable_x,
            smoothed_y_m   = stable_y,
            smoothed_z_m   = stable_z,
            confidence     = self.track_confidence,
            miss_count     = self.miss_count,
            v_z            = v_z,
            valid          = is_confirmed,
            validity_reason = validity_reason,
        )

        return context
