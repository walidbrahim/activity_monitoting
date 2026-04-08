"""
radar_engine.preprocessing.preprocessor
==========================================
RadarFramePreprocessor — low-level frame conditioning pipeline stage.

Responsibilities (Rule: single owner of frame preparation):
    1. Antenna sign/phase correction.
    2. Adaptive clutter-map update and subtraction (ClutterMap).
    3. Spectral history ring-buffer maintenance (SpectralHistory).
    4. Warmup gate: signals abort to downstream stages during calibration.

Extracted from ActivityPipeline._step1_hardware_and_detection() without
any numerical or behavioral changes.

The ``update_masking_state()`` method must be called by the engine at the
end of each frame so that the next frame's clutter masking uses correct
tracker state (the original code read tracker state from self.* which was
updated by step 5 of the previous frame).
"""

from __future__ import annotations

import logging

import numpy as np

from radar_engine.core.base import RadarModule
from radar_engine.core.context import RadarContext, PreprocessedFrame
from radar_engine.preprocessing.clutter import ClutterMap
from radar_engine.preprocessing.buffers import SpectralHistory

logger = logging.getLogger(__name__)



class RadarFramePreprocessor(RadarModule):
    """Low-level frame conditioner: correction → clutter → history → warmup gate.

    This module is the single owner of:
    - ``ClutterMap``     — adaptive background subtraction.
    - ``SpectralHistory``— ring buffer of per-antenna complex frames.
    - ``frame_count``    — monotonically increasing frame counter.

    It writes a ``PreprocessedFrame`` into ``context.preprocessed``.
    When ``warmup_active`` is True downstream modules must skip processing.
    """

    def __init__(
        self,
        num_bins:       int,
        num_antennas:   int,
        spectral_frames: int,
        alpha:          float,
        warmup_frames:  int,
        features,       # features config object (duck-typed, from config.pipeline.features)
        confidence_threshold: int = 3,
    ) -> None:
        """
        Args:
            num_bins:            Number of range bins.
            num_antennas:        Number of receive antennas.
            spectral_frames:     Capacity of the spectral history ring buffer
                                 (= resp_window_sec × frame_rate).
            alpha:               Global clutter-map EMA learning rate.
            warmup_frames:       Frames to discard during calibration.
            features:            Feature flag object (duck-typed from config).
            confidence_threshold: Tracker confidence counter threshold used
                                  for spatial masking decisions.
        """
        self.num_bins            = num_bins
        self.num_antennas        = num_antennas
        self.spectral_frames     = spectral_frames
        self.warmup_frames       = warmup_frames
        self.features            = features
        self.confidence_threshold = confidence_threshold

        # Owned sub-components
        self.clutter_map      = ClutterMap(num_bins, num_antennas, alpha)
        self.spectral_history = SpectralHistory(num_bins, num_antennas, spectral_frames)

        # Frame counter
        self.frame_count = 0

        # Masking state injected by the engine after each frame
        # (mirrors the post-step5 state that self.* had in the original)
        self._is_occupied         = False
        self._last_target_bin:    int | None = None
        self._track_confidence:   int = 0

    # ── Engine interface ─────────────────────────────────────────────────────

    def update_masking_state(
        self,
        is_occupied:      bool,
        last_target_bin:  int | None,
        track_confidence: int,
    ) -> None:
        """Update the tracker state used for clutter spatial masking.

        The engine calls this once at the end of each successful frame so
        that the next frame's ClutterMap.update_and_subtract() has the
        correct masking parameters — exactly as the original pipeline read
        ``self.is_occupied``, ``self.last_target_bin``, etc.
        """
        self._is_occupied      = is_occupied
        self._last_target_bin  = last_target_bin
        self._track_confidence = track_confidence

    # ── RadarModule interface ─────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all internal state (called on radar pose update or track loss)."""
        self.clutter_map.reset()
        self.spectral_history.reset()
        self.frame_count        = 0
        self._is_occupied       = False
        self._last_target_bin   = None
        self._track_confidence  = 0

    def process(self, context: RadarContext) -> RadarContext:
        """Run frame conditioning for one radar frame.

        Writes ``context.preprocessed`` and returns the updated context.
        When warmup is still active, ``context.preprocessed.warmup_active``
        will be True — downstream modules must check this and skip processing.
        """
        self.frame_count += 1
        fft_1d_data = context.raw_frame

        # ── 1. Antenna sign correction ────────────────────────────────────────
        # Even-indexed antennas (0,2,4,6) have inverted phase — flip sign.
        corrected_data = np.copy(fft_1d_data)
        corrected_data[:, [0, 2, 4, 6]] *= -1

        raw_mag_profile = np.sum(np.abs(corrected_data), axis=1)

        # ── 2. Clutter suppression ────────────────────────────────────────────
        # Spectral history write always uses dynamic_data (even without clutter
        # removal) so vital_analysis never silently lacks history.
        if not getattr(self.features, 'clutter_removal', True):
            dynamic_data        = corrected_data
            dynamic_mag_profile = raw_mag_profile
        else:
            dynamic_data = self.clutter_map.update_and_subtract(
                corrected_data,
                frame_count          = self.frame_count,
                warmup_frames        = self.warmup_frames,
                is_occupied          = self._is_occupied,
                last_target_bin      = self._last_target_bin,
                track_confidence     = self._track_confidence,
                confidence_threshold = self.confidence_threshold,
                target_protection    = self.features.target_protection,
            )
            dynamic_mag_profile = np.sum(np.abs(dynamic_data), axis=1)

        # ── 3. Warmup gate ────────────────────────────────────────────────────
        # Return a PreprocessedFrame with warmup_active=True; downstream skips.
        if self.frame_count <= self.warmup_frames:
            remaining = self.warmup_frames - self.frame_count
            context.diagnostics.update({
                "warmup_active":    True,
                "warmup_remaining": remaining,
                "warmup_status":    f"Calibrating ({remaining} frames)...",
            })
            logger.debug("Warmup: %d frames remaining.", remaining)
            context.preprocessed = PreprocessedFrame(
                corrected_data      = corrected_data,
                dynamic_data        = dynamic_data,
                dynamic_mag_profile = dynamic_mag_profile,
                raw_mag_profile     = raw_mag_profile,
                warmup_active       = True,
            )
            # Note: spectral history is NOT written during warmup
            return context

        # ── 4. Post-warmup one-time reset notification ────────────────────────
        if self.frame_count == self.warmup_frames + 1:
            context.diagnostics["warmup_complete"] = True
            logger.info("Warmup complete. Spectral history starts accumulating.")

        # ── 5. Spectral history ring-buffer write (O(1)) ──────────────────────
        # Always runs (even when clutter_removal is off) so downstream consumers
        # always have valid history.
        self.spectral_history.write(dynamic_data)

        context.preprocessed = PreprocessedFrame(
            corrected_data      = corrected_data,
            dynamic_data        = dynamic_data,
            dynamic_mag_profile = dynamic_mag_profile,
            raw_mag_profile     = raw_mag_profile,
            warmup_active       = False,
        )
        return context
