"""
radar_engine.preprocessing.buffers
=====================================
SpectralHistory — ring-buffered per-antenna complex frame history.

Extracted from ActivityPipeline.__init__ (allocation) and
_step1_hardware_and_detection (write + reorder logic) without any changes.

The buffer stores the last ``num_frames`` complex frames for all range bins
and antennas and can produce a chronologically ordered view on demand.

Ownership: RadarFramePreprocessor is the sole owner of a SpectralHistory instance.
Downstream consumers (VitalFeatureExtractor, RespirationExtractor) receive the
ordered cube through RadarContext.preprocessed or directly from the engine.
"""

from __future__ import annotations

import numpy as np


class SpectralHistory:
    """Ring-buffered 3-D complex frame history.

    Shape: (num_bins, num_antennas, num_frames)

    The write pointer advances one slot per frame (O(1), no allocation).

    When a chronological view is requested the buffer is reordered by
    rolling the frame axis so index 0 is the oldest frame and index -1
    is the most recent — matching the original ring-buffer reorder logic.
    """

    def __init__(self, num_bins: int, num_antennas: int, num_frames: int) -> None:
        """
        Args:
            num_bins:     Number of range bins.
            num_antennas: Number of receive antennas.
            num_frames:   Ring-buffer capacity (= resp_window_sec × frame_rate).
        """
        self.num_bins     = num_bins
        self.num_antennas = num_antennas
        self.num_frames   = num_frames

        self._buf      = np.zeros((num_bins, num_antennas, num_frames), dtype=complex)
        self._ring_idx = 0  # next write slot (unbounded; modulo for access)

    # ── Write ────────────────────────────────────────────────────────────────

    def write(self, frame_data: np.ndarray) -> None:
        """Store one complex frame into the ring buffer.

        Args:
            frame_data: Shape (num_bins, num_antennas) complex array.
        """
        slot = self._ring_idx % self.num_frames
        self._buf[:, :, slot] = frame_data
        self._ring_idx += 1

    # ── Read — ordered views ─────────────────────────────────────────────────

    def get_ordered_cube(self) -> np.ndarray:
        """Return a chronologically ordered 3-D cube.

        The returned array has shape (num_bins, num_antennas, num_frames)
        with index [:, :, 0] being the oldest stored frame and
        [:, :, -1] being the most recent.

        This exactly replicates the ring-buffer reorder used in
        ActivityPipeline.process_frame() before exporting raw_spectral_cube.
        """
        idx = self._ring_idx % self.num_frames
        if idx != 0:
            return np.concatenate(
                [self._buf[:, :, idx:], self._buf[:, :, :idx]], axis=2
            )
        return self._buf.copy()

    def get_ordered_2d(self) -> np.ndarray:
        """Return the antenna-summed, chronologically ordered 2-D history.

        Shape: (num_bins, num_frames).
        This is the ``spectral_history`` field consumed by RespiratoryPipelineV2
        (antenna sum over axis=1, same reorder as get_ordered_cube).
        """
        cube = self.get_ordered_cube()
        return np.sum(cube, axis=1)

    def get_bin_history(self, bin_idx: int) -> np.ndarray:
        """Return the chronologically ordered per-antenna history for one bin.

        Shape: (num_antennas, num_frames).
        Used by VitalFeatureExtractor when beamforming a specific candidate.
        Replicates the single-bin reorder in _step2_candidate_generation.
        """
        idx     = self._ring_idx % self.num_frames
        raw     = self._buf[bin_idx, :, :]         # (antennas, frames) view
        if idx != 0:
            return np.concatenate([raw[:, idx:], raw[:, :idx]], axis=1)
        return raw.copy()

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def ring_idx(self) -> int:
        """Current (unbounded) write index."""
        return self._ring_idx

    @property
    def frames_written(self) -> int:
        """Total number of frames written so far."""
        return self._ring_idx

    @property
    def is_full(self) -> bool:
        """True once the buffer has been filled at least once."""
        return self._ring_idx >= self.num_frames

    # ── State management ─────────────────────────────────────────────────────

    def reset(self) -> None:
        """Zero the buffer and reset the write pointer."""
        self._buf.fill(0)
        self._ring_idx = 0
