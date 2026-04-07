"""
radar_engine.preprocessing.clutter
=====================================
ClutterMap — spatial-masked exponential moving-average clutter suppression.

Ownership: RadarFramePreprocessor is the sole owner of a ClutterMap instance.
"""

from __future__ import annotations

import numpy as np


class ClutterMap:
    """Adaptive clutter map with spatially protected EMA learning.

    The clutter map builds a per-bin, per-antenna estimate of the static
    environment. Near a confirmed target the learning rate is frozen so that
    the target's own reflection does not evaporate into the background.

    State:
        _map:  Complex clutter estimate; shape (num_bins, num_antennas).
    """

    def __init__(self, num_bins: int, num_antennas: int, alpha: float) -> None:
        """
        Args:
            num_bins:     Number of range bins (first dimension).
            num_antennas: Number of receive antennas (second dimension).
            alpha:        Global EMA learning rate (0 < alpha < 1); smaller =
                          slower adaptation.  Typical value: 0.005–0.02.
        """
        self.num_bins     = num_bins
        self.num_antennas = num_antennas
        self.alpha        = alpha
        self._map         = np.zeros((num_bins, num_antennas), dtype=complex)

    # ── Core update ──────────────────────────────────────────────────────────

    def update_and_subtract(
        self,
        corrected_data:      np.ndarray,   # (num_bins, num_antennas) complex
        frame_count:         int,
        warmup_frames:       int,
        is_occupied:         bool,
        last_target_bin:     int | None,
        track_confidence:    int,
        confidence_threshold: int,
    ) -> np.ndarray:
        """Update the clutter map and return the clutter-subtracted frame.

        Learning rate rules (identical to original):
        - During warmup: fast convergence (alpha = 0.3) to bootstrap the map.
        - Post-warmup, unConfirmed track or empty room: global alpha.
        - Post-warmup, confirmed occupied target: spatially masked alpha
          (frozen near the target bin, full alpha far away).

        Args:
            corrected_data:       Antenna-corrected complex frame.
            frame_count:          Current cumulative frame index (1-based).
            warmup_frames:        Total number of warmup frames.
            is_occupied:          Whether a target is currently confirmed.
            last_target_bin:      Range bin of the last confirmed target; None
                                  if no target.
            track_confidence:     Current tracker confidence counter value.
            confidence_threshold: Threshold the counter must reach to be
                                  considered confirmed.

        Returns:
            dynamic_data: clutter_subtracted frame; same shape as corrected_data.
        """
        if frame_count <= warmup_frames:
            # Fast warmup convergence — same for all bins
            current_alpha_array = np.full(self.num_bins, 0.3)

        elif (not is_occupied or
              last_target_bin is None or
              track_confidence < confidence_threshold):
            # Evaporate ghosts globally while empty or unconfirmed
            current_alpha_array = np.full(self.num_bins, self.alpha)

        else:
            # Spatially Masked Learning:
            # freeze clutter learning near the confirmed target bin.
            # Protection radius: bins within 2 of the target are frozen
            # (alpha ≈ 0.001); alpha ramps back to global over the next 8 bins.
            dist_bins        = np.abs(np.arange(self.num_bins) - last_target_bin)
            protection_mask  = np.clip((dist_bins - 2.0) / 8.0, 0.0, 1.0)
            current_alpha_array = 0.001 + protection_mask * (self.alpha - 0.001)

        alpha_matrix = current_alpha_array[:, np.newaxis]  # broadcast over antennas
        self._map    = (alpha_matrix * corrected_data) + ((1.0 - alpha_matrix) * self._map)

        return corrected_data - self._map

    # ── State management ─────────────────────────────────────────────────────

    def reset(self) -> None:
        """Zero the clutter map (called on radar pose update or hard reset)."""
        self._map.fill(0)

    @property
    def map(self) -> np.ndarray:
        """Read-only view of the current clutter estimate."""
        return self._map
