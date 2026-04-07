"""
radar_engine.respiration.peaks
================================
detect_respiratory_peaks — robust inhale/exhale marker detection.

Extracted from respirationPipeline.py (top-level function, lines 31–53)
without any numerical changes.

The function detects breath cycle markers from the respiration waveform:
    troughs → inhale onsets (rising phase start)
    peaks   → exhale onsets (falling phase start)

Used by RespirationAnalyzer after the signal has been filtered and buffered.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def detect_respiratory_peaks(
    signal_data: np.ndarray,
    fs:          float,
) -> tuple[list[int], list[int]]:
    """Detect inhale-onset troughs and exhale-onset peaks in a respiration signal.

    Args:
        signal_data: 1-D respiratory displacement or velocity waveform.
        fs:          Signal sample rate [Hz].

    Returns:
        Tuple (troughs, peaks):
            troughs: Sorted list of buffer indices for inhale onsets.
            peaks:   Sorted list of buffer indices for exhale onsets.
    """
    signal_data = np.asarray(signal_data)

    # Distance constraint: minimum separation between consecutive peaks (~0.5 s)
    min_dist = max(1, int(fs * 0.5))

    # Dynamic prominence: peaks must be ≥ 25 % of the signal's peak-to-peak range
    sig_range  = np.ptp(signal_data)
    prominence = max(0.05, sig_range * 0.25)

    # Exhale onsets (peaks of the waveform)
    peaks,   _ = find_peaks( signal_data, distance=min_dist, prominence=prominence)
    # Inhale onsets (troughs → peaks of the negated waveform)
    troughs, _ = find_peaks(-signal_data, distance=min_dist, prominence=prominence)

    return troughs.tolist(), peaks.tolist()
