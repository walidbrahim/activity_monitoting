"""
radar_engine.detection.cfar
============================
Cell-Averaging CFAR (Constant False Alarm Rate) threshold computation.

Extracted from ActivityPipeline._compute_cfar_threshold() without any
numerical changes. This is a pure stateless function.

Ownership: TargetDetector calls this for each peak candidate.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_cfar_threshold(
    profile:  np.ndarray,
    bin_idx:  int,
    window:   int   = 6,
    guard:    int   = 2,
    scale:    float = 1.7,
) -> float:
    """Median CA-CFAR threshold for a single cell-under-test.

    Uses median instead of mean for the local noise floor estimate, making the
    threshold robust against sidelobe contamination from nearby strong movers.

    Args:
        profile:  1-D magnitude profile (e.g. dynamic_mag_profile).
        bin_idx:  Index of the cell under test.
        window:   Number of reference cells on each side of the CUT.
        guard:    Number of guard cells adjacent to the CUT excluded from the
                  noise estimate (prevents the CUT leaking into its own noise).
        scale:    Detection multiplier applied to the noise-floor estimate.

    Returns:
        CFAR threshold for this CUT. Always positive.
    """
    start = max(0, bin_idx - window)
    end   = min(len(profile), bin_idx + window + 1)
    sub_prof = profile[start:end]

    g_start = max(0, bin_idx - start - guard)
    g_end   = min(len(sub_prof), bin_idx - start + guard + 1)

    noise_cells = np.concatenate([sub_prof[:g_start], sub_prof[g_end:]])
    if len(noise_cells) == 0:
        return float(profile[bin_idx]) * 0.5

    cfar_thresh = float(np.median(noise_cells)) * scale
    logger.debug(
        "CFAR bin=%d noise_median=%.1f thresh=%.1f",
        bin_idx, np.median(noise_cells), cfar_thresh,
    )
    return cfar_thresh
