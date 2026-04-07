"""
radar_engine.detection.target_detector
=========================================
TargetDetector — per-frame candidate generation pipeline stage.

Extracted from ActivityPipeline._step2_candidate_generation() and
ActivityPipeline._score_candidates() without any numerical changes.

Orchestration:
    1. Peak finding in dynamic_mag_profile (scipy.signal.find_peaks + NMS).
    2. CFAR gate: reject peaks below their local CFAR threshold.
    3. Localization: estimate azimuth, elevation, world-frame XYZ per peak.
    4. Spatial zoning: assign zone label; reject out-of-bounds / ignored peaks.
    5. Vital features: compute VitalFeatures (aliveness, micro_state, vital_mult).
    6. Composite scoring: rank candidates.

Outputs list[TargetCandidate] + dict[int, VitalFeatures] into RadarContext.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.signal import find_peaks

from radar_engine.core.base import RadarModule
from radar_engine.core.context import RadarContext
from radar_engine.core.enums import MicroState
from radar_engine.core.models import TargetCandidate, VitalFeatures
from radar_engine.detection.cfar import compute_cfar_threshold
from radar_engine.detection.localization import estimate_candidate_geometry
from radar_engine.detection.zoning import ZoneEvaluator
from radar_engine.activity.vital_features import VitalFeatureExtractor

logger = logging.getLogger(__name__)

# ── Constants (mirrors ActivityPipeline._step2_candidate_generation) ──────────

_NON_MAX_WINDOW  = 2   # bins: suppress peaks closer than this to a stronger peak
_MIN_PEAK_HEIGHT = 50  # absolute magnitude floor before CFAR
_CFAR_WINDOW     = 6   # reference cells per side
_CFAR_GUARD      = 2   # guard cells per side
_CFAR_SCALE      = 1.7 # detection multiplier


class TargetDetector(RadarModule):
    """Per-frame candidate detector (stateless w.r.t. tracking).

    This module scores raw peaks from the magnitude profile and returns a
    typed list of TargetCandidate objects, an optional VitalFeatures per
    candidate, and marks the best candidate for the tracker.

    State:
        Only ``VitalFeatureExtractor`` carries inter-frame state (macro_timers).
        Reset propagates to it.
    """

    def __init__(
        self,
        range_resolution: float,
        frame_rate:       float,
        R:                np.ndarray,         # 3×3 rotation matrix from radar pose
        T:                np.ndarray,         # translation vector [x, y, z] metres
        features,
        zone_evaluator:   ZoneEvaluator | None = None,
    ) -> None:
        """
        Args:
            range_resolution: Metres per range bin.
            frame_rate:       Radar frame rate [Hz].
            R:                Rotation matrix (world ← radar-frame).
            T:                Translation vector (radar position in world frame).
            features:         Feature flags (duck-typed from config).
            zone_evaluator:   Injected ZoneEvaluator instance.  When None the
                              deprecated free-function shim is used instead
                              (backward compat for legacy call-sites).
        """
        self.range_res      = range_resolution
        self.features       = features
        self.R              = R
        self.T              = T
        if zone_evaluator is None:
            raise ValueError(
                "TargetDetector requires an explicit ZoneEvaluator. "
                "Pass one via the engine\'s cfg.layout."
            )
        self._zone_evaluate = zone_evaluator

        self._vital_extractor = VitalFeatureExtractor(frame_rate)

    # ── RadarModule interface ──────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear vital-feature inter-frame state (macro_timers)."""
        self._vital_extractor.reset()

    def process(self, context: RadarContext) -> RadarContext:
        """Run candidate detection for one radar frame.

        Reads:
            context.preprocessed — must not be None; must not be in warmup.

        Writes:
            context.candidates           — list[TargetCandidate] (valid + rejected)
            context.all_vital_features   — dict[bin_idx → VitalFeatures]
        """
        if context.preprocessed is None or context.preprocessed.warmup_active:
            context.candidates           = []
            context.all_vital_features   = {}
            return context

        pre             = context.preprocessed
        mag_profile     = pre.dynamic_mag_profile
        corrected_data  = pre.corrected_data
        dynamic_data    = pre.dynamic_data
        spectral_history = context.spectral_history

        # ── 1. Peak detection ─────────────────────────────────────────────────
        min_h    = _MIN_PEAK_HEIGHT
        raw_peaks, _ = find_peaks(mag_profile, height=min_h, distance=_NON_MAX_WINDOW)

        if len(raw_peaks) == 0:
            context.candidates         = []
            context.all_vital_features = {}
            return context

        # Sort descending by magnitude
        sorted_peaks = sorted(raw_peaks, key=lambda b: mag_profile[b], reverse=True)

        all_vital: dict[int, VitalFeatures] = {}
        candidates: list[TargetCandidate]   = []

        for cand_bin in sorted_peaks:
            mag = float(mag_profile[cand_bin])

            # ── 2. CFAR gate ──────────────────────────────────────────────────
            cfar_thresh = compute_cfar_threshold(
                mag_profile, cand_bin,
                window=_CFAR_WINDOW, guard=_CFAR_GUARD, scale=_CFAR_SCALE,
            )
            if mag < cfar_thresh:
                candidates.append(TargetCandidate(
                    bin_index=cand_bin, range_m=0.0, x_m=0.0, y_m=0.0, z_m=0.0,
                    magnitude=mag, azimuth_rad=0.0, elevation_rad=0.0,
                    zone="Unknown", valid=False, cfar_threshold=cfar_thresh, reject_reason="below_cfar",
                ))
                continue

            # ── 3. Localization ───────────────────────────────────────────────
            ch_cand     = corrected_data[cand_bin, :]
            cand_range  = cand_bin * self.range_res

            az, el, x, y, z = estimate_candidate_geometry(ch_cand, cand_range, self.R, self.T)

            # ── 4. Spatial zone gate ──────────────────────────────────────────
            zone_label, is_valid_zone = self._zone_evaluate(x, y, z)

            if not is_valid_zone:
                candidates.append(TargetCandidate(
                    bin_index=cand_bin, range_m=cand_range,
                    x_m=x, y_m=y, z_m=z, magnitude=mag,
                    azimuth_rad=az, elevation_rad=el,
                    zone=zone_label, valid=False,
                    cfar_threshold=cfar_thresh,
                    reject_reason="out_of_zone",
                ))
                continue

            # ── 5. Vital features ─────────────────────────────────────────────
            vital: VitalFeatures | None = None
            vital_mult = 1.0

            if spectral_history is not None:
                bin_hist = spectral_history.get_bin_history(cand_bin)
                vital    = self._vital_extractor.extract(bin_hist, ch_cand, cand_bin)
                vital_mult = vital.vital_multiplier
                all_vital[cand_bin] = vital

            # ── 6. Composite score ────────────────────────────────────────────
            # Mirrors original: magnitude × vital_multiplier × range weighting
            # Range deweighting: peaks at very short range are often clutter
            range_weight = min(1.0, cand_range / (3 * self.range_res + 1e-6))
            scored_mag   = mag * vital_mult * range_weight

            candidates.append(TargetCandidate(
                bin_index     = cand_bin,
                range_m       = cand_range,
                x_m           = x,
                y_m           = y,
                z_m           = z,
                magnitude     = scored_mag,
                azimuth_rad   = az,
                elevation_rad = el,
                zone          = zone_label,
                valid         = True,
                cfar_threshold= cfar_thresh,
                reject_reason = None,
            ))

        context.candidates           = candidates
        context.all_vital_features   = all_vital

        # Expose best candidate vital features at the top level
        valid_cands = [c for c in candidates if c.valid]
        if valid_cands:
            best_bin = max(valid_cands, key=lambda c: c.magnitude).bin_index
            context.vital_features = all_vital.get(best_bin)

        return context
