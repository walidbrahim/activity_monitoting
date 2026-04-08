"""
radar_engine.activity.vital_features
======================================
VitalFeatureExtractor — per-candidate phase-history feature computation.

Extracted from ActivityPipeline._step2_candidate_generation() (beamforming,
phase-based classification) and ActivityPipeline._compute_aliveness()
(spectral prominence, autocorrelation, entropy) without any numerical changes.

All thresholds and constants are preserved exactly as they appear in the
original implementation. The ``macro_timers`` dict is owned by this extractor
so micro-motion timing persists across frames for the same candidate.

Ownership: TargetDetector calls this once per accepted candidate peak.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
from scipy.signal import welch

from radar_engine.core.enums import MicroState
from radar_engine.core.models import VitalFeatures

logger = logging.getLogger(__name__)

# ── Constants (mirrors ActivityPipeline.__init__ class-level defaults) ────────

_MACRO_PHASE_PTP_THRESHOLD = 180.0   # degrees — strong motion burst
_MICRO_PHASE_VAR_THRESHOLD = 0.03    # phase variance — gentle micro-motion
_STATIC_GHOST_PTP          = 5.0     # degrees — essentially static
_MECH_ROTOR_DRIFT          = 400.0   # deg — monotonic phase ramp from rotating machinery
_MACRO_PERSIST_FRAMES      = 10      # frames macro label is held after burst
_RESP_BAND_LOW_HZ          = 0.1
_RESP_BAND_HIGH_HZ         = 0.5


class VitalFeatureExtractor:
    """Compute VitalFeatures for a specific range-bin candidate.

    State:
        _macro_timers: per-bin countdown; bin survives as MACRO_PHASE for this
                       many frames after a large phase burst even if the next
                       frame is quieter.
    """

    def __init__(self, frame_rate: float) -> None:
        """
        Args:
            frame_rate: Radar frame rate [Hz] — used for spectral axes.
        """
        self.frame_rate    = frame_rate
        self._macro_timers: dict[int, int] = defaultdict(int)

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(
        self,
        bin_history: np.ndarray,    # (num_antennas, num_frames) complex, ordered
        ch_snapshot: np.ndarray,    # (num_antennas,) complex, current-frame snapshot
        cand_bin:    int,
        force_psd:   bool = False,
    ) -> VitalFeatures:
        """Compute VitalFeatures for one candidate range bin.

        Args:
            bin_history:  Chronologically ordered per-antenna history from
                          SpectralHistory.get_bin_history(cand_bin).
            ch_snapshot:  Current-frame antenna snapshot at the candidate bin.
                          Used to derive beamforming weights.
            cand_bin:     Range-bin index (for persistent macro_timer lookup).
            force_psd:    If True, forces heavy _compute_aliveness math even
                          on fast-bailout paths (used by single-target GUI).

        Returns:
            VitalFeatures populated with all phase/spectral/aliveness fields.
        """
        # ── 1. Beamforming weights ───────────────────────────────────────────
        weights  = np.conj(ch_snapshot) / (np.abs(ch_snapshot) + 1e-9)

        # ── 2. Beamformed time-series ────────────────────────────────────────
        # bin_history: (antennas, frames)  weights: (antennas,)
        hist_bf  = np.sum(bin_history * weights[:, np.newaxis], axis=0)  # (frames,)

        # ── 3. Strict DC Removal ─────────────────────────────────────────────
        # If the target is 'protected', the ClutterMap freezes, allowing a slight 
        # complex DC offset to build up. This centers the orbital exactly at 0,0.
        hist_bf -= np.mean(hist_bf)

        # ── 4. Phase unwrap ──────────────────────────────────────────────────
        cand_phase = np.unwrap(np.angle(hist_bf)) * (180.0 / np.pi)   # degrees

        # ── 5. Standard classification features ──────────────────────────────
        n_frames  = len(cand_phase)
        phase_ptp = float(np.ptp(cand_phase))
        phase_var = float(np.var(np.diff(cand_phase)))
        phase_drift = float(abs(cand_phase[-1] - cand_phase[0]))
        
        micro_state = None
        vital_mult  = 1.0

        # ── 6. Classification (Fast Bailouts) ────────────────────────────────
        # Mechanical rotor: monotonically drifting phase ramp
        if phase_drift > _MECH_ROTOR_DRIFT:
            micro_state = MicroState.MECHANICAL_ROTOR
            vital_mult  = 0.01
            self._macro_timers[cand_bin] = 0

        # Large phase burst → macro motion
        elif phase_ptp > _MACRO_PHASE_PTP_THRESHOLD:
            micro_state = MicroState.MACRO_PHASE
            vital_mult  = 0.05
            self._macro_timers[cand_bin] = _MACRO_PERSIST_FRAMES

        # Carry-over from a recent burst
        elif self._macro_timers.get(cand_bin, 0) > 0:
            self._macro_timers[cand_bin] -= 1
            micro_state = MicroState.MACRO_PHASE
            vital_mult  = 0.05

        # Essentially static — probably ghost
        elif phase_ptp < _STATIC_GHOST_PTP:
            micro_state = MicroState.STATIC_GHOST
            vital_mult  = 0.1
            self._macro_timers[cand_bin] = 0

        # Gentle micro-motion (e.g. speaking, shifting)
        elif phase_var > _MICRO_PHASE_VAR_THRESHOLD:
            micro_state = MicroState.MICRO_PHASE
            vital_mult  = 1.0
            self._macro_timers[cand_bin] = 0
            
        else:
            # We strictly need PSD to tell if it's dead space or breathing
            force_psd = True

        # ── 7. Heavy PSD + Aliveness (Only if forced or needed) ─────────────
        aliveness       = 0.0
        spectral_prom   = 0.0
        autocorr_q      = 0.0
        spectral_ent    = 1.0
        displacement_mm = 0.0
        
        if force_psd:
            aliveness, spectral_prom, autocorr_q, spectral_ent, displacement_mm = \
                self._compute_aliveness(cand_phase, phase_ptp)
            
            # If we didn't hit a fast-bailout earlier, use PSD outcome to classify
            if micro_state is None:
                micro_state, vital_mult = self._classify_from_aliveness(
                    aliveness, spectral_prom, autocorr_q, phase_var)
                self._macro_timers[cand_bin] = 0

        return VitalFeatures(
            phase_ptp           = phase_ptp,
            displacement_mm     = displacement_mm,
            phase_variance      = phase_var,
            phase_drift         = phase_drift,
            spectral_prominence = spectral_prom,
            autocorr_quality    = autocorr_q,
            spectral_entropy    = spectral_ent,
            aliveness_score     = aliveness,
            micro_state         = micro_state,
            vital_multiplier    = vital_mult,
        )

    # ── RadarModule-compatible reset ─────────────────────────────────────────

    def reset(self) -> None:
        """Clear macro timers (call on track loss or hard reset)."""
        self._macro_timers.clear()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _compute_aliveness(
        self,
        cand_phase: np.ndarray,
        phase_ptp:  float,
    ) -> tuple[float, float, float, float, float]:
        """Compute spectral + autocorrelation aliveness scores.

        Preserves the exact computation from ActivityPipeline._compute_aliveness().

        Returns:
            (aliveness_score, spectral_prominence, autocorr_quality,
             spectral_entropy, displacement_mm)
        """
        n_frames = len(cand_phase)

        # ── Displacement (mm) from total phase swing ─────────────────────────
        # Using wavelength ≈ 3.9 mm for 77 GHz
        wavelength_mm  = 3.9
        displacement_mm = (phase_ptp / 360.0) * wavelength_mm  # full PTP

        # ── Welch PSD for spectral analysis ─────────────────────────────────
        nperseg = min(n_frames, 128)
        freqs, psd = welch(cand_phase, fs=self.frame_rate, nperseg=nperseg)

        # Respiratory band mask
        resp_mask = (freqs >= _RESP_BAND_LOW_HZ) & (freqs <= _RESP_BAND_HIGH_HZ)

        if resp_mask.any() and psd.sum() > 0:
            resp_power  = psd[resp_mask].sum()
            total_power = psd.sum()
            noise_power = total_power - resp_power

            # Spectral prominence: resp_power / noise_floor (with guard)
            spectral_prom = resp_power / (noise_power + 1e-6)

            # Spectral entropy (normalised)
            psd_norm      = psd / (total_power + 1e-6)
            spectral_ent  = float(
                -np.sum(psd_norm * np.log(psd_norm + 1e-10))
                / (np.log(len(psd)) + 1e-10)
            )
        else:
            spectral_prom = 0.0
            spectral_ent  = 1.0

        # ── Autocorrelation quality ──────────────────────────────────────────
        if n_frames >= 20:
            ac = np.correlate(cand_phase - cand_phase.mean(),
                              cand_phase - cand_phase.mean(), mode='full')
            ac = ac[n_frames - 1:]
            ac_norm = ac / (ac[0] + 1e-9)

            # Expected lag range for respiration (0.1–0.8 Hz → 1.25–10 s)
            min_lag = max(1, int(self.frame_rate / _RESP_BAND_HIGH_HZ))
            max_lag = min(n_frames - 1, int(self.frame_rate / _RESP_BAND_LOW_HZ))

            if max_lag > min_lag:
                autocorr_q = float(np.max(ac_norm[min_lag:max_lag]))
            else:
                autocorr_q = 0.0
        else:
            autocorr_q = 0.0

        # ── Composite aliveness score ────────────────────────────────────────
        # Mirrors the original weighted formula
        aliveness = float(np.clip(
            0.50 * min(spectral_prom / 10.0, 1.0)   # spectral prominence (cap at 10×)
            + 0.30 * max(0.0, autocorr_q)            # periodicity
            + 0.20 * min(displacement_mm / 3.0, 1.0) # amplitude
            , 0.0, 1.0
        ))

        return aliveness, float(spectral_prom), autocorr_q, float(spectral_ent), displacement_mm

    @staticmethod
    def _classify_from_aliveness(
        aliveness:      float,
        spectral_prom:  float,
        autocorr_q:     float,
        phase_var:      float,
    ) -> tuple[MicroState, float]:
        """Map aliveness scores to (MicroState, vital_multiplier)."""
        if aliveness >= 0.5 and spectral_prom > 2.0:
            return MicroState.ALIVE, 1.0
        elif aliveness >= 0.25 or autocorr_q > 0.3:
            return MicroState.WEAK_VITAL, 0.6
        else:
            return MicroState.DEAD_SPACE, 0.05
