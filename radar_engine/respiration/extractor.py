"""
radar_engine.respiration.extractor
=====================================
RespirationExtractor — signal conditioning and bin-locking pipeline stage.

Extracted from RespiratoryPipelineV2.process() (signal path only):
  - Bin lock and neighbor fusion (steps 1–2, lines 620–629).
  - Phase unwrap using the legacy ambiguity-correction algorithm (step 6).
  - Phase-velocity differentiation (step 7).
  - Lowpass filter (step 8).
  - Derivative (apnea path) computation (step 9).
  - Frozen rolling display buffer management (step 10).
  - Dynamic apnea threshold calibration (step 10.5).

The V1 (RespiratoryPipeline) signal chain is NOT migrated — V2 only.
V1 is retained exclusively in the original libs/ file for historical reference.

Ownership: RespirationExtractor writes RespirationSignal into RadarContext.
RespirationAnalyzer (separate module) consumes that and produces RespirationMetrics.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import signal as sp_signal

from radar_engine.core.base import RadarModule
from radar_engine.core.context import RadarContext
from radar_engine.core.enums import MicroState, RespirationState
from radar_engine.core.models import QualityFlag, RespirationSignal
from radar_engine.config.respiration import RespirationConfig

logger = logging.getLogger(__name__)


def _unwrap_phase_ambiguity(phase_arr: np.ndarray) -> np.ndarray:
    """Custom phase-ambiguity unwrapper (legacy algorithm from respirationPipeline.py).

    Unlike scipy.unwrap (which uses a ±π threshold in radians), this function
    operates in degrees with a ±180° threshold and includes the additional
    correction that scipy omits for exact ±180 transitions.

    Preserved verbatim from the top-level ``unwrap_phase_Ambiguity()`` function.
    """
    phase_arr     = np.asarray(phase_arr, dtype=float)
    phase_arr_ret = phase_arr.copy()
    cum_corr      = 0.0

    for i in range(1, len(phase_arr)):
        diff = phase_arr[i] - phase_arr[i - 1]
        if diff > 180:
            mod_factor = 1
        elif diff < -180:
            mod_factor = -1
        else:
            mod_factor = 0

        diff_mod = diff - mod_factor * 360
        if diff_mod == -180 and diff > 0:
            diff_mod = 180

        correction = diff_mod - diff
        if not (-180 < correction < 0 or 0 < correction < 180):
            # Only apply non-trivial corrections (multiples of ±360 wrap)
            cum_corr += correction

        phase_arr_ret[i] = phase_arr[i] + cum_corr

    return phase_arr_ret


class RespirationExtractor(RadarModule):
    """Extracts and conditions the respiratory waveform from the spectral history.

    Implements the V2 signal chain exactly:
        1. Target bin lock (or re-lock after MACRO_PHASE burst).
        2. Multi-bin spatial fusion: locked_bin ± 1 (1-2-1 weighted sum).
        3. Phase extraction with legacy ambiguity unwrapper.
        4. Phase-velocity differentiation (phase difference, Δφ).
        5. 4th-order Butterworth low-pass filter at 0.5 Hz.
        6. Derivative → normalised abs-derivative (apnea path).
        7. Frozen rolling display-buffer management.
        8. Dynamic apnea-threshold calibration (40 s warmup).

    Writes ``context.respiration_signal`` (RespirationSignal) on every call.
    """

    # ── V2 filter constants (verbatim from RespiratoryPipelineV2.process) ─────
    _LP_ORDER:      int   = 4
    _LP_CUTOFF:     float = 0.5    # Hz, Butterworth lowpass
    _DERIV_BASE_MAX: float = 0.2   # physical scale for normalised derivative
    _CALIB_FRAMES:  int   = 0      # computed in __init__ (40 s * fps)

    def __init__(
        self,
        cfg:        RespirationConfig | None = None,
        frame_rate: float                    = 25.0,
    ) -> None:
        """
        Args:
            cfg:        RespirationConfig dataclass.  None → dataclass defaults.
            frame_rate: Radar frame rate [Hz]; used for window sizing and filter.
        """
        if cfg is None:
            cfg = RespirationConfig()

        self.fps           = frame_rate
        self.window_frames = int(cfg.window_sec * self.fps)
        self._CALIB_FRAMES = int(40.0 * self.fps)
        self._LP_ORDER     = cfg.lowpass_order
        self._LP_CUTOFF    = cfg.lowpass_cutoff_hz

        # Rolling display buffers
        self._plot_resp_buf:  np.ndarray = np.zeros(self.window_frames)
        self._plot_deriv_buf: np.ndarray = np.zeros(self.window_frames)

        # State
        self.locked_bin:           int | None = None
        self._global_frame_idx:    int        = 0
        self.frames_since_present: int        = 0

        # Dynamic apnea threshold
        self.apnea_threshold:       float = 0.20  # fallback prior to calibration
        self.threshold_calibrated:  bool  = False

        # Pre-compute filter coefficients
        self._b, self._a = sp_signal.butter(
            self._LP_ORDER, self._LP_CUTOFF, btype='lowpass', fs=self.fps
        )

    # ── RadarModule interface ──────────────────────────────────────────────────

    def reset(self) -> None:
        """Full state reset — called on bed-zone exit or hard pipeline reset."""
        self.locked_bin            = None
        self._global_frame_idx     = 0
        self.frames_since_present  = 0
        self._plot_resp_buf        = np.zeros(self.window_frames)
        self._plot_deriv_buf       = np.zeros(self.window_frames)
        self.apnea_threshold       = 0.20
        self.threshold_calibrated  = False

    def process(self, context: RadarContext) -> RadarContext:
        """Run respiration signal extraction for one frame batch.

        Reads:
            context.spectral_history              — SpectralHistory instance (typed field).
            context.target_bin                    — confirmed range bin (typed field).
            context.activity.micro_state          — MicroState enum for re-lock gate.
            context.diagnostics["_frames"]        — number of frames this tick.

        Writes:
            context.respiration_signal — RespirationSignal.
        """
        frames        = int(context.diagnostics.get("frames_in_tick", 1))
        spectral_hist = context.spectral_history       # injected by engine (typed field)
        micro_state   = (
            context.activity.micro_state
            if context.activity is not None
            else None
        )

        self._global_frame_idx += frames

        # ── Eligibility gate ──────────────────────────────────────────────────
        # Mirrors V2: requires spectral_history + confirmed bin fields
        if spectral_hist is None:
            context.respiration_signal = RespirationSignal.inactive(self.window_frames)
            return context

        current_bin = context.target_bin     # typed field set by engine for confirmed track
        if current_bin is None:
            context.respiration_signal = RespirationSignal.inactive(self.window_frames)
            return context

        self.frames_since_present += frames

        # ── 1. Bin lock / re-lock ───────────────────────────────────────────────
        # Re-lock on MACRO_PHASE (large body motion invalidates current lock)
        if self.locked_bin is None or micro_state == MicroState.MACRO_PHASE:
            self.locked_bin = current_bin

        # ── 2. Multi-bin spatial fusion (±1 bin, simple sum) ─────────────────
        # V2 uses the antenna-summed 2-D spectral history (sum over axis=1 done
        # by SpectralHistory.get_ordered_2d). Shape: (bins, frames).
        sh_2d    = spectral_hist.get_ordered_2d()    # (num_bins, window_frames)
        n_bins   = sh_2d.shape[0]
        b_start  = max(0, self.locked_bin - 1)
        b_end    = min(n_bins, self.locked_bin + 2)

        fused_complex = np.sum(sh_2d[b_start:b_end, :], axis=0)  # (window_frames,)

        # ── 3. Phase extraction + legacy ambiguity unwrap ─────────────────────
        raw_phase   = np.angle(fused_complex, deg=True)
        target_data = _unwrap_phase_ambiguity(raw_phase)

        # ── 4. Phase-velocity differentiation (Δφ, length-preserving) ────────
        diff_phase       = np.zeros_like(target_data)
        diff_phase[1:]   = target_data[1:] - target_data[:-1]

        # ── 5. Lowpass filter ─────────────────────────────────────────────────
        if len(diff_phase) > 15:
            resp_signal_raw = sp_signal.lfilter(self._b, self._a, diff_phase)
        else:
            resp_signal_raw = diff_phase

        # ── 6. Normalised abs-derivative (apnea detection path) ───────────────
        first_deriv         = np.gradient(resp_signal_raw)
        abs_deriv           = np.abs(first_deriv)
        raw_scale_derivative = abs_deriv / self._DERIV_BASE_MAX

        # ── 7. Frozen rolling display buffers ─────────────────────────────────
        self._plot_resp_buf  = np.roll(self._plot_resp_buf,  -frames)
        self._plot_resp_buf[-frames:]  = resp_signal_raw[-frames:]

        self._plot_deriv_buf = np.roll(self._plot_deriv_buf, -frames)
        self._plot_deriv_buf[-frames:] = raw_scale_derivative[-frames:]

        display_signal  = self._plot_resp_buf[-len(resp_signal_raw):]
        scale_derivative = self._plot_deriv_buf[-len(resp_signal_raw):]

        # ── 8. Dynamic apnea threshold calibration (40 s) ─────────────────────
        if self._global_frame_idx >= self._CALIB_FRAMES and not self.threshold_calibrated:
            p95 = float(np.percentile(self._plot_deriv_buf, 95))
            self.apnea_threshold      = max(0.05, p95 * 0.25)
            self.threshold_calibrated = True
            logger.info(
                "Dynamic apnea threshold locked: %.3f (P95=%.3f)",
                self.apnea_threshold, p95,
            )

        # ── Build RespirationSignal ───────────────────────────────────────────
        is_calib = self._global_frame_idx < self._CALIB_FRAMES

        if is_calib:
            quality = QualityFlag.invalid("calibrating")
            state   = RespirationState.ACQUIRE
        else:
            quality = QualityFlag.valid(score=0.9)
            state   = RespirationState.TRACK

        context.respiration_signal = RespirationSignal(
            state                = state,
            locked_bin           = self.locked_bin,
            live_signal          = display_signal.copy(),
            phase_signal         = target_data,
            derivative_signal    = scale_derivative.copy(),
            apnea_threshold      = self.apnea_threshold,
            threshold_calibrated = self.threshold_calibrated,
            quality              = quality,
            frames_since_present = self.frames_since_present,
        )

        # Keep only frame-batch diagnostics; signal/threshold transfer now lives
        # on typed RespirationSignal fields.
        context.diagnostics["frames_in_tick"]       = frames

        return context
