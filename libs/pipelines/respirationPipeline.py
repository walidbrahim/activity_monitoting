import numpy as np
from scipy import signal
from scipy.signal import find_peaks
from config import config

def unwrap_phase_Ambiguity(phase_queue):
    phase_arr = phase_queue
    phase_arr_ret = phase_arr.copy()
    phase_diff_correction_cum = 0
    for i in range(len(phase_arr)):
        if not i:
            continue
        else:
            phase_diff = phase_arr[i] - phase_arr[i - 1]
            if phase_diff > 180:
                mod_factor = 1
            elif phase_diff < -180:
                mod_factor = -1
            else:
                mod_factor = 0
        phase_diff_mod = phase_diff - mod_factor * 2 * 180
        if phase_diff_mod == -180 and phase_diff > 0:
            phase_diff_mod = 180
        phase_diff_correction = phase_diff_mod - phase_diff
        if (phase_diff_correction < 180 and phase_diff_correction > 0) or (phase_diff_correction > -180 and phase_diff_correction < 0):
            phase_diff_correction = 0
        phase_diff_correction_cum += phase_diff_correction
        phase_arr_ret[i] = phase_arr[i] + phase_diff_correction_cum
    return phase_arr_ret

def detect_respiratory_peaks(signal_data, fs):
    """
    Robust peak/trough detection for respiration signal.
    Returns (troughs, peaks) indices.
    troughs -> Inhale Onsets
    peaks -> Exhale Onsets
    """
    signal_data = np.asarray(signal_data)
    # Distance constraint (e.g., 0.5 sec) to avoid noise
    min_dist = max(1, int(fs * 0.5))
    
    # Dynamic prominence threshold based on signal range
    # Peaks must be at least 25% of the signal's range to be counted
    sig_range = np.ptp(signal_data)
    prominence = max(0.05, sig_range * 0.25) 
    
    # Exhale Onsets (Peaks)
    peaks, _ = find_peaks(signal_data, distance=min_dist, prominence=prominence)
    
    # Inhale Onsets (Troughs -> Peaks of inverted signal)
    troughs, _ = find_peaks(-signal_data, distance=min_dist, prominence=prominence)
    
    return troughs, peaks



# ---------------------------------------------------------------------------
# Helper: Apnea Tracker (Stateful, cross-frame deduplication)
# ---------------------------------------------------------------------------
class ApneaTracker:
    """
    Tracks apnea events across frames using frame-index-based deduplication.
    Merges overlapping/adjacent segments to avoid double-counting.
    """
    def __init__(self):
        self.count = 0
        self.durations_sec = []
        self._global_events = []   # list of [global_start, global_end] in frame indices

    def update(self, segments, fps, current_buffer_len, global_frame_idx):
        """
        segments: list of (start_idx, end_idx) relative to current live_signal buffer
        fps: frames per second
        current_buffer_len: length of the live_signal buffer
        global_frame_idx: cumulative frame count since tracking started
        """
        global_offset = global_frame_idx - current_buffer_len

        for (s, e) in segments:
            if e <= 0 or s >= current_buffer_len:
                continue

            g_start = global_offset + s
            g_end = global_offset + e

            # Check overlap with the last recorded event (tolerance: ~0.5s)
            tolerance = int(0.5 * fps)
            matched = False

            if self._global_events:
                last_ev = self._global_events[-1]
                if g_start <= (last_ev[1] + tolerance):
                    matched = True
                    if g_end > last_ev[1]:
                        last_ev[1] = g_end
                        self.durations_sec[-1] = (last_ev[1] - last_ev[0]) / fps

            if not matched:
                self._global_events.append([g_start, g_end])
                self.count += 1
                self.durations_sec.append((g_end - g_start) / fps)

    def reset(self):
        self.count = 0
        self.durations_sec = []
        self._global_events = []


# ---------------------------------------------------------------------------
# Helper: Breath Cycle Tracker (Frame-based timing)
# ---------------------------------------------------------------------------
class BreathCycleTracker:
    """
    Tracks breath cycles using exhale peak frame indices.
    Uses frame-count-based timing for deterministic behavior.
    """
    def __init__(self, history_size=5, fps=25):
        self.history_size = history_size
        self.fps = fps
        self.count = 0
        self.cycle_durations = []
        self._last_peak_global_frame = None

    def update(self, peak_indices, global_frame_idx, buffer_len):
        """
        peak_indices: sorted list of peak indices within the current live_signal buffer
        global_frame_idx: cumulative frame count since tracking started
        buffer_len: length of the live_signal buffer
        """
        if len(peak_indices) == 0:
            return

        global_offset = global_frame_idx - buffer_len

        for idx in peak_indices:
            global_frame = global_offset + idx

            if self._last_peak_global_frame is None:
                self._last_peak_global_frame = global_frame
                continue

            duration_sec = (global_frame - self._last_peak_global_frame) / self.fps

            # Sanity filter: valid breath duration 0.5s–6.0s (~10–120 BPM)
            if 0.5 < duration_sec < 6.0:
                self.cycle_durations.append(duration_sec)
                self.count += 1
                self._last_peak_global_frame = global_frame
            elif duration_sec >= 6.0:
                # Too long — likely apnea gap. Reset anchor without counting.
                self._last_peak_global_frame = global_frame
            # If < 0.5s, ignore (noise) but don't update anchor

    @property
    def last_duration(self):
        return self.cycle_durations[-1] if self.cycle_durations else 0.0

    def get_rr_avg(self):
        """Average RR over last N cycles in BPM."""
        if not self.cycle_durations:
            return 0.0
        last_n = self.cycle_durations[-self.history_size:]
        avg_dur = np.mean(last_n)
        return 60.0 / avg_dur if avg_dur > 0 else 0.0

    def get_brv(self, n=20):
        """Breathing Rate Variability: std of last n cycle durations (SDNN analogue)."""
        if len(self.cycle_durations) < 5:
            return 0.0
        return float(np.std(self.cycle_durations[-n:]))

    def reset(self):
        self.count = 0
        self.cycle_durations = []
        self._last_peak_global_frame = None


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
class RespiratoryPipeline:
    # Phase-to-displacement conversion factor for 60GHz FMCW radar
    # λ = c / f = 3e8 / 60e9 = 5mm → displacement = phase * λ / (4π)
    PHASE_TO_MM = 5.0 / (4.0 * np.pi)  # ≈ 0.398 mm/rad

    def __init__(self, fps=25, window_seconds=30, baseline_seconds=40):
        self.fps = fps
        self.window_frames = int(window_seconds * self.fps)
        self.baseline_frames = int(baseline_seconds * self.fps)

        # State & History Buffers
        self.history_buffer = np.zeros(self.baseline_frames)
        self.rr_history_buffer = np.zeros(self.window_frames)

        # Apnea trace (boolean array synced with live_signal frame-by-frame)
        self.apnea_trace = np.zeros(self.window_frames, dtype=bool)

        self.apnea_active = False
        self.apnea_duration = 0.0

        self.current_rr = 0.0
        self.cycle_duration = 0.0
        self.depth_status = "unknown"
        self.confidence = 0.0
        self.locked_bin = None
        self.frames_since_present = 0

        # State Machine Flags
        self.state = "OFF"
        self.stable_resp_frames = 0
        self.acquire_frames = int(2.0 * self.fps)
        self._apnea_timer = 0
        
        # Stateful trackers
        self.apnea_tracker = ApneaTracker()
        self.cycle_tracker = BreathCycleTracker(
            history_size=getattr(config.respiration, 'cycle_tracker_history', 5),
            fps=self.fps
        )
        self._global_frame_idx = 0

    def process(self, act_pipe_out_dict, frames=1):
        """
        Activity-gated respiration processing.
        """
        self._global_frame_idx += frames

        # 1. Extract activity gates from act_pipe_out_dict
        zone_str = act_pipe_out_dict.get("zone", "No Occupant Detected")
        base_zone = zone_str.split(" - ")[0]
        subzone   = zone_str.split(" - ")[1] if " - " in zone_str else ""
        
        # Monitor type check (feedback: enabled in any zone of type "monitor")
        is_monitor = config.layout.get(base_zone, {}).get("type", "") == "monitor"
        
        posture = act_pipe_out_dict.get("posture", "")
        motion  = act_pipe_out_dict.get("motion_str", "")
        micro   = act_pipe_out_dict.get("micro_state", "STABLE")
        occ_conf = act_pipe_out_dict.get("occ_confidence", 0)

        track_ok = (
            act_pipe_out_dict.get("is_valid", False) and
            act_pipe_out_dict.get("final_bin") is not None and
            occ_conf >= 50 and
            micro not in {"STATIC_GHOST", "MECHANICAL_ROTOR", "DEAD_SPACE"}
        )

        # Full Respiration Eligibility: Monitor Zone + Lying Down + Resting
        full_resp_ok = (
            track_ok and
            is_monitor and
            posture == "Lying Down" and
            motion == "Resting/Breathing"
        )
        
        # RR-only Eligibility: Monitor Zone + (Lying/Sitting) + (Resting/Fidgeting)
        rr_only_ok = (
            track_ok and
            is_monitor and
            posture in {"Lying Down", "Sitting"} and
            motion in {"Resting/Breathing", "Restless/Fidgeting"}
        )

        # 2. State Machine Transitions
        prev_state = self.state
        if not track_ok:
            self.state = "OFF"
            self._reset_state_soft()
        elif full_resp_ok or rr_only_ok:
            if self.stable_resp_frames >= self.acquire_frames:
                self.locked_bin = act_pipe_out_dict["final_bin"]
                self.state = "TRACK"
            else:
                self.state = "ACQUIRE"
            self.stable_resp_frames += frames
        elif motion == "Postural Shift":
            self.state = "HOLD"
        elif motion in {"Major Movement", "Walking"} or posture in {"Standing", "Fallen"} or not is_monitor:
            self.state = "SUSPEND"
            self.stable_resp_frames = 0
        else:
            self.state = "OFF"
            self._reset_state_soft()

        is_active = self.state in ["TRACK", "HOLD"]

        # Reset if moving from active to non-active states sharply
        if prev_state in ["TRACK", "HOLD"] and self.state in ["OFF"]:
            self._reset_state_soft()

        # 3. Signal Extraction (Beamformed per-antenna extraction)
        fused_complex = np.zeros(frames, dtype=complex)
        
        if is_active and self.locked_bin is not None:
            raw_cube = act_pipe_out_dict.get('raw_spectral_cube')
                
            if raw_cube is not None:
                # time-ordered raw spectral cube: (bins, antennas, frames)
                # Use phase-aligned coherent sum across antennas
                snapshot = raw_cube[self.locked_bin, :, -1]
                weights = np.conj(snapshot) / (np.abs(snapshot) + 1e-9)

                # Beamformed complex time series for target and neighbors
                bf_center = np.sum(raw_cube[self.locked_bin, :, :] * weights[:, None], axis=0)
                bf_left   = np.sum(raw_cube[max(0, self.locked_bin - 1), :, :] * weights[:, None], axis=0)
                bf_right  = np.sum(raw_cube[min(raw_cube.shape[0]-1, self.locked_bin + 1), :, :] * weights[:, None], axis=0)

                # Spatial smoothing (1-2-1 bin weighting)
                fused_complex = 0.25 * bf_left + 0.50 * bf_center + 0.25 * bf_right
            else:
                # Fallback to summed history (legacy)
                spectral_hist = act_pipe_out_dict.get('spectral_history')
                if spectral_hist is not None:
                    start_bin = max(0, self.locked_bin - 1)
                    end_bin = min(spectral_hist.shape[0], self.locked_bin + 2)
                    fused_complex = np.sum(spectral_hist[start_bin:end_bin, :], axis=0)

        # 4. Signal Chain: Unwrap → Detrend → Bandpass
        if len(fused_complex) == 0:
            fused_complex = np.array([1e-10+1e-10j]*frames)

        raw_phase = np.unwrap(np.angle(fused_complex))
        diff_phase = np.concatenate(([0.0], np.diff(raw_phase)))

        # SQI (Spectral Quality Index)
        win = np.hanning(len(diff_phase))
        fft_mag = np.abs(np.fft.rfft(diff_phase * win))
        freqs = np.fft.rfftfreq(len(diff_phase), d=(1.0 / self.fps))
        breathing_mask = (freqs >= 0.15) & (freqs <= 0.6)
        total_mask = (freqs >= 0.15) & (freqs <= 3.0)
        sqi = np.sum(fft_mag[breathing_mask]) / (np.sum(fft_mag[total_mask]) + 1e-6)

        # 5. Fused Confidence
        zone_w = {
            "Center": 1.0, "Head Edge": 0.95, "Left Edge": 0.85, 
            "Right Edge": 0.85, "Foot Edge": 0.65, "Corner": 0.60
        }.get(subzone, 0.8 if is_monitor else 0.0)
        posture_w = {"Lying Down": 1.0, "Sitting": 0.55}.get(posture, 0.0)
        motion_w = {
            "Resting/Breathing": 1.0, "Restless/Fidgeting": 0.65, "Restless/Shifting": 0.35,
            "Postural Shift": 0.10, "Major Movement": 0.0, "Walking": 0.0,
        }.get(motion, 0.0)

        gate_conf = zone_w * posture_w * motion_w
        if not is_active: gate_conf = 0.0
        signal_conf = np.clip(sqi * 100, 0, 100)

        self.confidence = (0.45 * signal_conf + 0.35 * occ_conf + 0.20 * (100.0 * gate_conf))
        if self.state in ["OFF", "SUSPEND"]: self.confidence = 0.0

        # 6. Filtering & Displacement
        lp_cutoff = getattr(config.respiration, 'resp_lowpass_cutoff', 0.5)
        lp_order = getattr(config.respiration, 'resp_lowpass_order', 4)
        b, a = signal.butter(lp_order, lp_cutoff, btype='low', fs=self.fps)
        
        if len(diff_phase) > 15:
            filtered_velocity = signal.filtfilt(b, a, diff_phase)
        else:
            filtered_velocity = diff_phase

        integrated_phase = np.cumsum(filtered_velocity)
        integrated_phase = signal.detrend(integrated_phase, type='linear')
        filtered_resp = integrated_phase * self.PHASE_TO_MM  # mm
        live_signal = filtered_resp[-self.window_frames:]

        # Zero signal if unconfident or inactive
        self.frames_since_present += frames
        if self.frames_since_present < self.window_frames or not is_active:
            if not is_active: live_signal[:] = 0.0
            else: live_signal[:-self.frames_since_present] = 0.0

        # 7. Baseline History Update (Limited)
        if self.state == "TRACK" and full_resp_ok and self.confidence > 40.0:
            self.history_buffer = np.roll(self.history_buffer, -frames)
            self.history_buffer[-frames:] = filtered_resp[-frames:]

        # 8. Apnea Logic (Split into pause/event)
        norm_derivative = np.zeros_like(live_signal)
        apnea_segments  = []
        apnea_status    = "normal"

        apnea_eligible = (
            is_monitor and 
            posture == "Lying Down" and 
            motion == "Resting/Breathing" and
            self.state == "TRACK"
        )
        
        min_warmup_frames = int(5.0 * self.fps)
        has_warmup = self.frames_since_present > min_warmup_frames

        if has_warmup and is_active:
            first_derivative = np.gradient(live_signal)
            abs_derivative = np.abs(first_derivative)
            baseline_deriv = np.max(np.abs(np.gradient(self.history_buffer[-self.window_frames:])))
            if baseline_deriv > 1e-6:
                norm_derivative = np.clip(abs_derivative / baseline_deriv, 0.0, 1.0)

            resp_thresh = getattr(config.respiration, 'resp_threshold', 0.15)
            hold_window = int(getattr(config.respiration, 'apnea_hold_window_sec', 3.0) * self.fps)
            merge_gap   = int(getattr(config.respiration, 'apnea_merge_gap_sec', 0.5) * self.fps)

            raw_apnea_segments = []
            for i in range(hold_window, len(norm_derivative)):
                if np.all(norm_derivative[i-hold_window:i] <= resp_thresh):
                    raw_apnea_segments.append((i-hold_window, i))

            if raw_apnea_segments:
                cur_start, cur_end = raw_apnea_segments[0]
                for s, e in raw_apnea_segments[1:]:
                    if s - cur_end <= merge_gap: cur_end = e
                    else:
                        apnea_segments.append((cur_start, cur_end))
                        cur_start, cur_end = s, e
                apnea_segments.append((cur_start, cur_end))

        # Apnea State machine
        self.apnea_trace = np.roll(self.apnea_trace, -frames)
        self.apnea_trace[-frames:] = False
        
        if self.state == "HOLD":
            pass
        elif apnea_eligible and has_warmup and self.confidence >= 50.0:
            recent_signal = self.history_buffer[-int(5 * self.fps):]
            signal_mean_abs = np.mean(np.abs(recent_signal))
            if signal_mean_abs > 1e-6:
                norm_var = np.var(recent_signal) / (signal_mean_abs ** 2)
                norm_range = np.ptp(recent_signal) / signal_mean_abs
                is_flat = (norm_var < 0.05 and norm_range < 0.3)
            else: is_flat = True
                
            if is_flat:
                self._apnea_timer += frames
                if self._apnea_timer >= int(10.0 * self.fps):
                    self.apnea_active = True
                    apnea_status = "apnea_event"
                elif self._apnea_timer >= int(4.0 * self.fps):
                    apnea_status = "pause_candidate"
            else:
                self._apnea_timer = 0
                self.apnea_active = False

            if self.apnea_active: self.apnea_duration += (frames / float(self.fps))
            else: self.apnea_duration = 0.0
        else:
            self.apnea_active = False
            self.apnea_duration = 0.0
            self._apnea_timer = 0

        if self.apnea_active: self.apnea_trace[-frames:] = True
        self.apnea_tracker.update(apnea_segments, self.fps, len(live_signal), self._global_frame_idx)

        # 9. Peak Detection & RR
        inhales, exhales = [], []
        sig_range = np.ptp(live_signal)
        is_low_amplitude = sig_range < 0.1

        if (self.state == "TRACK" and self.confidence > 20.0 and not self.apnea_active and not is_low_amplitude):
            min_dist = max(1, int(self.fps * 0.5))
            prominence = max(0.01, sig_range * 0.25)
            exhales_arr, _ = signal.find_peaks(live_signal, distance=min_dist, prominence=prominence)
            inhales_arr, _ = signal.find_peaks(-live_signal, distance=min_dist, prominence=prominence)
            exhales, inhales = exhales_arr.tolist(), inhales_arr.tolist()

            if len(inhales) >= 2:
                intervals = np.diff(inhales) / self.fps
                valid_intervals = intervals[(intervals > 0.5) & (intervals < 6.0)]
                if len(valid_intervals) > 0:
                    self.cycle_duration = np.median(valid_intervals)
                    self.current_rr = 60.0 / self.cycle_duration
                else: self.current_rr = 0.0
            else: self.current_rr = 0.0
            self.cycle_tracker.update(exhales, self._global_frame_idx, len(live_signal))
        else: self.current_rr = 0.0

        # 10. Depth Classification
        if self.state == "TRACK" and self.confidence > 20.0 and not self.apnea_active:
            baseline_mean = np.mean(self.history_buffer)
            baseline_amplitude = np.mean(np.abs(self.history_buffer - baseline_mean))
            recent_mean = np.mean(live_signal)
            recent_amplitude = np.mean(np.abs(live_signal - recent_mean))
            if baseline_amplitude > 1e-4:
                ratio = recent_amplitude / baseline_amplitude
                if ratio > 1.3: self.depth_status = "deep"
                elif ratio < 0.7: self.depth_status = "shallow"
                else: self.depth_status = "normal"
            else: self.depth_status = "unknown"
        elif self.apnea_active: self.depth_status = "apnea"
        else: self.depth_status = "unknown"

        self.rr_history_buffer = np.roll(self.rr_history_buffer, -frames)
        self.rr_history_buffer[-frames:] = self.current_rr
        brv_n = getattr(config.respiration, 'brv_history_size', 20)
        brv_value = self.cycle_tracker.get_brv(n=brv_n)

        return {
            "live_signal": live_signal,
            "derivative_signal": norm_derivative,
            "inhales": inhales,
            "exhales": exhales,
            "cycle_duration": self.cycle_duration,
            "rr_current": self.current_rr,
            "rr_history": self.rr_history_buffer,
            "apnea_active": self.apnea_active,
            "apnea_duration": self.apnea_duration,
            "apnea_trace": self.apnea_trace,
            "apnea_segments": apnea_segments,
            "apnea_status": apnea_status,
            "apnea_count": self.apnea_tracker.count,
            "apnea_durations": list(self.apnea_tracker.durations_sec),
            "depth": self.depth_status,
            "confidence": self.confidence,
            "motion_status": self.state,
            "locked_bin": self.locked_bin,
            "cycle_count": self.cycle_tracker.count,
            "brv_value": brv_value,
            "last_cycle_duration": self.cycle_tracker.last_duration,
        }

    def _reset_state_soft(self):
        self.apnea_active = False
        self.apnea_duration = 0.0
        self.depth_status = "unknown"
        self.confidence = 0.0
        self.current_rr = 0.0
        self.cycle_duration = 0.0
        self._apnea_timer = 0
        
    def _reset_state(self):
        self._reset_state_soft()
        self.apnea_trace = np.zeros(self.window_frames, dtype=bool)
        self.rr_history_buffer = np.zeros(self.window_frames)
        self.locked_bin = None
        self.history_buffer = np.zeros(self.baseline_frames)
        self.frames_since_present = 0
        self.apnea_tracker.reset()
        self.cycle_tracker.reset()
        self._global_frame_idx = 0
        self.state = "OFF"
        self.stable_resp_frames = 0

    def _get_empty_dict(self):
        return {
            "live_signal": np.zeros(self.window_frames),
            "derivative_signal": np.zeros(self.window_frames),
            "inhales": [], "exhales": [],
            "rr_current": 0.0,
            "rr_history": np.zeros(self.window_frames),
            "apnea_active": False, "apnea_duration": 0.0,
            "apnea_trace": np.zeros(self.window_frames, dtype=bool),
            "apnea_segments": [],
            "apnea_count": 0, "apnea_durations": [],
            "cycle_duration": 0.0, "depth": "unknown",
            "confidence": 0.0, "motion_status": "STABLE",
            "cycle_count": 0, "brv_value": 0.0,
            "last_cycle_duration": 0.0,
        }

class RespiratoryPipelineV2:
    """
    Experimental respiration pipeline built from the test_respiration harness. 
    Prioritizes filtered-velocity plotting with physically frozen past-trace GUI buffers
    and robust, dynamic Apnea calibration sequences.
    """
    def __init__(self):
        self.fps = config.radar.frame_rate
        self.window_frames = int(config.respiration.resp_window_sec * self.fps)
        self.range_res = config.radar.range_resolution
        
        self.locked_bin = None
        
        # State & History Buffers
        self.plot_resp_buffer = np.zeros(self.window_frames)
        self.plot_deriv_buffer = np.zeros(self.window_frames)
        self.rr_history_buffer = np.zeros(self.window_frames)
        self.apnea_trace = np.zeros(self.window_frames, dtype=bool)

        self.apnea_active = False
        self.live_apnea_frames = 0
        
        self.apnea_tracker = ApneaTracker()
        self.cycle_tracker = BreathCycleTracker(
            history_size=getattr(config.respiration, 'cycle_tracker_history', 5),
            fps=self.fps
        )
        self._global_frame_idx = 0
        self.frames_since_present = 0
        
        self.unwrap_fn = unwrap_phase_Ambiguity

    def _reset_state(self):
        self.locked_bin = None
        self.plot_resp_buffer = np.zeros(self.window_frames)
        self.plot_deriv_buffer = np.zeros(self.window_frames)
        self.rr_history_buffer = np.zeros(self.window_frames)
        self.apnea_trace = np.zeros(self.window_frames, dtype=bool)

        self.apnea_active = False
        self.live_apnea_frames = 0
        self.frames_since_present = 0
        
        self.apnea_tracker.reset()
        self.cycle_tracker.reset()
        self._global_frame_idx = 0
        self.threshold_calibrated = False

    def process(self, act_pipe_out_dict, frames=1):
        """
        Process exactly one or more new frames from the Activity Pipeline's sliding spectral_history window.
        """
        # Guard against incomplete Pipeline output
        required_keys = ['final_bin', 'motion_str', 'spectral_history']
        if not all(k in act_pipe_out_dict for k in required_keys):
            return None

        current_bin = act_pipe_out_dict['final_bin']
        motion_str = act_pipe_out_dict['motion_str']
        self._global_frame_idx += frames
        
        # 1. Target Lock
        if self.locked_bin is None or motion_str == "MACRO_PHASE":
            self.locked_bin = current_bin
            
        # 2. Multi-Bin Spatial Fusion
        spectral_hist = act_pipe_out_dict['spectral_history']
        start_bin = max(0, self.locked_bin - 1)
        end_bin = min(spectral_hist.shape[0], self.locked_bin + 2)
        
        fused_complex = np.sum(spectral_hist[start_bin:end_bin, :], axis=0) # Sum across bins
        
        # 6. Extract raw phase data and unwrap using legacy utils algorithm for standard impulses
        raw_phase = np.angle(fused_complex, deg=True)
        target_data = self.unwrap_fn(raw_phase)
        
        # 7. Phase difference (remove DC) strictly yielding streaming Phase Velocity
        # Padding with 0 to maintain length N (was N-1)
        difference_data = np.zeros_like(target_data)
        difference_data[1:] = target_data[1:] - target_data[:-1]
        
        # 8. Lowpass filter the raw phase velocity
        b, a = signal.butter(4, 0.5, 'lowpass', fs=self.fps) 
        if len(difference_data) > 15:
            respiration_signal = signal.lfilter(b, a, difference_data)
        else:
            respiration_signal = difference_data
            
        # ── APNEA DETECTION PATH (unchanged) ──────────────────────────────
        # Uses phase velocity (difference + lowpass) -> derivative -> abs
        # 9. Metrics (Apnea, Depth, RR)
        first_derivative = np.gradient(respiration_signal)
        abs_derivative = np.abs(first_derivative)
        
        # Fixed physical scale (dynamic scaling disabled)
        self.deriv_base_max = 0.2
        sig_range = self.deriv_base_max
        raw_scale_derivative = abs_derivative / sig_range

        # ── DISPLAY PATH: Bandpass on raw phase (0.05–0.5 Hz) ────────────────
        # Applies a highpass filter (0.05 Hz) to the unwrapped phase (before
        # differentiation) to remove slow drift from posture/position changes.
        # Combined with the 0.5 Hz lowpass this is a bandpass that preserves
        # the sinusoidal chest displacement waveform. During apnea the signal
        # flattens because there are no oscillations in the 0.05-0.5 Hz band.
        # if len(target_data) > 15:
        #     b_hp, a_hp = signal.butter(2, 0.05, 'highpass', fs=self.fps)
        #     b_lp, a_lp = signal.butter(4, 0.5,  'lowpass',  fs=self.fps)
        #     hp_phase = signal.lfilter(b_hp, a_hp, target_data)
        #     bandpass_signal = signal.lfilter(b_lp, a_lp, hp_phase)
        # else:
        #     bandpass_signal = target_data

        # 10. Inject newest values into Frozen Local UI Buffers
        # Display buffer uses bandpass displacement (not velocity)
        self.plot_resp_buffer = np.roll(self.plot_resp_buffer, -frames)
        self.plot_resp_buffer[-frames:] = respiration_signal[-frames:]
        
        self.plot_deriv_buffer = np.roll(self.plot_deriv_buffer, -frames)
        self.plot_deriv_buffer[-frames:] = raw_scale_derivative[-frames:]
        
        # Use frozen arrays for the rest of calculation
        scale_derivative = self.plot_deriv_buffer[-len(raw_scale_derivative):]
        display_signal = self.plot_resp_buffer[-len(respiration_signal):]
        
        # 10.5 Determine Dynamic Apnea Threshold (10s warmup + 30s calibration = 40s)
        if not hasattr(self, 'apnea_threshold'):
            self.apnea_threshold = 0.20  # Fallback prior to calibration
            self.threshold_calibrated = False

        calc_frame = int(40.0 * self.fps)
        if self._global_frame_idx >= calc_frame and not self.threshold_calibrated:
            normal_peak_deriv = np.percentile(self.plot_deriv_buffer, 95)
            # Threshold is dynamically set to 25% of their "normal" peak breathing speed
            self.apnea_threshold = max(0.05, normal_peak_deriv * 0.25)
            self.threshold_calibrated = True
            print(f"Dynamic Apnea Threshold locked mathematically: {self.apnea_threshold:.3f}")

        # Detect Apnea
        apnea_len = int(5.0 * self.fps)
        if len(scale_derivative) > apnea_len and self.threshold_calibrated:
            recent_deriv = scale_derivative[-apnea_len:]
            # Use 95th percentile against the personalized dynamic threshold
            curr_apnea = bool(np.percentile(recent_deriv, 95) <= self.apnea_threshold)
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
            
        # Calculate array of current (local_start, local_end) segments inside 30s buffer
        trace_view = self.apnea_trace[-len(scale_derivative):].astype(int)
        diffs = np.diff(np.concatenate(([0], trace_view, [0])))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        apnea_segments = list(zip(starts, ends))
            
        self.apnea_tracker.update(apnea_segments, self.fps, len(scale_derivative), self._global_frame_idx)
                
        # Peaks & RR Cycle Tracker runs stably on the frozen UI display_signal
        troughs, peaks = detect_respiratory_peaks(display_signal, self.fps)
        if len(peaks) > 0:
            self.cycle_tracker.update(peaks, self._global_frame_idx, len(display_signal))
            self.last_peak_global = self._global_frame_idx - len(display_signal) + peaks[-1]
            
        # Roll RR history natively and smoothly drop it when peaks stop arriving (No EMA per user spec)
        base_rr = self.cycle_tracker.get_rr_avg()
        if hasattr(self, 'last_peak_global') and base_rr > 0:
            time_since_last_peak = (self._global_frame_idx - self.last_peak_global) / self.fps
            average_dur = 60.0 / base_rr
            if time_since_last_peak > average_dur:
                current_rr = 60.0 / time_since_last_peak
            else:
                current_rr = base_rr
        else:
            current_rr = base_rr
            
        if self.apnea_active and current_rr < 6.0:
            current_rr = 0.0

        self.rr_history_buffer = np.roll(self.rr_history_buffer, -frames)
        self.rr_history_buffer[-frames:] = current_rr
        
        # Depth Estimation: only from recent peaks (last 10s), suppressed during apnea
        depth_str = "--"
        if not self.apnea_active and len(peaks) > 0 and len(troughs) > 0:
            recent_cutoff = max(0, len(display_signal) - int(10 * self.fps))
            recent_peaks   = [p for p in peaks   if p >= recent_cutoff]
            recent_troughs = [t for t in troughs if t >= recent_cutoff]
            if recent_peaks and recent_troughs:
                last_peak   = recent_peaks[-1]
                last_trough = recent_troughs[-1]
                if last_peak < len(display_signal) and last_trough < len(display_signal):
                    depth_val = abs(display_signal[last_peak] - display_signal[last_trough])
                    if depth_val < 5.0:
                        depth_str = "shallow"
                    elif depth_val > 15.0:
                        depth_str = "deep"
                    else:
                        depth_str = "normal"
                    
        # BRV Estimation (Variance of breath cycle durations in seconds)
        brv_val = 0.0
        if hasattr(self.cycle_tracker, 'cycle_durations') and len(self.cycle_tracker.cycle_durations) > 1:
            brv_val = float(np.std(self.cycle_tracker.cycle_durations))
                
        # Handle "Calibration" generic state
        is_calib = self._global_frame_idx < calc_frame

        return {
            'live_signal': display_signal,
            'derivative_signal': scale_derivative,
            'locked_bin': self.locked_bin,
            'confidence': 90.0 if not is_calib else 0.0,
            'is_calibrating': is_calib,
            'rr_current': current_rr,
            'cycle_count': self.cycle_tracker.count,
            'apnea_count': self.apnea_tracker.count,
            'depth': depth_str,
            'inhales': troughs,
            'exhales': peaks,
            'rr_history': np.copy(self.rr_history_buffer),
            'apnea_active': self.apnea_active,
            'apnea_segments': apnea_segments,
            'apnea_duration': self.live_apnea_frames / self.fps,
            'last_cycle_duration': self.cycle_tracker.last_duration,
            'brv_value': brv_val,
            'Motion_State_bin': motion_str
        }

    def _get_empty_dict(self):
        return {
            "live_signal": np.zeros(self.window_frames),
            "derivative_signal": np.zeros(self.window_frames),
            "inhales": [], "exhales": [],
            "rr_current": 0.0,
            "rr_history": np.zeros(self.window_frames),
            "apnea_active": False, "apnea_duration": 0.0,
            "apnea_trace": np.zeros(self.window_frames, dtype=bool),
            "apnea_segments": [],
            "apnea_count": 0, "apnea_durations": [],
            "cycle_duration": 0.0, "depth": "unknown",
            "confidence": 0.0, "motion_status": "STABLE",
            "cycle_count": 0, "brv_value": 0.0,
            "last_cycle_duration": 0.0,
            "is_calibrating": False
        }
