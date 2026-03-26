import numpy as np
from scipy import signal
import time
from collections import deque
from config import config

class RespiratoryPipeline:
    def __init__(self, fps=25, window_seconds=30, baseline_seconds=40):
        self.fps = fps
        self.window_frames = int(window_seconds * self.fps)
        self.baseline_frames = int(baseline_seconds * self.fps) # Baseline needs to be longer than the window
        
        # State & History Buffers
        self.history_buffer = np.zeros(self.baseline_frames)
        self.rr_history_buffer = np.zeros(self.window_frames) # Size RR history directly to match live window time domain
        
        # Apnea trace (True/False boolean array synced with live_signal frame-by-frame)
        self.apnea_trace = np.zeros(self.window_frames, dtype=bool)

        self.apnea_active = False
        self.apnea_duration = 0.0
        
        self.current_rr = 0.0
        self.cycle_duration = 0.0
        self.depth_status = "unknown"
        self.confidence = 0.0
        self.locked_bin = None
        self.frames_since_present = 0

    def process(self, act_pipe_out_dict, frames=1):
        """
        act_pipe_out_dict: Output dictionary from ActivityPipeline containing bin and history
        """
        current_bin = act_pipe_out_dict['final_bin']
        motion_str = act_pipe_out_dict['motion_str']
        
        # 1. Target Lock (Hysteresis)
        # Only switch respiratory tracking bin if there was macro motion, or no lock exists
        if self.locked_bin is None or motion_str == "MACRO_PHASE":
            self.locked_bin = current_bin
            
        # 2. Multi-Bin Spatial Fusion
        # Grab exactly 1 bin on each side of the locked bin to smoothly average spatial shifts
        spectral_hist = act_pipe_out_dict['spectral_history']
        start_bin = max(0, self.locked_bin - 1)
        end_bin = min(spectral_hist.shape[0], self.locked_bin + 2)
        
        fused_complex = np.sum(spectral_hist[start_bin:end_bin, :], axis=0)

        # 3. Phase unwrapping, detrending, and SQI calculation
        raw_phase = np.unwrap(np.angle(fused_complex))
        detrended_phase = signal.detrend(raw_phase)
        
        # Calculate Signal Quality Index (SQI)
        window = np.hanning(len(detrended_phase))
        fft_mag = np.abs(np.fft.rfft(detrended_phase * window))
        freqs = np.fft.rfftfreq(len(detrended_phase), d=(1.0/self.fps))

        breathing_mask = (freqs >= 0.15) & (freqs <= 0.5)
        total_mask = (freqs >= 0.15) & (freqs <= 3.0)
        
        sqi = np.sum(fft_mag[breathing_mask]) / (np.sum(fft_mag[total_mask]) + 1e-6)

        # 2. Confidence Metric Evaluation
        if act_pipe_out_dict['motion_str'] == "MACRO_PHASE":
            self.confidence = 0.0   # Total corruption during huge movement
        elif act_pipe_out_dict['motion_str'] == "MICRO_PHASE":
            self.confidence = min(30.0, sqi * 100) # Capped at 30% during fidgeting
        else: # STABLE
            self.confidence = min(100.0, sqi * 200) # e.g. SQI of 0.45 = 90% confidence
            
        # 3. Bandpass filtering for Live UI
        b, a = signal.butter(2, [0.15, 0.5], btype='bandpass', fs=self.fps)
        filtered_resp = signal.filtfilt(b, a, detrended_phase)
        live_signal = filtered_resp[-self.window_frames:]
        
        # Mask out "ghost" phases before the subject physically sat down
        self.frames_since_present += frames
        if self.frames_since_present < self.window_frames:
            live_signal[:-self.frames_since_present] = 0.0

        # Update Long-term baseline history 
        if self.confidence > 40.0:
            self.history_buffer = np.roll(self.history_buffer, -len(filtered_resp))
            self.history_buffer[-len(filtered_resp):] = filtered_resp

        # 4. Scale-Invariant Apnea Detection
        self.apnea_trace = np.roll(self.apnea_trace, -frames) # Shift the visual trace left by frames
        self.apnea_trace[-frames:] = False

        if act_pipe_out_dict['motion_str'] != "MACRO_PHASE":
            recent_signal = self.history_buffer[-int(5 * self.fps):]
            signal_mean_abs = np.mean(np.abs(recent_signal))
            
            if signal_mean_abs > 1e-6:
                norm_var = np.var(recent_signal) / (signal_mean_abs ** 2)
                norm_range = np.ptp(recent_signal) / signal_mean_abs
                self.apnea_active = (norm_var < 0.05 and norm_range < 0.3)
            else:
                self.apnea_active = True

            if self.apnea_active: self.apnea_duration += (frames / float(self.fps))
            else: self.apnea_duration = 0.0
        else:
            self.apnea_active = False
            self.apnea_duration = 0.0

        # Mark the most recent frames in the UI visual trace
        if self.apnea_active:
            self.apnea_trace[-frames:] = True

        # --- Peak Detection (Inhales = Troughs, Exhales = Peaks) ---
        inhales = []
        exhales = []
        
        if self.confidence > 20.0 and not self.apnea_active:
            min_dist = max(1, int(self.fps * 0.5))
            sig_range = np.ptp(live_signal)
            prominence = max(0.05, sig_range * 0.25)
            
            exhales_arr, _ = signal.find_peaks(live_signal, distance=min_dist, prominence=prominence)
            inhales_arr, _ = signal.find_peaks(-live_signal, distance=min_dist, prominence=prominence)
            exhales = exhales_arr.tolist()
            inhales = inhales_arr.tolist()
            
            # --- Respiration Rate (RR) & Cycle Duration ---
            if len(inhales) >= 2:
                # Cycle duration based on Inhale-to-Inhale times
                intervals = np.diff(inhales) / self.fps
                valid_intervals = intervals[(intervals > 0.5) & (intervals < 6.0)]
                if len(valid_intervals) > 0:
                    self.cycle_duration = np.median(valid_intervals)
                    self.current_rr = 60.0 / self.cycle_duration
                else:
                    self.current_rr = 0.0
            else:
                self.current_rr = 0.0
        else:
            self.current_rr = 0.0

        # --- Depth Classification ---
        if self.confidence > 20.0 and not self.apnea_active:
            baseline_mean = np.mean(self.history_buffer)
            baseline_amplitude = np.mean(np.abs(self.history_buffer - baseline_mean))
            
            recent_mean = np.mean(live_signal)
            recent_amplitude = np.mean(np.abs(live_signal - recent_mean))
            
            if baseline_amplitude > 1e-4:
                amplitude_ratio = recent_amplitude / baseline_amplitude
                if amplitude_ratio > 1.3:
                    self.depth_status = "deep"
                elif amplitude_ratio < 0.7:
                    self.depth_status = "shallow"
                else:
                    self.depth_status = "normal"
            else:
                self.depth_status = "unknown"
        elif self.apnea_active:
            self.depth_status = "apnea"
        else:
            self.depth_status = "unknown"
        
        # Roll RR history for plot synced to the live_signal time axis
        self.rr_history_buffer = np.roll(self.rr_history_buffer, -frames)
        self.rr_history_buffer[-frames:] = self.current_rr

        return {
            "live_signal": live_signal,
            "inhales": inhales,
            "exhales": exhales,
            "cycle_duration": self.cycle_duration,
            "rr_current": self.current_rr,
            "rr_history": self.rr_history_buffer, # Length = self.window_frames
            "apnea_active": self.apnea_active,
            "apnea_duration": self.apnea_duration,
            "apnea_trace": self.apnea_trace, # Length = self.window_frames
            "depth": self.depth_status,
            "confidence": self.confidence,
            "motion_status": act_pipe_out_dict['motion_str']
        }
        
    def _reset_state(self):
        self.apnea_active = False
        self.apnea_duration = 0.0
        self.apnea_trace = np.zeros(self.window_frames, dtype=bool)
        self.rr_history_buffer = np.zeros(self.window_frames)
        self.depth_status = "unknown"
        self.confidence = 0.0
        self.locked_bin = None
        self.history_buffer = np.zeros(self.baseline_frames)
        self.current_rr = 0.0
        self.cycle_duration = 0.0
        self.frames_since_present = 0
        
    def _get_empty_dict(self):
        return {
            "live_signal": np.zeros(self.window_frames),
            "inhales": [], "exhales": [],
            "rr_current": 0.0, 
            "rr_history": np.zeros(self.window_frames),
            "apnea_active": False, "apnea_duration": 0.0, 
            "apnea_trace": np.zeros(self.window_frames, dtype=bool),
            "cycle_duration": 0.0, "depth": "unknown",
            "confidence": 0.0, "motion_status": "STABLE"
        }

