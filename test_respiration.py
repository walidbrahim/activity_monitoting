"""
Standalone Respiration Pipeline Test App
========================================
Uses REAL radar data but bypasses ActivityPipeline and RobotController.
Builds the spectral_history ring buffer in a minimal way, and feeds the
real RespiratoryPipeline to test all breathing features in isolation.

Usage:
    python test_respiration.py
"""

import sys
import queue
import logging
import numpy as np
import scipy.signal as signal
from scipy.signal import find_peaks

from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                              QLabel, QFrame, QApplication, QSlider, QPushButton,
                              QGroupBox, QComboBox, QTabWidget)
from PyQt6.QtCore import pyqtSlot, Qt, QTimer
from PyQt6.QtGui import QFont, QColor
import pyqtgraph as pg

from config import config
from libs.pipelines.respirationPipeline import RespiratoryPipeline
from libs.controllers.radarController import RadarController

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Legacy Signal Processor — Replicates 'others/data_processor.py' approach
# ──────────────────────────────────────────────────────────────────────────────
def detect_respiratory_peaks(signal_data, fs):
    """
    Robust peak/trough detection for respiration signal.
    Returns (troughs, peaks) indices.
    """
    signal_data = np.asarray(signal_data)
    min_dist = max(1, int(fs * 0.5))
    sig_range = np.ptp(signal_data)
    prominence = max(0.05, sig_range * 0.25) 
    peaks, _ = find_peaks(signal_data, distance=min_dist, prominence=prominence)
    troughs, _ = find_peaks(-signal_data, distance=min_dist, prominence=prominence)
    return troughs, peaks

def unwrap_phase_Ambiguity(phase_queue):
    phase_arr = np.copy(phase_queue)
    phase_arr_ret = np.copy(phase_arr)
    phase_diff_correction_cum = 0
    for i in range(len(phase_arr)):
        if not i:
            continue
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
        if (phase_diff_correction < 180 and phase_diff_correction > 0) \
                or (phase_diff_correction > -180 and phase_diff_correction < 0):
            phase_diff_correction = 0
        phase_diff_correction_cum += phase_diff_correction
        phase_arr_ret[i] = phase_arr[i] + phase_diff_correction_cum
    return phase_arr_ret

class LegacyRespiratoryPipeline:
    def __init__(self, fps):
        self.fps = fps
        self.window_frames = config.respiration.resp_window_sec * fps
        self.num_range_bins = config.radar.range_idx_num
        self.range_res = config.radar.range_resolution
        
        self.frame_buffer = [] 
        
        from libs.pipelines.respirationPipeline import ApneaTracker, BreathCycleTracker 
        self.apnea_tracker = ApneaTracker()
        self.cycle_tracker = BreathCycleTracker(history_size=5, fps=fps)
        self.rr_history_buffer = np.zeros(self.window_frames)
        
        self.locked_bin = 5
        self.global_frame_count = 0
        self.warmup_frames = int(fps * config.tuning.warmup_seconds)
        self.in_apnea = False

    def _reset_respiratory_state(self):
        """Hard wipe the internal logic trace buffers when someone leaves the bed so ghost Apnea instances never persist."""
        self.in_apnea = False
        if hasattr(self, 'live_apnea_frames'): self.live_apnea_frames = 0
        if hasattr(self, 'apnea_trace'): self.apnea_trace.fill(False)
        self.apnea_tracker.reset()
        self.cycle_tracker.reset()
        self.smoothed_rr = 0.0
        self.smoothed_ibi = 0.0
        self.last_peak_global = None
        if hasattr(self, 'plot_resp_buffer'):
            self.plot_resp_buffer.fill(0)
            self.plot_deriv_buffer.fill(0)
        if hasattr(self, 'rr_history_buffer'): self.rr_history_buffer.fill(0)
        if hasattr(self, 'ibi_history_buffer'): self.ibi_history_buffer.fill(0)

    def process_frame(self, fft_1d_data, frames_processed=1):
        self.global_frame_count += frames_processed

        # We only use ANTENNA 0 for this pipeline, exactly like legacy code
        # _range_matrix_queue = clutter_remove(_range_matrix_queue[:, :, 0])
        ant0_data = fft_1d_data[:, 0]
            
        self.frame_buffer.append(ant0_data)
        if len(self.frame_buffer) > self.window_frames:
            self.frame_buffer.pop(0)
            
        is_calibrating = self.global_frame_count <= self.warmup_frames
        if is_calibrating or len(self.frame_buffer) < self.fps * 2:
            return {'is_calibrating': True, 'confidence': 0}
            
        buf_np = np.array(self.frame_buffer)
        
        # 2. Clutter remove (Append strictly newest frame to avoid retroactive background morphing that causes 'stitched' filtering artifacts)
        if not hasattr(self, 'clutter_baseline'):
            self.clutter_baseline = np.mean(buf_np, axis=0)
            
            # Back-fill the first cleanly captured array 
            clut_np = np.copy(buf_np)
            for b_idx in range(self.num_range_bins):
                clut_np[:, b_idx] -= self.clutter_baseline[b_idx]
            self.cluttered_buffer = clut_np.tolist()
        else:
            self.clutter_baseline = 0.999 * self.clutter_baseline + 0.001 * ant0_data
            current_cluttered = ant0_data - self.clutter_baseline
            self.cluttered_buffer.append(current_cluttered)
            
        while len(self.cluttered_buffer) > self.window_frames:
            self.cluttered_buffer.pop(0)
            
        cluttered_np = np.array(self.cluttered_buffer)
            
        # 3. Target localization (Constrained strictly to the Bed: ~2.5m)
        dynamic_mag = np.abs(cluttered_np[-1])
        min_bin = max(2, int(config.tuning.min_search_range / self.range_res))
        max_bin = int(2.5 / self.range_res) 
        
        peak_mag = np.max(dynamic_mag[min_bin:max_bin]) if min_bin < max_bin and len(dynamic_mag) > max_bin else 0.0
        
        # Binary Occupancy Check (using identical tuning threshold directly without scaling)
        is_occupied = peak_mag >= config.pipeline.detection_threshold
        
        if not is_occupied:
            # Ghost detected mathematically empty: Force the baseline to aggressively absorb the inanimate bed
            self.clutter_baseline = 0.5 * self.clutter_baseline + 0.5 * ant0_data
            self._reset_respiratory_state()
            return {
                'status': 'Empty', 
                'motion': 'Still',
                'confidence': 0, 
                'live_signal': [],
                'rr_bpm': 0,
                'is_calibrating': False
            }
        
        if peak_mag > 0 and min_bin < max_bin:
            best_cand = min_bin + np.argmax(dynamic_mag[min_bin:max_bin])
            
            # HYSTERESIS TETHERING: Prevent the target bin from frantically jumping between adjacent centimeters.
            # Only jump if the new candidate strictly overpowers the tethered bin by +25%, or the old bin dies entirely.
            if not hasattr(self, 'locked_bin') or self.locked_bin is None:
                self.locked_bin = best_cand
            else:
                current_mag = dynamic_mag[self.locked_bin] if self.locked_bin < len(dynamic_mag) else 0.0
                cand_mag = dynamic_mag[best_cand] if best_cand < len(dynamic_mag) else 0.0
                
                # If current bin evaporated (< 50) or candidate is massively stronger, break the tether!
                if current_mag < 50.0 or cand_mag > current_mag * 1.25:
                    self.locked_bin = best_cand
        elif min_bin < len(dynamic_mag):
            self.locked_bin = min_bin + np.argmax(dynamic_mag[min_bin:])
        else:
            self.locked_bin = min_bin
            
        # 4. Target bins (+/- 1)
        tbins = [
            max(0, self.locked_bin - 1),
            self.locked_bin,
            min(self.num_range_bins - 1, self.locked_bin + 1)
        ]
        
        # 5. Sum Complex Data from Target Bins
        complex_subset = cluttered_np[:, tbins]
        combined_complex = np.sum(complex_subset, axis=1)
        
        # 6. Extract raw phase data and unwrap mathematically via numpy to destroy phase impulse jumps
        raw_phase = np.angle(combined_complex, deg=False)
        target_data = np.unwrap(raw_phase)
        target_data = np.degrees(target_data) # MUST mathematically scale back to Degrees!
        
        # 7. Phase Based Velocity Ghost Evaporator
        # Phase velocity is strictly used internally for purely algorithmic aliveness thresholding
        difference_data = target_data[1:] - target_data[:-1]
        phase_var = np.var(difference_data[-15:]) if len(difference_data) > 15 else 100.0
        if phase_var < 0.05:
            # Absorb the ghost shadow into the baseline immediately
            self.clutter_baseline = 0.5 * self.clutter_baseline + 0.5 * ant0_data
            self._reset_respiratory_state()
            return {
                'status': 'Empty', 
                'motion': 'Still',
                'confidence': 0, 
                'live_signal': [],
                'rr_bpm': 0,
                'is_calibrating': False,
                'apnea_calibrated': False
            }
            
        # 8. Non-Ringing Displacement Filter (1st-Order HP + 4th-Order LP)
        # We explicitly avoid Bandpass filters because their high-Q pole pairs 'ring'
        # into fake oscillations during flat apneas! A 1st-order HP only exponentially decays.
        if len(target_data) > 15:
            # Soft Highpass (0.05 Hz) perfectly anchors the DC wander so it stays in the [-30, 30] GUI box
            sos_hp = signal.butter(1, 0.05, btype='highpass', fs=self.fps, output='sos')
            centered_target = signal.sosfiltfilt(sos_hp, target_data, padlen=min(15, len(target_data)-1))
            
            # Smooth Lowpass (0.5 Hz) 
            sos_lp = signal.butter(4, 0.5, btype='lowpass', fs=self.fps, output='sos') 
            respiration_signal = signal.sosfiltfilt(sos_lp, centered_target, padlen=min(15, len(centered_target)-1))
        else:
            respiration_signal = np.zeros_like(target_data)
            
        # 9. Metrics (Apnea, Depth, RR)
        first_derivative = np.gradient(respiration_signal)
        abs_derivative = np.abs(first_derivative)
        
        # Standardize derivative actively using asymmetric dynamic tracking!
        if not hasattr(self, 'deriv_base_max'):
            self.deriv_base_max = 5.0
            self.deriv_base_min = 0.0

        current_max = np.percentile(abs_derivative, 95) if len(abs_derivative) > 20 else self.deriv_base_max
        self.deriv_base_min = np.min(abs_derivative) if len(abs_derivative) > 20 else 0.0

        # Autoscale instantly for deep breathing, but decay VERY slowly so apnea doesn't falsely zoom into the noise floor
        if current_max > self.deriv_base_max:
            self.deriv_base_max = 0.8 * self.deriv_base_max + 0.2 * current_max
        elif not getattr(self, 'in_apnea', False):
            self.deriv_base_max = 0.999 * self.deriv_base_max + 0.001 * current_max

        sig_range = max(self.deriv_base_max - self.deriv_base_min, 0.05)
        raw_scale_derivative = (abs_derivative - self.deriv_base_min) / sig_range
        
        # 10. Inject into Frozen Local UI Buffers (This mathematically absolutely guarantees the past plot never wiggles)
        if not hasattr(self, 'plot_resp_buffer'):
            self.plot_resp_buffer = np.zeros(self.window_frames)
            self.plot_deriv_buffer = np.zeros(self.window_frames)
            
        self.plot_resp_buffer = np.roll(self.plot_resp_buffer, -frames_processed)
        self.plot_resp_buffer[-frames_processed:] = respiration_signal[-frames_processed:]
        
        self.plot_deriv_buffer = np.roll(self.plot_deriv_buffer, -frames_processed)
        self.plot_deriv_buffer[-frames_processed:] = raw_scale_derivative[-frames_processed:]
        
        # Use frozen arrays for the rest of calculation
        scale_derivative = self.plot_deriv_buffer[-len(raw_scale_derivative):]
        display_signal = self.plot_resp_buffer[-len(respiration_signal):]
        
        # 10.5 Determine Dynamic Apnea Threshold 
        # (10s warmup + 30s calibration = 40s)
        if not hasattr(self, 'apnea_threshold'):
            self.apnea_threshold = 0.10  # Fallback prior to calibration
            self.threshold_calibrated = False

        calc_frame = int(40.0 * self.fps)
        if self.global_frame_count >= calc_frame and not self.threshold_calibrated:
            normal_peak_deriv = np.percentile(self.plot_deriv_buffer, 95)
            # Threshold is dynamically set to 25% of their "normal" peak breathing speed
            self.apnea_threshold = 0.1 #max(0.05, normal_peak_deriv * 0.25)
            self.threshold_calibrated = True
            logger.info(f"Dynamic Apnea Threshold Set: {self.apnea_threshold:.3f}")
            print(f"Dynamic Apnea Threshold Set: {self.apnea_threshold:.3f}")

        # Detect Apnea
        apnea_len = int(5.0 * self.fps)
        
        # Expand Apnea Segments over the display using a continuous boolean trace
        if not hasattr(self, 'apnea_trace'):
            self.apnea_trace = np.zeros(self.window_frames, dtype=bool)
            self.live_apnea_frames = 0
        
        if len(scale_derivative) > apnea_len and self.threshold_calibrated:
            recent_deriv = scale_derivative[-apnea_len:]
            # Use 95th percentile against the personalized dynamic threshold
            curr_apnea = bool(np.percentile(recent_deriv, 95) <= self.apnea_threshold)
            if curr_apnea and not self.in_apnea:
                self.in_apnea = True
                # Retroactively mark the 4-second waiting period directly so the zone overlays exactly from the start of the flatline!
                self.apnea_trace[-apnea_len:] = True
                self.live_apnea_frames = apnea_len
            elif not curr_apnea:
                self.in_apnea = False
                
        self.apnea_trace = np.roll(self.apnea_trace, -frames_processed)
        self.apnea_trace[-frames_processed:] = self.in_apnea
        
        if self.in_apnea:
            self.live_apnea_frames += frames_processed
        else:
            self.live_apnea_frames = 0
            
        # Calculate exactly the array of current (local_start, local_end) segments inside the whole 30s buffer
        trace_view = self.apnea_trace[-len(scale_derivative):].astype(int)
        diffs = np.diff(np.concatenate(([0], trace_view, [0])))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        apnea_segments = list(zip(starts, ends))
            
        self.apnea_tracker.update(apnea_segments, self.fps, len(scale_derivative), self.global_frame_count)
                
        # Peaks & RR Cycle Tracker runs stably on the frozen UI display_signal
        troughs, peaks = detect_respiratory_peaks(display_signal, self.fps)
        if len(peaks) > 0:
            self.cycle_tracker.update(peaks, self.global_frame_count, len(display_signal))
            self.last_peak_global = self.global_frame_count - len(display_signal) + peaks[-1]
            
        # 11. Roll IBI history natively and derive RR
        # Grab the pure average cycle duration in seconds
        avg_cycle_dur = 0.0
        if hasattr(self.cycle_tracker, 'cycle_durations') and self.cycle_tracker.cycle_durations:
            hist_size = self.cycle_tracker.history_size
            avg_cycle_dur = float(np.mean(self.cycle_tracker.cycle_durations[-hist_size:]))

        if hasattr(self, 'last_peak_global') and avg_cycle_dur > 0:
            time_since_last_peak = (self.global_frame_count - self.last_peak_global) / self.fps
            if time_since_last_peak > avg_cycle_dur:
                # Breath is late! The interval is dynamically extending
                current_ibi = time_since_last_peak
            else:
                current_ibi = avg_cycle_dur
        else:
            current_ibi = avg_cycle_dur
            
        # Ensure it zeroes out fully if Apnea is actively declared
        if self.in_apnea and current_ibi > 10.0:
            current_ibi = 0.0

        # Apply a smooth exponential average to visual IBI to eliminate sawtooth jumping!
        if not hasattr(self, 'smoothed_ibi'):
            self.smoothed_ibi = current_ibi
        self.smoothed_ibi = 0.98 * self.smoothed_ibi + 0.02 * current_ibi
        
        # RR is mathematically derived perfectly from the smoothed IBI
        self.smoothed_rr = (60.0 / self.smoothed_ibi) if self.smoothed_ibi >= 1.0 else 0.0

        if not hasattr(self, 'ibi_history_buffer'):
            self.ibi_history_buffer = np.zeros(self.window_frames)
        self.ibi_history_buffer = np.roll(self.ibi_history_buffer, -frames_processed)
        self.ibi_history_buffer[-frames_processed:] = self.smoothed_ibi

        self.rr_history_buffer = np.roll(self.rr_history_buffer, -frames_processed)
        self.rr_history_buffer[-frames_processed:] = self.smoothed_rr
        
        # 12. Depth Estimation (Peak to Trough Amplitude)
        depth_str = "normal"
        if len(peaks) > 0 and len(troughs) > 0:
            last_peak = peaks[-1]
            last_trough = troughs[-1]
            if last_peak < len(display_signal) and last_trough < len(display_signal):
                depth_val = abs(display_signal[last_peak] - display_signal[last_trough])
                if depth_val < 5.0:
                    depth_str = "shallow"
                elif depth_val > 15.0:
                    depth_str = "deep"
                    
        # 13. BRV Estimation (Variance of breath cycle durations in seconds)
        brv_val = 0.0
        if hasattr(self.cycle_tracker, 'cycle_durations') and len(self.cycle_tracker.cycle_durations) > 1:
            brv_val = float(np.std(self.cycle_tracker.cycle_durations))
                
        # 14. Motion vs Still (Derivative threshold logic)
        motion_mean = np.mean(abs_derivative[-15:]) if len(abs_derivative) > 15 else 0.0
        # If the chest is violently moving (mean dev > 2.0), the person is actively shifting or moving limbs
        motion_str = "Moving" if motion_mean > 0.5 else "Still"
        
        # Map signal to dict output expected by GUI
        return {
            'status': 'Occupied',
            'motion': motion_str,
            'live_signal': display_signal,
            'derivative_signal': scale_derivative,
            'locked_bin': self.locked_bin,
            'confidence': 90,
            'is_calibrating': False,
            'apnea_calibrated': getattr(self, 'threshold_calibrated', False),
            'rr_current': self.smoothed_rr,
            'cycle_count': self.cycle_tracker.count,
            'apnea_count': self.apnea_tracker.count,
            'depth': depth_str,
            'inhales': troughs,
            'exhales': peaks,
            'rr_history': np.copy(self.rr_history_buffer[-len(respiration_signal):]),
            'apnea_active': self.in_apnea,
            'apnea_segments': apnea_segments,
            'apnea_duration': self.live_apnea_frames / self.fps,
            'last_cycle_duration': self.cycle_tracker.last_duration,
            'brv_value': brv_val,
            'ibi_current': current_ibi,
            'ibi_history': np.copy(self.ibi_history_buffer[-len(respiration_signal):]),
            'Motion_State_bin': "STABLE"
        }


# ──────────────────────────────────────────────────────────────────────────────
# Test GUI Window
# ──────────────────────────────────────────────────────────────────────────────
class RespTestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🫁 Respiration Pipeline — Standalone Test")
        self.resize(1100, 750)
        self.setStyleSheet(f"background-color: {config.gui_theme.fig_bg};")

        # Time axis
        fps = config.radar.frame_rate
        window_sec = config.respiration.resp_window_sec
        hist_len = int(window_sec * fps)
        self.x_axis = np.linspace(-window_sec, 0, hist_len)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # ── Row 1: Breathing Signal ──
        self.resp_plot = self._create_plot("🫁 Live Breathing Signal", "Time (s)", "Breathing Signal (a.u.)")
        self.resp_plot.setXRange(-window_sec, 0, padding=0)
        # autoscale y axis
        # Dynamics Phase plot autoscaling!
        self.resp_plot.setYRange(-30, 30)
        self.resp_plot.enableAutoRange(axis='x', enable=False)
        self.resp_plot.enableAutoRange(axis='y', enable=False)

        self.curve_resp = self.resp_plot.plot(pen=pg.mkPen(color=config.gui_theme.occupant, width=2))
        self.curve_resp_zero = self.resp_plot.plot(pen=pg.mkPen(None))
        occ_c = QColor(config.gui_theme.occupant)
        self.fill_resp = pg.FillBetweenItem(self.curve_resp, self.curve_resp_zero,
                                             brush=pg.mkBrush(occ_c.red(), occ_c.green(), occ_c.blue(), 35))
        self.resp_plot.addItem(self.fill_resp)

        self.scatter_inhale = pg.ScatterPlotItem(size=10, brush=pg.mkBrush('green'), symbol='t')
        self.scatter_exhale = pg.ScatterPlotItem(size=10, brush=pg.mkBrush('red'), symbol='t1')
        self.resp_plot.addItem(self.scatter_inhale)
        self.resp_plot.addItem(self.scatter_exhale)

        # Annotations
        self._resp_ann_depth = pg.TextItem("", anchor=(0, 0), color=config.gui_theme.text)
        self._resp_ann_depth.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.resp_plot.addItem(self._resp_ann_depth)

        self._resp_ann_apnea = pg.TextItem("", anchor=(1, 0), color='#EF4444')
        self._resp_ann_apnea.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.resp_plot.addItem(self._resp_ann_apnea)

        self._resp_ann_cycles = pg.TextItem("", anchor=(0, 1), color=config.gui_theme.subtext)
        self._resp_ann_cycles.setFont(QFont("Arial", 10))
        self.resp_plot.addItem(self._resp_ann_cycles)

        self._resp_ann_conf = pg.TextItem("", anchor=(1, 1), color=config.gui_theme.subtext)
        self._resp_ann_conf.setFont(QFont("Arial", 10))
        self.resp_plot.addItem(self._resp_ann_conf)

        self._apnea_regions = []

        main_layout.addWidget(self.resp_plot, stretch=3)

        # ── Row 2: Analytics Tabs + Derivative side-by-side ──
        row2 = QHBoxLayout()

        self.analytics_tabs = QTabWidget()
        self.analytics_tabs.setStyleSheet(f"""
            QTabWidget::pane {{ border: 1px solid {config.gui_theme.grid}; border-radius: 5px; }}
            QTabBar::tab {{ background: {config.gui_theme.panel_bg}; color: {config.gui_theme.subtext}; padding: 8px 15px; border-top-left-radius: 4px; border-top-right-radius: 4px; margin-right: 2px; }}
            QTabBar::tab:selected {{ background: {config.gui_theme.card_bg}; color: {config.gui_theme.text}; font-weight: bold; border-bottom: 2px solid {config.gui_theme.occupant}; }}
        """)

        # Tab 1: RR Trend
        self.tab_rr = QWidget()
        rr_layout = QVBoxLayout(self.tab_rr)
        rr_layout.setContentsMargins(0, 0, 0, 0)

        self.rr_plot = self._create_plot("📈 Respiration Rate (RR)", "Time (s)", "BPM")
        self.rr_plot.setXRange(-window_sec, 0, padding=0)
        self.rr_plot.setYRange(0, 40)
        self.rr_plot.enableAutoRange(axis='x', enable=False)
        self.curve_rr = self.rr_plot.plot(pen=pg.mkPen(color=config.gui_theme.text, width=2))
        self.curve_rr_zero = self.rr_plot.plot(pen=pg.mkPen(None))
        txt_c = QColor(config.gui_theme.text)
        self.fill_rr = pg.FillBetweenItem(self.curve_rr, self.curve_rr_zero,
                                           brush=pg.mkBrush(txt_c.red(), txt_c.green(), txt_c.blue(), 25))
        self.rr_plot.addItem(self.fill_rr)

        self._rr_ann_current = pg.TextItem("", anchor=(1, 0), color=config.gui_theme.text)
        self._rr_ann_current.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.rr_plot.addItem(self._rr_ann_current)

        self._rr_ann_brv = pg.TextItem("", anchor=(0, 0), color=config.gui_theme.subtext)
        self._rr_ann_brv.setFont(QFont("Arial", 10))
        self.rr_plot.addItem(self._rr_ann_brv)

        self._rr_ann_cycle = pg.TextItem("", anchor=(0.5, 0), color=config.gui_theme.subtext)
        self._rr_ann_cycle.setFont(QFont("Arial", 10))
        self.rr_plot.addItem(self._rr_ann_cycle)

        rr_layout.addWidget(self.rr_plot)
        self.analytics_tabs.addTab(self.tab_rr, "📈 RR Trend")

        # Tab 2: IBI Trend
        self.tab_ibi = QWidget()
        ibi_layout = QVBoxLayout(self.tab_ibi)
        ibi_layout.setContentsMargins(0, 0, 0, 0)

        self.ibi_plot = self._create_plot("⏱️ Estimated IBI", "Time (s)", "Seconds")
        self.ibi_plot.setXRange(-window_sec, 0, padding=0)
        self.ibi_plot.setYRange(0, 15)  # Max interval ~15s (4 breaths/min)
        self.ibi_plot.enableAutoRange(axis='x', enable=False)
        self.curve_ibi = self.ibi_plot.plot(pen=pg.mkPen(color='#38BDF8', width=2))
        self.curve_ibi_zero = self.ibi_plot.plot(pen=pg.mkPen(None))
        self.fill_ibi = pg.FillBetweenItem(self.curve_ibi, self.curve_ibi_zero, brush=pg.mkBrush(56, 189, 248, 25))
        self.ibi_plot.addItem(self.fill_ibi)

        self._ibi_ann_current = pg.TextItem("", anchor=(1, 0), color='#38BDF8')
        self._ibi_ann_current.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.ibi_plot.addItem(self._ibi_ann_current)
        ibi_layout.addWidget(self.ibi_plot)
        self.analytics_tabs.addTab(self.tab_ibi, "⏱️ IBI Trend")

        row2.addWidget(self.analytics_tabs, stretch=2)

        # Derivative signal plot
        self.deriv_plot = self._create_plot("📉 Normalized Derivative", "Time (s)", "Norm")
        self.deriv_plot.setXRange(-window_sec, 0, padding=0)
        self.deriv_plot.setYRange(0, 1.05)
        self.deriv_plot.enableAutoRange(axis='x', enable=False)
        self.curve_deriv = self.deriv_plot.plot(pen=pg.mkPen(color='#F59E0B', width=1.5))
        # Threshold line
        thresh = config.respiration.resp_threshold
        self.deriv_plot.addItem(pg.InfiniteLine(
            pos=thresh, angle=0,
            pen=pg.mkPen(color='#EF4444', width=1, style=Qt.PenStyle.DashLine)
        ))
        self._deriv_ann_thresh = pg.TextItem(f"Apnea threshold: {thresh}", anchor=(1, 1), color='#EF4444')
        self._deriv_ann_thresh.setFont(QFont("Arial", 9))
        self._deriv_ann_thresh.setPos(0, thresh)
        self.deriv_plot.addItem(self._deriv_ann_thresh)

        row2.addWidget(self.deriv_plot, stretch=2)

        main_layout.addLayout(row2, stretch=2)

        # ── Row 3: Info bar ──
        self.info_bar = QLabel("Waiting for radar data...")
        self.info_bar.setStyleSheet(f"""
            color: {config.gui_theme.text}; 
            background-color: {config.gui_theme.card_bg}; 
            padding: 8px 15px; 
            border-radius: 6px; 
            font-size: 13px;
        """)
        self.info_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.info_bar)

        # Store window_sec for annotation positioning
        self._resp_window = window_sec

    def get_manual_bin(self):
        return None

    def _create_plot(self, title, xlabel, ylabel):
        p = pg.PlotWidget()
        p.setBackground(config.gui_theme.panel_bg)
        p.setTitle(title, color=config.gui_theme.text, size="13pt", bold=True)
        p.setLabel('bottom', xlabel, color=config.gui_theme.text)
        p.setLabel('left', ylabel, color=config.gui_theme.text)
        p.showGrid(x=True, y=True, alpha=0.3)
        p.getAxis('bottom').setPen(config.gui_theme.grid)
        p.getAxis('left').setPen(config.gui_theme.grid)
        p.getAxis('bottom').setTextPen(config.gui_theme.subtext)
        p.getAxis('left').setTextPen(config.gui_theme.subtext)
        return p

    @pyqtSlot(dict)
    def update_display(self, resp_dict):
        status = resp_dict.get('status', 'Empty')
        motion = resp_dict.get('motion', 'Still')

        if status == 'Empty' or not resp_dict.get('confidence', 0) > 0:
            if resp_dict.get('is_calibrating', False):
                self.info_bar.setText("⏳ Calibrating Radar & Phase Baseline... (Please keep bed empty)")
                self.info_bar.setStyleSheet(f"color: #F59E0B; background-color: {config.gui_theme.card_bg}; padding: 8px 15px; border-radius: 6px; font-weight: bold; font-size: 13px;")
            else:
                self.info_bar.setText("⚪ Room Empty - No Occupant Detected")
                self.info_bar.setStyleSheet(f"color: {config.gui_theme.text}; background-color: {config.gui_theme.card_bg}; padding: 8px 15px; border-radius: 6px; font-size: 13px;")
                
            self.curve_resp.setData([], [])
            self.scatter_inhale.setData([], [])
            self.scatter_exhale.setData([], [])
            
            # Wipe apnea blocks universally
            for region in self._apnea_regions:
                self.resp_plot.removeItem(region)
            self._apnea_regions.clear()
            
            # Wipe trace curves
            self.curve_rr.setData([], [])
            self.curve_rr_zero.setData([], [])
            self.curve_ibi.setData([], [])
            self.curve_ibi_zero.setData([], [])
            self.curve_deriv.setData([], [])
            
            # Reset texts
            self._resp_ann_apnea.setText("")
            self._resp_ann_cycles.setText("")
            self._rr_ann_current.setText("RR: --")
            self._ibi_ann_current.setText("IBI: --")
            return

        sig = resp_dict.get('live_signal', np.array([]))
        if len(sig) == 0:
            return

        is_apnea_calibrated = resp_dict.get('apnea_calibrated', False)
        if not is_apnea_calibrated:
            self.info_bar.setText(f"👤 Occupied   |   {motion}   |   ⏳ Profiling Breathing Scale (~30s)")
            self.info_bar.setStyleSheet(f"color: #38BDF8; background-color: {config.gui_theme.card_bg}; padding: 8px 15px; border-radius: 6px; font-weight: bold; font-size: 13px;")
        else:
            self.info_bar.setText(f"👤 Occupied   |   {motion}   |   ✅ System Ready")
            self.info_bar.setStyleSheet(f"color: #22C55E; background-color: {config.gui_theme.card_bg}; padding: 8px 15px; border-radius: 6px; font-weight: bold; font-size: 13px;")

        x_sig = self.x_axis[-len(sig):]

        # ── Breathing signal ──
        self.curve_resp.setData(x_sig, sig)
        self.curve_resp_zero.setData(x_sig, np.zeros(len(sig)))

        # Scatter markers
        inhs = resp_dict.get('inhales', [])
        exhs = resp_dict.get('exhales', [])
        if len(inhs) > 0 and len(sig) > 0:
            self.scatter_inhale.setData(x_sig[inhs], sig[inhs])
        else:
            self.scatter_inhale.setData([], [])
        if len(exhs) > 0 and len(sig) > 0:
            self.scatter_exhale.setData(x_sig[exhs], sig[exhs])
        else:
            self.scatter_exhale.setData([], [])

        # Apnea red zones
        for region in self._apnea_regions:
            self.resp_plot.removeItem(region)
        self._apnea_regions.clear()
        for (s, e) in resp_dict.get('apnea_segments', []):
            if 0 <= s < len(x_sig) and 0 < e <= len(x_sig):
                region = pg.LinearRegionItem(
                    values=[x_sig[s], x_sig[min(e, len(x_sig)-1)]],
                    brush=pg.mkBrush(239, 68, 68, 40),
                    movable=False
                )
                region.setZValue(-10)
                self.resp_plot.addItem(region)
                self._apnea_regions.append(region)

        # Annotations — Respiration Plot
        rx_min, rx_max = -self._resp_window, 0
        vr = self.resp_plot.getPlotItem().getViewBox().viewRange()
        ry_min, ry_max = vr[1]

        depth = resp_dict.get('depth', 'unknown')
        depth_colors = {'normal': '#22C55E', 'deep': '#38BDF8', 'shallow': '#F59E0B', 'apnea': '#EF4444'}
        self._resp_ann_depth.setText(f"Depth: {depth.capitalize()}")
        self._resp_ann_depth.setColor(QColor(depth_colors.get(depth, config.gui_theme.subtext)))
        self._resp_ann_depth.setPos(rx_min, ry_max)

        apnea_count = resp_dict.get('apnea_count', 0)
        apnea_text = f"Apnea: {apnea_count}"
        if resp_dict.get('apnea_active', False):
            apnea_text += f" (active {resp_dict.get('apnea_duration', 0):.1f}s)"
        self._resp_ann_apnea.setText(apnea_text)
        self._resp_ann_apnea.setPos(rx_max, ry_max)

        self._resp_ann_cycles.setText(f"Cycles: {resp_dict.get('cycle_count', 0)}")
        self._resp_ann_cycles.setPos(rx_min, ry_min)

        conf = resp_dict.get('confidence', 0)
        conf_color = '#22C55E' if conf > 60 else ('#F59E0B' if conf > 30 else '#EF4444')
        self._resp_ann_conf.setText(f"Confidence: {conf:.0f}%")
        self._resp_ann_conf.setColor(QColor(conf_color))
        self._resp_ann_conf.setPos(rx_max, ry_min)

        # ── RR plot ──
        rr_hist = resp_dict.get('rr_history', np.array([]))
        if len(rr_hist) > 0:
            x_rr = self.x_axis[-len(rr_hist):]
            self.curve_rr.setData(x_rr, rr_hist)
            self.curve_rr_zero.setData(x_rr, np.zeros(len(rr_hist)))

        rrx_min, rrx_max, rry_max = -self._resp_window, 0, 40
        rr_val = resp_dict.get('rr_current', 0)
        if rr_val > 0:
            rr_color = '#22C55E' if 6 <= rr_val <= 30 else '#F59E0B'
            self._rr_ann_current.setText(f"RR: {rr_val:.1f} bpm")
            self._rr_ann_current.setColor(QColor(rr_color))
        else:
            self._rr_ann_current.setText("RR: --")
            self._rr_ann_current.setColor(QColor(config.gui_theme.subtext))
        self._rr_ann_current.setPos(rrx_max, rry_max)

        brv = resp_dict.get('brv_value', 0)
        self._rr_ann_brv.setText(f"BRV: {brv:.3f}s" if brv > 0 else "BRV: --")
        self._rr_ann_brv.setPos(rrx_min, rry_max)

        last_cyc = resp_dict.get('last_cycle_duration', 0)
        self._rr_ann_cycle.setText(f"Last cycle: {last_cyc:.2f}s" if last_cyc > 0 else "Last cycle: --")
        self._rr_ann_cycle.setPos((rrx_min + rrx_max) / 2, rry_max)

        # ── IBI plot ──
        ibi_hist = resp_dict.get('ibi_history', np.array([]))
        if len(ibi_hist) > 0:
            x_ibi = self.x_axis[-len(ibi_hist):]
            self.curve_ibi.setData(x_ibi, ibi_hist)
            self.curve_ibi_zero.setData(x_ibi, np.zeros(len(ibi_hist)))
            
        ibi_val = resp_dict.get('ibi_current', 0)
        if ibi_val > 0:
            self._ibi_ann_current.setText(f"IBI: {ibi_val:.2f} s")
        else:
            self._ibi_ann_current.setText("IBI: --")
        self._ibi_ann_current.setPos(0, 12)

        # ── Derivative plot ──
        deriv = resp_dict.get('derivative_signal', np.array([]))
        if len(deriv) > 0:
            x_d = self.x_axis[-len(deriv):]
            self.curve_deriv.setData(x_d, deriv)

        # ── Info bar ──
        locked_bin = resp_dict.get('locked_bin', 0)
        bin_dist = (locked_bin or 0) * config.radar.range_resolution
        parts = [
            f"👤 {status}",
            f"🚶‍♂️ {motion}",
            f"🎯 Bin: {locked_bin} ({bin_dist:.2f} m)",
            f"📊 Confidence: {conf:.0f}%",
            f"💨 RR: {rr_val:.1f} bpm" if rr_val > 0 else "💨 RR: --",
            f"🔄 Cycles: {resp_dict.get('cycle_count', 0)}",
            f"🚫 Apnea: {apnea_count}",
            f"📏 Depth: {depth}",
            f"📐 Amplitude: {np.ptp(sig):.2f} mm",
        ]
        self.info_bar.setText("   |   ".join(parts))


# ──────────────────────────────────────────────────────────────────────────────
# Processing Thread (reads real radar, bypasses activity pipeline)
# ──────────────────────────────────────────────────────────────────────────────
from PyQt6.QtCore import QThread, pyqtSignal


class RespTestThread(QThread):
    """Reads from the radar queue, builds spectral history, runs respiration pipeline."""
    result_ready = pyqtSignal(dict)

    def __init__(self, pt_fft_q, get_manual_bin_func=None, parent=None):
        super().__init__(parent)
        self.pt_fft_q = pt_fft_q
        self.get_manual_bin_func = get_manual_bin_func
        self.running = True

        self.pipeline = LegacyRespiratoryPipeline(fps=config.radar.frame_rate)

    def run(self):
        while self.running:
            try:
                fft_frame = self.pt_fft_q.get(timeout=0.1)
                frames_processed = 1

                manual_bin = self.get_manual_bin_func() if self.get_manual_bin_func else None

                # Process the first frame immediately so we don't overwrite and drop it!
                resp_dict = self.pipeline.process_frame(fft_frame, frames_processed=1)

                # Drain and process extra frames one by one without skipping
                while not self.pt_fft_q.empty():
                    try:
                        fft_frame = self.pt_fft_q.get_nowait()
                        resp_dict = self.pipeline.process_frame(fft_frame, frames_processed=1)
                        frames_processed += 1
                    except queue.Empty:
                        break

                if resp_dict.get('is_calibrating', False):
                    self.result_ready.emit({'confidence': 0})
                    continue

                try:
                    self.result_ready.emit(resp_dict)
                except Exception as e:
                    logger.error("Resp pipeline error: %s", e, exc_info=True)

            except queue.Empty:
                continue

    def stop(self):
        self.running = False
        self.wait()


# ──────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────────────────────────────
def main():
    import multiprocessing

    logging.basicConfig(level=config.app.log_level)

    app = QApplication(sys.argv)

    # Shared queue for radar FFT data
    pt_fft_q = multiprocessing.Queue(maxsize=100)

    # Start radar controller (reads real radar data)
    print("Starting Respiration Test App...")
    print("Connecting to radar...\n")

    radar = RadarController(
        multiprocessing.Queue(),  # state_q (unused)
        pt_fft_q=pt_fft_q,
        start_radar_flag=multiprocessing.Event()
    )
    radar.daemon = True
    radar.start()

    # Create GUI
    window = RespTestWindow()

    # Start processing thread with manual bin callback
    proc_thread = RespTestThread(pt_fft_q, get_manual_bin_func=window.get_manual_bin)
    proc_thread.result_ready.connect(window.update_display)
    proc_thread.start()

    window.show()

    ret = app.exec()

    # Cleanup
    proc_thread.stop()
    radar.terminate()
    radar.join(timeout=2)
    print("\nTest app closed.")
    sys.exit(ret)


if __name__ == "__main__":
    main()
