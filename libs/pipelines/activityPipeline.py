import logging
import numpy as np
from collections import Counter, deque
from scipy import signal
from config import config

logger = logging.getLogger(__name__)

class ActivityPipeline:
    """
    This class is responsible for processing the radar data and detecting occupancy in specified zones of a given layout
    """
    def __init__(self, num_range_bins, range_resolution):
        self.num_range_bins = num_range_bins
        self.range_res = range_resolution
        self.clutter_map = np.zeros((num_range_bins, config.radar.antennas), dtype=complex)
        self.alpha = config.pipeline.alpha 
        self.detection_threshold = config.pipeline.detection_threshold
        self.frame_count = 0

        # Dual-Metric State Machine
        self.last_target_bin = None   # Remember where they were sitting
        self.last_target_coords = (0.0, 0.0, 0.0)
        self.occupied_reflection = None
        self.current_active_zone = "No Occupant Detected"
        self.is_occupied = False

        # Coordinate Tracking (Smooths the red dot)
        self.track_x = None
        self.track_y = None
        self.track_z = None

        # Zone Debouncing (Stops the text from flickering)
        self.current_stable_zone = "No Occupant Detected"
        self.zone_history = deque(maxlen=config.pipeline.frame_to_confirm_zone)
        self.frames_to_confirm_zone = config.pipeline.frame_to_confirm_zone

        # Zone occupancy duration tracker
        self.zone_timer_zone = None
        self.zone_timer_start = None
        self.zone_timer_last_seen = None
        self.zone_timer_hold_sec = 3.0   # tolerate short detection dropouts
                
        # Radar Location (Boot default fallback cascade)
        d_zone = getattr(config.app, "default_radar_pose", "Room")
        d_pose = config.layout.get(d_zone, {}).get("radar_pose", None)
        room_pose = config.layout.get("Room", {}).get("radar_pose", {})
        
        self.radar_x = d_pose.get("x") if d_pose else room_pose.get("x", 1.22)
        self.radar_y = d_pose.get("y") if d_pose else room_pose.get("y", 3.27)
        self.radar_z = d_pose.get("z") if d_pose else room_pose.get("z", 1.03)
        self.yaw_deg = d_pose.get("yaw_deg", 180) if d_pose else room_pose.get("yaw_deg", 180)
        self.pitch_deg = d_pose.get("pitch_deg", 0) if d_pose else room_pose.get("pitch_deg", 0)

        self.T, self.R = self._build_rotation(self.radar_x, self.radar_y, self.radar_z, self.yaw_deg, self.pitch_deg)

        # Tracking & Persistence Variables
        self.coord_buffer = deque(maxlen=config.pipeline.buffer_size)
        self.buffer_size = config.pipeline.buffer_size        
        self.track_confidence = 0   
        self.macro_timers = {}
        self.confidence_threshold = 3 
        self.miss_allowance = config.pipeline.miss_allowance       
        self.miss_counter = 0

        # Feature Flags
        self.features = config.pipeline.features

        # ==========================================
        # Fall Detection Parameters
        # ==========================================
        self.z_history = deque(maxlen=50)
        self.z_history_size = 50           # Store last ~2 second of Z data 
        self.fall_threshold_z = config.posture.fall_threshold
        self.fall_velocity_threshold = config.posture.fall_velocity_threshold
        self.fall_persistence_frames = 0
        self.is_fallen = False

        # Warmup Parameters
        self.warmup_frames = int(config.radar.frame_rate * config.tuning.warmup_seconds)
        
        # Per-antenna spectral history for beamformed aliveness scoring — ring buffered
        # Shape: (num_range_bins, antennas, frames)
        self.spectral_frames = config.respiration.resp_window_sec * config.radar.frame_rate
        self.spectral_history = np.zeros(
            (self.num_range_bins, config.radar.antennas, self.spectral_frames), dtype=complex
        )

        # Ring buffer write index
        self._ring_idx_spectral = 0

        # Aliveness Check: Layer 2 — Position Stability
        self._pos_history = deque(maxlen=config.tuning.position_stability_window)

        # Last-valid-frame bin/mag cache — kept coherent with the persisted track
        # so that tolerated-miss frames never mix position telemetry from the old
        # track with bin/mag from an unrelated fallback peak.
        self._last_valid_bin       = None
        self._last_valid_dynamic_mag = 0.0

        # ── Display Stabilisation ────────────────────────────────────────────
        # Majority-vote buffer for the *displayed* final_bin (tracking uses raw).
        _bvw = int(getattr(config.tuning, 'bin_vote_window', 7))
        self._bin_history  = deque(maxlen=_bvw)
        self._stable_bin   = None

        # Separate, longer debounce deque for bed sub-zones only.
        _szd = int(getattr(config.tuning, 'subzone_debounce_frames', 15))
        self._subzone_history       = deque(maxlen=_szd)
        self._stable_subzone_label  = ""   # e.g. "Center", "Head Edge"

        # Current hysteresis posture — prevents flicker near thresholds.
        self._stable_posture = "Lying Down"

        # Walking detection: rolling buffer of the smoothed track XY positions.
        # Walk is declared when net displacement over the window exceeds the threshold.
        self._xy_track_hist = deque(maxlen=config.motion.walk_window_frames)

        # Monitored-zone continuity hysteresis:
        # counts consecutive frames where reflection dropped below the floor threshold.
        # A hard reset is only issued after reflection_dip_tolerance consecutive dips.
        self._reflection_dip_frames = 0

        # Fall-logic zone-transition cooldown:
        # suppresses fall scoring for N frames after the track was last seen in a
        # non-floor zone to avoid nuisance triggers during bed exit / transit.
        self._zone_transition_cooldown = 0

        self.output_dict = {}
        self.apnea_frames = 0
        self.entry_frames = 0
        self.frames_to_occupy = int(config.radar.frame_rate * config.tuning.entry_hold_seconds)
        self.empty_room()

    @staticmethod
    def _build_rotation(x, y, z, yaw_deg, pitch_deg):
        """Build translation vector and rotation matrix from radar pose."""
        T = np.array([x, y, z])
        pitch_rad = np.radians(pitch_deg)
        yaw_rad = np.radians(yaw_deg)
        
        R_pitch = np.array([
            [1, 0, 0],
            [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
            [0, np.sin(pitch_rad), np.cos(pitch_rad)]
        ])
        
        R_yaw = np.array([
            [np.cos(yaw_rad), np.sin(yaw_rad), 0],
            [-np.sin(yaw_rad), np.cos(yaw_rad), 0],
            [0, 0, 1]
        ])
        
        R = np.dot(R_yaw, R_pitch)
        return T, R



    def _compute_aliveness(self, cand_phase, cand_bin):
        """
        Multi-metric aliveness score (Layer 3).
        Returns (aliveness_score: float [0-1], micro_state: str).
        """
        detrended = signal.detrend(cand_phase)
        window = np.hanning(len(detrended))
        windowed = detrended * window
        fft_result = np.fft.rfft(windowed)
        fft_mag = np.abs(fft_result)
        freqs = np.fft.rfftfreq(len(detrended), d=(1.0 / config.radar.frame_rate))

        vital_mask = (freqs >= 0.15) & (freqs <= 0.7)
        eval_mask  = (freqs >= 0.15) & (freqs <= 3.0)

        vital_fft = fft_mag[vital_mask]
        eval_fft  = fft_mag[eval_mask]

        if len(vital_fft) == 0 or len(eval_fft) == 0 or np.sum(eval_fft) < 1e-9:
            return 0.0, "DEAD_SPACE"

        # --- Metric 1: Peak Prominence ---
        peak_power    = np.max(vital_fft)
        median_power  = np.median(eval_fft) + 1e-9
        raw_prominence = peak_power / median_power
        prominence_score = np.clip((raw_prominence - 1.5) / (5.0 - 1.5), 0.0, 1.0)

        # --- Metric 2: Autocorrelation Quality ---
        norm_signal = detrended - np.mean(detrended)
        autocorr = np.correlate(norm_signal, norm_signal, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]
        if autocorr[0] > 1e-9:
            autocorr = autocorr / autocorr[0]
        else:
            autocorr = np.zeros_like(autocorr)
        min_lag = int(1.4 * config.radar.frame_rate)
        max_lag = min(int(6.7 * config.radar.frame_rate), len(autocorr) - 1)
        autocorr_peak = np.max(autocorr[min_lag:max_lag]) if min_lag < max_lag else 0.0
        autocorr_score = np.clip((autocorr_peak - 0.05) / (0.3 - 0.05), 0.0, 1.0)

        # --- Metric 3: Displacement Amplitude ---
        # λ/(4π) for 60 GHz
        phase_ptp = np.ptp(cand_phase)
        displacement_mm = (phase_ptp * 5.0) / (4.0 * np.pi)
        if config.tuning.breathing_displacement_min <= displacement_mm <= config.tuning.breathing_displacement_max:
            amplitude_score = 1.0
        elif displacement_mm < config.tuning.breathing_displacement_min:
            amplitude_score = np.clip(displacement_mm / config.tuning.breathing_displacement_min, 0.0, 1.0)
        else:
            amplitude_score = np.clip(1.0 - (displacement_mm - config.tuning.breathing_displacement_max) / 20.0, 0.0, 1.0)

        # --- Metric 4: Spectral Entropy ---
        p = eval_fft / (np.sum(eval_fft) + 1e-9)
        p = p[p > 1e-12]
        entropy = -np.sum(p * np.log(p))
        max_entropy = np.log(len(eval_fft))
        normalized_entropy = entropy / (max_entropy + 1e-9)
        entropy_score = np.clip(1.0 - normalized_entropy, 0.0, 1.0)

        aliveness = (0.30 * prominence_score) + (0.30 * autocorr_score) + \
                    (0.20 * amplitude_score)  + (0.20 * entropy_score)

        logger.debug(
            "Aliveness bin=%d: prom=%.2f autocorr=%.2f ampl=%.2f(%.1fmm) entropy=%.2f → %.2f",
            cand_bin, prominence_score, autocorr_score, amplitude_score,
            displacement_mm, entropy_score, aliveness
        )

        if aliveness >= config.tuning.aliveness_threshold:
            micro_state = "ALIVE"
        elif aliveness >= config.tuning.aliveness_threshold * 0.5:
            micro_state = "WEAK_VITAL"
        else:
            micro_state = "DEAD_SPACE"

        return aliveness, micro_state

    def _score_candidates(self, candidates, max_mag, use_tethering):
        best, best_s = None, -float('inf')
        
        track_in_monitor = False
        if self.track_x is not None and self.current_active_zone is not None:
            is_bed_zone = "Bed" in self.current_active_zone
            is_lying_down = getattr(self, 'track_z', 0.0) < config.posture.sitting_threshold
            is_actively_moving = getattr(self, 'motion_level', 0.0) > 0.25
            if is_bed_zone and is_lying_down and not is_actively_moving:
                track_in_monitor = True

        for c in candidates:
            mag_ratio = c['mag'] / max_mag
            
            if track_in_monitor:
                s = (mag_ratio * 0.1) + (c['vital_mult'] * 2.5) 
            else:
                s = (mag_ratio * 0.3) + (c['vital_mult'] * 1.5)
            
            if c['zone'] not in ('Floor / Transit', 'Out of Bounds (Ghost)'):
                s += 0.15
                
            if use_tethering and self.is_occupied and self.last_target_bin is not None:
                bin_distance = abs(c['bin'] - self.last_target_bin)
                if bin_distance == 0:
                    s += 0.50
                elif bin_distance == 1:
                    s += 0.25

            if use_tethering and self.track_x is not None:
                xy_dist = np.sqrt((c['x'] - self.track_x)**2 + (c['y'] - self.track_y)**2)
                
                if track_in_monitor:
                    if xy_dist > 0.4:
                        s -= 2.0
                    elif xy_dist > 0.15:
                        s -= min(0.8, (xy_dist - 0.15) * 1.5)
                else:
                    if xy_dist > 0.15: 
                        s -= min(0.4, (xy_dist - 0.15) * 0.4)
                    
                z_jump = abs(c['z'] - self.track_z)
                s -= min(0.3, z_jump * 0.5)
                
            if s > best_s:
                best_s = s
                best = c
                
        return best, best_s

    def update_radar_pose(self, x, y, z, yaw_deg, pitch_deg):
        self.radar_x, self.radar_y, self.radar_z = x, y, z
        self.yaw_deg, self.pitch_deg = yaw_deg, pitch_deg
        self.T, self.R = self._build_rotation(x, y, z, yaw_deg, pitch_deg)
        self.clutter_map.fill(0)
        self.spectral_history.fill(0)
        self._ring_idx_spectral = 0
        self.frame_count = 0
        self._reset_track()  # single source of truth for all transient state

    def empty_room(self):
        self.motion_level = 0.0
        self.current_micro_state = "STABLE"
        self._pos_history.clear()
        self.macro_timers.clear()
        self.output_dict = {
            "X": None,
            "Y": None,
            "Z": None,
            "Range": 0.0,
            "Azimuth": None,
            "Elevation": None,
            "zone": "No Occupant Detected",
            "status": "",
            "occ_confidence": 0,
            "posture_confidence": 0,
            "posture": "",
            "motion_str": "",
            "duration_str": "",
            "fall_confidence": 0,
            "micro_state": "STABLE",
            "is_valid": False,
        }

    def _update_zone_timer(self, zone_name, valid_detection, now):
        """
        Tracks continuous time spent in the current zone.
        Bed sub-zones ("Bed - Center", "Bed - Edge" …) are normalised to their
        base name so a sub-zone transition does NOT reset the duration counter.
        """
        # Normalise: treat all "Bed - *" sub-zones as the same timer bucket.
        base_zone = zone_name.split(" - ")[0] if " - " in zone_name else zone_name
        tracked_zone = base_zone if valid_detection else None

        if tracked_zone is not None:
            if self.zone_timer_zone != tracked_zone:
                self.zone_timer_zone  = tracked_zone
                self.zone_timer_start = now
            self.zone_timer_last_seen = now
        else:
            if self.zone_timer_zone is not None and self.zone_timer_last_seen is not None:
                if now - self.zone_timer_last_seen > self.zone_timer_hold_sec:
                    self.zone_timer_zone      = None
                    self.zone_timer_start     = None
                    self.zone_timer_last_seen = None

    def _format_duration(self, seconds):
        seconds = int(max(0, seconds))
        if seconds < 60:
            return f"{seconds}s"
        minutes = seconds // 60
        sec = seconds % 60
        if minutes < 60:
            return f"{minutes}m {sec:02d}s"
        hours = minutes // 60
        minutes = minutes % 60
        return f"{hours}h {minutes:02d}m"

    def evaluate_spatial_zone(self, x, y, z):
        """
        Evaluates a 3D point against the defined geofences efficiently.
        Returns: (Zone String, is_valid_target)
        """
        # 1. Global Boundary Check (Fast Fail)
        room = config.layout.get("Room")
        if room:
            if not (room["x"][0] <= x <= room["x"][1] and 
                    room["y"][0] <= y <= room["y"][1] and 
                    room["z"][0] <= z <= room["z"][1]):
                return "Out of Bounds (Ghost)", False
                
        # 2. Strict Interference Check — 'ignore' zones override everything
        for name, bounds in config.layout.items():
            if bounds.get("type") == "ignore":
                if (bounds["x"][0] <= x <= bounds["x"][1] and
                    bounds["y"][0] <= y <= bounds["y"][1] and
                    bounds["z"][0] <= z <= bounds["z"][1]):
                    return f"Ignored ({name})", False

        # 3. Target Zone Check (Monitored zones)
        for name, bounds in config.layout.items():
            zone_type = bounds.get("type")
            
            if zone_type == "monitor":
                if (bounds["x"][0] <= x <= bounds["x"][1] and
                    bounds["y"][0] <= y <= bounds["y"][1] and
                    bounds["z"][0] <= z <= bounds["z"][1]):
                    
                    if name == "Bed":
                        x_min, x_max = bounds["x"]
                        y_min, y_max = bounds["y"]
                        m_x = bounds.get("margin_x", [0.2, 0.2]) 
                        m_y = bounds.get("margin_y", [0.2, 0.2])
                        
                        is_center_x = (x_min + m_x[0]) <= x <= (x_max - m_x[1])
                        is_center_y = (y_min + m_y[0]) <= y <= (y_max - m_y[1])
                        
                        if is_center_x and is_center_y: return f"{name} - Center", True
                        elif x < (x_min + m_x[0]): return f"{name} - Right Edge", True
                        elif x > (x_max - m_x[1]): return f"{name} - Left Edge", True
                        elif y < (y_min + m_y[0]): return f"{name} - Foot Edge", True 
                        elif y > (y_max - m_y[1]): return f"{name} - Head Edge", True 
                        else: return f"{name} - Corner", True
                        
                    return name, True
                    
        # 4. Fallback: Inside the room, but not in a predefined zone
        return "Floor / Transit", True

    # ═══════════════════════════════════════════════════════════════════════
    # 5-STEP PIPELINE
    # ═══════════════════════════════════════════════════════════════════════

    def _step1_hardware_and_detection(self, fft_1d_data):
        """
        Step 1 — Hardware Correction & Clutter Suppression.
        Applies phase correction, updates the spatially-protected clutter map,
        writes per-antenna complex history, and gates warmup frames.
        """
        self.frame_count += 1
        corrected_data = np.copy(fft_1d_data)
        corrected_data[:, [0, 2, 4, 6]] *= -1

        raw_mag_profile = np.sum(np.abs(corrected_data), axis=1)

        # --- Clutter suppression (optional, configurable via features.clutter_removal) ---
        # Warmup gating and spectral_history writes always run regardless of this flag
        # so that vital_analysis and downstream consumers are never silently starved.
        if not getattr(self.features, 'clutter_removal', True):
            dynamic_data        = corrected_data
            dynamic_mag_profile = raw_mag_profile
        else:
            if self.frame_count <= self.warmup_frames:
                current_alpha_array = np.full(self.num_range_bins, 0.3)
            elif not self.is_occupied or self.last_target_bin is None or \
                    self.track_confidence < self.confidence_threshold:
                # Evaporate ghosts globally while empty or unconfirmed
                current_alpha_array = np.full(self.num_range_bins, self.alpha)
            else:
                # Spatially Masked Learning: freeze clutter map near the confirmed target
                dist_bins = np.abs(np.arange(self.num_range_bins) - self.last_target_bin)
                protection_mask = np.clip((dist_bins - 2.0) / 8.0, 0.0, 1.0)
                current_alpha_array = 0.001 + protection_mask * (self.alpha - 0.001)

            alpha_matrix = current_alpha_array[:, np.newaxis]
            self.clutter_map    = (alpha_matrix * corrected_data) + ((1.0 - alpha_matrix) * self.clutter_map)
            dynamic_data        = corrected_data - self.clutter_map
            dynamic_mag_profile = np.sum(np.abs(dynamic_data), axis=1)

        # --- Warmup gate (always runs, independent of clutter_removal) ---
        if self.frame_count <= self.warmup_frames:
            remaining = self.warmup_frames - self.frame_count
            self.output_dict["status"] = f"Calibrating ({remaining} frames)..."
            self.output_dict["zone"]   = "Calibrating"
            self.output_dict["Range"]  = 0.0
            self.output_dict["X"], self.output_dict["Y"], self.output_dict["Z"] = 0, 0, 0
            print(f"Pipeline warming up: {remaining} frames remaining")
            return {"abort": True}

        # Reset track state once immediately after warmup ends
        if self.frame_count == self.warmup_frames + 1:
            self.track_x, self.track_y, self.track_z = None, None, None
            self.coord_buffer.clear()
            self.track_confidence = 0

        # Ring-buffered per-antenna history write (O(1), no allocation)
        # Always runs so that vital_analysis has valid history even when clutter_removal is off.
        self.spectral_history[:, :, self._ring_idx_spectral % self.spectral_frames] = dynamic_data
        self._ring_idx_spectral += 1

        return {
            "corrected_data": corrected_data,
            "dynamic_data": dynamic_data,
            "dynamic_mag_profile": dynamic_mag_profile,
            "raw_mag_profile": raw_mag_profile
        }

    def _compute_cfar_threshold(self, profile, bin_idx, window=6, guard=2, scale=1.7):
        """
        Median CA-CFAR: uses median instead of mean for the local noise floor,
        making the threshold robust against sidelobe contamination from nearby
        strong movers. `scale` is the detection multiplier above the noise floor;
        `window` is the number of reference cells on each side; `guard` cells
        adjacent to the CUT are excluded from the noise estimate.
        """
        start = max(0, bin_idx - window)
        end   = min(len(profile), bin_idx + window + 1)
        sub_prof = profile[start:end]

        g_start = max(0, bin_idx - start - guard)
        g_end   = min(len(sub_prof), bin_idx - start + guard + 1)

        noise_cells = np.concatenate([sub_prof[:g_start], sub_prof[g_end:]])
        if len(noise_cells) == 0:
            return profile[bin_idx] * 0.5
        # Median is more robust than mean against sidelobe pull-up
        cfar_thresh = float(np.median(noise_cells)) * scale
        logger.debug("CFAR bin=%d noise_median=%.1f thresh=%.1f", bin_idx, np.median(noise_cells), cfar_thresh)
        # print(f"CFAR bin={bin_idx} noise_median={np.median(noise_cells)} thresh={cfar_thresh}")
        return cfar_thresh

    def _step2_candidate_generation(self, dynamic_data, dynamic_mag_profile):
        """
        Step 2 — Candidate Generation.
        Uses scipy find_peaks (NMS) instead of argpartition to avoid picking
        adjacent bins from the same scatterer. Applies median CA-CFAR + fixed-floor
        dual gate. Extracts phase via a phase-aligned coherent sum across antennas
        (weights steered by the observed snapshot, not a geometry-driven delay-
        and-sum beamformer) for a higher-SNR aliveness score.
        """
        min_search_bin = int(config.tuning.min_search_range / self.range_res)
        use_cfar = getattr(config.pipeline, 'use_cfar', True)

        # --- Non-Maximum Suppression via find_peaks ---
        peaks, _ = signal.find_peaks(dynamic_mag_profile[min_search_bin:], distance=3)
        peaks += min_search_bin

        if len(peaks) == 0:
            if self.is_occupied:
                print(f"Rejection: No peaks found above detection_threshold ({self.detection_threshold})")
            return {
                "is_jump": False, "is_valid_point": False,
                "final_peak_bin": None, "dynamic_peak_bin": min_search_bin,
                "raw_x": 0.0, "raw_y": 0.0, "raw_z": 0.0
            }

        sorted_peaks = peaks[np.argsort(dynamic_mag_profile[peaks])][::-1]
        sorted_peaks = sorted_peaks[:min(config.tuning.num_candidates, len(sorted_peaks))]

        valid_candidates = []
        final_peak_bin   = None
        raw_x = raw_y = raw_z = 0.0
        is_jump        = False
        is_valid_point = False

        for cand_bin in sorted_peaks:
            cand_range = cand_bin * self.range_res
            ch_cand    = dynamic_data[cand_bin, :]

            S_cand = np.array([
                [ch_cand[3], ch_cand[1]],
                [ch_cand[2], ch_cand[0]],
                [ch_cand[7], ch_cand[5]],
                [ch_cand[6], ch_cand[4]]
            ])
            om_az = np.angle(np.sum(S_cand[:, 0]    * np.conj(S_cand[:, 1])))
            om_el = np.angle(np.sum(S_cand[0:3, :] * np.conj(S_cand[1:4, :])))

            az_cand = np.arcsin(np.clip(om_az / np.pi, -1.0, 1.0))
            el_cand = np.arcsin(np.clip(om_el / np.pi, -1.0, 1.0))

            Pr_c = np.array([
                cand_range * np.sin(az_cand) * np.cos(el_cand),
                cand_range * np.cos(az_cand) * np.cos(el_cand),
                cand_range * np.sin(el_cand)
            ])
            Pb_c = np.dot(self.R, Pr_c) + self.T

            # --- CFAR / fixed-floor dual gate ---
            dynamic_threshold = self.detection_threshold
            if use_cfar:
                cfar_thresh = self._compute_cfar_threshold(dynamic_mag_profile, cand_bin)
                dynamic_threshold = max(self.detection_threshold, cfar_thresh)

            if dynamic_mag_profile[cand_bin] < dynamic_threshold:
                continue

            zone_name, is_valid = self.evaluate_spatial_zone(Pb_c[0], Pb_c[1], Pb_c[2])

            if not is_valid or zone_name == "Out of Bounds (Ghost)" or zone_name.startswith("Ignored"):
                continue

            # --- Beamformed Aliveness (Target-Specific Delay-and-Sum) ---
            cand_micro_state = "STABLE"
            vital_multiplier = 0.1

            if not getattr(self.features, 'vital_analysis', True):
                vital_multiplier = 1.0
            elif (self.frame_count - self.warmup_frames) > self.spectral_frames:
                # Weights steered toward this candidate's angle
                weights  = np.conj(ch_cand) / (np.abs(ch_cand) + 1e-9)

                # Reorder ring buffer for this bin only: shape (A, F)
                hist_raw = self.spectral_history[cand_bin, :, :]
                idx = self._ring_idx_spectral % self.spectral_frames
                if idx != 0:
                    hist_raw = np.concatenate([hist_raw[:, idx:], hist_raw[:, :idx]], axis=1)

                # Beamformed complex time series
                hist_bf    = np.sum(hist_raw * weights[:, np.newaxis], axis=0)  # (F,)
                hist_bf    = np.where(hist_bf == 0, 1e-10 + 1e-10j, hist_bf)
                cand_phase = np.unwrap(np.angle(hist_bf))

                phase_ptp       = np.ptp(cand_phase)
                displacement_mm = (phase_ptp * 5.0) / (4.0 * np.pi)   # λ/(4π) for 60 GHz
                phase_var       = np.var(np.diff(cand_phase))
                phase_drift     = abs(cand_phase[-1] - cand_phase[0])

                if phase_var < config.tuning.ghost_phase_threshold:
                    vital_multiplier, cand_micro_state = 0.01, "STATIC_GHOST"
                    self.macro_timers[cand_bin] = 0
                elif phase_drift > (20 * np.pi):
                    vital_multiplier, cand_micro_state = 0.01, "MECHANICAL_ROTOR"
                    self.macro_timers[cand_bin] = 0
                elif displacement_mm > config.tuning.macro_displacement_mm:
                    cand_micro_state = "MACRO_PHASE"
                    self.macro_timers[cand_bin] = self.macro_timers.get(cand_bin, 0) + 1
                    vital_multiplier = 0.1 if self.macro_timers[cand_bin] > 5 * config.radar.frame_rate else 0.9
                elif phase_var > 0.3 and dynamic_mag_profile[cand_bin] > self.detection_threshold * 2.0:
                    vital_multiplier, cand_micro_state = 0.7, "MICRO_PHASE"
                    self.macro_timers[cand_bin] = 0
                else:
                    self.macro_timers[cand_bin] = 0
                    aliveness, cand_micro_state = self._compute_aliveness(cand_phase, cand_bin)
                    if aliveness >= config.tuning.aliveness_threshold:
                        vital_multiplier = 1.0
                    elif aliveness >= config.tuning.aliveness_threshold * 0.5:
                        vital_multiplier = 0.5
                    else:
                        vital_multiplier = 0.05

                # Position stability penalty (Layer 2)
                if len(self._pos_history) >= config.tuning.position_stability_window // 2:
                    pos_arr = np.array(list(self._pos_history))
                    pos_var = np.var(pos_arr[:, 0]) + np.var(pos_arr[:, 1])
                    if pos_var > config.tuning.max_clutter_position_var:
                        penalty = np.clip(pos_var / config.tuning.max_clutter_position_var, 1.0, 3.0)
                        vital_multiplier *= (1.0 / penalty)
                        logger.debug("Position stability penalty: var=%.4f, penalty=%.2f", pos_var, penalty)

            valid_candidates.append({
                'bin': cand_bin,
                'x': Pb_c[0], 'y': Pb_c[1], 'z': Pb_c[2],
                'azimuth': az_cand, 'elevation': el_cand,
                'mag': dynamic_mag_profile[cand_bin],
                'zone': zone_name,
                'vital_mult': vital_multiplier,
                'micro_state': cand_micro_state
            })
        if valid_candidates:
            is_valid_point = True
            max_mag = max(c['mag'] for c in valid_candidates)
            best_cand, _ = self._score_candidates(
                valid_candidates, max_mag,
                use_tethering=getattr(self.features, 'tethering', True)
            )

            final_peak_bin = best_cand['bin']
            raw_x, raw_y, raw_z = best_cand['x'], best_cand['y'], best_cand['z']
            raw_z = np.clip(raw_z, config.tuning.z_clip_min, config.tuning.z_clip_max)

            # Jump check BEFORE committing any state — a rejected candidate must not
            # contaminate current_active_zone, current_micro_state, or _pos_history.
            if self.track_x is not None and getattr(self.features, 'tethering', True):
                jump_dist = np.sqrt((raw_x - self.track_x)**2 + (raw_y - self.track_y)**2)
                if jump_dist > config.tuning.jump_reject_distance:
                    is_valid_point = False
                    final_peak_bin = None
                    is_jump = True
                    print(f"Rejection: Jump reject! dist={jump_dist:.2f}m > {config.tuning.jump_reject_distance}m")

            # Only mutate shared state when the candidate is not rejected
            if is_valid_point:
                self.current_active_zone = best_cand['zone']
                self.current_micro_state = best_cand.get('micro_state', 'STABLE')
                self._pos_history.append((raw_x, raw_y))

        dynamic_peak_bin = final_peak_bin if final_peak_bin is not None else sorted_peaks[0]

        # Only export geometry when the candidate passed the jump check.
        # On rejected frames we leave the previous accepted values in output_dict
        # rather than writing mismatched telemetry (angles from best_cand, Range from
        # a fallback peak that may differ).
        if is_valid_point and valid_candidates:
            self.output_dict["Azimuth"]   = best_cand['azimuth']
            self.output_dict["Elevation"] = best_cand['elevation']
            self.output_dict["Range"]     = final_peak_bin * self.range_res
            logger.debug(
                "peak_mag=%.1f threshold=%.1f az=%.1f° el=%.1f°",
                dynamic_mag_profile[final_peak_bin], self.detection_threshold,
                np.degrees(best_cand['azimuth']),
                np.degrees(best_cand['elevation'])
            )

        return {
            "is_jump": is_jump, "is_valid_point": is_valid_point,
            "dynamic_peak_bin": dynamic_peak_bin,
            "raw_x": raw_x, "raw_y": raw_y, "raw_z": raw_z
        }

    def _step3_tracking(self, is_valid_point, raw_x, raw_y, raw_z):
        """
        Step 3 — Tracking.
        Merges temporal persistence (confidence gate + miss counter) with
        adaptive EMA smoothing into one cohesive stage. Returns smoothed
        track position and vertical velocity.
        """
        if not getattr(self.features, 'temporal_persistence', True):
            if is_valid_point:
                self.track_confidence = self.confidence_threshold
                self.coord_buffer = deque([(raw_x, raw_y, raw_z)], maxlen=self.buffer_size)
            else:
                self.track_confidence = 0
                print("Rejection: Track lost (Persistence disabled)")
                return {"abort": True, "kill_track": True}
        else:
            if is_valid_point:
                self.track_confidence = min(self.track_confidence + 1, self.confidence_threshold)
                self.miss_counter = 0
                self.coord_buffer.append((raw_x, raw_y, raw_z))
            else:
                self.miss_counter += 1
                if self.miss_counter > self.miss_allowance:
                    self.track_confidence = 0
                    print(f"Rejection: Missed too many frames ({self.miss_counter}/{self.miss_allowance}), resetting track")
                    return {"abort": True, "kill_track": True}

        if self.track_confidence < self.confidence_threshold:
            if is_valid_point:
                print(f"Rejection: Track not yet confirmed (Confidence {self.track_confidence}/{self.confidence_threshold})")
            return {"abort": True, "kill_track": False}

        recent_coords = np.array(list(self.coord_buffer))
        med_x = np.median(recent_coords[:, 0])
        med_y = np.median(recent_coords[:, 1])
        med_z = np.median(recent_coords[:, 2])

        if not getattr(self.features, 'adaptive_smoothing', True):
            self.track_x, self.track_y, self.track_z = med_x, med_y, med_z
            self.motion_level = 0.5
            return {"X_b": med_x, "Y_b": med_y, "Z_b": med_z, "v_z": 0.0}

        if self.track_x is None:
            self.track_x, self.track_y, self.track_z = med_x, med_y, med_z
            self.motion_level = 0.0
        else:
            shift_distance = np.sqrt(
                (med_x - self.track_x)**2 +
                (med_y - self.track_y)**2 +
                (med_z - self.track_z)**2
            )
            self.motion_level = (0.2 * shift_distance) + (0.8 * self.motion_level)

            track_zone, _ = self.evaluate_spatial_zone(self.track_x, self.track_y, self.track_z)
            if track_zone != self.current_active_zone:
                adaptive_alpha = 0.4
                recent = list(self.coord_buffer)[-3:] if len(self.coord_buffer) >= 3 else list(self.coord_buffer)
                self.coord_buffer.clear()
                self.coord_buffer.extend(recent)
            elif shift_distance > 0.50:
                adaptive_alpha = 0.5
                recent = list(self.coord_buffer)[-5:] if len(self.coord_buffer) >= 5 else list(self.coord_buffer)
                self.coord_buffer.clear()
                self.coord_buffer.extend(recent)
            elif shift_distance < 0.12:
                adaptive_alpha = 0.05    # TEST (original 0.02)
            else:
                adaptive_alpha = 0.1

            self.track_x = (adaptive_alpha * med_x) + ((1 - adaptive_alpha) * self.track_x)
            self.track_y = (adaptive_alpha * med_y) + ((1 - adaptive_alpha) * self.track_y)
            self.track_z = (adaptive_alpha * med_z) + ((1 - adaptive_alpha) * self.track_z)

        self.z_history.append(self.track_z)

        velocity_window = int(config.radar.frame_rate * 0.4)
        if len(self.z_history) >= velocity_window:
            z_list = list(self.z_history)
            dz  = z_list[-1] - z_list[-velocity_window]
            v_z = dz / (velocity_window / config.radar.frame_rate)
        else:
            v_z = 0.0

        # Feed walk-detection history with the latest smoothed position
        if self.track_x is not None:
            self._xy_track_hist.append((self.track_x, self.track_y))

        return {"X_b": self.track_x, "Y_b": self.track_y, "Z_b": self.track_z, "v_z": v_z}

    def _step4_activity_inference(self, X_b, Y_b, Z_b, v_z):
        """
        Step 4 — Activity Inference.
        Translates the smoothed 3-D position into a confirmed zone, posture
        label, and motion string. Ghost / ignored zones kill the track.
        """
        final_zone, _ = self.evaluate_spatial_zone(X_b, Y_b, Z_b)

        # Normalise to base zone before writing to zone_history so that
        # current_stable_zone is always coarse (e.g. "Bed", not "Bed - Center").
        # The sub-zone suffix is handled separately by _subzone_history below.
        base_zone_for_debounce = final_zone.split(" - ")[0] if " - " in final_zone else final_zone
        self.zone_history.append(base_zone_for_debounce)

        if len(self.zone_history) == self.frames_to_confirm_zone:
            most_common = Counter(self.zone_history).most_common(1)[0][0]
            self.current_stable_zone = most_common

        # Restore the full sub-zone label so _subzone_history debounce below can apply it.
        stable_base = getattr(self, 'current_stable_zone', base_zone_for_debounce)
        if " - " in final_zone and final_zone.startswith(stable_base):
            # coarse zone confirmed — preserve suffix for sub-zone debounce
            pass
        else:
            # coarse zone changed or no suffix — use stable base without suffix
            final_zone = stable_base

        # Ghost / ignored zone → kill track immediately
        if final_zone == "Out of Bounds (Ghost)" or final_zone.startswith("Ignored"):
            print(f"Rejection: Target in {final_zone}, resetting track")
            return {"abort": True, "kill_track": True}

        # ── Motion label ───────────────────────────────────────────────────────────
        # Walking check: net XY translation over the history window.
        # In Floor / Transit zones this is more reliable than Z_b height for posture.
        is_walking = False
        if (final_zone == "Floor / Transit" and
                len(self._xy_track_hist) == self._xy_track_hist.maxlen):
            oldest = self._xy_track_hist[0]
            dx = self.track_x - oldest[0]
            dy = self.track_y - oldest[1]
            walk_disp = np.sqrt(dx * dx + dy * dy)
            if walk_disp >= config.motion.walk_displacement_m:
                is_walking = True

        if is_walking:
            motion_str = "Walking"
            # print(f"Walking: {self.motion_level}")
        elif self.motion_level > config.motion.restless_max:
            motion_str = "Major Movement"
            # print(f"Major Movement: {self.motion_level}")
        elif getattr(self, 'current_micro_state', 'STABLE') == "MACRO_PHASE" and (
                "Bed" in self.current_active_zone or "Chair" in self.current_active_zone):
            motion_str = "Postural Shift"
            # print(f"Postural Shift: {self.motion_level}")
        elif self.motion_level > config.motion.rest_max:
            motion_str = "Restless/Shifting"
            # print(f"Restless/Shifting: {self.motion_level}")
        elif getattr(self, 'current_micro_state', 'STABLE') == "MICRO_PHASE":
            motion_str = "Restless/Fidgeting"
            # print(f"Restless/Fidgeting: {self.motion_level}")
        else:
            motion_str = "Resting/Breathing"
            # print(f"Resting/Breathing: {self.motion_level}")

        if not getattr(self.features, 'fall_posture', True):
            return {"final_zone": final_zone, "posture": "Unknown", "motion_str": motion_str}

        # ── Walking prior: if walking, person is guaranteed standing ────────────
        # This behavioural prior is more reliable than the Z_b height estimate
        # in Floor / Transit zones where floor bounce makes Z_b noisy.
        if is_walking:
            self._stable_posture = "Standing"   # seed hysteresis so stop-walking falls through correctly

        # ── Posture with hysteresis (prevents threshold-boundary flicker) ────
        # Margin from config; default 0.05 m.  The stable_posture state only
        # changes when Z_b crosses threshold ± margin, not the bare threshold.
        hm = float(getattr(config.tuning, 'posture_hysteresis_m', 0.05))
        stand_hi = config.posture.standing_threshold + hm
        stand_lo = config.posture.standing_threshold - hm
        sit_hi   = config.posture.sitting_threshold  + hm
        sit_lo   = config.posture.sitting_threshold  - hm

        # ── Transit Standing Bias ───────────────────────────────────────────────
        # In Floor / Transit, Z_b is noisier (floor bounce, no blanket, open space)
        # so we widen the exit-Standing margin and raise the enter-Sitting bar,
        # giving extra "stickiness" to Standing without requiring walking.
        if final_zone == "Floor / Transit":
            tb = float(getattr(config.tuning, 'transit_standing_bias_m', 0.08))
            stand_lo -= tb   # need to drop further before leaving Standing
            sit_hi   += tb   # need to go lower before entering Sitting
            sit_lo   += tb   # same for Lying Down entry from Sitting

        if self._stable_posture == "Standing":
            # Exit standing only when Z_b drops clearly below threshold - margin
            if Z_b < stand_lo:
                self._stable_posture = "Sitting" if Z_b >= sit_lo else "Lying Down"
        elif self._stable_posture == "Sitting":
            if Z_b > stand_hi:
                self._stable_posture = "Standing"
            elif Z_b < sit_lo:
                self._stable_posture = "Lying Down"
        else:  # Lying Down
            if Z_b > stand_hi:
                self._stable_posture = "Standing"
            elif Z_b > sit_hi:
                self._stable_posture = "Sitting"

        posture = self._stable_posture

        # ── Bed sub-zone debounce ─────────────────────────────────────────────
        # The coarse zone ("Bed") is already debounced by zone_history.
        # We apply a *separate* majority-vote window just for the sub-zone suffix
        # so "Center" / "Head Edge" labels are calmer than the raw geometry check.
        if " - " in final_zone:
            base_z, sub_z = final_zone.split(" - ", 1)
            self._subzone_history.append(sub_z)
            if len(self._subzone_history) == self._subzone_history.maxlen:
                stable_sub = Counter(self._subzone_history).most_common(1)[0][0]
                self._stable_subzone_label = stable_sub
            # Reconstruct zone with stable sub-label
            if self._stable_subzone_label:
                final_zone = f"{base_z} - {self._stable_subzone_label}"
        else:
            # Leaving the bed — clear sub-zone history so it starts fresh on re-entry
            self._subzone_history.clear()
            self._stable_subzone_label = ""

        return {"final_zone": final_zone, "posture": posture, "motion_str": motion_str}

    def _step5_alert_logic(self, is_valid_point, is_jump, dynamic_peak_bin,
                           dynamic_mag_profile, raw_mag_profile,
                           raw_x, raw_y, raw_z,
                           final_zone, posture, v_z, motion_str, Z_b):
        """
        Step 5 — Alert Logic.
        Contains the occupancy state machine (breathing → apnea transitions),
        dual confidence indices, and fall detection safeguards — all decoupled
        from detection and tracking.
        """
        status = self.output_dict.get("status", "")
        current_raw_reflection = 0.0

        # ── 1. Occupancy State Machine ──────────────────────────────────────
        is_active_target = False
        micro = getattr(self, 'current_micro_state', 'STABLE')

        if getattr(self.features, 'apnea_state', True):
            if micro not in ["DEAD_SPACE", "STATIC_GHOST"]: #, "MECHANICAL_ROTOR"]:
                if not self.is_occupied and micro == "WEAK_VITAL" and \
                        dynamic_mag_profile[dynamic_peak_bin] < self.detection_threshold * 2.0:
                    is_active_target = False
                elif is_valid_point and not is_jump and \
                        dynamic_mag_profile[dynamic_peak_bin] >= self.detection_threshold:
                    is_active_target = True
        else:
            if is_valid_point and not is_jump and \
                    dynamic_mag_profile[dynamic_peak_bin] >= self.detection_threshold:
                is_active_target = True

        # Entry debounce: decay accumulator on empty frames
        if not is_active_target and not self.is_occupied and self.entry_frames > 0:
            self.entry_frames = max(0, self.entry_frames - 1)

        if is_active_target:
            if not self.is_occupied:
                self.entry_frames += 1
                if self.entry_frames < self.frames_to_occupy:
                    print(f"Rejection: Occupancy not yet stable (EntryFrames={self.entry_frames}/{self.frames_to_occupy})")
                    return {"abort": True}
            self.is_occupied        = True
            self.apnea_frames       = 0
            self.last_target_bin    = dynamic_peak_bin
            self.last_target_coords = (raw_x, raw_y, raw_z)

            # Asymmetric EMA for occupied_reflection:
            # rise slowly (resist blanket/motion spikes) but fall faster so the
            # baseline tracks real posture-driven drops without staying erroneously high.
            current_peak = float(np.max(
                raw_mag_profile[max(0, dynamic_peak_bin - 1):
                                min(self.num_range_bins, dynamic_peak_bin + 2)]
            ))
            if self.occupied_reflection is None:
                self.occupied_reflection = current_peak
            elif current_peak >= self.occupied_reflection:
                alpha_ref = 0.005   # ~200 frames to fully converge upward
                self.occupied_reflection = (alpha_ref * current_peak) + \
                                           ((1.0 - alpha_ref) * self.occupied_reflection)
            else:
                alpha_ref = 0.03    # ~33 frames to follow a real drop
                self.occupied_reflection = (alpha_ref * current_peak) + \
                                           ((1.0 - alpha_ref) * self.occupied_reflection)
            self._reflection_dip_frames = 0  # reset dip counter on active detection
            status = "Occupied (Breathing/Moving)"

        elif self.is_occupied and self.last_target_bin is not None:
            curr_loc = (self.track_x, self.track_y, self.track_z) \
                       if self.track_x is not None else self.last_target_coords
            last_zone_name, _ = self.evaluate_spatial_zone(*curr_loc)
            base_zone = last_zone_name.split(" - ")[0]
            is_monitored_zone = config.layout.get(base_zone, {}).get("type") == "monitor"

            if is_monitored_zone:
                # Monitored zone: reflection continuity / apnea logic applies.
                # Only here is a static floor reflection meaningful as an aliveness gate.
                #
                # Wider search window (±3 bins) so a small range-bin shift from a
                # posture change does not artificially drop the reading to zero.
                search_half = 3
                current_raw_reflection = float(np.max(
                    raw_mag_profile[max(0, self.last_target_bin - search_half):
                                    min(self.num_range_bins, self.last_target_bin + search_half + 1)]
                ))
                continuity_ratio = getattr(config.tuning, 'continuity_ratio', 0.60)
                threshold = (self.occupied_reflection or 2000) * continuity_ratio
                dip_tolerance = int(getattr(config.tuning, 'reflection_dip_tolerance',
                                            config.radar.frame_rate))  # default 1 sec
                # print(f"Step 5 Reflection Check: curr={current_raw_reflection:.0f}, base={self.occupied_reflection or 0.0:.0f}, thresh={threshold:.0f}, micro={micro}, dip={self._reflection_dip_frames}")

                if current_raw_reflection > threshold:
                    # Reflection is healthy — reset dip counter and run normal apnea logic.
                    self._reflection_dip_frames = 0
                    if micro == "STATIC_GHOST":
                        print("Target identified as STATIC_GHOST, resetting track")
                        return {"abort": True, "kill_track": True}
                    self.apnea_frames += 1
                    if self.apnea_frames >= config.radar.frame_rate * 5:
                        status     = "Possible Apnea"
                        motion_str = "Static"
                    else:
                        status = "Still / Monitoring..."
                else:
                    # Reflection dropped below floor.
                    self._reflection_dip_frames += 1
                    if self._reflection_dip_frames < dip_tolerance:
                        # Soft-fail: degrade status but keep track alive during hysteresis window.
                        status = "Occupied / weak signal"
                        logger.debug(
                            "Reflection dip %d/%d: curr=%.0f thresh=%.0f — soft-fail, track preserved.",
                            self._reflection_dip_frames, dip_tolerance,
                            current_raw_reflection, threshold
                        )
                    else:
                        print(f"Target reflection lost: {current_raw_reflection:.0f} < {threshold:.0f} for {self._reflection_dip_frames} frames, resetting track")
                        return {"abort": True, "kill_track": True}

            else:
                # Non-monitored zone: reflection continuity is not applicable.
                # The person may be moving/standing in a transit or activity zone.
                # Step 3 miss-counter and step 4 zone-validity own the track lifetime here.
                # Reset dip counter so a stale monitored-zone streak cannot carry back
                # into the next bed episode.
                self.apnea_frames          = 0
                self._reflection_dip_frames = 0
                status = "Room Presence"
                logger.debug("Non-monitored zone (%s): track kept alive by persistence.", base_zone)

        else:
            # No existing track — nothing to persist.
            # Only print if we were trying to detect something
            if is_valid_point:
                print("Rejection: Frame aborted – no existing track to persist candidate.")
            return {"abort": True, "kill_track": True}

        # ── 2. Confidence Indices ────────────────────────────────────────────
        temporal_conf = (self.track_confidence / self.confidence_threshold) * 100.0
        signal_conf   = 100.0

        if "Breathing" in status or "Moving" in status:
            margin = (dynamic_mag_profile[dynamic_peak_bin] - self.detection_threshold) / self.detection_threshold
            signal_conf = min(100.0, 50.0 + margin * 100.0)
            if self.motion_level > 0.05:
                signal_conf = 100.0
        elif "Apnea" in status or "Monitoring" in status:
            occupied  = self.occupied_reflection or 2000
            continuity_ratio = getattr(config.tuning, 'continuity_ratio', 0.60)
            threshold = occupied * continuity_ratio
            margin    = (current_raw_reflection - threshold) / (occupied - threshold + 1e-6)
            signal_conf = min(100.0, 50.0 + max(0.0, margin) * 50.0)
        elif "weak signal" in status:
            # Hysteresis window: confidence degrades linearly with dip count.
            dip_tolerance = int(getattr(config.tuning, 'reflection_dip_tolerance',
                                        config.radar.frame_rate))
            signal_conf = max(10.0, 50.0 * (1.0 - self._reflection_dip_frames / dip_tolerance))

        occ_confidence = max(
            0.0,
            0.6 * temporal_conf + 0.4 * signal_conf
            - (self.miss_counter / max(1, self.miss_allowance)) * 100.0
        )

        # ── 3. Fall Safeguards ───────────────────────────────────────────────
        # NOTE: posture_confidence is computed AFTER the fall block so that
        # a posture of "Fallen" gets its correct confidence branch.
        fall_confidence = 0.0

        if getattr(self.features, 'fall_posture', True) and final_zone == "Floor / Transit":
            status = status.replace("Occupied", "In the Room")

            # Cooldown: suppress fall scoring for N frames after a zone transition
            # from a non-floor zone (e.g., bed exit) to avoid nuisance triggers.
            if self._zone_transition_cooldown > 0:
                self._zone_transition_cooldown -= 1
                logger.debug("Fall cooldown active: %d frames remaining.", self._zone_transition_cooldown)
            else:
                if v_z < self.fall_velocity_threshold and not self.is_fallen:
                    self.is_fallen = True
                    self.fall_persistence_frames = 0
                    logger.warning("Fall triggered by velocity: v_z=%.2f m/s", v_z)

                if Z_b <= self.fall_threshold_z or self.is_fallen:
                    self.is_fallen = True
                    self.fall_persistence_frames += 1

                    # Post-fall immobility override: cancel fall if subject is actively moving
                    if self.motion_level > getattr(config.motion, 'rest_max', 0.15) and \
                            self.fall_persistence_frames > self.z_history_size // 2:
                        self.is_fallen = False
                        self.fall_persistence_frames = 0
                        logger.warning("Fall aborted: persistent motion detected after descent.")
                    else:
                        h_score = max(0.0, (self.fall_threshold_z - Z_b) / self.fall_threshold_z) * 100.0
                        p_score = min(100.0, (self.fall_persistence_frames / self.z_history_size) * 100.0)
                        v_score = min(100.0, abs(v_z) / abs(self.fall_velocity_threshold) * 100.0) \
                                  if v_z < 0 else 0.0
                        fall_confidence = 0.3 * h_score + 0.4 * p_score + 0.3 * v_score

                        if fall_confidence > 60.0:
                            status  = "CRITICAL: Fall Detected!"
                            posture = "Fallen"
                            logger.warning(
                                "Fall verified. Conf=%.1f (H:%.1f P:%.1f V:%.1f)",
                                fall_confidence, h_score, p_score, v_score
                            )
                else:
                    self.fall_persistence_frames = max(0, self.fall_persistence_frames - 2)
        else:
            # Track is not in floor zone — arm cooldown for the next floor entry
            # so bed-exit transitions don't immediately trigger fall scoring.
            if not (self.is_fallen or self.fall_persistence_frames > 0):
                self._zone_transition_cooldown = int(getattr(config.tuning, 'fall_cooldown_frames', 15))
            self.is_fallen = False
            self.fall_persistence_frames = 0

        # ── Posture Confidence (computed after fall decision) ──────────────────
        # Walking is a strong behavioural prior for Standing: override confidence
        # directly instead of relying on the noisy Z_b distance formula.
        if motion_str == "Walking":
            posture_confidence = float(config.motion.walk_posture_conf)
        else:
            if posture == "Fallen":
                # confidence scales with how far below floor threshold the subject is
                dist_to_thresh = abs(self.fall_threshold_z - Z_b)
            elif posture == "Standing":
                dist_to_thresh = abs(Z_b - config.posture.standing_threshold)
            elif posture == "Sitting":
                dist_to_thresh = min(
                    abs(Z_b - config.posture.sitting_threshold),
                    abs(config.posture.standing_threshold - Z_b)
                )
            else:  # Lying Down
                dist_to_thresh = abs(config.posture.sitting_threshold - Z_b)

            base_posture_conf  = min(100.0, 50.0 + (dist_to_thresh / 0.125) * 50.0)
            posture_confidence = max(50.0, base_posture_conf - min(40.0, self.motion_level * 200.0))


        return {
            "status": status,
            "motion_str": motion_str, "posture": posture,
            "occ_confidence": occ_confidence,
            "posture_confidence": posture_confidence,
            "fall_confidence": fall_confidence
        }

    def process_frame(self, fft_1d_data):
        """5-step pipeline: Detection → Candidates → Tracking → Inference → Alerts."""

        # ── Step 1: Hardware Correction & Clutter Suppression ───────────────
        s1 = self._step1_hardware_and_detection(fft_1d_data)
        if s1.get("abort"):
            return self.output_dict

        # ── Step 2: Candidate Generation (CFAR + Beamformed Aliveness) ──────
        s2 = self._step2_candidate_generation(s1["dynamic_data"], s1["dynamic_mag_profile"])

        # ── Step 3: Tracking (Persistence + Adaptive Smoothing) ─────────────
        s3 = self._step3_tracking(s2["is_valid_point"], s2["raw_x"], s2["raw_y"], s2["raw_z"])
        if s3.get("abort"):
            if s3.get("kill_track"):
                self._reset_track()
                # self.clutter_map = s1["corrected_data"].copy() 
            return self.output_dict

        # ── Step 4: Activity Inference (Zone, Posture, Motion) ───────────────
        s4 = self._step4_activity_inference(s3["X_b"], s3["Y_b"], s3["Z_b"], s3["v_z"])
        if s4.get("abort"):
            if s4.get("kill_track"):
                self._reset_track()
            return self.output_dict

        # ── Step 5: Alert Logic (State Machine, Confidence, Falls) ───────────
        s5 = self._step5_alert_logic(
            s2["is_valid_point"], s2["is_jump"], s2["dynamic_peak_bin"],
            s1["dynamic_mag_profile"], s1["raw_mag_profile"],
            s2["raw_x"], s2["raw_y"], s2["raw_z"],
            s4["final_zone"], s4["posture"], s3["v_z"], s4["motion_str"], s3["Z_b"]
        )
        if s5.get("abort"):
            if s5.get("kill_track"):
                self._reset_track()
                # self.clutter_map = s1["corrected_data"].copy()
            return self.output_dict

        # ── Zone Timer ───────────────────────────────────────────────────────
        now = self.frame_count / config.radar.frame_rate
        self._update_zone_timer(s4["final_zone"], s2["is_valid_point"], now)
        duration_str = "--"
        if self.zone_timer_zone is not None and self.zone_timer_start is not None:
            duration_str = self._format_duration(now - self.zone_timer_start)

        # ── Sum per-antenna history → 2-D for respirationPipeline ───────────
        # The antenna sum (axis=1) is order-independent, but the FRAME axis is
        # still in ring-buffer circular order. The respiration pipeline does
        # phase unwrapping on the frame axis, which requires chronological order.
        # Reorder along the frame axis first, then collapse antennas.
        idx = self._ring_idx_spectral % self.spectral_frames
        if idx != 0:
            raw3d = np.concatenate(
                [self.spectral_history[:, :, idx:], self.spectral_history[:, :, :idx]], axis=2
            )
        else:
            raw3d = self.spectral_history
        ordered_spectral = np.sum(raw3d, axis=1)   # (bins, frames) — time-ordered, backward-compatible

        # ── Cache bin/mag from confirmed frames; reuse during tolerated misses ──
        # On a miss frame dynamic_peak_bin is a fallback unrelated to the persisted
        # track, so its bin index and magnitude are meaningless for the track.
        if s2["is_valid_point"]:
            self._last_valid_bin         = s2["dynamic_peak_bin"]
            self._last_valid_dynamic_mag = s1["dynamic_mag_profile"][s2["dynamic_peak_bin"]]
            # Majority-vote stable bin for display (internal tethering uses raw bin above)
            self._bin_history.append(self._last_valid_bin)
            if len(self._bin_history) >= 1:
                self._stable_bin = Counter(self._bin_history).most_common(1)[0][0]

        self.output_dict.update({
            "X": s3["X_b"], "Y": s3["Y_b"], "Z": s3["Z_b"],
            "final_bin":           self._stable_bin if self._stable_bin is not None else self._last_valid_bin,
            "dynamic_mag":         self._last_valid_dynamic_mag,
            "detection_threshold": self.detection_threshold,
            "zone":                s4["final_zone"],
            "status":              s5["status"],
            "occ_confidence":      s5["occ_confidence"],
            "posture_confidence":  s5["posture_confidence"],
            "posture":             s5["posture"],
            "motion_str":          s5["motion_str"],
            "duration_str":        duration_str,
            "fall_confidence":     s5["fall_confidence"],
            "micro_state":         getattr(self, 'current_micro_state', 'STABLE'),
            "spectral_history":    ordered_spectral,
            "is_valid":            True,
        })
        return self.output_dict

    def _reset_track(self):
        """
        Single source of truth for clearing all transient detection state.
        Called on track loss, ghost kill, zone abort, and radar pose update.
        """
        # Tracking
        self.is_occupied      = False
        self.last_target_bin  = None
        self.last_target_coords = (0.0, 0.0, 0.0)
        self.track_confidence = 0
        self.miss_counter     = 0
        self.coord_buffer.clear()
        self.z_history.clear()
        self.track_x = self.track_y = self.track_z = None

        # Zone debounce — clear so new episode starts fresh
        self.zone_history.clear()
        self.current_stable_zone = "No Occupant Detected"
        self.current_active_zone = "No Occupant Detected"

        # Zone timer
        self.zone_timer_zone      = None
        self.zone_timer_start     = None
        self.zone_timer_last_seen = None

        # Occupancy episode counters
        self.entry_frames          = 0
        self.apnea_frames          = 0
        self.occupied_reflection   = None  # reset so EMA restarts cleanly

        # Continuity hysteresis counters
        self._reflection_dip_frames    = 0
        self._zone_transition_cooldown = 0

        # Last-valid cache
        self._last_valid_bin         = None
        self._last_valid_dynamic_mag = 0.0

        # Display stabilisation state
        self._bin_history.clear()
        self._stable_bin            = None
        self._subzone_history.clear()
        self._stable_subzone_label  = ""
        self._stable_posture        = "Lying Down"   # default posture on fresh episode
        self._xy_track_hist.clear()

        # Fall state
        self.is_fallen             = False
        self.fall_persistence_frames = 0

        self.empty_room()
