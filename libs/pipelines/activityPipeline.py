import numpy as np
from scipy import signal
import time
from collections import deque
from config import config

class ActivityPipeline:
    """
    This class is responsible for processing the radar data and detecting occupancy in specified zones of a given layout
    """
    def __init__(self, num_range_bins, range_resolution):
        self.num_range_bins = num_range_bins
        self.range_res = range_resolution
        self.clutter_map = np.zeros((num_range_bins, config.radar.antennas), dtype=complex)
        self.alpha = config.pipeline.alpha 
        self.detection_threshold = config.pipeline.detection_threshold # need to tune this threshold!
        self.frame_count = 0

        # Dual-Metric State Machine
        self.baseline_profile = None  # The raw absolute reflection of the empty bed
        self.static_margin = config.pipeline.static_margin     # How much raw signal a static body adds over the empty bed
        self.last_target_bin = None   # Remember where they were sitting
        self.is_occupied = False

        # Coordinate Tracking (Smooths the red dot)
        self.track_x = None
        self.track_y = None
        self.track_z = None
        self.track_alpha = config.pipeline.track_alpha # Coordinate smoothing factor (lower = smoother but slower to update)

        # Zone Debouncing (Stops the text from flickering)
        self.current_stable_zone = "No Occupant Detected"
        self.zone_history = []
        self.frames_to_confirm_zone = config.pipeline.frame_to_confirm_zone # Require 2 second (50 frames) of stability to change config.layout

        # Zone occupancy duration tracker
        self.zone_timer_zone = None
        self.zone_timer_start = None
        self.zone_timer_last_seen = None
        self.zone_timer_hold_sec = 3.0   # tolerate short detection dropouts
                
        # Radar Location (Room coordinates)
        self.radar_x = config.layout["Radar"]["x"]
        self.radar_y = config.layout["Radar"]["y"]
        self.radar_z = config.layout["Radar"]["z"]
        self.yaw_deg = config.layout["Radar"]["yaw_deg"]
        self.pitch_deg = config.layout["Radar"]["pitch_deg"]

        self.T = np.array([self.radar_x, self.radar_y, self.radar_z])
    
        # 2. Rotation Matrices (Which way is it looking?)
        pitch_rad = np.radians(self.pitch_deg)
        yaw_rad = np.radians(self.yaw_deg)
        
        # Pitch: Rotates the beam up/down (around the X axis)
        R_pitch = np.array([
            [1, 0, 0],
            [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
            [0, np.sin(pitch_rad), np.cos(pitch_rad)]
        ])
        
        # Yaw: Rotates the beam left/right (around the Z axis)   --->  (0 = +Y, 90 = +X, -90 = -X)
        R_yaw = np.array([
            [np.cos(yaw_rad), np.sin(yaw_rad), 0],
            [-np.sin(yaw_rad), np.cos(yaw_rad), 0],
            [0, 0, 1]
        ])
        
        # Master Rotation Matrix: Apply Pitch, then Yaw
        self.R = np.dot(R_yaw, R_pitch)

        # Tracking & Persistence Variables
        self.coord_buffer = []      
        self.buffer_size = config.pipeline.buffer_size        
        self.track_confidence = 0   
        self.confidence_threshold = 3 
        self.miss_allowance = config.pipeline.miss_allowance       
        self.miss_counter = 0

        # Track reassessment: periodically score without tethering to escape bad locks
        self.reassess_interval = config.radar.frame_rate * 3  # Every 3 seconds
        self.frames_since_reassess = 0

        # ==========================================
        # Fall Detection Parameters
        # ==========================================
        self.z_history = []
        self.z_history_size = 50           # Store last ~2 second of Z data 
        self.fall_threshold_z = config.posture.fall_threshold       # Height below which a person is 'on the floor' (meters)
        self.fall_velocity_threshold = config.posture.fall_velocity_threshold # Velocity threshold for a rapid drop (m/s)
        self.fall_persistence_frames = 0   # How long have they been on the floor
        self.is_fallen = False             # Latching boolean for fall state

        # Warmup Parameters
        self.warmup_frames = config.radar.frame_rate * 1 
        
        # Vital Gating History Buffer
        # Keep 5 seconds of complex data for all bins to evaluate biological motion
        self.vital_gate_frames = 10*config.radar.frame_rate 
        self.complex_history = np.zeros((self.num_range_bins, self.vital_gate_frames), dtype=complex)

        # 10 seconds of history for high-res breathing extraction
        self.spectral_frames = config.respiration.resp_window_sec * config.radar.frame_rate 
        self.spectral_history = np.zeros((self.num_range_bins, self.spectral_frames), dtype=complex)

        self.output_dict = {}
        self.apnea_frames = 0
        self.entry_frames = 0
        self.frames_to_occupy = int(config.radar.frame_rate * 3.0)
        self.empty_room()

    def _score_candidates(self, candidates, max_mag, use_tethering):
        best, best_s = None, -float('inf')
        
        # --- THE ESCAPE CLAUSE ---
        # Only apply strict bed physics if they are in the bed AND lying down
        track_in_monitor = False
        if self.track_x is not None and self.current_active_zone is not None:
            is_bed_zone = "Bed" in self.current_active_zone
            is_lying_down = getattr(self, 'track_z', 0.0) < config.posture.sitting_threshold
            
            # The Breakout: If they are actively shifting/moving out of the zone, let them go!
            is_actively_moving = getattr(self, 'motion_level', 0.0) > 0.25 
            
            if is_bed_zone and is_lying_down and not is_actively_moving:
                track_in_monitor = True

        for c in candidates:
            # Base score: heavily normalize the magnitude so it doesn't overpower vitals
            mag_ratio = c['mag'] / max_mag
            
            # --- 1. VITAL OVERRIDE (Focus on the Chest) ---
            if track_in_monitor:
                # If they are in bed, we don't care how "loud" the reflection is.
                # We care almost exclusively about how rhythmic it is.
                s = (mag_ratio * 0.2) + (c['vital_mult'] * 2.0) 
            else:
                # Standard scoring for walking around the room
                s = mag_ratio * c['vital_mult']
            
            # Zone preference bonus   # TODO: make it relative to the zone type not the name
            if c['zone'] not in ('Floor / Transit', 'Out of Bounds (Ghost)'):
                s += 0.15
                
            # --- 2. STICKY BIN BONUS ---
            if use_tethering and self.is_occupied and self.last_target_bin is not None:
                bin_distance = abs(c['bin'] - self.last_target_bin)
                if bin_distance == 0:
                    s += 0.50  # Center of the chest
                elif bin_distance == 1:
                    s += 0.25  # Edge of the chest (safe 15cm slide)

            # --- 3. ANATOMY TETHERING ---
            if use_tethering and self.track_x is not None:
                xy_dist = np.sqrt((c['x'] - self.track_x)**2 + (c['y'] - self.track_y)**2)
                
                if track_in_monitor:
                    # STRICT TETHER: The person is lying down. The center of mass 
                    # should absolutely not jump 1 meter to the feet.
                    if xy_dist > 0.4: # > 40 cm jump (e.g., jumping from chest to knees)
                        s -= 2.0      # Massive penalty. Kills the candidate entirely.
                    elif xy_dist > 0.15:
                        s -= min(0.8, (xy_dist - 0.15) * 1.5)
                else:
                    # LOOSE TETHER: They are walking around the room. 
                    # Allow larger jumps for normal gait speed.
                    if xy_dist > 0.15: 
                        s -= min(0.4, (xy_dist - 0.15) * 0.4)
                    
                # Penalize vertical jumps (people don't suddenly teleport to the ceiling)
                z_jump = abs(c['z'] - self.track_z)
                s -= min(0.3, z_jump * 0.5)
                
            if s > best_s:
                best_s = s
                best = c
                
        return best, best_s

    def empty_room(self):
        self.motion_level = 0.0
        self.current_micro_state = "STABLE"
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
        }

    def _update_zone_timer(self, zone_name, valid_detection, now):
        """
        Tracks continuous time spent in the current valid zone.
        Resets when the zone changes or when detection is lost
        for longer than zone_timer_hold_sec.
        """

        tracked_zone = zone_name if valid_detection else None

        if tracked_zone is not None:
            # New zone entered
            if self.zone_timer_zone != tracked_zone:
                self.zone_timer_zone = tracked_zone
                self.zone_timer_start = now

            self.zone_timer_last_seen = now

        else:
            # No valid detection: keep timer briefly alive to avoid flicker
            if self.zone_timer_zone is not None and self.zone_timer_last_seen is not None:
                if now - self.zone_timer_last_seen > self.zone_timer_hold_sec:
                    self.zone_timer_zone = None
                    self.zone_timer_start = None
                    self.zone_timer_last_seen = None
        return

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
                
        # 2. Strict Interference Check
        # We process 'ignore' zones first so they safely override overlapping target zones
        for name, bounds in config.layout.items():
            if bounds.get("type") == "ignore":
                if (bounds["x"][0] <= x <= bounds["x"][1] and
                    bounds["y"][0] <= y <= bounds["y"][1] and
                    bounds["z"][0] <= z <= bounds["z"][1]):
                    return f"Ignored ({name})", False

        # 3. Target Zone Check (Beds and Monitors)
        for name, bounds in config.layout.items():
            zone_type = bounds.get("type")
            
            if zone_type == "monitor":
                if (bounds["x"][0] <= x <= bounds["x"][1] and
                    bounds["y"][0] <= y <= bounds["y"][1] and
                    bounds["z"][0] <= z <= bounds["z"][1]):
                    
                    if name == "Bed":
                        # They are in the bed; calculate the sub-zone
                        x_min, x_max = bounds["x"]
                        y_min, y_max = bounds["y"]
                        m_x = bounds.get("margin_x", [0.2, 0.2]) 
                        m_y = bounds.get("margin_y", [0.2, 0.2])
                        
                        is_center_x = (x_min + m_x[0]) <= x <= (x_max - m_x[1])
                        is_center_y = (y_min + m_y[0]) <= y <= (y_max - m_y[1])
                        
                        if is_center_x and is_center_y: return f"{name} - Center", True
                        elif x < (x_min + m_x[0]):      return f"{name} - Right Edge", True
                        elif x > (x_max - m_x[1]):      return f"{name} - Left Edge", True
                        elif y < (y_min + m_y[0]):      return f"{name} - Foot Edge", True 
                        elif y > (y_max - m_y[1]):      return f"{name} - Head Edge", True 
                        else:                           return f"{name} - Corner", True
                        
                    # If it's just a monitor, return the name
                    return name, True
                    
        # 4. Fallback: Inside the room, but not in a predefined zone
        return "Floor / Transit", True

    def _step1_hardware_correction(self, fft_1d_data):
        self.frame_count += 1
        corrected_data = np.copy(fft_1d_data)
        corrected_data[:, [0, 2, 4, 6]] *= -1 

        raw_mag_profile = np.sum(np.abs(corrected_data), axis=1)

        if self.baseline_profile is None:
            self.baseline_profile = raw_mag_profile.copy()
        elif not self.is_occupied and self.frame_count > self.warmup_frames:
            self.baseline_profile = (0.01 * raw_mag_profile) + (0.99 * self.baseline_profile)

        if self.frame_count <= self.warmup_frames:
            current_alpha = 0.3
        elif self.is_occupied:
            current_alpha = 0.01
        else:
            current_alpha = 0.05
            
        self.clutter_map = (current_alpha * corrected_data) + ((1 - current_alpha) * self.clutter_map)
        dynamic_data = corrected_data - self.clutter_map
        dynamic_mag_profile = np.sum(np.abs(dynamic_data), axis=1)

        if self.frame_count <= self.warmup_frames:
            if self.frame_count == self.warmup_frames:
                self.baseline_profile = raw_mag_profile.copy()
            remaining = self.warmup_frames - self.frame_count
            self.output_dict["status"] = f"Calibrating ({remaining} frames)..."
            self.output_dict["zone_name"] = "Calibrating"
            self.output_dict["Range"] = 0.0
            self.output_dict["x"], self.output_dict["y"], self.output_dict["z"] = 0, 0, 0
            return {"abort": True}

        if self.frame_count == self.warmup_frames + self.vital_gate_frames + 1:
            self.track_x, self.track_y, self.track_z = None, None, None
            self.coord_buffer.clear()
            self.track_confidence = 0

        self.complex_history = np.roll(self.complex_history, -1, axis=1)
        self.complex_history[:, -1] = np.sum(dynamic_data, axis=1)

        self.spectral_history = np.roll(self.spectral_history, -1, axis=1)
        self.spectral_history[:, -1] = np.sum(dynamic_data, axis=1)

        return {
            "corrected_data": corrected_data,
            "dynamic_data": dynamic_data,
            "dynamic_mag_profile": dynamic_mag_profile,
            "raw_mag_profile": raw_mag_profile
        }

    def _step2_spatial_candidates(self, dynamic_data, dynamic_mag_profile):
        min_search_bin = int(0.30 / self.range_res)
        is_valid_point = False
        num_candidates = 15 

        actual_candidates = min(num_candidates, len(dynamic_mag_profile[min_search_bin:]))
        if actual_candidates == 0:
            return {"is_jump": False, "is_valid_point": False, "final_peak_bin": None, "dynamic_peak_bin": min_search_bin, "raw_x":0, "raw_y":0, "raw_z":0}

        all_peaks = np.argpartition(dynamic_mag_profile[min_search_bin:], -actual_candidates)[-actual_candidates:]
        all_peaks += min_search_bin

        sorted_peaks = all_peaks[np.argsort(dynamic_mag_profile[all_peaks])][::-1]
        final_peak_bin = None

        valid_candidates = []
        raw_x, raw_y, raw_z = 0, 0, 0
        is_jump = False
        
        for cand_bin in sorted_peaks:
            cand_range = cand_bin * self.range_res
            ch_cand = dynamic_data[cand_bin, :]
            S_cand = np.array([[ch_cand[3], ch_cand[1]], [ch_cand[2], ch_cand[0]], [ch_cand[7], ch_cand[5]], [ch_cand[6], ch_cand[4]]])

            om_az = np.angle(np.sum(S_cand[:, 0] * np.conj(S_cand[:, 1])))
            om_el = np.angle(np.sum(S_cand[0:3, :] * np.conj(S_cand[1:4, :])))
            
            az_cand = np.arcsin(np.clip(om_az / np.pi, -1.0, 1.0))
            el_cand = np.arcsin(np.clip(om_el / np.pi, -1.0, 1.0))

            Pr_c = np.array([cand_range * np.sin(az_cand) * np.cos(el_cand),
                            cand_range * np.cos(az_cand) * np.cos(el_cand),
                            cand_range * np.sin(el_cand)])
            Pb_c = np.dot(self.R, Pr_c) + self.T
            
            zone_name, is_valid = self.evaluate_spatial_zone(Pb_c[0], Pb_c[1], Pb_c[2])
            if is_valid and zone_name != "Out of Bounds (Ghost)":
                cand_micro_state = "STABLE"
                vital_multiplier = 0.1
                
                if (self.frame_count - self.warmup_frames) > self.spectral_frames:
                    cand_history = self.spectral_history[cand_bin, :]
                    cand_history_safe = np.where(cand_history == 0, 1e-10 + 1e-10j, cand_history)
                    cand_phase = np.unwrap(np.angle(cand_history_safe))
                    
                    phase_ptp = np.ptp(cand_phase)
                    displacement_mm = (phase_ptp * 5.0) / (4.0 * np.pi)
                    
                    phase_diff = np.diff(cand_phase)
                    phase_var = np.var(phase_diff)
                    
                    if displacement_mm > 15.0:
                        vital_multiplier = 0.9 
                        cand_micro_state = "MACRO_PHASE"
                    elif phase_var > 0.3: 
                        vital_multiplier = 0.7
                        cand_micro_state = "MICRO_PHASE"
                    else:
                        detrended_phase = signal.detrend(cand_phase)
                        window = np.hanning(self.spectral_frames)
                        windowed_phase = detrended_phase * window
                        
                        fft_result = np.fft.rfft(windowed_phase)
                        fft_mag = np.abs(fft_result)
                        freqs = np.fft.rfftfreq(self.spectral_frames, d=(1.0/config.radar.frame_rate))
                        
                        vital_band_mask = (freqs >= 0.15) & (freqs <= 0.7)
                        eval_band_mask = (freqs >= 0.15) & (freqs <= 3.0) 
                        
                        vital_energy = np.sum(fft_mag[vital_band_mask])
                        total_energy = np.sum(fft_mag[eval_band_mask])
                        
                        sqi = vital_energy / (total_energy + 1e-6)
                        
                        if sqi > 0.45:
                            vital_multiplier = 1.0
                        elif sqi > 0.25:
                            vital_multiplier = 0.5
                        else:
                            vital_multiplier = 0.05
                            cand_micro_state = "DEAD_SPACE"
                
                valid_candidates.append({
                    'bin': cand_bin,
                    'x': Pb_c[0], 'y': Pb_c[1], 'z': Pb_c[2],
                    'azimuth': az_cand,
                    'elevation': el_cand,
                    'mag': dynamic_mag_profile[cand_bin],
                    'zone': zone_name,
                    'vital_mult': vital_multiplier,
                    'micro_state': cand_micro_state
                })
                
        if valid_candidates:
            is_valid_point = True
            max_mag = max(c['mag'] for c in valid_candidates)

            best_cand, best_score = self._score_candidates(valid_candidates, max_mag, use_tethering=False)
            
            final_peak_bin = best_cand['bin']
            self.current_active_zone = best_cand['zone']
            self.current_micro_state = best_cand.get('micro_state', 'STABLE')

            raw_x, raw_y, raw_z = best_cand['x'], best_cand['y'], best_cand['z']
            raw_z = np.clip(raw_z, 0.05, 1.8)

            if self.track_x is not None:
                jump_dist = np.sqrt((raw_x - self.track_x)**2 + (raw_y - self.track_y)**2)
                if jump_dist > 1.5:  
                    is_valid_point = False
                    final_peak_bin = None
                    is_jump = True
                        
        if final_peak_bin is None:
            dynamic_peak_bin = sorted_peaks[0]
        else:
            dynamic_peak_bin = final_peak_bin

        self.output_dict["Azimuth"] = best_cand['azimuth'] if valid_candidates else 0.0 
        self.output_dict["Elevation"] = best_cand['elevation'] if valid_candidates else 0.0
        self.output_dict["Range"] = dynamic_peak_bin * self.range_res

        return {
            "is_jump": is_jump,
            "is_valid_point": is_valid_point,
            "final_peak_bin": final_peak_bin,
            "dynamic_peak_bin": dynamic_peak_bin,
            "raw_x": raw_x,
            "raw_y": raw_y,
            "raw_z": raw_z
        }

    def _step3_state_machine(self, dynamic_mag_profile, dynamic_peak_bin, raw_x, raw_y, raw_z, is_jump, raw_mag_profile, corrected_data):
        current_threshold = self.detection_threshold if not self.is_occupied else (self.detection_threshold * 0.75)  
        
        status = self.output_dict.get("status", "")

        if not is_jump and dynamic_mag_profile[dynamic_peak_bin] >= self.detection_threshold:
            if not self.is_occupied:
                self.entry_frames += 1
                if self.entry_frames < self.frames_to_occupy:
                    return {"abort": True}
            
            self.is_occupied = True
            self.apnea_frames = 0
            self.last_target_bin = dynamic_peak_bin   
            self.last_target_coords = (raw_x, raw_y, raw_z)      
            status = "Occupied (Breathing/Moving)"     
        
        elif self.is_occupied and self.last_target_bin is not None:
            if self.track_x is not None:
                curr_loc = (self.track_x, self.track_y, self.track_z)
            else:
                curr_loc = getattr(self, 'last_target_coords', (0,0,0))
                
            last_zone_name, _ = self.evaluate_spatial_zone(curr_loc[0], curr_loc[1], curr_loc[2])
            
            try:
                is_monitored_zone = config.layout[last_zone_name]["type"] == "monitor"
            except KeyError:
                is_monitored_zone = False

            current_raw_reflection = np.max(raw_mag_profile[self.last_target_bin - 1 : self.last_target_bin + 2])
            empty_bed_reflection = self.baseline_profile[self.last_target_bin]
            
            if is_monitored_zone and current_raw_reflection > (empty_bed_reflection + self.static_margin):
                self.apnea_frames += 1
                apnea_limit = config.radar.frame_rate * 5 
                if self.apnea_frames >= apnea_limit:
                    status = "Possible Apnea"
                else:
                    status = "Still / Monitoring..."
            else:
                self.is_occupied = False
                self.apnea_frames = 0
                self.last_target_bin = None
                self.track_confidence = 0
                self.coord_buffer.clear()
                self.z_history.clear()
                self.track_x, self.track_y, self.track_z = None, None, None
                self.empty_room()
                self.clutter_map = corrected_data.copy()
                return {"abort": True}
        else:
            self.empty_room()
            return {"abort": True}

        return {"status": status, "current_raw_reflection": locals().get('current_raw_reflection', 0.0), "empty_bed_reflection": locals().get('empty_bed_reflection', 0.0)}

    def _step4_temporal_persistence(self, is_valid_point, raw_x, raw_y, raw_z):
        if is_valid_point:
            self.track_confidence = min(self.track_confidence + 1, self.confidence_threshold)
            self.miss_counter = 0
            self.coord_buffer.append((raw_x, raw_y, raw_z))
            if len(self.coord_buffer) > self.buffer_size:
                self.coord_buffer.pop(0)
        else:
            self.miss_counter += 1
            if self.miss_counter > self.miss_allowance:
                self.is_occupied = False
                self.last_target_bin = None
                self.track_confidence = 0
                self.coord_buffer.clear()
                self.z_history.clear()
                self.track_x, self.track_y, self.track_z = None, None, None
                self.empty_room()
                return {"abort": True}
        return {}

    def _step5_adaptive_smoothing(self):
        if self.track_confidence >= self.confidence_threshold:
            recent_coords = np.array(self.coord_buffer)
            med_x = np.median(recent_coords[:, 0])
            med_y = np.median(recent_coords[:, 1])
            med_z = np.median(recent_coords[:, 2])

            if self.track_x is None:
                self.track_x, self.track_y, self.track_z = med_x, med_y, med_z
                self.motion_level = 0.0
            else:
                shift_distance = np.sqrt((med_x - self.track_x)**2 + (med_y - self.track_y)**2 + (med_z - self.track_z)**2)
                self.motion_level = (0.2 * shift_distance) + (0.8 * self.motion_level)

                track_zone, _ = self.evaluate_spatial_zone(self.track_x, self.track_y, self.track_z)
                candidate_zone = self.current_active_zone  
                if track_zone != candidate_zone:
                    adaptive_alpha = 0.4
                elif shift_distance < 0.12:
                    adaptive_alpha = 0.02  
                elif shift_distance < 0.50:
                    adaptive_alpha = 0.1   
                else:
                    adaptive_alpha = 0.5   
                
                self.track_x = (adaptive_alpha * med_x) + ((1 - adaptive_alpha) * self.track_x)
                self.track_y = (adaptive_alpha * med_y) + ((1 - adaptive_alpha) * self.track_y)
                self.track_z = (adaptive_alpha * med_z) + ((1 - adaptive_alpha) * self.track_z)

            X_b, Y_b, Z_b = med_x, med_y, med_z  

            self.z_history.append(Z_b)
            if len(self.z_history) > self.z_history_size:  
                self.z_history.pop(0)
                
            velocity_window_frames = int(config.radar.frame_rate * 0.4) 
            if len(self.z_history) >= velocity_window_frames:
                dz = self.z_history[-1] - self.z_history[-velocity_window_frames]
                dt = velocity_window_frames * (1.0 / config.radar.frame_rate) 
                v_z = dz / dt  
            else:
                v_z = 0.0

            return {"X_b": X_b, "Y_b": Y_b, "Z_b": Z_b, "v_z": v_z}
        else:
            self.empty_room()
            return {"abort": True}

    def _step6_posture_and_motion(self, X_b, Y_b, Z_b, status):
        final_zone, _ = self.evaluate_spatial_zone(X_b, Y_b, Z_b)

        self.zone_history.append(final_zone)
        if len(self.zone_history) > self.frames_to_confirm_zone:
            self.zone_history.pop(0)

        if len(self.zone_history) == self.frames_to_confirm_zone:
            most_common_zone = max(set(self.zone_history), key=self.zone_history.count)
            self.current_stable_zone = most_common_zone
            
        final_zone = self.current_stable_zone

        if Z_b > config.posture.standing_threshold:
            posture = "Standing"
        elif Z_b > config.posture.sitting_threshold:
            posture = "Sitting"
        else:
            posture = "Lying Down"

        if "Apnea" in status:
            motion_str = "Static"
        elif self.motion_level > config.motion.restless_max: 
            motion_str = "Major Movement"
        elif getattr(self, 'current_micro_state', 'STABLE') == "MACRO_PHASE" and ("Bed" in self.current_active_zone or "Chair" in self.current_active_zone):
            motion_str = "Postural Shift"
        elif self.motion_level > config.motion.rest_max: 
            motion_str = "Restless/Shifting"
        elif getattr(self, 'current_micro_state', 'STABLE') == "MICRO_PHASE":
            motion_str = "Restless/Fidgeting"
        else:
            motion_str = "Resting/Breathing"
        
        fall_confidence = 0.0
        
        if final_zone in ["Out of Bounds (Ghost)", "Ignored"]:
            self.is_occupied = False
            self.last_target_bin = None
            return {"abort": True}
            
        elif final_zone == "Floor / Transit":
            status = status.replace("Occupied", "In the Room") 
            if Z_b <= self.fall_threshold_z or self.is_fallen:
                self.is_fallen = True
                self.fall_persistence_frames += 1
                
                h_score = max(0.0, (self.fall_threshold_z - Z_b) / self.fall_threshold_z) * 100.0
                p_score = min(100.0, (self.fall_persistence_frames / self.z_history_size) * 100.0) 
                
                fall_confidence = (0.4 * h_score) + (0.6 * p_score)
                
                if fall_confidence > 60.0:
                    status = "CRITICAL: Fall Detected!"
                    posture = "Fallen"
            else:
                self.fall_persistence_frames = max(0, self.fall_persistence_frames - 2)
        else:
            self.is_fallen = False
            self.fall_persistence_frames = 0

        return {"final_zone": final_zone, "posture": posture, "motion_str": motion_str, "status": status, "fall_confidence": fall_confidence}

    def _step7_confidence_metrics(self, status, dynamic_mag_profile, dynamic_peak_bin, current_raw_reflection, empty_bed_reflection, Z_b, posture):
        temporal_conf = (self.track_confidence / self.confidence_threshold) * 100.0
        signal_conf = 100.0
        
        if "Breathing" in status:
            margin = (dynamic_mag_profile[dynamic_peak_bin] - self.detection_threshold) / self.detection_threshold
            signal_conf = min(100.0, 50.0 + (margin * 100.0))
            if self.motion_level > 0.05:
                signal_conf = 100.0 
        elif "Apnea" in status:
            margin = (current_raw_reflection - empty_bed_reflection - self.static_margin) / self.static_margin
            if margin <= 0: margin = 0
            signal_conf = min(100.0, 50.0 + (margin * 100.0))
            
        occ_confidence = (0.6 * temporal_conf) + (0.4 * signal_conf)
        if self.miss_counter > 0:
            occ_confidence = max(0.0, occ_confidence - ((self.miss_counter / self.miss_allowance) * 100.0))

        if posture == "Standing":
            dist_to_thresh = abs(Z_b - config.posture.standing_threshold)
        elif posture == "Sitting":
            dist_to_thresh = min(abs(Z_b - config.posture.sitting_threshold), abs(config.posture.standing_threshold - Z_b))
        elif posture == "Fallen":
            dist_to_thresh = abs(self.fall_threshold_z - Z_b)
        else: 
            dist_to_thresh = abs(config.posture.sitting_threshold - Z_b)
        
        base_posture_conf = min(100.0, 50.0 + ((dist_to_thresh / 0.125) * 50.0))
        motion_penalty = min(40.0, self.motion_level * 200.0) 
        posture_confidence = max(50.0, base_posture_conf - motion_penalty)

        return {"occ_confidence": occ_confidence, "posture_confidence": posture_confidence}

    def process_frame(self, fft_1d_data):
        # Step 1: Hardware Correction & Background
        s1 = self._step1_hardware_correction(fft_1d_data)
        if s1.get("abort"): return self.output_dict

        # Step 2: Spatial-Aware Candidate Selection
        s2 = self._step2_spatial_candidates(s1["dynamic_data"], s1["dynamic_mag_profile"])

        # Step 3: State Machine
        s3 = self._step3_state_machine(
            s1["dynamic_mag_profile"], s2["dynamic_peak_bin"], 
            s2["raw_x"], s2["raw_y"], s2["raw_z"], 
            s2["is_jump"], s1["raw_mag_profile"], s1["corrected_data"]
        )
        if s3.get("abort"): return self.output_dict

        # Step 4: Temporal Persistence
        s4 = self._step4_temporal_persistence(s2["is_valid_point"], s2["raw_x"], s2["raw_y"], s2["raw_z"])
        if s4.get("abort"): return self.output_dict

        # Step 5: Adaptive Smoothing
        s5 = self._step5_adaptive_smoothing()
        if s5.get("abort"): return self.output_dict

        # Step 6: Posture Logic & Motion Tagging
        s6 = self._step6_posture_and_motion(s5["X_b"], s5["Y_b"], s5["Z_b"], s3["status"])
        if s6.get("abort"): return self.output_dict

        # Step 7: Confidence Indices
        s7 = self._step7_confidence_metrics(
            s6["status"], s1["dynamic_mag_profile"], s2["dynamic_peak_bin"], 
            s3["current_raw_reflection"], s3["empty_bed_reflection"], 
            s5["Z_b"], s6["posture"]
        )

        # Step 8: Update Zone Timers
        now = self.frame_count / config.radar.frame_rate
        self._update_zone_timer(s6["final_zone"], s2["is_valid_point"], now)

        duration_str = "--"
        if self.zone_timer_zone is not None and self.zone_timer_start is not None:
            duration_str = self._format_duration(now - self.zone_timer_start)

        # Finalize Output
        self.output_dict.update({
            "X": s5["X_b"],
            "Y": s5["Y_b"],
            "Z": s5["Z_b"],
            "final_bin": s2["dynamic_peak_bin"],
            "zone": s6["final_zone"],
            "status": s6["status"],
            "occ_confidence": s7["occ_confidence"],
            "posture_confidence": s7["posture_confidence"],
            "posture": s6["posture"],
            "motion_str": s6["motion_str"],
            "duration_str": duration_str,
            "fall_confidence": s6["fall_confidence"],
            "spectral_history": self.spectral_history,
        })
        return self.output_dict
