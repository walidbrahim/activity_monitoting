import multiprocessing
import queue
from collections import deque
import logging
import time
import struct
import traceback
import numpy as np
from scipy import signal
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as mtransforms
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Wedge, Circle
import platform
if platform.system() == 'Darwin':
    matplotlib.use('MacOSX')
else:
    matplotlib.use('TkAgg')
import requests
import threading
import serial

# ==========================================
# Setup Parameters
# ==========================================

# --- 3D Geofencing Dictionary ---
LAYOUT = {
    "Room": {
        "type": "boundary",
        "x": [0.0, 2.58], "y": [0.0, 3.5], "z": [0.0, 2.7] 
    },
    "Radar": {
        "type": "sensor",
        "x": 1.15, "y": 3.27, "z": 1,
        "yaw_deg": 210, # 0 is up, 90 is right
        "pitch_deg": -30, # 0 is level
        "fov_deg": 120
    },
    "Bed": {
        "type": "monitor", 
        "x": [0.0, 1.05], "y": [1.45, 3.5], "z": [0, 2.7],
        "margin_x": [0.3, 0.3], # 0.3 at the left/right
        "margin_y": [0.3, 0]  # 0.3 at the footer
    },
    "Desk": {
        "type": "ignore", # Standard bounding box, no sub-LAYOUT
        "x": [1.95, 2.58], "y": [1.6, 2.64], "z": [0, 1.2]
    },
    "Chair": {
        "type": "monitor", # Standard bounding box, no sub-LAYOUT
        "x": [1.5, 2], "y": [1.6, 2.64], "z": [0, 1.2]
    }
}

# Pushover keys
PUSHOVER_USER_KEY = "ua846nsmwipsvkvk2r7guc1koekhnf"
PUSHOVER_API_TOKEN = "arh5d8i8nprcfovit7kyobse1p3ey4"

# Radar parameters
RADAR_CONFIG_FILE_PATH = 'profiles/60ghz_25fps_6843_8ant_15cm.txt'
ANTENNAS = 8
RANGE_RESOLUTION = 0.15
FRAME_RATE = 25
RANGE_IDX_NUM = 35
TI_1DFFT_QUEUE_LEN = 25
if platform.system() == 'Darwin':
    TI_CLI_SERIAL_PORT ="/dev/cu.usbserial-00E2410B0"  # Home:"/dev/cu.usbserial-00E243020"  
    SERIAL_PORT_NAME = "/dev/cu.usbserial-00E2410B1" # Home: "/dev/cu.usbserial-00E243021"      
else:
    TI_CLI_SERIAL_PORT =  "/dev/ttyUSB0"
    SERIAL_PORT_NAME ="/dev/ttyUSB1"
MAGIC_WORD = [0x708050603040101, 0x708050603040102, 0x708050603040103, 0x708050603040104,
                      0x708050603040105, 0x708050603040106, 0x708050603040107, 0x708050603040108]
START_FREQUENCY = 60  # GHz   
CHIRP_LOOP = 16
RANGE_BINS = 64
CHIRP_SLOPE = 32  # MHz/us
CHIRP_END_TIME = 36 # us
ADC_START = 2 # us
CHIRP_IDLE_TIME =   15  # us
ADC_SAMPLE_RATE =  2000   #ksps  ---> Fast time sampling (over 1 chirp)
TX_ANT = 2
RX_ANT = 4

# Occupancy Detection PipelineParameters
DETECTION_THRESHOLD = 150.0   # Threshold for detecting a moving object
STATIC_MARGIN = 200            # How much raw signal a static body adds over the empty bed
ALPHA = 0.05                  # Moving average filter coefficient
TRACK_ALPHA = 0.2             # Coordinate smoothing factor (lower = smoother but slower to update)
FRAME_TO_CONFIRM_ZONE = 50    # Require 1 second (25 frames) of stability to change LAYOUT
BUFFER_SIZE = 10              # Buffer size for coordinate tracking
MISS_ALLOWANCE = 25           # Allow 50 frames of no detection before clearing the track

# Posture parameters
FALL_DETECTION_ENABLE = True
FALL_THRESHOLD = 0.5    # m
FALL_VELOCITY_THRESHOLD = -1.2   # m/s

SITTING_THRESHOLD = 0.6 
STANDING_THRESHOLD = 1.1

# Motion thresholds
REST_MAX = 0.1
RESTLESS_MAX = 0.3

# Respiration Processing
RESP_WINDOW_SEC = 30 # seconds

# Others
LOG_LEVEL = 10
NEED_SEND_TI_CONFIG = True
SEND_ALERT = False    # Send alert to phone/watch

# GUI Theme
FIG_BG = "#111827"
PANEL_BG = "#1F2937"
CARD_BG = "#243447"
CARD_OK = "#166534"
CARD_WARN = "#92400E"
CARD_ALERT = "#7F1D1D"

TEXT = "#E5E7EB"
SUBTEXT = "#9CA3AF"
GRID = "#334155"

BED = "#4A90E2"
CHAIR = "#14B8A6"
MONITOR = "#6B8E7A"
IGNORE = "#DC2626"

OCCUPANT = "#22D3EE"
ROOM_EDGE = "#CBD5E1"
FOV = "#EF4444"
RADAR = "#F43F5E"
ORIGIN = "#FACC15"

# TODO: Make it abstract
def send_watch_alert(state_message):
    """Sends a push notification in the background."""
    def _send():
        try:
            requests.post("https://api.pushover.net/1/messages.json", data={
                "token": PUSHOVER_API_TOKEN,
                "user": PUSHOVER_USER_KEY,
                "message": f"Update: {state_message}",
                "title": "Room Activity Monitor",
                "sound": "intermission" # watch ping sound
            }, timeout=5)
        except Exception as e:
            print(f"Failed to send alert: {e}")
            
    # Start the web request in a separate thread so the radar doesn't lag
    threading.Thread(target=_send, daemon=True).start()

# ==========================================
# 1. Radar Controller
# ==========================================
class WisSerial:
    def __init__(self, port=SERIAL_PORT_NAME, baudrate=921600):
        self.ser = serial.Serial()
        self.ser.baudrate = baudrate
        self.ser.timeout = 2
        self.ser.port = port

    def connect(self):
        try:
            if self.ser.is_open:
                print('will close %s' % (self.ser.port,))
                self.ser.close()
            self.ser.open()
            if self.ser.is_open:
                return
        except:
            pass

        if self.ser.is_open:
            print(' Open UART success!')
            time.sleep(0.1)
            self.ser.reset_input_buffer()
        else:
            print(' Open UART fail')

    def read(self, buf_len):
        rxbuf = self.ser.read(buf_len)
        return rxbuf
    
    def read_buffer_line(self):
        rxbuf_readline = self.ser.readline()
        return rxbuf_readline

    def write(self, content):
        if isinstance(content, str):
            content = content.encode('utf-8')
        self.ser.write(content)

    def close(self):
        try:
            self.ser.close()
        except:
            pass

    def is_open(self):
        try:
            return self.ser.is_open
        except:
            return False

class RadarController(multiprocessing.Process):
    def __init__(self, state_q, calculation_status=0, **kwargs):
        super().__init__()
        self.state_q = state_q
        self.out_fft_array_q = kwargs.get('out_fft_array_q')
        self.save_fft_array_q = kwargs.get('save_fft_array_q')
        self.pt_fft_q = kwargs.get('pt_fft_q')
        self.start_radar_flag = kwargs.get('start_radar_flag')
        self.calculation_status = calculation_status
        self.range_matrix_queue = np.zeros((0, RANGE_IDX_NUM, ANTENNAS), dtype=complex)
        self.fft_matrix_queue = np.zeros((0, RANGE_IDX_NUM, ANTENNAS), dtype=complex)
        self.able_to_calculate_flag = False
        self.RangeMatrixQueueLen = TI_1DFFT_QUEUE_LEN
        self.able_put_flag = False
        self.recording_flag = False
        self.order = 0
        self.put_fft = np.zeros((RANGE_IDX_NUM, ANTENNAS), dtype=complex)

    def run(self):
        logging.basicConfig(level=LOG_LEVEL)
        ti_cli_ser = WisSerial(SERIAL_PORT_NAME, baudrate=921600)
        self.wait_for_reconnect(ti_cli_ser, True)

    @staticmethod
    def send_ti_config(is_new_config):
        if is_new_config:
            ti_cli_ser = WisSerial(TI_CLI_SERIAL_PORT, baudrate=115200)
            ti_cli_ser.connect()
            config_path = RADAR_CONFIG_FILE_PATH
            with open(config_path, 'r') as f:
                print('\nSending Configuration to radar ...')
                config_line = f.readline()
                while config_line:
                    ti_cli_ser.write(config_line)
                    time.sleep(0.1)
                    feedback = ti_cli_ser.read_buffer_line()
                    time.sleep(0.1)
                    config_line = f.readline()
                print('Radar started ...\n')

    def read_data(self, ti_cli_ser):
        is_error = False
        chirp_temp = b''
        flag_header = False
        while not is_error:
            try:
                temp_buffer = ti_cli_ser.read_buffer_line()  
                if b'\x01\x04\x03\x06\x05\x08\x07' in temp_buffer and b'TIAOP\r\n' in temp_buffer:
                    chirp_temp = temp_buffer[1:]
                    if len(chirp_temp) == 163:
                        self.analyticalBuffer(chirp_temp)
                elif b'\x01\x04\x03\x06\x05\x08\x07' in temp_buffer and b'TIAOP\r\n' not in temp_buffer:
                    flag_header = True
                    chirp_temp = temp_buffer[1:]
                elif flag_header:
                    chirp_temp += temp_buffer
                    if b'TIAOP\r\n' in temp_buffer:
                        flag_header = False
                        if len(chirp_temp) == 163:
                            self.analyticalBuffer(chirp_temp)          
            except:
                traceback.print_exc()
                print('traceback.format_exc():\n%s' % traceback.format_exc())

    def wait_for_reconnect(self, ti_cli_ser, is_first_connect=False):
        while not ti_cli_ser.is_open():
            self.send_ti_config(NEED_SEND_TI_CONFIG)
            ti_cli_ser.connect()
        self.read_data(ti_cli_ser)

    def analyticalBuffer(self, data):
        header_length = 8
        timeLen = 4
        step_size = 4
        magic = struct.unpack('Q', data[:header_length])
        timeStamp = struct.unpack('I', data[header_length:(header_length + timeLen)])      
        if magic[0] == MAGIC_WORD[self.order]:
            content_start = header_length + timeLen
            range_matrix_real = np.zeros(RANGE_IDX_NUM, dtype=int)
            range_matrix_imag = np.zeros(RANGE_IDX_NUM, dtype=int)
            output_idx = 0
            for rangeIdx in range(0, RANGE_IDX_NUM * step_size, step_size):
                temp_real = struct.unpack('<h', data[(content_start + rangeIdx):(content_start + rangeIdx + 2)])
                temp_imag = struct.unpack('<h', data[(content_start + rangeIdx + 2):(content_start + rangeIdx + 4)])
                range_matrix_real[output_idx] = temp_real[0]
                range_matrix_imag[output_idx] = temp_imag[0]
                output_idx = output_idx + 1
            
            endtimestamp = struct.unpack('I', data[(content_start+RANGE_IDX_NUM*step_size):(content_start+RANGE_IDX_NUM*step_size+4)])

            range_matrix_all_ant_real = range_matrix_real.reshape(RANGE_IDX_NUM, 1)
            range_matrix_all_ant_imag = range_matrix_imag.reshape(RANGE_IDX_NUM, 1)
            range_fft = range_matrix_all_ant_real + 1j * range_matrix_all_ant_imag

            self.put_fft[:, self.order] = range_fft.reshape(-1)
            self.order = self.order + 1

            if self.order == ANTENNAS:
                self.order = 0
                self.pt_fft_q.put(self.put_fft)
                
                # Create a fresh array for the next frame to prevent overwriting data in the queue
                self.put_fft = np.zeros((RANGE_IDX_NUM, ANTENNAS), dtype=complex)
        else:
            self.order = 0

# ==================================================
# 2. Indoor Activity Monitoring Processing Pipeline
# ==================================================
class ActivityPipeline:
    """
    This class is responsible for processing the radar data and detecting occupancy in specified zones of a given layout
    """
    def __init__(self, num_range_bins, range_resolution):
        self.num_range_bins = num_range_bins
        self.range_res = range_resolution
        self.clutter_map = np.zeros((num_range_bins, ANTENNAS), dtype=complex)
        self.alpha = ALPHA 
        self.detection_threshold = DETECTION_THRESHOLD # need to tune this threshold!
        self.frame_count = 0

        # Dual-Metric State Machine
        self.baseline_profile = None  # The raw absolute reflection of the empty bed
        self.static_margin = STATIC_MARGIN     # How much raw signal a static body adds over the empty bed
        self.last_target_bin = None   # Remember where they were sitting
        self.is_occupied = False

        # Coordinate Tracking (Smooths the red dot)
        self.track_x = None
        self.track_y = None
        self.track_z = None
        self.track_alpha = TRACK_ALPHA # Coordinate smoothing factor (lower = smoother but slower to update)

        # Zone Debouncing (Stops the text from flickering)
        self.current_stable_zone = "No Occupant Detected"
        self.zone_history = []
        self.frames_to_confirm_zone = FRAME_TO_CONFIRM_ZONE # Require 2 second (50 frames) of stability to change LAYOUT

        # Zone occupancy duration tracker
        self.zone_timer_zone = None
        self.zone_timer_start = None
        self.zone_timer_last_seen = None
        self.zone_timer_hold_sec = 3.0   # tolerate short detection dropouts
                
        # Radar Location (Room coordinates)
        self.radar_x = LAYOUT["Radar"]["x"]
        self.radar_y = LAYOUT["Radar"]["y"]
        self.radar_z = LAYOUT["Radar"]["z"]
        self.yaw_deg = LAYOUT["Radar"]["yaw_deg"]
        self.pitch_deg = LAYOUT["Radar"]["pitch_deg"]

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
        self.buffer_size = BUFFER_SIZE        
        self.track_confidence = 0   
        self.confidence_threshold = 3 
        self.miss_allowance = MISS_ALLOWANCE       
        self.miss_counter = 0

        # Track reassessment: periodically score without tethering to escape bad locks
        self.reassess_interval = FRAME_RATE * 3  # Every 3 seconds
        self.frames_since_reassess = 0

        # ==========================================
        # Fall Detection Parameters
        # ==========================================
        self.z_history = []
        self.z_history_size = 50           # Store last ~2 second of Z data 
        self.fall_threshold_z = FALL_THRESHOLD       # Height below which a person is 'on the floor' (meters)
        self.fall_velocity_threshold = FALL_VELOCITY_THRESHOLD # Velocity threshold for a rapid drop (m/s)
        self.fall_persistence_frames = 0   # How long have they been on the floor
        self.is_fallen = False             # Latching boolean for fall state

        # Warmup Parameters
        self.warmup_frames = FRAME_RATE * 1 
        
        # Vital Gating History Buffer
        # Keep 5 seconds of complex data for all bins to evaluate biological motion
        self.vital_gate_frames = 10*FRAME_RATE 
        self.complex_history = np.zeros((self.num_range_bins, self.vital_gate_frames), dtype=complex)

        # 10 seconds of history for high-res breathing extraction
        self.spectral_frames = RESP_WINDOW_SEC * FRAME_RATE 
        self.spectral_history = np.zeros((self.num_range_bins, self.spectral_frames), dtype=complex)

        self.output_dict = {}
        self.apnea_frames = 0
        self.entry_frames = 0
        self.frames_to_occupy = int(FRAME_RATE * 3.0)
        self.empty_room()

    def _score_candidates(self, candidates, max_mag, use_tethering):
        best, best_s = None, -float('inf')
        
        # --- THE ESCAPE CLAUSE ---
        # Only apply strict bed physics if they are in the bed AND lying down
        track_in_monitor = False
        if self.track_x is not None and self.current_active_zone is not None:
            is_bed_zone = "Bed" in self.current_active_zone
            is_lying_down = getattr(self, 'track_z', 0.0) < SITTING_THRESHOLD
            
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
        room = LAYOUT.get("Room")
        if room:
            if not (room["x"][0] <= x <= room["x"][1] and 
                    room["y"][0] <= y <= room["y"][1] and 
                    room["z"][0] <= z <= room["z"][1]):
                return "Out of Bounds (Ghost)", False
                
        # 2. Strict Interference Check
        # We process 'ignore' zones first so they safely override overlapping target zones
        for name, bounds in LAYOUT.items():
            if bounds.get("type") == "ignore":
                if (bounds["x"][0] <= x <= bounds["x"][1] and
                    bounds["y"][0] <= y <= bounds["y"][1] and
                    bounds["z"][0] <= z <= bounds["z"][1]):
                    return f"Ignored ({name})", False

        # 3. Target Zone Check (Beds and Monitors)
        for name, bounds in LAYOUT.items():
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

    def process_frame(self, fft_1d_data):
        # ==========================================
        # Step 1: Hardware Correction & Background
        # ==========================================
        self.frame_count += 1
        is_jump = False
        fall_confidence = 0.0
        corrected_data = np.copy(fft_1d_data)
        corrected_data[:, [0, 2, 4, 6]] *= -1 

        # Calculate the raw magnitude profile
        raw_mag_profile = np.sum(np.abs(corrected_data), axis=1)

        # Slowly update the empty bed baseline ONLY when the room is empty
        if self.baseline_profile is None:
            self.baseline_profile = raw_mag_profile.copy()
        elif not self.is_occupied and self.frame_count > self.warmup_frames:
            self.baseline_profile = (0.01 * raw_mag_profile) + (0.99 * self.baseline_profile)

        # Dynamic Clutter Filter (Freeze background learning if someone is in the room)
        if self.frame_count <= self.warmup_frames:
            current_alpha = 0.3  # Fast convergence during calibration
        elif self.is_occupied:
            current_alpha = 0.01
        else:
            current_alpha = 0.05
        self.clutter_map = (current_alpha * corrected_data) + ((1 - current_alpha) * self.clutter_map)
        dynamic_data = corrected_data - self.clutter_map
        dynamic_mag_profile = np.sum(np.abs(dynamic_data), axis=1)

        # --- Calibration period: let filters converge before making any decisions ---
        if self.frame_count <= self.warmup_frames:
            # Snapshot baseline at the END of warmup (when clutter map has converged)
            if self.frame_count == self.warmup_frames:
                self.baseline_profile = raw_mag_profile.copy()
            remaining = self.warmup_frames - self.frame_count
            self.output_dict["status"] = f"Calibrating ({remaining} frames)..."
            self.output_dict["zone_name"] = "Calibrating"
            self.output_dict["Range"] = 0.0
            self.output_dict["x"], self.output_dict["y"], self.output_dict["z"] = 0, 0, 0
            return self.output_dict

        # Clear any track established during warmup once vital gating kicks in
        if self.frame_count == self.warmup_frames + self.vital_gate_frames + 1:
            self.track_x, self.track_y, self.track_z = None, None, None
            self.coord_buffer.clear()
            self.track_confidence = 0

        self.complex_history = np.roll(self.complex_history, -1, axis=1)
        self.complex_history[:, -1] = np.sum(dynamic_data, axis=1)

        self.spectral_history = np.roll(self.spectral_history, -1, axis=1)
        self.spectral_history[:, -1] = np.sum(dynamic_data, axis=1)

        # ==========================================
        # Step 2: Spatial-Aware Candidate Selection
        # ==========================================
        min_search_bin = int(0.30 / self.range_res)   # distance of 30 cm to avoid noise from antennas
        is_valid_point = False
        num_candidates = 15 

        actual_candidates = min(num_candidates, len(dynamic_mag_profile[min_search_bin:]))
        
        all_peaks = np.argpartition(dynamic_mag_profile[min_search_bin:], -actual_candidates)[-actual_candidates:]
        all_peaks += min_search_bin

        sorted_peaks = all_peaks[np.argsort(dynamic_mag_profile[all_peaks])][::-1]
        final_peak_bin = None

        valid_candidates = []
        raw_x, raw_y, raw_z = 0, 0, 0
        
        for cand_bin in sorted_peaks:
            # 1. Quick Spatial Check for this specific peak
            cand_range = cand_bin * self.range_res
            
            # Use the 8-channel data for THIS bin to find its angle
            ch_cand = dynamic_data[cand_bin, :]
            S_cand = np.array([[ch_cand[3], ch_cand[1]], [ch_cand[2], ch_cand[0]], [ch_cand[7], ch_cand[5]], [ch_cand[6], ch_cand[4]]])

            om_az = np.angle(np.sum(S_cand[:, 0] * np.conj(S_cand[:, 1])))
            om_el = np.angle(np.sum(S_cand[0:3, :] * np.conj(S_cand[1:4, :])))
            
            az_cand = np.arcsin(np.clip(om_az / np.pi, -1.0, 1.0))
            el_cand = np.arcsin(np.clip(om_el / np.pi, -1.0, 1.0))

            self.output_dict["azimuth"] = az_cand
            self.output_dict["elevation"] = el_cand

            # 2. Project this candidate to World Coordinates
            Pr_c = np.array([cand_range * np.sin(az_cand) * np.cos(el_cand),
                            cand_range * np.cos(az_cand) * np.cos(el_cand),
                            cand_range * np.sin(el_cand)])
            Pb_c = np.dot(self.R, Pr_c) + self.T
            
            # 3. TEST: Is this candidate actually in the room?
            zone_name, is_valid = self.evaluate_spatial_zone(Pb_c[0], Pb_c[1], Pb_c[2])
            if is_valid and zone_name != "Out of Bounds (Ghost)":

                cand_micro_state = "STABLE"

                # --- VITAL CONTENT GATING ---
                vital_multiplier = 0.1
                if (self.frame_count - self.warmup_frames) > self.spectral_frames:
                    cand_history = self.spectral_history[cand_bin, :]
                    cand_history_safe = np.where(cand_history == 0, 1e-10 + 1e-10j, cand_history)
                    
                    # 1. Extract and Unwrap Phase
                    cand_phase = np.unwrap(np.angle(cand_history_safe))
                    
                    # 2. Calculate Physical Displacement
                    # At 60GHz, lambda = 5mm. Displacement = (phase_shift * lambda) / (4 * pi)
                    phase_ptp = np.ptp(cand_phase)
                    displacement_mm = (phase_ptp * 5.0) / (4.0 * np.pi)
                    
                    # 3. Calculate Phase Variance (First Derivative)
                    # High variance catches jerky, non-sinusoidal micro-motions (typing, fidgeting)
                    phase_diff = np.diff(cand_phase)
                    phase_var = np.var(phase_diff)
                    # print(f'Cand {cand_bin}: phase_var={phase_var:.2f}')
                    
                    # --- MOTION EVALUATION GATES ---
                    
                    # GATE A: MACRO-MOTION (Postural Shifts)
                    if displacement_mm > 15.0:
                        # They moved more than 15mm. The phase is too tangled for a clean vital FFT.
                        # They are clearly active, so reward the candidate, but skip the spectrum math.
                        vital_multiplier = 0.9 
                        # print(f'Cand {cand_bin}: MACRO-MOTION ({displacement_mm:.1f}mm)')
                        cand_micro_state = "MACRO_PHASE"

                    # GATE B: MICRO-MOTION (Fidgeting / Talking)
                    elif phase_var > 0.3: 
                        # The overall displacement is small, but it's erratic. 
                        # 0.25 rad^2 is a tunable threshold for "jagged" movement.
                        vital_multiplier = 0.7
                        # print(f'Cand {cand_bin}: MICRO-MOTION (var={phase_var:.2f})')
                        cand_micro_state = "MICRO_PHASE"

                    # GATE C: STILLNESS (Evaluate Vitals)
                    else:
                        # The target is still enough to extract a clean breathing rhythm.
                        detrended_phase = signal.detrend(cand_phase)
                        window = np.hanning(self.spectral_frames)
                        windowed_phase = detrended_phase * window
                        
                        # Slow-Time FFT
                        fft_result = np.fft.rfft(windowed_phase)
                        fft_mag = np.abs(fft_result)
                        freqs = np.fft.rfftfreq(self.spectral_frames, d=(1.0/FRAME_RATE))
                        
                        # Define Biological Bands
                        # Expanded to 0.7 Hz (42 BPM) to catch heavy/distress breathing
                        vital_band_mask = (freqs >= 0.15) & (freqs <= 0.7)
                        
                        # Total dynamic spectrum (ignoring DC/VLF baseline wander below 0.15Hz)
                        eval_band_mask = (freqs >= 0.15) & (freqs <= 3.0) 
                        
                        vital_energy = np.sum(fft_mag[vital_band_mask])
                        total_energy = np.sum(fft_mag[eval_band_mask])
                        
                        # Signal Quality Index (SQI): Ratio of vital energy to total energy
                        sqi = vital_energy / (total_energy + 1e-6)
                        
                        if sqi > 0.45:
                            vital_multiplier = 1.0  # Clean, dominant breathing rhythm
                            # peak_bin = np.where(vital_band_mask)[0][np.argmax(fft_mag[vital_band_mask])]
                            # breathing_rate_hz = freqs[peak_bin]
                        elif sqi > 0.25:
                            vital_multiplier = 0.5  # Weak breathing, or buried in slight noise
                        else:
                            vital_multiplier = 0.05 # Dead space (just static clutter noise)
                            cand_micro_state = "DEAD_SPACE"
                
                valid_candidates.append({
                    'bin': cand_bin,
                    'x': Pb_c[0], 'y': Pb_c[1], 'z': Pb_c[2],
                    'azimuth': az,
                    'elevation': el,
                    'mag': dynamic_mag_profile[cand_bin],
                    'zone': zone_name,
                    'vital_mult': vital_multiplier,
                    'micro_state': cand_micro_state
                })
                
        # --- CANDIDATE SCORING ---
        if valid_candidates:
            is_valid_point = True
            max_mag = max(c['mag'] for c in valid_candidates)

            #  Normal scoring (with tethering)
            best_cand, best_score = self._score_candidates(valid_candidates, max_mag, use_tethering=False)
            
            # # Periodic reassessment: score WITHOUT tethering every N frames
            # self.frames_since_reassess += 1
            # if self.track_x is not None and self.frames_since_reassess >= self.reassess_interval:
            #     self.frames_since_reassess = 0
            #     untethered_best, untethered_score = self._score_candidates(valid_candidates, max_mag, use_tethering=False)
                
            #     # If the untethered winner is significantly better, switch to it
            #     if untethered_best['bin'] != best_cand['bin'] and untethered_score > best_score + 0.2:
            #         best_cand = untethered_best
                    
            #         # Reset track so the smoothing starts fresh on the new target
            #         self.track_x, self.track_y, self.track_z = None, None, None
            #         self.coord_buffer.clear()
            #         self.track_confidence = 0

            final_peak_bin = best_cand['bin']
            self.current_active_zone = best_cand['zone']

            # Extract the winning micro-motion state
            self.current_micro_state = best_cand.get('micro_state', 'STABLE')

            raw_x, raw_y, raw_z = best_cand['x'], best_cand['y'], best_cand['z']
            raw_z = np.clip(raw_z, 0.05, 1.8)

            # --- TELEPORTATION GUARD ---
            # If we have an existing track and the new candidate is too far,
            # reject it — let the state machine handle exit via apnea at old location
            if self.track_x is not None:
                jump_dist = np.sqrt((raw_x - self.track_x)**2 + (raw_y - self.track_y)**2)
                if jump_dist > 1.5:  # More than 1.5 meter jump
                    # Don't accept this candidate — fall through to apnea path
                    is_valid_point = False
                    final_peak_bin = None
                    is_jump = True
                        
        # Fallback: If no candidate is in the room, default to the loudest 
        # (This allows Step 3/4 to handle the 'Empty Room' logic)
        if final_peak_bin is None:
            dynamic_peak_bin = sorted_peaks[0]
        else:
            dynamic_peak_bin = final_peak_bin

        self.output_dict["Azimuth"] = best_cand['azimuth']
        self.output_dict["Elevation"] = best_cand['elevation']
        self.output_dict["Range"] = dynamic_peak_bin * self.range_res
 
        # ==========================================
        # Step 3: State Machine
        # ==========================================     
        ## 1. Determine the detection threshold with HYSTERESIS
        current_threshold = self.detection_threshold if not self.is_occupied else (self.detection_threshold * 0.75)  
        print(f"[Threshold] dynamic_mag={dynamic_mag_profile[dynamic_peak_bin]:.1f}, threshold={current_threshold}")
        
        # 2. Check if the peak we found in Step 2 is strong enough to be "Active"
        if not is_jump and dynamic_mag_profile[dynamic_peak_bin] >= self.detection_threshold:  # current_threshold:
            # --- THE ENTRY DEBOUNCE FILTER ---
            if not self.is_occupied:
                self.entry_frames += 1
                if self.entry_frames < self.frames_to_occupy:
                    return self.output_dict # Abort early, don't update coordinates yet
            
            # STATE 1: ACTIVE / BREATHING
            self.is_occupied = True
            self.apnea_frames = 0
            self.last_target_bin = dynamic_peak_bin   
            self.last_target_coords = (raw_x, raw_y, raw_z)      
            status = "Occupied (Breathing/Moving)"     
        
        # 3. If weak signal, check for Stillness (ONLY if previously occupied)
        elif self.is_occupied and self.last_target_bin is not None:
            # Check the last known zone
            if self.track_x is not None:
                curr_loc = (self.track_x, self.track_y, self.track_z)
            else:
                # Use the coordinates saved when we first saw them
                curr_loc = getattr(self, 'last_target_coords', (0,0,0))
                
            last_zone_name, _ = self.evaluate_spatial_zone(curr_loc[0], curr_loc[1], curr_loc[2])
            
            # --- ZONE GATING ---
            # Only perform Apnea check in monitored zones (Bed, Chair, etc)
            try:
                is_monitored_zone = LAYOUT[last_zone_name]["type"] == "monitor"
            except KeyError:
                is_monitored_zone = False

            # Look at the raw power at the specific distance we last saw the user
            current_raw_reflection = np.max(raw_mag_profile[self.last_target_bin - 1 : self.last_target_bin + 2])
            empty_bed_reflection = self.baseline_profile[self.last_target_bin]
            
            # If the reflection is still 'thicker' than an empty bed, they haven't left
            # print(f"[Apnea Check] raw={current_raw_reflection:.1f}, baseline={empty_bed_reflection:.1f}, margin={self.static_margin}, diff={current_raw_reflection - empty_bed_reflection:.1f}")
            if is_monitored_zone and current_raw_reflection > (empty_bed_reflection + self.static_margin):
                # S STATE 2: STILLNESS / MONITORING
                self.apnea_frames += 1
                
                apnea_limit = FRAME_RATE * 5 # 5 seconds
                if self.apnea_frames >= apnea_limit:
                    status = "Possible Apnea"
                else:
                    status = "Still / Monitoring..."
            else:
                # STATE 3: CONFIRMED EXIT 
                self.is_occupied = False
                self.apnea_frames = 0
                self.last_target_bin = None
                self.track_confidence = 0
                self.coord_buffer.clear()
                self.z_history.clear()
                self.track_x, self.track_y, self.track_z = None, None, None
                self.empty_room()
                
                # Instantly absorb the exit disturbances into the clutter map
                # This mathematically silences the room until a real person walks back in.
                self.clutter_map = corrected_data.copy()
                
                return self.output_dict
        else:
            # STATE 4: TRULY EMPTY
            self.empty_room()
            return self.output_dict

        # ==========================================
        # Step 4: Temporal Persistence (Hit/Miss)
        # ==========================================
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
                return self.output_dict

        # ==========================================
        # Step 5: Adaptive Smoothing & Actigraphy
        # ==========================================
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

                # Adaptive Deadband Logic
                # Check if the winning candidate's zone disagrees with where the track currently sits
                track_zone, _ = self.evaluate_spatial_zone(self.track_x, self.track_y, self.track_z)
                candidate_zone = self.current_active_zone  # Set from the winning candidate
                if track_zone != candidate_zone:
                    # Zone mismatch: the scoring is pulling us somewhere different.
                    # Use aggressive alpha to catch up quickly instead of being stuck.
                    adaptive_alpha = 0.4
                elif shift_distance < 0.12:
                    adaptive_alpha = 0.02  # Freeze the UI dot (just breathing)
                elif shift_distance < 0.50:
                    adaptive_alpha = 0.1   # Glide smoothly to new position
                else:
                    adaptive_alpha = 0.5   # Failsafe
                
                self.track_x = (adaptive_alpha * med_x) + ((1 - adaptive_alpha) * self.track_x)
                self.track_y = (adaptive_alpha * med_y) + ((1 - adaptive_alpha) * self.track_y)
                self.track_z = (adaptive_alpha * med_z) + ((1 - adaptive_alpha) * self.track_z)

            # X_b, Y_b, Z_b = self.track_x, self.track_y, self.track_z
            X_b, Y_b, Z_b = med_x, med_y, med_z  # Use median directly, skip EMA

            # --- Z-History Update for Fall Detection ---
            self.z_history.append(Z_b)
            if len(self.z_history) > self.z_history_size:  # Still stores 50 frames (2 seconds)
                self.z_history.pop(0)
                
            # --- The Short-Window Derivative (Velocity) ---  # TODO: Need to further work on this
            # Calculate velocity over the last 0.4 seconds to catch the peak fall speed
            # without diluting it over the full 2-second persistence buffer.
            velocity_window_frames = int(FRAME_RATE * 0.4) # e.g., 10 frames at 25 FPS
            
            if len(self.z_history) >= velocity_window_frames:
                # Compare current Z to the Z from 0.4 seconds ago
                dz = self.z_history[-1] - self.z_history[-velocity_window_frames]
                dt = velocity_window_frames * (1.0 / FRAME_RATE) 
                v_z = dz / dt  # Vertical velocity in m/s (negative means falling)
                # print(f"\nCurrent Z: {Z_b:.2f}m | Drop Speed: {v_z:.2f} m/s")
            else:
                v_z = 0.0
        else:
            self.empty_room()
            return self.output_dict

        # ==========================================
        # Step 6: Posture Logic & Motion Tagging
        # ==========================================
        # Re-evaluate the zone with the smoothed coordinates
        final_zone, _ = self.evaluate_spatial_zone(X_b, Y_b, Z_b)

        # --- ZONE DEBOUNCING (Majority Vote) ---
        self.zone_history.append(final_zone)
        if len(self.zone_history) > self.frames_to_confirm_zone:
            self.zone_history.pop(0)

        # Only update the UI if the buffer is full
        if len(self.zone_history) == self.frames_to_confirm_zone:
            # Find the most common zone in the recent history
            most_common_zone = max(set(self.zone_history), key=self.zone_history.count)
            self.current_stable_zone = most_common_zone
            
        final_zone = self.current_stable_zone

        # Posture Recognition logic
        if Z_b > STANDING_THRESHOLD:
            posture = "Standing"
        elif Z_b > SITTING_THRESHOLD:
            posture = "Sitting"
        else:
            posture = "Lying Down"

        ### Unified Motion Tagging Hierarchy
        if "Apnea" in status:
            motion_str = "Static"
            
        # 1. Look for massive spatial coordinate changes
        elif self.motion_level > RESTLESS_MAX: 
            motion_str = "Major Movement"
            
        # 2. Look for large in-place phase displacements (rolling over in bed)
        elif getattr(self, 'current_micro_state', 'STABLE') == "MACRO_PHASE" and ("Bed" in self.current_active_zone or "Chair" in self.current_active_zone):
            motion_str = "Postural Shift"

        # 3. Look for moderate spatial drift
        elif self.motion_level > REST_MAX: 
            motion_str = "Restless/Shifting"
            
        # 4. Look for sub-millimeter phase chaos (typing, talking, twitching)
        elif getattr(self, 'current_micro_state', 'STABLE') == "MICRO_PHASE":
            motion_str = "Restless/Fidgeting"
            
        # 5. Signal is clean and coordinates are locked. Safe to read vitals.
        else:
            motion_str = "Resting/Breathing"
        
        # Contextual Overrides
        if final_zone in ["Out of Bounds (Ghost)", "Ignored"]:
            # The math tried to drag the dot outside the wall. Instantly kill the track.
            self.is_occupied = False
            self.last_target_bin = None
            return self.output_dict
            
        elif final_zone == "Floor / Transit":
            # The radar sees them, but they are not in bed or any "monitor"-type zone
            status = status.replace("Occupied", "In the Room") 
            
            # FALL DETECTION GATE
            if Z_b <= self.fall_threshold_z or self.is_fallen:
                # If they dropped fast OR they were already flagged as fallen
                # if v_z <= self.fall_velocity_threshold or self.is_fallen:
                self.is_fallen = True
                self.fall_persistence_frames += 1
                
                # 1. Height Score (Closer to floor = worse)
                h_score = max(0.0, (self.fall_threshold_z - Z_b) / self.fall_threshold_z) * 100.0
                
                # 2. Time Score (Longer on floor = worse) 
                # They must stay down for the full 2 seconds (50 frames) to hit 100%
                p_score = min(100.0, (self.fall_persistence_frames / self.z_history_size) * 100.0) 
                
                fall_confidence = (0.4 * h_score) + (0.6 * p_score)
                
                # Require high confidence (meaning they fell fast AND stayed down)
                if fall_confidence > 60.0:
                    status = "CRITICAL: Fall Detected!"
                    posture = "Fallen"
            else:
                # They are low, but lowered slowly (e.g., sitting on the floor safely)
                # Slowly decrement the persistence so it doesn't instantly clear
                self.fall_persistence_frames = max(0, self.fall_persistence_frames - 2)
        else:
            # They are back up (above 40cm). Reset fall logic.
            self.is_fallen = False
            self.fall_persistence_frames = 0
            
        # ==========================================
        # Step 7: Confidence Index Generation
        # ==========================================
        # Occupancy Confidence
        temporal_conf = (self.track_confidence / self.confidence_threshold) * 100.0
        signal_conf = 100.0
        
        if "Breathing" in status:
            margin = (dynamic_mag_profile[dynamic_peak_bin] - self.detection_threshold) / self.detection_threshold
            signal_conf = min(100.0, 50.0 + (margin * 100.0))
            if self.motion_level > 0.05:
                signal_conf = 100.0 
        elif "Apnea" in status:
            margin = (current_raw_reflection - empty_bed_reflection - self.static_margin) / self.static_margin
            signal_conf = min(100.0, 50.0 + (margin * 100.0))
            
        occ_confidence = (0.6 * temporal_conf) + (0.4 * signal_conf)
        if self.miss_counter > 0:
            occ_confidence = max(0.0, occ_confidence - ((self.miss_counter / self.miss_allowance) * 100.0))

        # Posture Confidence
        if posture == "Standing":
            dist_to_thresh = abs(Z_b - STANDING_THRESHOLD)
        elif posture == "Sitting":
            dist_to_thresh = min(abs(Z_b - SITTING_THRESHOLD), abs(STANDING_THRESHOLD - Z_b))
        elif posture == "Fallen":
            dist_to_thresh = abs(self.fall_threshold_z - Z_b) # Custom for fallen
        else: 
            dist_to_thresh = abs(SITTING_THRESHOLD - Z_b)
        
        base_posture_conf = min(100.0, 50.0 + ((dist_to_thresh / 0.125) * 50.0))
        motion_penalty = min(40.0, self.motion_level * 200.0) 
        posture_confidence = max(50.0, base_posture_conf - motion_penalty)

        # ==========================================
        # Step 8: Zone Timer Update
        # ==========================================
        now = self.frame_count / FRAME_RATE
        self._update_zone_timer(final_zone, is_valid_point, now)

        duration_str = "--"
        if self.zone_timer_zone is not None and self.zone_timer_start is not None:
            duration_str = self._format_duration(now - self.zone_timer_start)

        # Output
        self.output_dict = {
            "X": X_b,
            "Y": Y_b,
            "Z": Z_b,
            "Range": self.output_dict["Range"],
            "Azimuth": self.output_dict["Azimuth"],
            "Elevation": self.output_dict["Elevation"],
            "final_bin": dynamic_peak_bin,
            "zone": final_zone,
            "status": status,
            "occ_confidence": occ_confidence,
            "posture_confidence": posture_confidence,
            "posture": posture,
            "motion_str": motion_str,
            "duration_str": duration_str,
            "fall_confidence": fall_confidence,
            "spectral_history": self.spectral_history,
        }
        return self.output_dict

# ==========================================
# 3. GUI Visualizer (dashboard)
# ==========================================
class Visualizer:
    """
    Dashboard-style visualizer for room occupancy / posture / motion monitoring.
    Keeps the same update(...) interface as your current class.
    """

    def __init__(self, act_pipeline, resp_pipeline=None, history_len=None):
        self.p = act_pipeline
        self.resp_pipeline = resp_pipeline
        self.resp_window_sec = RESP_WINDOW_SEC if 'RESP_WINDOW_SEC' in globals() else 30
        self.history_len = history_len if history_len is not None else int(self.resp_window_sec * FRAME_RATE)

        # -----------------------------
        # Theme
        # -----------------------------
        self.FIG_BG = FIG_BG
        self.PANEL_BG = PANEL_BG
        self.CARD_BG = CARD_BG
        self.CARD_OK = CARD_OK
        self.CARD_WARN = CARD_WARN
        self.CARD_ALERT = CARD_ALERT

        self.TEXT = TEXT
        self.SUBTEXT = SUBTEXT
        self.GRID = GRID

        self.BED = BED
        self.CHAIR = CHAIR
        self.MONITOR = MONITOR
        self.IGNORE = IGNORE

        self.OCCUPANT = OCCUPANT
        self.ROOM_EDGE = ROOM_EDGE
        self.FOV = FOV
        self.RADAR = RADAR
        self.ORIGIN = ORIGIN

        plt.ion()

        # -----------------------------
        # Figure layout
        # -----------------------------
        self.fig = plt.figure(figsize=(15.5, 8.4), facecolor=self.FIG_BG)

        # Main layout: Left (Cards) | Middle (Room Map) | Right (Plots)
        outer = self.fig.add_gridspec(
            1, 3,
            width_ratios=[0.55, 1.0, 1.20],
            wspace=0.25
        )

        # Left layout: Vertically stacked cards
        left_gs = outer[0].subgridspec(
            3, 1,
            hspace=0.25
        )
        self.ax_occ = self.fig.add_subplot(left_gs[0])
        self.ax_posture = self.fig.add_subplot(left_gs[1])
        self.ax_system = self.fig.add_subplot(left_gs[2])

        # Middle layout: Room Map
        self.ax = self.fig.add_subplot(outer[1])

        # Right layout: Vertically stacked plots
        right_gs = outer[2].subgridspec(
            3, 1,
            height_ratios=[1.5, 0.8, 1.2],
            hspace=0.35
        )
        self.ax_live_resp = self.fig.add_subplot(right_gs[0])  # live breathing signal
        self.ax_rr_trend = self.fig.add_subplot(right_gs[1], sharex=self.ax_live_resp)   # RR trend
        self.ax_trend = self.fig.add_subplot(right_gs[2])   # combined trends

        self.fig.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.06)

        # -----------------------------
        # Main room panel styling
        # -----------------------------
        self.ax.set_facecolor(self.PANEL_BG)
        self.ax.set_title("Real-Time Room Activity Monitoring", color=self.TEXT, fontsize=18, pad=20, fontweight="bold")
        self.ax.set_xlabel("Width (m)", color=self.TEXT, fontsize=11)
        self.ax.set_ylabel("Depth (m)", color=self.TEXT, fontsize=11)
        self.ax.set_aspect('equal', adjustable='box')

        self.ax.set_anchor('E')

        for spine in self.ax.spines.values():
            spine.set_color("#475569")

        self.ax.tick_params(colors=self.SUBTEXT, labelsize=10)
        self.ax.grid(True, linestyle='--', linewidth=0.45, alpha=0.18, color=self.GRID)

        room = LAYOUT.get("Room")
        if room:
            self.ax.set_xlim(room["x"][0] - 0.03, room["x"][1] + 0.06)
            self.ax.set_ylim(room["y"][0] - 0.03, room["y"][1] + 0.08)
        else:
            self.ax.set_xlim(-0.03, 2.70)
            self.ax.set_ylim(-0.03, 3.60)

        # -----------------------------
        # Card axes setup
        # -----------------------------
        for a in [self.ax_occ, self.ax_posture, self.ax_system]:
            a.set_facecolor(self.FIG_BG)
            a.set_xlim(0, 1)
            a.set_ylim(0, 1)
            a.axis("off")

        # -----------------------------
        # Trend panel styling
        # -----------------------------
        for trend_ax, title in [
                (self.ax_trend, "Motion / Activity")
            ]:
                trend_ax.set_facecolor(self.PANEL_BG)
                for spine in trend_ax.spines.values():
                    spine.set_color("#475569")
        trend_ax.tick_params(colors=self.SUBTEXT, labelsize=9)
        trend_ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.35, color=self.GRID)
        trend_ax.set_title(title, color=self.TEXT, fontsize=12, pad=10, fontweight="bold")
        trend_ax.set_xlabel("Time (s)", color=self.TEXT, fontsize=10)
        trend_ax.set_ylabel("Normalized", color=self.TEXT, fontsize=10)

        # -----------------------------
        # Histories
        # -----------------------------
        self.motion_hist = deque(maxlen=history_len)
        self.motion_smooth_hist = deque(maxlen=history_len)
        self.occ_hist = deque(maxlen=history_len)
        self.posture_hist = deque(maxlen=history_len)
        self.activity_hist = deque(maxlen=history_len)
        self.fall_hist = deque(maxlen=history_len)
        self.occ_fill = None

        # -----------------------------
        # Static environment
        # -----------------------------
        self.zone_patches = {}
        self.zone_labels = {}
        self.room_patch = None
        self.fov_patch = None
        self.radar_artist = None
        self.origin_artist = None

        self._draw_static_environment()

        # -----------------------------
        # Dynamic occupant artists
        # -----------------------------
        self.confidence_glow = Circle((0, 0), radius=0.01, color=self.OCCUPANT, alpha=0.0, zorder=6)
        self.occupant_circle = Circle((0, 0), radius=0.05, color=self.OCCUPANT, alpha=1.0, zorder=7)
        self.ax.add_patch(self.confidence_glow)
        self.ax.add_patch(self.occupant_circle)

        self.occupant_label = self.ax.text(
            0, 0, "", color=self.TEXT, fontsize=9, zorder=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="#0F172A", ec="none", alpha=0.85)
        )

        self._hide_occupant()

        # -----------------------------
        # Trend lines
        # -----------------------------
        self.occ_line, = self.ax_trend.plot([], [], linewidth=2.0, color="#38BDF8", label="Occupancy conf.")
        self.posture_line, = self.ax_trend.plot([], [], linewidth=2.0, color="#F59E0B", label="Posture conf.")
        self.motion_smooth_line, = self.ax_trend.plot([], [], linewidth=2.0, linestyle="--", color="#22C55E", label="Motion")
        self.fall_line, = self.ax_trend.plot([], [], linewidth=2.0, color="#EF4444", label="Fall conf.")

        leg = self.ax_trend.legend(frameon=True, fontsize=9, loc="upper right")
        leg.get_frame().set_facecolor("#0F172A")
        leg.get_frame().set_edgecolor("#334155")
        for txt in leg.get_texts():
            txt.set_color(self.TEXT)

        # Fall detection thresholds
        self.fall_warn_threshold = 50.0
        self.fall_alert_threshold = 80.0

        # Initial cards
        self._draw_card(
            self.ax_occ, "Occupancy",
            [("Zone", "--"), ("State", "Initializing"), ("Confidence", "--"), ("Duration", "--")],
            facecolor=self.CARD_BG
        )
        self._draw_card(
            self.ax_posture, "Posture & Motion",
            [("Posture", "--"), ("Posture conf.", "--"), ("Height", "--"), ("Motion", "--")],
            facecolor=self.CARD_BG
        )
        self._draw_card(
            self.ax_system,
            "System / Safety",
            [
                ("Radar", "Online"),
                ("Tracking", "Waiting ..."),
                ("Fall state", "Normal"),
                ("Fall conf.", "--"),
            ],
            facecolor=self.CARD_BG
        )

        self._style_trend_axis(self.ax_trend, "Confidence: Occupancy / Posture / Motion / Fall Trends", "Normalized")

        self.resp_window_sec = RESP_WINDOW_SEC if 'RESP_WINDOW_SEC' in globals() else 30
        self.time_axis_seconds = np.linspace(-self.resp_window_sec, 0, self.resp_window_sec * FRAME_RATE)

        self.line_resp, = self.ax_live_resp.plot(self.time_axis_seconds, np.zeros_like(self.time_axis_seconds), color=self.OCCUPANT, linewidth=1.5)
        self.line_rr, = self.ax_rr_trend.plot(self.time_axis_seconds, np.zeros_like(self.time_axis_seconds), color=self.TEXT, linewidth=1.5)

        self.scatter_inhale = self.ax_live_resp.scatter([], [], color='green', marker='^', zorder=5)
        self.scatter_exhale = self.ax_live_resp.scatter([], [], color='red', marker='v', zorder=5)

        self.apnea_fill = None

        self.apnea_text_ui = self.ax_live_resp.text(
            0.02, 0.92, "Breathing: Unknown", 
            transform=self.ax_live_resp.transAxes, 
            color=self.TEXT, fontsize=11, fontweight="bold", va="top"
        )
        self.cycle_text_ui = self.ax_rr_trend.text(
            0.02, 0.85, "RR: 0.0", 
            transform=self.ax_rr_trend.transAxes, 
            color=self.SUBTEXT, fontsize=10, fontweight="bold", va="top"
        )
        
        for base_ax, title in [(self.ax_live_resp, "Live Breathing Signal"), (self.ax_rr_trend, "Respiration Rate")]:
            base_ax.set_facecolor(self.PANEL_BG)
            for spine in base_ax.spines.values(): spine.set_color("#475569")
            base_ax.tick_params(colors=self.SUBTEXT, labelsize=9)
            base_ax.grid(True, linestyle='--', linewidth=0.55, alpha=0.28, color=self.GRID)
            if title: base_ax.set_title(title, color=self.TEXT, fontsize=11, fontweight="bold", pad=5)
            
        self.ax_live_resp.set_xlim(-self.resp_window_sec, 0)
        self.ax_rr_trend.set_xlim(-self.resp_window_sec, 0)
        self.ax_live_resp.set_ylim(-3.14, 3.14) 
        self.ax_rr_trend.set_ylim(0, 40) 
        
        plt.setp(self.ax_live_resp.get_xticklabels(), visible=False)

    # =========================================================
    # Helpers
    # =========================================================
    def _compact_state_text(self, state_name):
        s = str(state_name)
        s_low = s.lower()

        if "breathing" in s_low and "moving" in s_low:
            return "Occupied"
        if "no occupant" in s_low:
            return "Empty"
        if len(s) > 20:
            return s[:20] + "..."
        return s

    def _draw_card(self, ax, title, lines, facecolor=None):
        ax.cla()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_facecolor(self.FIG_BG)

        bg = facecolor if facecolor is not None else self.CARD_BG

        card = FancyBboxPatch(
            (0.03, 0.06), 0.94, 1,
            boxstyle="round,pad=0.02,rounding_size=0.035",
            linewidth=0,
            facecolor=bg,
            alpha=0.98
        )
        ax.add_patch(card)

        ax.text(0.04, 0.84, title, fontsize=12, fontweight="bold",
        color=self.TEXT, va="center", ha="left", clip_on=False)

        row_y = [0.66, 0.48, 0.30, 0.14]
        for (label, value), y in zip(lines, row_y):
            ax.text(0.04, y, str(label), fontsize=9, color=self.SUBTEXT,
                    va="center", ha="left", clip_on=False)
            ax.text(0.96, y, str(value), fontsize=9, color=self.TEXT,
                    va="center", ha="right", fontweight="bold", clip_on=False)

    def _zone_color(self, zone_type):
        if zone_type == "bed":
            return self.BED
        if zone_type == "chair":
            return self.CHAIR
        if zone_type == "monitor":
            return self.MONITOR
        if zone_type == "ignore":
            return self.IGNORE
        return "#64748B"

    def _draw_static_environment(self):
        radar_x = None
        radar_y = None

        # Draw room / objects
        for name, bounds in LAYOUT.items():
            btype = bounds["type"]

            if btype == "sensor":
                radar_x = self.p.radar_x
                radar_y = self.p.radar_y
                continue

            x_min, x_max = bounds["x"]
            y_min, y_max = bounds["y"]
            width = x_max - x_min
            height = y_max - y_min

            if btype == "boundary":
                self.room_patch = patches.Rectangle(
                    (x_min, y_min), width, height,
                    linewidth=2.2,
                    edgecolor=self.ROOM_EDGE,
                    facecolor=(1, 1, 1, 0.03),
                    zorder=1
                )
                self.ax.add_patch(self.room_patch)

            elif btype in ["bed", "chair", "monitor"]:
                color = self._zone_color(btype)
                rect = patches.Rectangle(
                    (x_min, y_min), width, height,
                    linewidth=1.5,
                    edgecolor=color,
                    facecolor=color,
                    alpha=0.28,
                    zorder=2
                )
                self.ax.add_patch(rect)
                self.zone_patches[name] = rect

                label = self.ax.text(
                    x_min + width / 2, y_min + height / 2,
                    name,
                    ha='center', va='center',
                    color=self.TEXT,
                    fontsize=11,
                    fontweight='bold' if btype in ["bed", "chair"] else "normal",
                    zorder=3
                )
                self.zone_labels[name] = label

                if btype == "bed":
                    m_x = bounds.get("margin_x", [0.2, 0.2])
                    m_y = bounds.get("margin_y", [0.2, 0.2])
                    inner = patches.Rectangle(
                        (x_min + m_x[0], y_min + m_y[0]),
                        width - m_x[0] - m_x[1],
                        height - m_y[0] - m_y[1],
                        fill=False,
                        edgecolor="#22C55E",
                        linestyle='--',
                        linewidth=1.0,
                        alpha=0.35,
                        zorder=3
                    )
                    self.ax.add_patch(inner)

            elif btype == "ignore":
                rect = patches.Rectangle(
                    (x_min, y_min), width, height,
                    linewidth=1.5,
                    edgecolor=self.IGNORE,
                    facecolor=self.IGNORE,
                    alpha=0.18,
                    hatch='//',
                    zorder=2
                )
                self.ax.add_patch(rect)
                self.zone_patches[name] = rect
                self.zone_labels[name] = self.ax.text(
                    x_min + width / 2, y_min + height / 2,
                    name,
                    ha='center', va='center',
                    color=self.TEXT, fontsize=10, zorder=3
                )

        # Radar + FoV
        if radar_x is not None and radar_y is not None:
            self.radar_artist, = self.ax.plot(
                radar_x, radar_y, '^',
                markersize=10, color=self.RADAR, zorder=5
            )
            self.ax.text(radar_x, radar_y + 0.10, "Radar", color=self.TEXT, fontsize=9, ha='center', zorder=5)

            fov_deg = LAYOUT["Radar"]["fov_deg"]

            # Yaw=0 looks toward +Y => Matplotlib angle basis from +X, so convert
            center_deg = 90 - self.p.yaw_deg
            theta1 = center_deg - fov_deg / 2
            theta2 = center_deg + fov_deg / 2

            self.fov_patch = Wedge(
                center=(radar_x, radar_y),
                r=5.25,
                theta1=theta1,
                theta2=theta2,
                facecolor=self.FOV,
                edgecolor=self.FOV,
                linewidth=1.4,
                linestyle='--',
                alpha=0.08,
                zorder=2
            )
            self.ax.add_patch(self.fov_patch)

        # Origin
        self.origin_artist, = self.ax.plot(
            0, 0, 'o',
            color=self.ORIGIN, markersize=8,
            markeredgecolor='black', zorder=5
        )
        self.ax.text(0.08, -0.04, "Origin", color=self.SUBTEXT, fontsize=9)

    def _hide_occupant(self):
        self.occupant_circle.set_visible(False)
        self.confidence_glow.set_visible(False)
        self.occupant_label.set_visible(False)

    def _show_occupant(self, x, y, occ_confidence, zone_name, marker_color=None):
        color = marker_color if marker_color is not None else self.OCCUPANT

        # Normalize confidence to 0-1
        occ_norm = max(0.0, min(1.0, occ_confidence / 100.0))

        self.occupant_circle.center = (x, y)
        self.occupant_circle.radius = 0.045 + 0.03 * occ_norm
        self.occupant_circle.set_facecolor(color)
        self.occupant_circle.set_edgecolor(color)
        self.occupant_circle.set_visible(True)

        self.confidence_glow.center = (x, y)
        self.confidence_glow.radius = 0.12 + 0.18 * occ_norm
        self.confidence_glow.set_facecolor(color)
        self.confidence_glow.set_alpha(0.12 + 0.18 * occ_norm)
        self.confidence_glow.set_visible(True)

    def _normalize_motion(self, motion_str):
        s = (motion_str or "").lower()
        if "still" in s:
            return 0.10
        if "breath" in s:
            return 0.35
        if "rest" in s:
            return 0.25
        if "move" in s:
            return 0.75
        if "active" in s:
            return 0.90
        return 0.20

    def _status_color(self, zone_name, state_name):
        zs = (zone_name or "").lower()
        ss = (state_name or "").lower()

        if "apnea" in ss or "alert" in ss:
            return self.CARD_ALERT
        if "floor" in zs or "out of bounds" in zs or "uncertain" in ss:
            return self.CARD_WARN
        if "occupied" in ss:
            return self.CARD_OK
        return self.CARD_BG

    def _tracking_quality(self, occ_confidence, posture_confidence, is_inside):
        if not is_inside:
            return "Lost"
        avg = 0.5 * occ_confidence + 0.5 * posture_confidence
        if avg >= 80:
            return "Good"
        if avg >= 55:
            return "Fair"
        return "Weak"

    def _update_zone_highlight(self, active_zone_name):
        # reset
        for name, patch in self.zone_patches.items():
            zone_type = LAYOUT[name]["type"]
            color = self._zone_color(zone_type)
            patch.set_facecolor(color)
            patch.set_edgecolor(color)
            patch.set_linewidth(1.3)

            if zone_type == "ignore":
                patch.set_alpha(0.18)
            else:
                patch.set_alpha(0.24)

        # highlight active
        if active_zone_name and "out of bounds" not in active_zone_name.lower() and "floor" not in active_zone_name.lower():
            for name, patch in self.zone_patches.items():
                if name in active_zone_name:
                    patch.set_alpha(0.42)
                    patch.set_edgecolor(self.OCCUPANT)
                    patch.set_linewidth(2.8)
                    break

    def _update_trends(self):
        curr_len = len(self.occ_hist)
        if curr_len == 0:
            return
            
        elapsed_sec = curr_len / float(FRAME_RATE)
        x = np.linspace(-elapsed_sec, 0, curr_len)

        self.occ_line.set_data(x, list(self.occ_hist))
        self.posture_line.set_data(x, list(self.posture_hist))
        self.motion_smooth_line.set_data(x, list(self.motion_smooth_hist))
        self.fall_line.set_data(x, list(self.fall_hist))

        self.ax_trend.set_xlim(-self.resp_window_sec, 0)
        self.ax_trend.set_ylim(0, 1.05)
        self.ax_trend.margins(x=0.02)

        if self.occ_fill is not None:
            self.occ_fill.remove()

        if len(x) > 0:
            occ_band = 0.10 * np.asarray(self.activity_hist, dtype=float)
            self.occ_fill = self.ax_trend.fill_between(
                x, 0, occ_band,
                step='pre',
                alpha=0.10,
                color="#38BDF8"
            )

    def _style_trend_axis(self, ax, title, ylabel):
        ax.set_facecolor(self.PANEL_BG)
        for spine in ax.spines.values():
            spine.set_color("#475569")
        ax.tick_params(colors=self.SUBTEXT, labelsize=9)
        ax.grid(True, linestyle='--', linewidth=0.55, alpha=0.28, color=self.GRID)
        ax.set_title(title, color=self.TEXT, fontsize=12, pad=10, fontweight="bold")
        ax.set_xlabel("Frames", color=self.TEXT, fontsize=10)
        ax.set_ylabel(ylabel, color=self.TEXT, fontsize=10)
        ax.set_ylim(0, 1.05)

    def _style_reserved_axis(self, ax, title):
        ax.set_facecolor(self.PANEL_BG)
        for spine in ax.spines.values():
            spine.set_color("#475569")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, color=self.TEXT, fontsize=12, pad=10, fontweight="bold")



    def _fall_status(self, fall_confidence):
        fc = 0.0 if fall_confidence is None else float(fall_confidence)

        if fc >= self.fall_alert_threshold:
            return "Fall Detected", self.CARD_ALERT, "#EF4444"
        elif fc >= self.fall_warn_threshold:
            return "Fall Risk", self.CARD_WARN, "#F59E0B"
        else:
            return "Normal", self.CARD_BG, self.OCCUPANT

    # =========================================================
    # Main update
    # =========================================================
    def update(self, occ_out_dict, resp_dict=None, frames=1):
        """
        Update dashboard using the same API as your current visualizer.
        """
        # print(f'Height: {Z_b}, Zone: {zone_name}, Fall confidence: {fall_confidence}')
        room_limit_x = LAYOUT["Room"]["x"]
        room_limit_y = LAYOUT["Room"]["y"]

        # unpack the output dictionary
        X_b = occ_out_dict["X"]
        Y_b = occ_out_dict["Y"]
        Z_b = occ_out_dict["Z"]
        zone_name = occ_out_dict["zone"]
        # print(f"\n[Visualizer] zone_name: {zone_name}")
        state_name = occ_out_dict["status"]
        occ_confidence = occ_out_dict["occ_confidence"]
        posture_confidence = occ_out_dict["posture_confidence"]
        posture = occ_out_dict["posture"]
        motion_str = occ_out_dict["motion_str"]
        duration_str = occ_out_dict["duration_str"]
        fall_confidence = occ_out_dict["fall_confidence"]
        Range = occ_out_dict["Range"]
        # print(f"[Visualizer] Range: {Range}")
        Azimuth = occ_out_dict["Azimuth"]
        Elevation = occ_out_dict["Elevation"]
        
        # initialize variables
        is_inside = False
        valid_detection = False

        if X_b is not None and Y_b is not None:
            is_inside = (room_limit_x[0] <= X_b <= room_limit_x[1]) and (room_limit_y[0] <= Y_b <= room_limit_y[1])

        if X_b is not None and Y_b is not None and is_inside and "No Occupant" not in str(state_name):
            valid_detection = True

        # Fall detection check
        fall_state, fall_card_color, fall_marker_color = self._fall_status(fall_confidence)

        # Update zone highlighting
        self._update_zone_highlight(zone_name if valid_detection else None)

        # Update histories
        motion_value = self._normalize_motion(motion_str)
        occ_value = np.clip((occ_confidence or 0) / 100.0, 0, 1)
        posture_value = np.clip((posture_confidence or 0) / 100.0, 0, 1)
        activity_value = 1.0 if valid_detection else 0.0
        fall_value = np.clip((fall_confidence or 0) / 100.0, 0, 1)

        if len(self.motion_smooth_hist) == 0:
            motion_smooth = motion_value
        else:
            motion_smooth = 0.85 * self.motion_smooth_hist[-1] + 0.15 * motion_value

        for _ in range(frames):
            self.motion_hist.append(motion_value)
            self.motion_smooth_hist.append(motion_smooth)
            self.occ_hist.append(occ_value)
            self.posture_hist.append(posture_value)
            self.fall_hist.append(fall_value)
            self.activity_hist.append(activity_value)

        self._update_trends()

        # Update occupant marker
        if valid_detection:
            self._show_occupant(X_b, Y_b, occ_confidence, zone_name, fall_marker_color)
        else:
            self._hide_occupant()

        # Header
        tracking = self._tracking_quality(occ_confidence or 0, posture_confidence or 0, is_inside)
        state_header = "Normal"
        if "apnea" in str(state_name).lower():
            state_header = "Alert"
        elif not valid_detection:
            state_header = "Scanning"
        elif "floor" in str(zone_name).lower():
            state_header = "Caution"

        # Cards
        occ_card_color = self._status_color(zone_name, state_name)

        self._draw_card(
            self.ax_occ,
            "Occupancy",
            [
                ("Zone", str(zone_name)),
                ("State", self._compact_state_text(state_name)),
                ("Confidence", f"{int(occ_confidence)}%"),
                ("Duration", duration_str),
            ],
            facecolor=occ_card_color
        )

        z_text = "--" if Z_b is None else f"{Z_b:.2f} m"
        self._draw_card(
            self.ax_posture,
            "Posture & Motion",
            [
                ("Posture", str(posture)),
                ("Posture conf.", f"{int(posture_confidence)}%"),
                ("Height (Range)", f"{z_text} ({Range:.2f} m)"),
                ("Motion", str(motion_str)),
            ],
            facecolor=self.CARD_BG
        )

        self._draw_card(
            self.ax_system,
            "System",
            [
                ("Radar", "Online"),
                ("Tracking", tracking),
                ("Fall state", fall_state),
                ("Fall conf.", f"{int(fall_confidence)}%"),
            ],
            facecolor=fall_card_color
        )

        # Draw Respiratory Data if available
        if resp_dict is not None and resp_dict.get("confidence", 0) > 0:
            # 1. Update lines
            self.line_resp.set_ydata(resp_dict["live_signal"])
            self.line_rr.set_ydata(resp_dict["rr_history"])
            
            # 2. Update Scatters (Peaks)
            inhales = resp_dict["inhales"]
            exhales = resp_dict["exhales"]
            if len(inhales) > 0:
                self.scatter_inhale.set_offsets(np.c_[self.time_axis_seconds[inhales], resp_dict["live_signal"][inhales]])
            else:
                self.scatter_inhale.set_offsets(np.empty((0, 2)))
                
            if len(exhales) > 0:
                self.scatter_exhale.set_offsets(np.c_[self.time_axis_seconds[exhales], resp_dict["live_signal"][exhales]])
            else:
                self.scatter_exhale.set_offsets(np.empty((0, 2)))
                
            # 3. Update Apnea Shading
            if self.apnea_fill is not None:
                self.apnea_fill.remove()
                self.apnea_fill = None
            
            apnea_trace = resp_dict.get("apnea_trace", np.zeros_like(self.time_axis_seconds, dtype=bool))
            if np.any(apnea_trace):
                self.apnea_fill = self.ax_live_resp.fill_between(
                    self.time_axis_seconds, -3.14, 3.14,
                    where=apnea_trace,
                    color=self.CARD_ALERT, alpha=0.25, zorder=1
                )
                
            # 4. Update Text Overlays
            status_text = "Apnea Warning" if resp_dict.get("apnea_active") else f"Monitoring ({int(resp_dict['confidence'])}%)"
            self.apnea_text_ui.set_text(f"Breathing: {status_text} | Depth: {resp_dict.get('depth', 'unknown').title()}")
            self.apnea_text_ui.set_color("#EF4444" if resp_dict.get("apnea_active") else self.TEXT)
            
            self.cycle_text_ui.set_text(f"RR: {resp_dict['rr_current']:.1f} bpm | Cycle: {resp_dict['cycle_duration']:.1f}s")
        else:
            # Clear or grey-out the plots if confidence is lost or no data
            self.line_resp.set_ydata(np.zeros_like(self.time_axis_seconds))
            self.line_rr.set_ydata(np.zeros_like(self.time_axis_seconds))
            self.scatter_inhale.set_offsets(np.empty((0, 2)))
            self.scatter_exhale.set_offsets(np.empty((0, 2)))
            if self.apnea_fill is not None:
                self.apnea_fill.remove()
                self.apnea_fill = None
            self.apnea_text_ui.set_text("Breathing: Waiting/Unstable...")
            self.apnea_text_ui.set_color(self.SUBTEXT)
            self.cycle_text_ui.set_text("RR: 0.0")

        # Draw refresh
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

# ==========================================
# 4. Respiratory Pipeline
# ==========================================
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


# ==========================================
# 5. Main Application Loop
# ==========================================

if __name__ == "__main__":
    
    print("Starting Bed Occupation Visualizer...")

    # 1. Setup Queues
    state_q = multiprocessing.Queue()
    pt_fft_q = multiprocessing.Queue()
    
    # 2. Start Radar Process
    radar_process = RadarController(state_q=state_q, pt_fft_q=pt_fft_q)
    radar_process.start()
    
    # 3. Initialize Pipelines and Visualizer
    act_pipeline = ActivityPipeline(num_range_bins=RANGE_IDX_NUM, range_resolution=RANGE_RESOLUTION)
    resp_pipeline = RespiratoryPipeline(fps=FRAME_RATE, window_seconds=RESP_WINDOW_SEC)
    visualizer = Visualizer(act_pipeline, resp_pipeline)
    last_notified_state = ""
    
    print("Waiting for radar data...")
    frame_count = 0
    
    try:
        while plt.fignum_exists(visualizer.fig.number):
            try:
                # 1. Block and wait for at least one frame
                fft_frame = pt_fft_q.get(timeout=1.0) 
                occ_out_dict = act_pipeline.process_frame(fft_frame)
                frames_processed = 1
                
                # 2. Rapidly drain any EXTRA frames that piled up during the last screen draw
                while not pt_fft_q.empty():
                    try:
                        fft_frame = pt_fft_q.get_nowait()
                        occ_out_dict = act_pipeline.process_frame(fft_frame)
                        frames_processed += 1
                    except queue.Empty:
                        break # Queue is fully caught up!
                
                # 3. Respiratory Processing for Monitoring Zones
                resp_dict = None
                zone_name = occ_out_dict.get('zone', 'Unknown')
                
                # Identify if current position explicitly qualifies for breathing tracking
                is_monitor = False
                if zone_name in LAYOUT and LAYOUT[zone_name].get('type') == 'monitor':
                    is_monitor = True
                elif 'Bed' in zone_name:
                    is_monitor = True
                    
                if is_monitor and occ_out_dict.get('status') != "No Occupant":
                    try:
                        resp_dict = resp_pipeline.process(occ_out_dict, frames=frames_processed)
                    except Exception as e:
                        print(f"[Main] Respiratory Error: {e}")
                        resp_pipeline._reset_state()
                else:
                    if resp_pipeline.frames_since_present > 0:
                        print(f"[Main] Run off-zone reset. Clearing ghost memory.")
                    resp_pipeline._reset_state()

                # 4. Update the display exactly ONCE with the absolute freshest data
                visualizer.update(occ_out_dict, resp_dict, frames=frames_processed)
                
                # 4. Send Alert 
                if SEND_ALERT and occ_out_dict['status'] != last_notified_state and "Initializing" not in occ_out_dict['status'] and (occ_out_dict['posture'] == "Fallen" or 'Apnea' in occ_out_dict['status']):
                    print(f"State changed to: {occ_out_dict['status']}. Sending alert to Watch...")
                    send_watch_alert(occ_out_dict['status'])  # Notification to Apple iPhone/Watch  (PushOver)
                    last_notified_state = occ_out_dict['status']
            except queue.Empty:
                act_pipeline.empty_room()
                resp_pipeline._reset_state()
                visualizer.update(act_pipeline.output_dict, None)
                continue
            
    except KeyboardInterrupt:
        print("\nStopping Application (Ctrl+C detected)...")
    finally:
        print("\nCleaning up processes...")
        if radar_process.is_alive():
            radar_process.terminate()
            radar_process.join()
        plt.ioff()
    plt.close('all')
    print("Application Closed Gracefully.")
    