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
        "type": "bed", 
        "x": [0.0, 1.05], "y": [1.45, 3.5], "z": [0, 2.7],
        "margin_x": [0.3, 0.3], # 0.3 at the left/right
        "margin_y": [0.3, 0]  # 0.3 at the footer
    },
    "Desk": {
        "type": "monitor", # Standard bounding box, no sub-LAYOUT
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
TRACK_ALPHA = 0.05             # Coordinate smoothing factor (lower = smoother but slower to update)
FRAME_TO_CONFIRM_ZONE = 25    # Require 1 second (25 frames) of stability to change LAYOUT
BUFFER_SIZE = 25              # Buffer size for coordinate tracking
MISS_ALLOWANCE = 50           # Allow 50 frames of no detection before clearing the track

FALL_DETECTION_ENABLE = True
FALL_DURATION = 2 # seconds
FALL_THRESHOLD = 0.5
FALL_VELOCITY_THRESHOLD = 0

SITTING_THRESHOLD = 0.6 
STANDING_THRESHOLD = 1.1

# Motion thresholds
REST_MAX = 0.1
RESTLESS_MAX = 0.3

# Others
LOG_LEVEL = 10
NEED_SEND_TI_CONFIG = True
SEND_ALERT = False

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
        self.frame_count = 0
        self.warmup_frames = FRAME_RATE * 1 
        
        # Vital Gating History Buffer
        # Keep 5 seconds of complex data for all bins to evaluate biological motion
        self.vital_gate_frames = 10*FRAME_RATE 
        self.complex_history = np.zeros((self.num_range_bins, self.vital_gate_frames), dtype=complex)

        # 10 seconds of history for high-res breathing extraction
        self.spectral_frames = 10 * FRAME_RATE 
        self.spectral_history = np.zeros((self.num_range_bins, self.spectral_frames), dtype=complex)

        self.output_dict = {}
        self.empty_room()

    def _score_candidates(self, candidates, max_mag, use_tethering):
        best, best_s = None, -float('inf')
        for c in candidates:
            s = (c['mag'] / max_mag) * c['vital_mult']
            # Zone preference bonus
            if c['zone'] not in ('Floor / Transit', 'Out of Bounds (Ghost)'):
                s += 0.15
            # Spatial tethering (only when enabled AND track exists)
            if use_tethering and self.track_x is not None:
                xy_dist = np.sqrt((c['x'] - self.track_x)**2 + (c['y'] - self.track_y)**2)
                s -= min(0.4, xy_dist * 0.3)
                z_jump = abs(c['z'] - self.track_z)
                s -= min(0.2, z_jump * 0.5)
            if s > best_s:
                best_s = s
                best = c
        return best, best_s

    def empty_room(self):
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
        Evaluates a 3D point against the defined geofences.
        Returns: (Zone String, is_valid_target)
        """
        # 1. Global Boundary Check
        room = LAYOUT.get("Room")
        if room:
            if not (room["x"][0] <= x <= room["x"][1] and 
                    room["y"][0] <= y <= room["y"][1] and 
                    room["z"][0] <= z <= room["z"][1]):
                return "Out of Bounds (Ghost)", False
                    
        # 2. Interference Check (Are they in a 'Keep-Out' zone?)
        for name, bounds in LAYOUT.items():
            if bounds["type"] == "ignore":
                # Check if the X, Y, Z coordinates fall inside this ignore box
                in_x = bounds["x"][0] <= x <= bounds["x"][1]
                in_y = bounds["y"][0] <= y <= bounds["y"][1]
                in_z = bounds["z"][0] <= z <= bounds["z"][1]
                
                if in_x and in_y and in_z:
                    # The point is inside a noise source! Tell the pipeline to reject it.
                    return f"Ignored ({name})", False

        # 3. Dynamic Bed & Sub-Zone Check
        for name, bounds in LAYOUT.items():
            if bounds["type"] == "bed":
                # Check if the person is inside the overall 3D bed box
                if (bounds["x"][0] <= x <= bounds["x"][1] and
                    bounds["y"][0] <= y <= bounds["y"][1] and
                    bounds["z"][0] <= z <= bounds["z"][1]):
                    
                    # They are in the bed! Now calculate exactly where.
                    x_min, x_max = bounds["x"]
                    y_min, y_max = bounds["y"]
                    m_x = bounds.get("margin_x", [0.2, 0.2]) # Default to 20cm if not provided
                    m_y = bounds.get("margin_y", [0.2, 0.2])
                    
                    # Are they safely inside the center margins?
                    is_center_x = (x_min + m_x[0]) <= x <= (x_max - m_x[1])
                    is_center_y = (y_min + m_y[0]) <= y <= (y_max - m_y[1])
                    
                    if is_center_x and is_center_y:
                        return f"{name} - Center", True
                    elif x < (x_min + m_x[0]):
                        return f"{name} - Right Edge", True
                    elif x > (x_max - m_x[1]):
                        return f"{name} - Left Edge", True
                    elif y < (y_min + m_y[0]):
                        return f"{name} - Foot Edge", True 
                    elif y > (y_max - m_y[1]):
                        return f"{name} - Head Edge", True 
                    else:
                        return f"{name} - Corner", True

        # 4. Standard Furniture Check ("monitor")
        for name, bounds in LAYOUT.items():
            if bounds["type"] == "monitor":
                if (bounds["x"][0] <= x <= bounds["x"][1] and
                    bounds["y"][0] <= y <= bounds["y"][1] and
                    bounds["z"][0] <= z <= bounds["z"][1]):
                    return name, True
                    
        # 5. Fallback
        return "Floor / Transit", True

    def process_frame(self, fft_1d_data):
        # ==========================================
        # Step 1: Hardware Correction & Background
        # ==========================================
        self.frame_count += 1
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

            # 2. Project this candidate to World Coordinates
            Pr_c = np.array([cand_range * np.sin(az_cand) * np.cos(el_cand),
                            cand_range * np.cos(az_cand) * np.cos(el_cand),
                            cand_range * np.sin(el_cand)])
            Pb_c = np.dot(self.R, Pr_c) + self.T
            
            # 3. TEST: Is this candidate actually in the room?
            zone_name, is_valid = self.evaluate_spatial_zone(Pb_c[0], Pb_c[1], Pb_c[2])
            
            if is_valid and zone_name != "Out of Bounds (Ghost)":

                # --- VITAL CONTENT GATING ---
                vital_multiplier = 0.1
                if (self.frame_count - self.warmup_frames) > self.spectral_frames:
                    cand_history = self.spectral_history[cand_bin, :]
                    cand_history_safe = np.where(cand_history == 0, 1e-10 + 1e-10j, cand_history)
                    
                    # 1. Extract and Unwrap
                    cand_phase = np.unwrap(np.angle(cand_history_safe))
                    phase_ptp = np.ptp(cand_phase) # Peak-to-peak phase shift
                    
                    # 2. Detrend to remove slow posture shifts (DC drift)
                    detrended_phase = signal.detrend(cand_phase)
                    
                    # 3. Apply a Hanning window to prevent edge-effect noise in the FFT
                    window = np.hanning(self.spectral_frames)
                    windowed_phase = detrended_phase * window
                    
                    # 4. Perform the Slow-Time FFT
                    # rfft is highly optimized for purely real input arrays
                    fft_result = np.fft.rfft(windowed_phase)
                    fft_mag = np.abs(fft_result)
                    
                    # Calculate the frequency corresponding to each FFT bin
                    freqs = np.fft.rfftfreq(self.spectral_frames, d=(1.0/FRAME_RATE))
                    
                    # 5. Define the Biological Breathing Band
                    vital_band_mask = (freqs >= 0.15) & (freqs <= 0.5)
                    noise_band_mask = (freqs > 1.0) & (freqs <= 3.0) # High freq noise
                    
                    # Calculate energy inside the breathing band vs the noise floor
                    vital_energy = np.sum(fft_mag[vital_band_mask])
                    noise_energy = np.sum(fft_mag[noise_band_mask])
                    
                    # Find the peak frequency inside the vital band
                    if vital_energy > 0:
                        vital_bins = np.where(vital_band_mask)[0]
                        peak_bin = vital_bins[np.argmax(fft_mag[vital_band_mask])]
                        breathing_rate_hz = freqs[peak_bin]
                        
                        # Calculate Signal-to-Noise Ratio (SNR) for the breathing signal
                        # Add tiny epsilon to prevent divide by zero
                        vital_snr = vital_energy / (noise_energy + 1e-6) 
                        print(f'\n\ncand_bin: {cand_bin} - zone: {zone_name} - vital_snr: {vital_snr} - breathing_rate_hz: {breathing_rate_hz}')
                        if vital_snr > 1.0: 
                            vital_multiplier = 1.0  # Clean breathing confirmed
                            print(f'Candidate {cand_bin} in {zone_name} has clean breathing signal')
                        elif vital_snr > 0.6:
                            # Some rhythm is present, but noisy.
                            vital_multiplier = 0.5  
                            print(f'Candidate {cand_bin} in {zone_name} has active motion')
                        elif phase_ptp > 10.0:
                            # FFT logic fails on major motion (high phase noise floor).
                            # Fall back to time-domain logic: they are clearly active.
                            vital_multiplier = 0.9  # They are moving too much for a clean FFT, but clearly active
                            print(f'Candidate {cand_bin} in {zone_name} has active motion')
                        else:
                            vital_multiplier = 0.1  # Just static noise
                            print(f'Candidate {cand_bin} in {zone_name} is static')
                    else:
                        vital_multiplier = 0.05 # Dead space
                        print(f'Candidate {cand_bin} in {zone_name} is dead space')
                
                valid_candidates.append({
                    'bin': cand_bin,
                    'x': Pb_c[0], 'y': Pb_c[1], 'z': Pb_c[2],
                    'mag': dynamic_mag_profile[cand_bin],
                    'zone': zone_name,
                    'vital_mult': vital_multiplier
                })
                
        # --- CANDIDATE SCORING ---
        if valid_candidates:
            is_valid_point = True
            max_mag = max(c['mag'] for c in valid_candidates)

            #  Normal scoring (with tethering)
            best_cand, best_score = self._score_candidates(valid_candidates, max_mag, use_tethering=True)
            
            # Periodic reassessment: score WITHOUT tethering every N frames
            self.frames_since_reassess += 1
            if self.track_x is not None and self.frames_since_reassess >= self.reassess_interval:
                self.frames_since_reassess = 0
                untethered_best, untethered_score = self._score_candidates(valid_candidates, max_mag, use_tethering=False)
                
                # If the untethered winner is significantly better, switch to it
                if untethered_best['bin'] != best_cand['bin'] and untethered_score > best_score + 0.2:
                    best_cand = untethered_best
                    
                    # Reset track so the smoothing starts fresh on the new target
                    self.track_x, self.track_y, self.track_z = None, None, None
                    self.coord_buffer.clear()
                    self.track_confidence = 0

            final_peak_bin = best_cand['bin']
            self.current_active_zone = best_cand['zone']
            raw_x, raw_y, raw_z = best_cand['x'], best_cand['y'], best_cand['z']
            raw_z = np.clip(raw_z, 0.05, 1.8)

            # --- TELEPORTATION GUARD ---
            # If we have an existing track and the new candidate is too far,
            # reject it — let the state machine handle exit via apnea at old location
            if self.track_x is not None:
                jump_dist = np.sqrt((raw_x - self.track_x)**2 + (raw_y - self.track_y)**2)
                if jump_dist > 1.0:  # More than 1 meter jump
                    # Don't accept this candidate — fall through to apnea path
                    is_valid_point = False
                    final_peak_bin = None
                        
        # Fallback: If no candidate is in the room, default to the loudest 
        # (This allows Step 3/4 to handle the 'Empty Room' logic)
        if final_peak_bin is None:
            dynamic_peak_bin = sorted_peaks[0]
        else:
            dynamic_peak_bin = final_peak_bin

        self.output_dict["Range"] = dynamic_peak_bin * self.range_res
 
        # ==========================================
        # Step 3: State Machine
        # ==========================================       
        # 1. Check if the peak we found in Step 2 is strong enough to be "Active"
        print(f"[Threshold] dynamic_mag={dynamic_mag_profile[dynamic_peak_bin]:.1f}, threshold={self.detection_threshold}")
        if dynamic_mag_profile[dynamic_peak_bin] >= self.detection_threshold:
            # STATE 1: ACTIVE / BREATHING
            self.is_occupied = True
            self.last_target_bin = dynamic_peak_bin         
            status = "Occupied (Breathing/Moving)"     
        
        # 2. If the signal is weak, check if someone was PREVIOUSLY there (Stillness/Apnea)
        elif self.is_occupied and self.last_target_bin is not None:
            # Look at the raw power at the specific distance we last saw the user
            current_raw_reflection = raw_mag_profile[self.last_target_bin]
            empty_bed_reflection = self.baseline_profile[self.last_target_bin]
            
            # If the reflection is still 'thicker' than an empty bed, they haven't left
            print(f"[Apnea Check] raw={current_raw_reflection:.1f}, baseline={empty_bed_reflection:.1f}, margin={self.static_margin}, diff={current_raw_reflection - empty_bed_reflection:.1f}")
            if current_raw_reflection > (empty_bed_reflection + self.static_margin):
                # STATE 2: APNEA (Use safely buffered coordinates)
                if self.track_x is not None:
                    raw_x, raw_y, raw_z = self.track_x, self.track_y, self.track_z
                elif len(self.coord_buffer) > 0:
                    raw_x, raw_y, raw_z = self.coord_buffer[-1]
                else:
                    # No track data at all — can't maintain apnea state
                    self.is_occupied = False
                    self.last_target_bin = None
                    self.empty_room()
                    return self.output_dict

                status = "Still / Possible Apnea"
            else:
                # STATE 3: CONFIRMED EXIT (Power dropped to baseline)
                self.is_occupied = False
                self.last_target_bin = None
                self.track_confidence = 0
                self.coord_buffer.clear()
                self.z_history.clear() # Clear Z-history on exit
                self.track_x, self.track_y, self.track_z = None, None, None
                self.empty_room()

                # Force baseline to re-learn the truly empty room
                self.baseline_profile = raw_mag_profile.copy()  # Snapshot the empty state
                
                return self.output_dict
        else:
            # STATE 4: TRULY EMPTY
            self.empty_room()
            return self.output_dict

        # # ==========================================
        # # Step 4: Outlier Rejection (Movement Logic)
        # # ==========================================        
        # # 1. TELEPORTATION CHECK
        # # If the dot jumps more than 1.5 meters in 1/10th of a second, it's a glitch.
        # if is_valid_point and self.track_x is not None and "Apnea" not in status:
        #     jump_distance = np.sqrt((raw_x - self.track_x)**2 + (raw_y - self.track_y)**2)
        #     if jump_distance > 1.5:  
        #         is_valid_point = False 

        # ==========================================
        # Step 5: Temporal Persistence (Hit/Miss)
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
        # Step 6: Adaptive Smoothing & Actigraphy
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
            if len(self.z_history) > self.z_history_size:
                self.z_history.pop(0)
                
            # Calculate Vertical Velocity (V_z) in m/s (Assuming ~50ms per frame)
            dt = len(self.z_history) * 0.04 
            if dt > 0:
                v_z = (self.z_history[-1] - self.z_history[0]) / dt
            else:
                v_z = 0.0
        else:
            self.empty_room()
            return self.output_dict

        # ==========================================
        # Step 7: Posture Logic & Motion Tagging
        # ==========================================
        # Re-evaluate the zone with the smoothed coordinates
        final_zone, _ = self.evaluate_spatial_zone(X_b, Y_b, Z_b)

        # --- ZONE DEBOUNCING ---
        # Don't switch the displayed zone until the new zone persists for N frames
        if final_zone != self.current_stable_zone:
            self.zone_history.append(final_zone)
            # Check if the last N entries all agree on the new zone
            if len(self.zone_history) >= self.frames_to_confirm_zone:
                recent = self.zone_history[-self.frames_to_confirm_zone:]
                if all(z == final_zone for z in recent):
                    self.current_stable_zone = final_zone  # Confirmed switch
            # Keep using the old stable zone until confirmed
            final_zone = self.current_stable_zone
        else:
            self.zone_history.clear()  # Reset counter when back to stable zone

        # Posture Recognition logic
        if Z_b > STANDING_THRESHOLD:
            posture = "Standing"
        elif Z_b > SITTING_THRESHOLD:
            posture = "Sitting"
        else:
            posture = "Lying Down"

        # Motion Tagging logic
        if "Apnea" in status:
            motion_str = "Static"
        elif self.motion_level < REST_MAX: 
            motion_str = "Resting/Breathing"
        elif self.motion_level < RESTLESS_MAX: 
            motion_str = "Restless/Shifting"
        else:
            motion_str = "Major Movement"
        
        # Contextual Overrides
        fall_confidence = 0.0
        
        if final_zone in ["Out of Bounds (Ghost)", "Ignored"]:
            # The math tried to drag the dot outside the wall. Instantly kill the track.
            self.is_occupied = False
            self.last_target_bin = None
            return self.output_dict
            
        elif final_zone == "Floor / Transit":
            # The radar sees them, but they are not in bed or any "monitor"-type zone
            status = status.replace("Occupied", "In the Room") 
            
            # FALL DETECTION GATE
            if Z_b <= self.fall_threshold_z:
                # If they dropped fast OR they were already flagged as fallen
                print(f'Height: {Z_b}, Zone: {zone_name}, Velocity: {v_z}')
                if v_z <= self.fall_velocity_threshold or self.is_fallen:
                    self.is_fallen = True
                    self.fall_persistence_frames += 1
                    
                    # 1. Height Score (Closer to floor = worse)
                    h_score = max(0.0, (self.fall_threshold_z - Z_b) / self.fall_threshold_z) * 100.0
                    # 2. Time Score (Longer on floor = worse)
                    p_score = min(100.0, (self.fall_persistence_frames / 50.0) * 100.0) # 50 frames = 2 secs
                    
                    fall_confidence = (0.4 * h_score) + (0.6 * p_score)
                    
                    if fall_confidence > 60.0:
                        status = "CRITICAL: Fall Detected!"
                        posture = "Fallen"
                else:
                    # They are low, but lowered slowly (e.g., sitting on the floor to play)
                    self.fall_persistence_frames = max(0, self.fall_persistence_frames - 1)
            else:
                self.is_fallen = False
                self.fall_persistence_frames = 0
        else:
            # They are safe in a bed or chair. Reset fall logic.
            self.is_fallen = False
            self.fall_persistence_frames = 0
            
        # ==========================================
        # Step 8: Confidence Index Generation
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
        # Step 9: Zone Timer Update
        # ==========================================
        now = time.monotonic()
        self._update_zone_timer(final_zone, is_valid_point, now)

        duration_str = "--"
        if self.zone_timer_zone is not None and self.zone_timer_start is not None:
            duration_str = self._format_duration(now - self.zone_timer_start)

        self.output_dict = {
            "X": X_b,
            "Y": Y_b,
            "Z": Z_b,
            "Range": self.output_dict["Range"],
            "Azimuth": self.output_dict["Azimuth"],
            "Elevation": self.output_dict["Elevation"],
            "zone": final_zone,
            "status": status,
            "occ_confidence": occ_confidence,
            "posture_confidence": posture_confidence,
            "posture": posture,
            "motion_str": motion_str,
            "duration_str": duration_str,
            "fall_confidence": fall_confidence,
        }
        return self.output_dict


# ==========================================
# 3. GUI Visualizer (dashboard)
# ==========================================
class ActivityVisualizer:
    """
    Dashboard-style visualizer for room occupancy / posture / motion monitoring.
    Keeps the same update(...) interface as your current class.
    """

    def __init__(self, pipeline, history_len=120):
        self.p = pipeline
        self.history_len = history_len

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
        self.fig = plt.figure(figsize=(13, 8.4), facecolor=self.FIG_BG)

        # Main layout: top content + bottom cards
        outer = self.fig.add_gridspec(
            2, 1,
            height_ratios=[3.0, 1.05],
            hspace=0.25
        )

        # Top layout:
        # room map | small spacer | right-side stacked plots
        top = outer[0].subgridspec(
            2, 3,
            width_ratios=[0.86, 0.2, 1],
            height_ratios=[0.84, 1.16],
            wspace=0.00,
            hspace=0.3
        )

        self.ax = self.fig.add_subplot(top[:, 0])         # room map
        # top[:, 1] intentionally left empty as spacer
        self.ax_breath = self.fig.add_subplot(top[0, 2])  # breathing reserved
        self.ax_trend = self.fig.add_subplot(top[1, 2])   # combined trends

        # Bottom layout: centered narrower cards
        bottom = outer[1].subgridspec(
            1, 5,
            width_ratios=[0.22, 1.0, 1.0, 1.0, 0.22],
            wspace=0.14
        )

        self.ax_occ = self.fig.add_subplot(bottom[0, 1])
        self.ax_posture = self.fig.add_subplot(bottom[0, 2])
        self.ax_system = self.fig.add_subplot(bottom[0, 3])

        self.fig.subplots_adjust(left=0.06, right=0.96, top=0.90, bottom=0.08)

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
                (self.ax_breath, "Confidence Trends"),
                (self.ax_trend, "Motion / Activity")
            ]:
                trend_ax.set_facecolor(self.PANEL_BG)
                for spine in trend_ax.spines.values():
                    spine.set_color("#475569")
        trend_ax.tick_params(colors=self.SUBTEXT, labelsize=9)
        trend_ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.35, color=self.GRID)
        trend_ax.set_title(title, color=self.TEXT, fontsize=12, pad=10, fontweight="bold")
        trend_ax.set_xlabel("Frames", color=self.TEXT, fontsize=10)
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

        self._style_reserved_axis(self.ax_breath, "Breathing Signal (Reserved)")
        self._style_trend_axis(self.ax_trend, "Confidence: Occupancy / Posture / Motion / Fall Trends", "Normalized")

        self.ax_breath.text(
            0.5, 0.60,
            "Reserved for future breathing waveform",
            transform=self.ax_breath.transAxes,
            ha="center", va="center",
            color=self.TEXT, fontsize=11, fontweight="bold"
        )
        self.ax_breath.text(
            0.5, 0.38,
            "Respiration signal / RR / breathing confidence",
            transform=self.ax_breath.transAxes,
            ha="center", va="center",
            color=self.SUBTEXT, fontsize=10
        )

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

        ax.text(0.08, 0.84, title, fontsize=15, fontweight="bold",
        color=self.TEXT, va="center", ha="left", clip_on=False)

        row_y = [0.66, 0.48, 0.30, 0.14]
        for (label, value), y in zip(lines, row_y):
            ax.text(0.08, y, str(label), fontsize=10, color=self.SUBTEXT,
                    va="center", ha="left", clip_on=False)
            ax.text(0.91, y, str(value), fontsize=11, color=self.TEXT,
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
        x = np.arange(len(self.occ_hist))

        self.occ_line.set_data(x, list(self.occ_hist))
        self.posture_line.set_data(x, list(self.posture_hist))
        self.motion_smooth_line.set_data(x, list(self.motion_smooth_hist))
        self.fall_line.set_data(x, list(self.fall_hist))

        self.ax_trend.set_xlim(0, max(self.history_len - 1, 20))
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

    def update_breath_panel(self, waveform=None, rr_bpm=None, breath_conf=None):
        self.ax_breath.cla()
        self._style_reserved_axis(self.ax_breath, "Breathing Signal")

        if waveform is None or len(waveform) == 0:
            self.ax_breath.text(
                0.5, 0.60,
                "Reserved for future breathing waveform",
                transform=self.ax_breath.transAxes,
                ha="center", va="center",
                color=self.TEXT, fontsize=11, fontweight="bold"
            )
            self.ax_breath.text(
                0.5, 0.38,
                "Respiration signal / RR / breathing confidence",
                transform=self.ax_breath.transAxes,
                ha="center", va="center",
                color=self.SUBTEXT, fontsize=10
            )
            return

        x = np.arange(len(waveform))
        self.ax_breath.plot(x, waveform, linewidth=1.8)
        self.ax_breath.set_xticks([])
        self.ax_breath.set_yticks([])

        if rr_bpm is not None:
            self.ax_breath.text(
                0.03, 0.92,
                f"RR: {rr_bpm:.1f} bpm",
                transform=self.ax_breath.transAxes,
                color=self.TEXT, fontsize=10, fontweight="bold", va="top"
            )
        if breath_conf is not None:
            self.ax_breath.text(
                0.97, 0.92,
                f"Conf: {breath_conf:.0f}%",
                transform=self.ax_breath.transAxes,
                color=self.TEXT, fontsize=10, fontweight="bold", ha="right", va="top"
            )

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
    def update(self, occ_out_dict):
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
        print(f"\n[Visualizer] zone_name: {zone_name}")
        state_name = occ_out_dict["status"]
        occ_confidence = occ_out_dict["occ_confidence"]
        posture_confidence = occ_out_dict["posture_confidence"]
        posture = occ_out_dict["posture"]
        motion_str = occ_out_dict["motion_str"]
        duration_str = occ_out_dict["duration_str"]
        fall_confidence = occ_out_dict["fall_confidence"]
        Range = occ_out_dict["Range"]
        print(f"[Visualizer] Range: {Range}")
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

        # Draw refresh
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

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
    
    # 3. Initialize Pipeline and Visualizer
    pipeline = ActivityPipeline(num_range_bins=RANGE_IDX_NUM, range_resolution=RANGE_RESOLUTION)
    visualizer = ActivityVisualizer(pipeline)
    last_notified_state = ""
    
    print("Waiting for radar data...")
    frame_count = 0
    
    try:
        while plt.fignum_exists(visualizer.fig.number):
            try:
                # 1. Block and wait for at least one frame
                fft_frame = pt_fft_q.get(timeout=1.0) 
                occ_out_dict = pipeline.process_frame(fft_frame)
                
                # 2. Rapidly drain any EXTRA frames that piled up during the last screen draw
                while not pt_fft_q.empty():
                    try:
                        fft_frame = pt_fft_q.get_nowait()
                        occ_out_dict = pipeline.process_frame(fft_frame)

                    except queue.Empty:
                        break # Queue is fully caught up!
                
                # 3. Update the display exactly ONCE with the absolute freshest data
                visualizer.update(occ_out_dict)
                
                # 4. Send Alert 
                if SEND_ALERT and occ_out_dict['status'] != last_notified_state and "Initializing" not in occ_out_dict['status'] and (occ_out_dict['posture'] == "Fallen" or 'Apnea' in occ_out_dict['status']):
                    print(f"State changed to: {occ_out_dict['status']}. Sending alert to Watch...")
                    send_watch_alert(occ_out_dict['status'])  # Notification to Apple iPhone/Watch  (PushOver)
                    last_notified_state = occ_out_dict['status']
            except queue.Empty:
                pipeline.empty_room()
                visualizer.update(pipeline.output_dict)
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
    