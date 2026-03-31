import yaml
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Union, Optional
import os

class LayoutItem(BaseModel):
    type: str
    x: Union[float, List[float]]
    y: Union[float, List[float]]
    z: Union[float, List[float]]
    yaw_deg: Optional[float] = None
    pitch_deg: Optional[float] = None
    fov_deg: Optional[float] = None
    margin_x: Optional[List[float]] = None
    margin_y: Optional[List[float]] = None

class PushoverConfig(BaseModel):
    user_key: str
    api_token: str

class RadarConfig(BaseModel):
    config_file_path: str
    antennas: int
    range_resolution: float
    frame_rate: int
    range_idx_num: int
    ti_1dfft_queue_len: int
    magic_word: List[int]
    start_frequency: int
    chirp_loop: int
    range_bins: int
    chirp_slope: int
    chirp_end_time: int
    adc_start: int
    chirp_idle_time: int
    adc_sample_rate: int
    tx_ant: int
    rx_ant: int
    mac_cli: str
    mac_data: str
    linux_cli: str
    linux_data: str
    win_cli: str
    win_data: str

class PipelineFeatures(BaseModel):
    clutter_removal: bool = True
    vital_analysis: bool = True
    tethering: bool = True
    apnea_state: bool = True
    temporal_persistence: bool = True
    adaptive_smoothing: bool = True
    fall_posture: bool = True

class PipelineConfig(BaseModel):
    detection_threshold: float
    static_margin: int
    alpha: float
    track_alpha: float
    frame_to_confirm_zone: int
    buffer_size: int
    miss_allowance: int
    features: PipelineFeatures = Field(default_factory=PipelineFeatures)

class PostureConfig(BaseModel):
    fall_detection_enable: bool
    fall_threshold: float
    fall_velocity_threshold: float
    sitting_threshold: float
    standing_threshold: float

class MotionConfig(BaseModel):
    rest_max: float
    restless_max: float
    # Walking detection
    walk_window_frames: int   = 15    # look-back window for net XY displacement (~0.6 s at 25 fps)
    walk_displacement_m: float = 0.30 # minimum net translation to classify as Walking
    walk_posture_conf: float  = 90.0  # posture_confidence floor when walking is confirmed

class RespirationConfig(BaseModel):
    resp_window_sec: int
    resp_lowpass_cutoff: float = 0.5
    resp_lowpass_order: int = 4
    resp_threshold: float = 0.1
    apnea_hold_window_sec: float = 3.0
    apnea_merge_gap_sec: float = 0.5
    brv_history_size: int = 20
    cycle_tracker_history: int = 5
    bin_stability_sec: float = 2.0

class TuningConfig(BaseModel):
    min_search_range: float = 0.30
    num_candidates: int = 15
    ghost_phase_threshold: float = 0.0005
    macro_displacement_mm: float = 15.0
    z_clip_min: float = 0.05
    z_clip_max: float = 1.80
    jump_reject_distance: float = 1.5
    warmup_seconds: float = 1.0
    entry_hold_seconds: float = 3.0
    reassess_seconds: float = 3.0
    # Aliveness Check: Layer 1 — SNR Floor
    min_person_snr: float = 80.0
    # Aliveness Check: Layer 2 — Position Stability
    position_stability_window: int = 75
    max_clutter_position_var: float = 0.08
    # Aliveness Check: Layer 3 — Multi-Metric Spectral Quality
    aliveness_threshold: float = 0.40
    breathing_displacement_min: float = 0.5
    breathing_displacement_max: float = 12.0

class AppFlagsConfig(BaseModel):
    log_level: int
    need_send_ti_config: bool
    send_alert: bool
    enable_robot_arm: bool = True
    default_radar_pose: Optional[str] = "Room"

class GuiThemeConfig(BaseModel):
    fig_bg: str
    panel_bg: str
    card_bg: str
    card_ok: str
    card_warn: str
    card_alert: str
    text: str
    subtext: str
    grid: str
    bed: str
    chair: str
    monitor: str
    ignore: str
    occupant: str
    room_edge: str
    fov: str
    radar: str
    origin: str

class AppConfig(BaseModel):
    layout: Dict[str, Dict[str, Any]]
    pushover: PushoverConfig
    radar: RadarConfig
    pipeline: PipelineConfig
    posture: PostureConfig
    motion: MotionConfig
    respiration: RespirationConfig
    app: AppFlagsConfig
    gui_theme: GuiThemeConfig
    tuning: TuningConfig = Field(default_factory=TuningConfig)

    @classmethod
    def load(cls, file_path: str = "profiles/app_config.yaml") -> "AppConfig":
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Config file not found: {file_path}")
            
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def save(self, file_path: str = "profiles/app_config.yaml"):
        with open(file_path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

# By default, we can provide a globally accessible config object 
# that loads when this module is imported.
try:
    config = AppConfig.load()
except Exception as e:
    print(f"Warning: Could not load default config: {e}")
    config = None
