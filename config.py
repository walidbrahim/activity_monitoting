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

class PipelineConfig(BaseModel):
    detection_threshold: float
    static_margin: int
    alpha: float
    track_alpha: float
    frame_to_confirm_zone: int
    buffer_size: int
    miss_allowance: int

class PostureConfig(BaseModel):
    fall_detection_enable: bool
    fall_threshold: float
    fall_velocity_threshold: float
    sitting_threshold: float
    standing_threshold: float

class MotionConfig(BaseModel):
    rest_max: float
    restless_max: float

class RespirationConfig(BaseModel):
    resp_window_sec: int

class AppFlagsConfig(BaseModel):
    log_level: int
    need_send_ti_config: bool
    send_alert: bool

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
