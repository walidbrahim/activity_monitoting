from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field


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


class AppFlagsConfig(BaseModel):
    log_level: int
    need_send_ti_config: bool
    send_alert: bool
    enable_robot_arm: bool = True
    default_radar_pose: str = "Room"
    imu_auto_align_enabled: bool = False
    radar_imu_index: int = 0
    imu_yaw_offset: float = 180.0
    imu_pitch_offset: float = 0.0
    radar_pitch_multiplier: float = -1.0


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


class CameraConfig(BaseModel):
    enabled: bool = True
    device_index: int = 0


class VernierConfig(BaseModel):
    enabled: bool = False
    rate_hz: int = 20
    use_ble: bool = False
    sensors: List[int] = Field(default_factory=lambda: [1])
    model: str = "GDX-RB 0K203641"
    mac: str = "BC:33:AC:AE:BE:35"


class WitMotionConfig(BaseModel):
    enabled: bool = False
    mac: str = ""
    rate_hz: int = 50
    location: str = "unknown"
    channel: str = "ACC"


class RecordingConfig(BaseModel):
    enabled: bool = False
    output_dir: str = "captures"
    session_name: str = ""
    save_raw_frames: bool = True
    save_frame_records: bool = True
    save_event_log: bool = True
    recorder_capacity: int = 5000
    event_capacity: int = 2000


class DatabaseConfig(BaseModel):
    enabled: bool = True
    path: str = "data/monitoring.sqlite3"
    rr_bucket_sec: int = 10
    session_end_grace_sec: int = 10
    subject_id: str = "subject_001"
    device_id: str = "radar_001"


class TISerialConfig(BaseModel):
    config_file: str = ""
    magic_word: List[int] = Field(default_factory=list)
    mac_cli: str = ""
    mac_data: str = ""
    linux_cli: str = ""
    linux_data: str = ""
    win_cli: str = ""
    win_data: str = ""
    start_frequency: int = 0
    chirp_loop: int = 0
    chirp_slope: int = 0
    chirp_end_time: int = 0
    adc_start: int = 0
    chirp_idle_time: int = 0
    adc_sample_rate: int = 0
    tx_ant: int = 0
    rx_ant: int = 0


class Dca1000Config(BaseModel):
    udp_port: int = 4098
    fpga_ip: str = "192.168.33.180"


class HardwareConfig(BaseModel):
    source: str = "ti_serial"
    antennas: int
    range_bins: int
    range_resolution: float
    frame_rate: float
    ti_serial: TISerialConfig = Field(default_factory=TISerialConfig)
    dca1000: Dca1000Config = Field(default_factory=Dca1000Config)


class PreprocessingFeaturesConfig(BaseModel):
    clutter_removal: bool = True
    vital_analysis: bool = True
    apnea_state: bool = True
    temporal_persistence: bool = True
    adaptive_smoothing: bool = True
    fall_posture: bool = True


class PreprocessingConfig(BaseModel):
    clutter_ema_alpha: float = 0.05
    static_clutter_margin: float = 200.0
    warmup_frames: int = 25
    features: PreprocessingFeaturesConfig = Field(default_factory=PreprocessingFeaturesConfig)


class DetectionConfig(BaseModel):
    use_cfar: bool = True
    detection_threshold: float = 150.0
    num_candidates: int = 15
    min_search_range_m: float = 0.30
    z_clip_min: float = 0.05
    z_clip_max: float = 1.80


class TrackingConfig(BaseModel):
    track_ema_alpha: float = 0.05
    miss_allowance: int = 25
    jump_reject_distance_m: float = 1.0
    coord_buffer_size: int = 50
    frames_to_occupy: Optional[int] = None


class PostureConfig(BaseModel):
    sitting_threshold_m: float = 0.45
    standing_threshold_m: float = 0.95
    hysteresis_m: float = 0.05
    transit_standing_bias_m: float = 0.08
    posture_z_neighborhood_m: float = 0.30
    posture_z_bias: float = 0.0
    posture_transit_z_bonus: float = 0.15
    fall_detection_enable: bool = False
    fall_threshold_m: float = 0.30
    fall_velocity_threshold: float = -1.2
    fall_cooldown_frames: int = 25


class MotionConfig(BaseModel):
    rest_max: float = 0.10
    restless_max: float = 0.30
    walk_window_frames: int = 15
    walk_displacement_m: float = 0.30
    walk_posture_conf: float = 90.0


class ActivityConfigSection(BaseModel):
    zone_debounce_frames: int = 50
    entry_hold_seconds: float = 3.0
    subzone_debounce_frames: int = 15
    bin_vote_window: int = 7
    continuity_ratio: float = 0.60
    reflection_dip_tolerance: int = 5
    posture: PostureConfig = Field(default_factory=PostureConfig)
    motion: MotionConfig = Field(default_factory=MotionConfig)


class VitalsConfig(BaseModel):
    aliveness_threshold: float = 0.40
    breathing_displacement_min: float = 0.5
    breathing_displacement_max: float = 12.0
    min_person_snr: float = 80.0
    ghost_phase_threshold: float = 0.0005
    macro_displacement_mm: float = 15.0
    position_stability_window: int = 75
    max_clutter_position_var: float = 0.08


class RespirationSectionConfig(BaseModel):
    window_sec: float = 30.0
    lowpass_cutoff_hz: float = 0.5
    lowpass_order: int = 4
    bin_stability_sec: float = 2.0
    apnea_hold_window_sec: float = 5.0
    apnea_merge_gap_sec: float = 0.5
    brv_history_size: int = 20
    cycle_tracker_history: int = 5
    resp_threshold: float = 0.1


class AppConfig(BaseModel):
    hardware: HardwareConfig
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    activity: ActivityConfigSection = Field(default_factory=ActivityConfigSection)
    vitals: VitalsConfig = Field(default_factory=VitalsConfig)
    respiration_cfg: RespirationSectionConfig = Field(default_factory=RespirationSectionConfig)
    layout: Dict[str, Dict[str, Any]]
    app: AppFlagsConfig
    gui_theme: GuiThemeConfig
    camera: CameraConfig = Field(default_factory=CameraConfig)
    vernier: VernierConfig = Field(default_factory=VernierConfig)
    witmotion1: WitMotionConfig = Field(default_factory=WitMotionConfig)
    witmotion2: WitMotionConfig = Field(default_factory=WitMotionConfig)
    recording: RecordingConfig = Field(default_factory=RecordingConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    tuning: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}

    @classmethod
    def load(cls, file_path: str = "profiles/base.yaml") -> "AppConfig":
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Config file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def save(self, file_path: str = "profiles/base.yaml") -> None:
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)


def load_profile(*file_paths: str) -> AppConfig:
    """Load and deep-merge one or more layered profile YAML files."""
    if not file_paths:
        raise ValueError("At least one profile file must be specified.")

    def _deep_merge(base: dict, override: dict) -> dict:
        result = copy.deepcopy(base)
        for key, val in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(val, dict):
                result[key] = _deep_merge(result[key], val)
            else:
                result[key] = copy.deepcopy(val)
        return result

    merged: dict = {}
    for path in file_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Profile not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        merged = _deep_merge(merged, data)

    return AppConfig(**merged)


class ConfigFactory:
    """Build typed radar_engine dataclass configs from AppConfig."""

    @staticmethod
    def engine_config(app_cfg: "AppConfig") -> "EngineConfig":
        from radar_engine.config.activity import ActivityConfig, MotionCfg, PostureCfg
        from radar_engine.config.detection import DetectionConfig as EngineDetectionConfig
        from radar_engine.config.engine import EngineConfig
        from radar_engine.config.hardware import RadarHardwareConfig
        from radar_engine.config.preprocessing import (
            PreprocessingConfig as EnginePreprocessingConfig,
        )
        from radar_engine.config.respiration import (
            RespirationConfig as EngineRespirationConfig,
        )
        from radar_engine.config.tracking import TrackingConfig as EngineTrackingConfig
        from radar_engine.config.vitals import VitalsConfig as EngineVitalsConfig

        t = app_cfg.tuning or {}

        def pick(default: Any, *keys: str) -> Any:
            for key in keys:
                if key in t:
                    return t[key]
            return default

        hw = app_cfg.hardware
        pp = app_cfg.preprocessing
        det = app_cfg.detection
        tr = app_cfg.tracking
        act = app_cfg.activity
        vit = app_cfg.vitals
        rsp = app_cfg.respiration_cfg
        pos = act.posture
        mot = act.motion

        hardware = RadarHardwareConfig(
            num_range_bins=int(pick(hw.range_bins, "range_bins")),
            num_antennas=int(pick(hw.antennas, "antennas")),
            frame_rate=float(pick(hw.frame_rate, "frame_rate")),
            range_resolution=float(pick(hw.range_resolution, "range_resolution")),
        )

        preprocessing = EnginePreprocessingConfig(
            clutter_ema_alpha=float(pick(pp.clutter_ema_alpha, "clutter_ema_alpha")),
            static_clutter_margin=float(
                pick(pp.static_clutter_margin, "static_clutter_margin")
            ),
            warmup_frames=int(pick(pp.warmup_frames, "warmup_frames")),
            features_clutter_removal=bool(
                pick(pp.features.clutter_removal, "features_clutter_removal")
            ),
            features_vital_analysis=bool(
                pick(pp.features.vital_analysis, "features_vital_analysis")
            ),
            features_temporal_persistence=bool(
                pick(pp.features.temporal_persistence, "features_temporal_persistence")
            ),
            features_adaptive_smoothing=bool(
                pick(pp.features.adaptive_smoothing, "features_adaptive_smoothing")
            ),
            features_fall_posture=bool(
                pick(pp.features.fall_posture, "features_fall_posture")
            ),
            features_apnea_state=bool(
                pick(pp.features.apnea_state, "features_apnea_state")
            ),
        )

        detection = EngineDetectionConfig(
            use_cfar=bool(pick(det.use_cfar, "use_cfar")),
            detection_threshold=float(
                pick(det.detection_threshold, "detection_threshold")
            ),
            num_candidates=int(pick(det.num_candidates, "num_candidates")),
            min_search_range_m=float(
                pick(det.min_search_range_m, "min_search_range_m")
            ),
            z_clip_min=float(pick(det.z_clip_min, "z_clip_min")),
            z_clip_max=float(pick(det.z_clip_max, "z_clip_max")),
        )

        frames_to_occupy = tr.frames_to_occupy
        if frames_to_occupy is None:
            frames_to_occupy = int(act.entry_hold_seconds * hardware.frame_rate)

        tracking = EngineTrackingConfig(
            track_ema_alpha=float(pick(tr.track_ema_alpha, "track_ema_alpha")),
            miss_allowance=int(pick(tr.miss_allowance, "miss_allowance")),
            jump_reject_distance_m=float(
                pick(tr.jump_reject_distance_m, "jump_reject_distance_m")
            ),
            coord_buffer_size=int(pick(tr.coord_buffer_size, "coord_buffer_size")),
            frames_to_occupy=int(pick(frames_to_occupy, "frames_to_occupy")),
        )

        posture_cfg = PostureCfg(
            sitting_threshold_m=float(
                pick(pos.sitting_threshold_m, "sitting_threshold_m")
            ),
            standing_threshold_m=float(
                pick(pos.standing_threshold_m, "standing_threshold_m")
            ),
            hysteresis_m=float(pick(pos.hysteresis_m, "hysteresis_m")),
            transit_standing_bias_m=float(
                pick(pos.transit_standing_bias_m, "transit_standing_bias_m")
            ),
            posture_z_neighborhood_m=float(
                pick(pos.posture_z_neighborhood_m, "posture_z_neighborhood_m")
            ),
            posture_z_bias=float(pick(pos.posture_z_bias, "posture_z_bias")),
            posture_transit_z_bonus=float(
                pick(pos.posture_transit_z_bonus, "posture_transit_z_bonus")
            ),
            fall_detection_enable=bool(
                pick(pos.fall_detection_enable, "fall_detection_enable")
            ),
            fall_threshold_m=float(pick(pos.fall_threshold_m, "fall_threshold_m")),
            fall_velocity_threshold=float(
                pick(pos.fall_velocity_threshold, "fall_velocity_threshold")
            ),
            fall_cooldown_frames=int(
                pick(pos.fall_cooldown_frames, "fall_cooldown_frames")
            ),
        )

        motion_cfg = MotionCfg(
            rest_max=float(pick(mot.rest_max, "rest_max")),
            restless_max=float(pick(mot.restless_max, "restless_max")),
            walk_window_frames=int(
                pick(mot.walk_window_frames, "walk_window_frames")
            ),
            walk_displacement_m=float(
                pick(mot.walk_displacement_m, "walk_displacement_m")
            ),
            walk_posture_conf=float(
                pick(mot.walk_posture_conf, "walk_posture_conf")
            ),
        )

        activity = ActivityConfig(
            zone_debounce_frames=int(
                pick(act.zone_debounce_frames, "zone_debounce_frames")
            ),
            entry_hold_seconds=float(
                pick(act.entry_hold_seconds, "entry_hold_seconds")
            ),
            subzone_debounce_frames=int(
                pick(act.subzone_debounce_frames, "subzone_debounce_frames")
            ),
            bin_vote_window=int(pick(act.bin_vote_window, "bin_vote_window")),
            continuity_ratio=float(pick(act.continuity_ratio, "continuity_ratio")),
            reflection_dip_tolerance=int(
                pick(act.reflection_dip_tolerance, "reflection_dip_tolerance")
            ),
            posture=posture_cfg,
            motion=motion_cfg,
        )

        vitals = EngineVitalsConfig(
            aliveness_threshold=float(
                pick(vit.aliveness_threshold, "aliveness_threshold")
            ),
            breathing_displacement_min=float(
                pick(vit.breathing_displacement_min, "breathing_displacement_min")
            ),
            breathing_displacement_max=float(
                pick(vit.breathing_displacement_max, "breathing_displacement_max")
            ),
            min_person_snr=float(pick(vit.min_person_snr, "min_person_snr")),
            ghost_phase_threshold=float(
                pick(vit.ghost_phase_threshold, "ghost_phase_threshold")
            ),
            macro_displacement_mm=float(
                pick(vit.macro_displacement_mm, "macro_displacement_mm")
            ),
            position_stability_window=int(
                pick(vit.position_stability_window, "position_stability_window")
            ),
            max_clutter_position_var=float(
                pick(vit.max_clutter_position_var, "max_clutter_position_var")
            ),
        )

        respiration = EngineRespirationConfig(
            window_sec=float(pick(rsp.window_sec, "window_sec")),
            lowpass_cutoff_hz=float(
                pick(rsp.lowpass_cutoff_hz, "lowpass_cutoff_hz")
            ),
            lowpass_order=int(pick(rsp.lowpass_order, "lowpass_order")),
            bin_stability_sec=float(pick(rsp.bin_stability_sec, "bin_stability_sec")),
            apnea_hold_window_sec=float(
                pick(rsp.apnea_hold_window_sec, "apnea_hold_window_sec")
            ),
            apnea_merge_gap_sec=float(
                pick(rsp.apnea_merge_gap_sec, "apnea_merge_gap_sec")
            ),
            brv_history_size=int(pick(rsp.brv_history_size, "brv_history_size")),
            cycle_tracker_history=int(
                pick(rsp.cycle_tracker_history, "cycle_tracker_history")
            ),
            resp_threshold=float(pick(rsp.resp_threshold, "resp_threshold")),
        )

        return EngineConfig(
            hardware=hardware,
            preprocessing=preprocessing,
            detection=detection,
            tracking=tracking,
            activity=activity,
            vitals=vitals,
            respiration=respiration,
            layout=dict(app_cfg.layout),
        )


def _default_profile_stack_from_env() -> list[str]:
    base_profile = "profiles/base.yaml"
    selected = os.getenv("APP_PROFILE", "base").strip()
    if selected in ("", "base", "base.yaml", base_profile):
        return [base_profile]

    overlay = selected if selected.endswith(".yaml") else f"{selected}.yaml"
    if "/" not in overlay:
        overlay = f"profiles/{overlay}"
    return [base_profile, overlay]


try:
    config = load_profile(*_default_profile_stack_from_env())
except Exception as exc:  # pragma: no cover
    print(f"Warning: Could not load default profile stack: {exc}")
    config = None
