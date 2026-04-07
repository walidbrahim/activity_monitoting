"""
radar_engine.config
====================
Engine-owned typed configuration dataclasses.

Rules:
  - Pure Python dataclasses — NO Pydantic, NO YAML, NO external imports.
  - All fields have sensible numeric defaults: modules are unit-testable
    without any YAML file present.
  - The AppConfig (Pydantic, YAML) → EngineConfig mapping lives exclusively
    in ConfigFactory (config.py) — never here.
"""
from radar_engine.config.hardware      import RadarHardwareConfig
from radar_engine.config.preprocessing import PreprocessingConfig
from radar_engine.config.detection     import DetectionConfig
from radar_engine.config.tracking      import TrackingConfig
from radar_engine.config.activity      import ActivityConfig, PostureCfg, MotionCfg
from radar_engine.config.vitals        import VitalsConfig
from radar_engine.config.respiration   import RespirationConfig
from radar_engine.config.engine        import EngineConfig

__all__ = [
    "RadarHardwareConfig",
    "PreprocessingConfig",
    "DetectionConfig",
    "TrackingConfig",
    "ActivityConfig", "PostureCfg", "MotionCfg",
    "VitalsConfig",
    "RespirationConfig",
    "EngineConfig",
]
