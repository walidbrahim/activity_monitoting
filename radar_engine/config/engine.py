"""radar_engine.config.engine — Top-level engine configuration aggregate."""
from __future__ import annotations
from dataclasses import dataclass, field

from radar_engine.config.hardware      import RadarHardwareConfig
from radar_engine.config.preprocessing import PreprocessingConfig
from radar_engine.config.detection     import DetectionConfig
from radar_engine.config.tracking      import TrackingConfig
from radar_engine.config.activity      import ActivityConfig
from radar_engine.config.vitals        import VitalsConfig
from radar_engine.config.respiration   import RespirationConfig


@dataclass
class EngineConfig:
    """Aggregate configuration passed to RadarEngine / BedMonitoringEngine.

    Usage::

        # All defaults (for tests or simulation):
        cfg = EngineConfig.with_hardware(
            num_range_bins=35, num_antennas=8,
            frame_rate=25.0, range_resolution=0.15,
        )

        # From live AppConfig via ConfigFactory:
        cfg = ConfigFactory.engine_config(app_config)

    The ``layout`` dict preserves the spatial zone configuration in its
    original form (dict of zone_name → zone_dict) so the engine can
    determine zone membership for each candidate.
    """

    # Required — must be provided (no sensible universal default for hardware)
    hardware: RadarHardwareConfig

    # All other sub-configs have safe defaults for unit testing
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    detection:     DetectionConfig     = field(default_factory=DetectionConfig)
    tracking:      TrackingConfig      = field(default_factory=TrackingConfig)
    activity:      ActivityConfig      = field(default_factory=ActivityConfig)
    vitals:        VitalsConfig        = field(default_factory=VitalsConfig)
    respiration:   RespirationConfig   = field(default_factory=RespirationConfig)

    # Spatial layout: dict[zone_name, zone_dict] — kept as raw dict because
    # layout is highly variable (arbitrary zone names + nested radar_pose).
    layout: dict = field(default_factory=dict)

    @classmethod
    def with_hardware(
        cls,
        num_range_bins:   int,
        num_antennas:     int,
        frame_rate:       float,
        range_resolution: float,
        **overrides,
    ) -> "EngineConfig":
        """Convenience constructor for tests and scripting.

        All non-hardware sub-configs use their defaults.  Pass keyword
        arguments to override individual sub-configs, e.g.::

            cfg = EngineConfig.with_hardware(
                num_range_bins=35, num_antennas=8,
                frame_rate=25.0, range_resolution=0.15,
                activity=ActivityConfig(zone_debounce_frames=25),
            )
        """
        hw = RadarHardwareConfig(
            num_range_bins=num_range_bins,
            num_antennas=num_antennas,
            frame_rate=frame_rate,
            range_resolution=range_resolution,
        )
        return cls(hardware=hw, **overrides)
