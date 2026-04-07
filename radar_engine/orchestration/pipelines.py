"""
radar_engine.orchestration.pipelines
=======================================
Application-specific engine factories.

BedMonitoringEngine wraps RadarEngine with bed-monitoring defaults and
provides application-level construction defaults.
"""

from __future__ import annotations

from radar_engine.orchestration.engine import RadarEngine


class BedMonitoringEngine(RadarEngine):
    """RadarEngine pre-configured for bed / clinical monitoring.

    Passes ``with_respiration=True`` and forwards all keyword arguments
    (including ``cfg: EngineConfig``) straight to ``RadarEngine``.

    Usage::

        from radar_engine.config.engine import EngineConfig
        from radar_engine.orchestration.pipelines import BedMonitoringEngine

        ec  = EngineConfig.with_hardware(35, 8, 25.0, 0.15)
        eng = BedMonitoringEngine(cfg=ec)
        out = eng.process_frame(raw_frame)
    """

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("with_respiration", True)
        super().__init__(**kwargs)
