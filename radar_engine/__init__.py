"""
radar_engine — Reusable, GUI-agnostic radar processing engine.

Three-layer architecture:
    Layer 1:  radar_engine/  (this package)  — algorithm & estimation
    Layer 2:  apps/*/        (controllers)    — display policy & app rules
    Layer 3:  libs/gui/ / p1_resp_gui.py      — presentation only

Public API (stable):
    from radar_engine.core.models import EngineOutput
    from radar_engine.orchestration.engine import RadarEngine
"""
__version__ = "0.1.0"
