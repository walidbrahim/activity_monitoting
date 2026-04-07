"""
radar_engine.detection.zoning
==============================
Spatial zone evaluation: maps a 3-D world-frame point to a named layout zone.

Extracted from ActivityPipeline.evaluate_spatial_zone() without any logic changes.

Entry point:
  - ``ZoneEvaluator`` class — receives layout at construction time,
    fully decoupled from the global config singleton.

Ownership: TargetDetector calls this per candidate after coordinate estimation.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class ZoneEvaluator:
    """Evaluates a 3-D world-frame point against a spatial layout dict.

    Args:
        layout: dict mapping zone_name → zone_dict, as loaded from YAML.
                Expected keys per zone: ``type``, ``x``, ``y``, ``z``,
                and optionally ``margin_x``, ``margin_y``.
    """

    def __init__(self, layout: dict) -> None:
        self._layout = layout

    def __call__(self, x: float, y: float, z: float) -> tuple[str, bool]:
        """Evaluate a point against the layout.  See module-level docstring."""
        return _evaluate(self._layout, x, y, z)

    def update_layout(self, layout: dict) -> None:
        """Replace the layout at runtime (e.g. after a pose update)."""
        self._layout = layout


def _evaluate(layout: dict, x: float, y: float, z: float) -> tuple[str, bool]:
    """Inner evaluation logic — pure function, no global state.

    Evaluation order (mirrors original):
    1. Room boundary (fast-fail out-of-bounds).
    2. Ignore/exclusion zones (override everything).
    3. Monitor zones (with Bed sub-zone classification).
    4. Transit fallback.

    Args:
        layout: Zone configuration dict.
        x, y, z: World-frame coordinates [m].

    Returns:
        Tuple (zone_label: str, is_valid_target: bool).
        ``is_valid_target`` is False for out-of-bounds or ignored zones.
    """
    # ── 1. Global boundary check (fast fail) ──────────────────────────────────
    room = layout.get("Room")
    if room:
        if not (
            room["x"][0] <= x <= room["x"][1] and
            room["y"][0] <= y <= room["y"][1] and
            room["z"][0] <= z <= room["z"][1]
        ):
            return "Out of Bounds (Ghost)", False

    # ── 2. Exclusion zones — override everything ──────────────────────────────
    for name, bounds in layout.items():
        if bounds.get("type") == "ignore":
            if (bounds["x"][0] <= x <= bounds["x"][1] and
                    bounds["y"][0] <= y <= bounds["y"][1] and
                    bounds["z"][0] <= z <= bounds["z"][1]):
                return f"Ignored ({name})", False

    # ── 3. Monitor zones ───────────────────────────────────────────────────────
    for name, bounds in layout.items():
        if bounds.get("type") == "monitor":
            if (bounds["x"][0] <= x <= bounds["x"][1] and
                    bounds["y"][0] <= y <= bounds["y"][1] and
                    bounds["z"][0] <= z <= bounds["z"][1]):

                # Bed gets sub-zone classification (center / edges / corners)
                if name == "Bed":
                    x_min, x_max = bounds["x"]
                    y_min, y_max = bounds["y"]
                    m_x = bounds.get("margin_x", [0.2, 0.2])
                    m_y = bounds.get("margin_y", [0.2, 0.2])

                    is_cx = (x_min + m_x[0]) <= x <= (x_max - m_x[1])
                    is_cy = (y_min + m_y[0]) <= y <= (y_max - m_y[1])

                    if is_cx and is_cy:
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

                return name, True

    # ── 4. Transit / floor fallback ────────────────────────────────────────────
    return "Floor / Transit", True
