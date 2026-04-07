"""
radar_engine.diagnostics.event_log
=====================================
EventLog — structured event journal for state transitions.

Tracks discrete state changes that are significant enough to log but too
fine-grained for a full FrameRecord: track gained/lost, zone transitions,
apnea start/stop, fall events, and occupancy state machine transitions.

Events are timestamped, typed, and carry a small payload dict.
The log is queryable by type and exportable to JSON / plain text.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Categories of tracked discrete events."""
    TRACK_GAINED          = "TRACK_GAINED"
    TRACK_LOST            = "TRACK_LOST"
    ZONE_CHANGE           = "ZONE_CHANGE"
    OCCUPANCY_CHANGE      = "OCCUPANCY_CHANGE"
    POSTURE_CHANGE        = "POSTURE_CHANGE"
    APNEA_START           = "APNEA_START"
    APNEA_END             = "APNEA_END"
    FALL_CANDIDATE        = "FALL_CANDIDATE"
    FALL_DETECTED         = "FALL_DETECTED"
    FALL_CLEARED          = "FALL_CLEARED"
    ENGINE_RESET          = "ENGINE_RESET"
    WARMUP_COMPLETE       = "WARMUP_COMPLETE"
    RR_LOCK               = "RR_LOCK"
    CUSTOM                = "CUSTOM"


@dataclass
class Event:
    """One discrete event entry."""
    event_type:  str
    timestamp:   float
    frame_index: int
    payload:     dict = field(default_factory=dict)

    @property
    def type_enum(self) -> EventType:
        try:
            return EventType(self.event_type)
        except ValueError:
            return EventType.CUSTOM

    def __str__(self) -> str:
        t = time.strftime("%H:%M:%S", time.localtime(self.timestamp))
        ms = int((self.timestamp - int(self.timestamp)) * 1000)
        payload_str = ", ".join(f"{k}={v}" for k, v in self.payload.items())
        return f"[{t}.{ms:03d}] fr#{self.frame_index:05d} {self.event_type} {payload_str}"


class EventLog:
    """Stateful event journal with duplicate-suppression and export.

    Records only genuine state *transitions* — repeated identical states are
    suppressed so the log stays meaningful rather than redundant.

    Args:
        capacity: Maximum events to retain (ring buffer).
    """

    def __init__(self, capacity: int = 500) -> None:
        self._buf:        deque[Event] = deque(maxlen=capacity)
        # Last-seen values for transition suppression
        self._last_zone:      str | None = None
        self._last_occupancy: str | None = None
        self._last_posture:   str | None = None
        self._last_track:     bool       = False
        self._last_apnea:     bool       = False
        self._last_fall:      str        = "none"

    # ── Auto-recording from EngineOutput ─────────────────────────────────────

    def update(self, output, frame_index: int | None = None) -> None:
        """Detect and record all relevant state transitions from an EngineOutput.

        Args:
            output:      Typed EngineOutput from RadarEngine.process_frame().
            frame_index: Override frame index (uses output.frame_index if None).
        """
        from radar_engine.core.models import EngineOutput
        fi = frame_index if frame_index is not None else output.frame_index
        ts = output.timestamp

        # Warmup completion
        if output.diagnostics.get("warmup_complete"):
            self._log(EventType.WARMUP_COMPLETE, ts, fi, {})

        # Track gained / lost
        has_track = output.has_target
        if has_track and not self._last_track:
            t = output.tracked_target
            self._log(EventType.TRACK_GAINED, ts, fi, {
                "x": round(t.x_m, 2),
                "y": round(t.y_m, 2),
                "z": round(t.z_m, 2),
            })
        elif not has_track and self._last_track:
            self._log(EventType.TRACK_LOST, ts, fi, {})
        self._last_track = has_track

        # Activity-level events
        act = output.activity
        if act and act.valid:
            # Zone change
            if act.zone != self._last_zone:
                self._log(EventType.ZONE_CHANGE, ts, fi, {
                    "from": self._last_zone or "--",
                    "to":   act.zone,
                })
                self._last_zone = act.zone

            # Occupancy change
            occ = act.occupancy.value
            if occ != self._last_occupancy:
                self._log(EventType.OCCUPANCY_CHANGE, ts, fi, {
                    "from": self._last_occupancy or "--",
                    "to":   occ,
                    "conf": round(act.occupancy_confidence, 1),
                })
                self._last_occupancy = occ

            # Posture change
            posture = act.posture.value
            if posture != self._last_posture:
                self._log(EventType.POSTURE_CHANGE, ts, fi, {
                    "from": self._last_posture or "--",
                    "to":   posture,
                })
                self._last_posture = posture

            # Fall state changes
            fall_v = act.fall_state.value
            if fall_v != self._last_fall:
                if fall_v == "detected":
                    self._log(EventType.FALL_DETECTED, ts, fi,
                              {"confidence": round(act.fall_confidence, 1)})
                elif fall_v == "candidate":
                    self._log(EventType.FALL_CANDIDATE, ts, fi,
                              {"confidence": round(act.fall_confidence, 1)})
                elif self._last_fall in ("detected", "candidate") and fall_v == "none":
                    self._log(EventType.FALL_CLEARED, ts, fi, {})
                self._last_fall = fall_v

        # Apnea transitions
        rm = output.respiration_metrics
        if rm:
            if rm.apnea_active and not self._last_apnea:
                self._log(EventType.APNEA_START, ts, fi, {})
            elif not rm.apnea_active and self._last_apnea:
                self._log(EventType.APNEA_END, ts, fi, {})
            self._last_apnea = rm.apnea_active

    def log_custom(
        self,
        label:       str,
        frame_index: int,
        payload:     dict | None = None,
    ) -> None:
        """Record a custom application-defined event."""
        self._log(EventType.CUSTOM, time.time(), frame_index, {
            "label": label, **(payload or {})
        })

    def log_reset(self, frame_index: int, reason: str = "") -> None:
        """Record an engine reset event."""
        self._log(EventType.ENGINE_RESET, time.time(), frame_index, {"reason": reason})

    # ── Query ─────────────────────────────────────────────────────────────────

    def all_events(self) -> list[Event]:
        return list(self._buf)

    def events_of_type(self, event_type: EventType) -> list[Event]:
        return [e for e in self._buf if e.event_type == event_type.value]

    def last_n(self, n: int) -> list[Event]:
        buf = list(self._buf)
        return buf[-n:]

    def since(self, timestamp: float) -> list[Event]:
        return [e for e in self._buf if e.timestamp >= timestamp]

    # ── Export ────────────────────────────────────────────────────────────────

    def export_json(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(e) for e in self._buf]
        with p.open("w") as f:
            json.dump(data, f, indent=2)
        logger.info("Exported %d events to %s", len(self._buf), p)
        return p

    def export_text(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            for ev in self._buf:
                f.write(str(ev) + "\n")
        return p

    def summary(self) -> dict[str, int]:
        """Return count of each event type in the log."""
        counts: dict[str, int] = {}
        for ev in self._buf:
            counts[ev.event_type] = counts.get(ev.event_type, 0) + 1
        return counts

    def reset(self) -> None:
        self._buf.clear()
        self._last_zone      = None
        self._last_occupancy = None
        self._last_posture   = None
        self._last_track     = False
        self._last_apnea     = False
        self._last_fall      = "none"

    # ── Private ───────────────────────────────────────────────────────────────

    def _log(
        self,
        event_type:  EventType,
        timestamp:   float,
        frame_index: int,
        payload:     dict,
    ) -> None:
        ev = Event(
            event_type  = event_type.value,
            timestamp   = timestamp,
            frame_index = frame_index,
            payload     = payload,
        )
        self._buf.append(ev)
        logger.debug("%s", ev)
