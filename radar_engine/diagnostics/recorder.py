"""
radar_engine.diagnostics.recorder
=====================================
FrameRecorder — lightweight per-frame telemetry sink.

Captures typed EngineOutput snapshots into a rolling in-memory ring buffer
with optional CSV/JSON export. Designed for:
  - Real-time dashboard debug overlays (querying last-N frames).
  - Automated regression baselines (export to JSON).
  - Post-mortem analysis of detection / tracking events.

No GUI or Qt dependencies — pure Python, importable from any layer.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from radar_engine.core.models import EngineOutput

logger = logging.getLogger(__name__)


@dataclass
class FrameRecord:
    """Compact per-frame snapshot stored in the ring buffer.

    Only scalar / small-dimensionality fields are captured to keep memory
    bounded.  NumPy arrays (signals, histories) are intentionally omitted;
    use a dedicated data-logger for full signal capture.
    """
    frame_index:         int
    timestamp:           float
    elapsed_ms:          float           # wall time since previous frame

    # Tracking
    has_target:          bool
    track_x:             float | None
    track_y:             float | None
    track_z:             float | None
    track_confidence:    int
    miss_count:          int
    v_z:                 float

    # Activity
    occupancy:           str
    zone:                str
    posture:             str
    motion:              str
    micro_state:         str
    is_walking:          bool
    fall_state:          str
    fall_confidence:     float
    occ_confidence:      float

    # Vital features
    aliveness_score:     float
    displacement_mm:     float
    vital_multiplier:    float

    # Respiration
    has_respiration:     bool
    rr_bpm:              float | None
    apnea_active:        bool
    breath_depth:        str

    # Diagnostics passthrough
    extras:              dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return a JSON-serialisable plain dict."""
        d = asdict(self)
        # Replace None with 0.0 / "" for CSV compat
        for k, v in d.items():
            if v is None:
                d[k] = ""
        return d


class FrameRecorder:
    """Rolling in-memory ring buffer of FrameRecord snapshots.

    Thread-safe read/write for use in the BedMonitorController background
    thread and GUI overlays.

    Args:
        capacity: Maximum number of frames to retain (older entries are evicted).
    """

    # CSV columns in output order (extras excluded)
    _CSV_FIELDS = [
        "frame_index", "timestamp", "elapsed_ms", "has_target",
        "track_x", "track_y", "track_z", "track_confidence", "miss_count", "v_z",
        "occupancy", "zone", "posture", "motion", "micro_state", "is_walking",
        "fall_state", "fall_confidence", "occ_confidence",
        "aliveness_score", "displacement_mm", "vital_multiplier",
        "has_respiration", "rr_bpm", "apnea_active", "breath_depth",
    ]

    def __init__(self, capacity: int = 1500) -> None:
        self._buf:         deque[FrameRecord] = deque(maxlen=capacity)
        self._prev_ts:     float | None       = None
        self._total_frames: int               = 0

    # ── Write ─────────────────────────────────────────────────────────────────

    def record(self, output: EngineOutput) -> None:
        """Capture a single EngineOutput frame into the ring buffer.

        Args:
            output: Typed EngineOutput from RadarEngine.process_frame().
        """
        now        = output.timestamp
        elapsed_ms = (now - self._prev_ts) * 1000.0 if self._prev_ts else 0.0
        self._prev_ts = now
        self._total_frames += 1

        # ── Unpack fields (safe defaults when stages were skipped) ────────────
        tracked    = output.tracked_target
        activity   = output.activity
        vf         = output.vital_features
        rm         = output.respiration_metrics

        rec = FrameRecord(
            frame_index      = output.frame_index,
            timestamp        = now,
            elapsed_ms       = round(elapsed_ms, 2),

            has_target       = output.has_target,
            track_x          = tracked.smoothed_x_m   if tracked and tracked.valid else None,
            track_y          = tracked.smoothed_y_m   if tracked and tracked.valid else None,
            track_z          = tracked.smoothed_z_m   if tracked and tracked.valid else None,
            track_confidence = tracked.confidence      if tracked else 0,
            miss_count       = tracked.miss_count      if tracked else 0,
            v_z              = tracked.v_z             if tracked and tracked.valid else 0.0,

            occupancy        = activity.occupancy.value   if activity else "empty",
            zone             = activity.zone              if activity else "No Occupant Detected",
            posture          = activity.posture.value     if activity else "unknown",
            motion           = activity.motion.value      if activity else "unknown",
            micro_state      = activity.micro_state.value if activity else "STABLE",
            is_walking       = activity.is_walking        if activity else False,
            fall_state       = activity.fall_state.value  if activity else "none",
            fall_confidence  = activity.fall_confidence   if activity else 0.0,
            occ_confidence   = activity.occupancy_confidence if activity else 0.0,

            aliveness_score  = vf.aliveness_score    if vf else 0.0,
            displacement_mm  = vf.displacement_mm    if vf else 0.0,
            vital_multiplier = vf.vital_multiplier   if vf else 1.0,

            has_respiration  = output.has_respiration,
            rr_bpm           = rm.rr_bpm             if rm else None,
            apnea_active     = rm.apnea_active        if rm else False,
            breath_depth     = rm.breath_depth        if rm else "--",

            extras           = dict(output.diagnostics),
        )
        self._buf.append(rec)

    # ── Read ──────────────────────────────────────────────────────────────────

    def last(self, n: int = 1) -> list[FrameRecord]:
        """Return the n most recent FrameRecord objects (newest last)."""
        buf = list(self._buf)
        return buf[-n:] if n <= len(buf) else buf

    def all_records(self) -> list[FrameRecord]:
        """Return all records currently in the buffer (oldest first)."""
        return list(self._buf)

    def query_last_seconds(self, seconds: float) -> list[FrameRecord]:
        """Return records from the last ``seconds`` of wall time."""
        cutoff = time.time() - seconds
        return [r for r in self._buf if r.timestamp >= cutoff]

    @property
    def total_frames(self) -> int:
        """Total number of frames recorded (including evicted)."""
        return self._total_frames

    @property
    def buffered_frames(self) -> int:
        """Number of frames currently in the ring buffer."""
        return len(self._buf)

    # ── Export ────────────────────────────────────────────────────────────────

    def export_csv(self, path: str | Path) -> Path:
        """Write all buffered records to a CSV file.

        Args:
            path: Target file path (.csv).

        Returns:
            Resolved Path to the written file.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for rec in self._buf:
                writer.writerow(rec.to_dict())
        logger.info("Exported %d frames to %s", len(self._buf), p)
        return p

    def export_json(self, path: str | Path) -> Path:
        """Write all buffered records to a JSON file.

        Args:
            path: Target file path (.json).

        Returns:
            Resolved Path to the written file.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        records = [r.to_dict() for r in self._buf]
        with p.open("w") as f:
            json.dump(records, f, indent=2)
        logger.info("Exported %d frames to %s", len(self._buf), p)
        return p

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Compute aggregate statistics over the buffered window.

        Returns:
            Dict with keys: n_frames, n_tracked, zone_counts, avg_rr_bpm,
            apnea_frames, avg_confidence, fps_estimate.
        """
        records = list(self._buf)
        if not records:
            return {}

        tracked_recs = [r for r in records if r.has_target]
        rr_vals      = [r.rr_bpm for r in records if r.rr_bpm and r.rr_bpm > 0]

        zone_counts: dict[str, int] = {}
        for r in records:
            zone_counts[r.zone] = zone_counts.get(r.zone, 0) + 1

        elapsed_ms = [r.elapsed_ms for r in records if r.elapsed_ms > 0]
        fps = 1000.0 / float(np.mean(elapsed_ms)) if elapsed_ms else 0.0

        return {
            "n_frames":       len(records),
            "n_tracked":      len(tracked_recs),
            "occupancy_pct":  round(100.0 * len(tracked_recs) / max(1, len(records)), 1),
            "zone_counts":    zone_counts,
            "avg_rr_bpm":     round(float(np.mean(rr_vals)), 2) if rr_vals else None,
            "apnea_frames":   sum(1 for r in records if r.apnea_active),
            "avg_confidence": round(float(np.mean([r.occ_confidence for r in records])), 1),
            "fps_estimate":   round(fps, 1),
        }

    def reset(self) -> None:
        """Clear the buffer and reset counters."""
        self._buf.clear()
        self._prev_ts      = None
        self._total_frames = 0
