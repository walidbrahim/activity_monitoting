"""SQLite-backed healthcare session logging for bed monitoring.

Session rule:
    A session starts when a confirmed target is in a monitor zone.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from dataclasses import dataclass
from typing import Any

from radar_engine.core.enums import FallState
from radar_engine.core.models import EngineOutput


@dataclass(frozen=True)
class DbConfig:
    enabled: bool = True
    path: str = "data/monitoring.sqlite3"
    rr_bucket_sec: int = 10
    session_end_grace_sec: int = 10
    subject_id: str = "subject_001"
    device_id: str = "radar_001"


class SessionDbLogger:
    """Persists sessions, healthcare events, and RR rollups."""

    def __init__(self, cfg: DbConfig, monitor_zone_names: set[str], fps: float) -> None:
        self.cfg = cfg
        self.monitor_zone_names = monitor_zone_names
        self.fps = max(1e-6, float(fps))
        self._conn: sqlite3.Connection | None = None
        self._active_session_id: str | None = None
        self._active_started_at: float | None = None
        self._last_in_monitor_ts: float | None = None
        self._last_posture: str | None = None
        self._last_occupancy: str | None = None
        self._last_zone: str | None = None
        self._last_apnea: bool = False
        self._last_fall: str = FallState.NONE.value

        if self.cfg.enabled:
            self._open()
            self._ensure_schema()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def shutdown(self, ts: float | None = None) -> None:
        """Gracefully end any active session then close DB connection."""
        if self._conn is None:
            return
        if self._active_session_id is not None:
            end_ts = float(ts if ts is not None else self._last_in_monitor_ts or 0.0)
            if end_ts <= 0:
                import time as _time
                end_ts = _time.time()
            self._end_session(end_ts, "app_shutdown")
        self.close()

    def ingest(self, output: EngineOutput, frames: int = 1) -> None:
        if not self.cfg.enabled or self._conn is None:
            return
        ts = float(output.timestamp)
        act = output.activity
        rr = output.respiration_metrics
        has_target = bool(output.has_target and act and act.valid)
        zone = act.zone if (act and act.valid) else "No Occupant Detected"
        base_zone = zone.split(" - ", 1)[0] if zone else ""
        in_monitor = has_target and base_zone in self.monitor_zone_names

        if in_monitor:
            self._last_in_monitor_ts = ts

        if self._active_session_id is None and in_monitor:
            self._start_session(ts, base_zone)
            self._log_event(
                ts=ts,
                event_type="BED_ENTRY",
                severity="info",
                payload={"zone": zone},
            )

        if self._active_session_id is not None:
            grace = float(self.cfg.session_end_grace_sec)
            if not in_monitor:
                if self._last_in_monitor_ts is None or (ts - self._last_in_monitor_ts) >= grace:
                    self._log_event(
                        ts=ts,
                        event_type="BED_EXIT",
                        severity="info",
                        payload={"zone": zone},
                    )
                    self._end_session(ts, "monitor_zone_exit")
                    return

            self._log_transitions(output)
            self._log_rr_bucket(ts, rr, frames)
            self._log_activity_bucket(ts, output, frames)

    def _open(self) -> None:
        os.makedirs(os.path.dirname(self.cfg.path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(
            self.cfg.path,
            timeout=30.0,
            check_same_thread=False,
        )
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")

    def _ensure_schema(self) -> None:
        assert self._conn is not None
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS monitoring_sessions (
              id TEXT PRIMARY KEY,
              subject_id TEXT NOT NULL,
              device_id TEXT NOT NULL,
              monitor_zone TEXT NOT NULL,
              started_at REAL NOT NULL,
              ended_at REAL,
              start_reason TEXT NOT NULL,
              end_reason TEXT,
              duration_sec REAL
            );

            CREATE TABLE IF NOT EXISTS session_events (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              session_id TEXT NOT NULL REFERENCES monitoring_sessions(id) ON DELETE CASCADE,
              ts REAL NOT NULL,
              event_type TEXT NOT NULL,
              severity TEXT NOT NULL,
              payload_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS session_rr_10s (
              session_id TEXT NOT NULL REFERENCES monitoring_sessions(id) ON DELETE CASCADE,
              bucket_start_ts REAL NOT NULL,
              rr_sum REAL NOT NULL DEFAULT 0,
              rr_sum_sq REAL NOT NULL DEFAULT 0,
              rr_min REAL,
              rr_max REAL,
              rr_samples INTEGER NOT NULL DEFAULT 0,
              quality_sum REAL NOT NULL DEFAULT 0,
              apnea_seconds REAL NOT NULL DEFAULT 0,
              PRIMARY KEY (session_id, bucket_start_ts)
            );

            CREATE TABLE IF NOT EXISTS session_activity_10s (
              session_id TEXT NOT NULL REFERENCES monitoring_sessions(id) ON DELETE CASCADE,
              bucket_start_ts REAL NOT NULL,
              posture_standing_sec REAL NOT NULL DEFAULT 0,
              posture_sitting_sec REAL NOT NULL DEFAULT 0,
              posture_lying_sec REAL NOT NULL DEFAULT 0,
              posture_fallen_sec REAL NOT NULL DEFAULT 0,
              posture_unknown_sec REAL NOT NULL DEFAULT 0,
              motion_score_sum REAL NOT NULL DEFAULT 0,
              motion_score_max REAL NOT NULL DEFAULT 0,
              motion_score_samples INTEGER NOT NULL DEFAULT 0,
              occ_conf_sum REAL NOT NULL DEFAULT 0,
              occ_conf_samples INTEGER NOT NULL DEFAULT 0,
              posture_conf_sum REAL NOT NULL DEFAULT 0,
              posture_conf_samples INTEGER NOT NULL DEFAULT 0,
              signal_conf_sum REAL NOT NULL DEFAULT 0,
              signal_conf_samples INTEGER NOT NULL DEFAULT 0,
              PRIMARY KEY (session_id, bucket_start_ts)
            );

            CREATE INDEX IF NOT EXISTS idx_sessions_started_at
              ON monitoring_sessions(started_at);
            CREATE INDEX IF NOT EXISTS idx_events_session_ts
              ON session_events(session_id, ts);
            CREATE INDEX IF NOT EXISTS idx_events_type_ts
              ON session_events(event_type, ts);
            CREATE INDEX IF NOT EXISTS idx_rr_bucket_ts
              ON session_rr_10s(bucket_start_ts);
            CREATE INDEX IF NOT EXISTS idx_activity_bucket_ts
              ON session_activity_10s(bucket_start_ts);

            CREATE VIEW IF NOT EXISTS v_rr_bucket_stats AS
            SELECT
              session_id,
              bucket_start_ts,
              rr_samples,
              CASE WHEN rr_samples > 0 THEN rr_sum / rr_samples END AS rr_mean,
              rr_min,
              rr_max,
              CASE
                WHEN rr_samples > 1
                THEN sqrt((rr_sum_sq / rr_samples) - ((rr_sum / rr_samples) * (rr_sum / rr_samples)))
                ELSE 0.0
              END AS rr_std,
              CASE WHEN rr_samples > 0 THEN quality_sum / rr_samples ELSE 0 END AS quality_mean,
              apnea_seconds
            FROM session_rr_10s;

            CREATE VIEW IF NOT EXISTS v_daily_summary AS
            SELECT
              date(started_at, 'unixepoch') AS day,
              COUNT(*) AS n_sessions,
              SUM(COALESCE(duration_sec, 0)) AS monitored_seconds,
              AVG(COALESCE(duration_sec, 0)) AS avg_session_seconds
            FROM monitoring_sessions
            GROUP BY day;

            CREATE VIEW IF NOT EXISTS v_weekly_summary AS
            SELECT
              strftime('%Y-W%W', started_at, 'unixepoch') AS year_week,
              COUNT(*) AS n_sessions,
              SUM(COALESCE(duration_sec, 0)) AS monitored_seconds
            FROM monitoring_sessions
            GROUP BY year_week;

            CREATE VIEW IF NOT EXISTS v_monthly_summary AS
            SELECT
              strftime('%Y-%m', started_at, 'unixepoch') AS year_month,
              COUNT(*) AS n_sessions,
              SUM(COALESCE(duration_sec, 0)) AS monitored_seconds
            FROM monitoring_sessions
            GROUP BY year_month;

            CREATE VIEW IF NOT EXISTS v_activity_bucket_stats AS
            SELECT
              session_id,
              bucket_start_ts,
              posture_standing_sec,
              posture_sitting_sec,
              posture_lying_sec,
              posture_fallen_sec,
              posture_unknown_sec,
              CASE
                WHEN motion_score_samples > 0
                THEN motion_score_sum / motion_score_samples
                ELSE 0
              END AS motion_score_mean,
              motion_score_max,
              CASE WHEN occ_conf_samples > 0 THEN occ_conf_sum / occ_conf_samples ELSE 0 END AS occ_conf_mean,
              CASE
                WHEN posture_conf_samples > 0
                THEN posture_conf_sum / posture_conf_samples
                ELSE 0
              END AS posture_conf_mean,
              CASE
                WHEN signal_conf_samples > 0
                THEN signal_conf_sum / signal_conf_samples
                ELSE 0
              END AS signal_conf_mean
            FROM session_activity_10s;
            """
        )
        self._conn.commit()

    def _start_session(self, ts: float, base_zone: str) -> None:
        assert self._conn is not None
        sid = str(uuid.uuid4())
        self._active_session_id = sid
        self._active_started_at = ts
        self._last_posture = None
        self._last_occupancy = None
        self._last_zone = None
        self._last_apnea = False
        self._last_fall = FallState.NONE.value
        self._conn.execute(
            """
            INSERT INTO monitoring_sessions
              (id, subject_id, device_id, monitor_zone, started_at, start_reason)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (sid, self.cfg.subject_id, self.cfg.device_id, base_zone, ts, "target_confirmed_in_monitor_zone"),
        )
        self._conn.commit()

    def _end_session(self, ts: float, reason: str) -> None:
        assert self._conn is not None
        if self._active_session_id is None or self._active_started_at is None:
            return
        duration = max(0.0, ts - self._active_started_at)
        self._conn.execute(
            """
            UPDATE monitoring_sessions
            SET ended_at = ?, end_reason = ?, duration_sec = ?
            WHERE id = ?
            """,
            (ts, reason, duration, self._active_session_id),
        )
        self._conn.commit()
        self._active_session_id = None
        self._active_started_at = None

    def _log_event(self, ts: float, event_type: str, severity: str, payload: dict[str, Any]) -> None:
        assert self._conn is not None
        if self._active_session_id is None:
            return
        self._conn.execute(
            """
            INSERT INTO session_events (session_id, ts, event_type, severity, payload_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                self._active_session_id,
                ts,
                event_type,
                severity,
                json.dumps(payload, separators=(",", ":"), ensure_ascii=True),
            ),
        )
        self._conn.commit()

    def _log_transitions(self, output: EngineOutput) -> None:
        act = output.activity
        rm = output.respiration_metrics
        if act is None or not act.valid:
            return
        ts = float(output.timestamp)
        posture = act.posture.value
        occupancy = act.occupancy.value
        zone = act.zone

        if self._last_zone is None:
            self._last_zone = zone
        elif zone != self._last_zone:
            self._log_event(ts, "ZONE_CHANGE", "info", {"from": self._last_zone, "to": zone})
            self._last_zone = zone

        if self._last_posture is None:
            self._last_posture = posture
        elif posture != self._last_posture:
            self._log_event(
                ts,
                "POSTURE_CHANGE",
                "info",
                {"from": self._last_posture, "to": posture, "confidence": act.posture_confidence},
            )
            self._last_posture = posture

        if self._last_occupancy is None:
            self._last_occupancy = occupancy
        elif occupancy != self._last_occupancy:
            self._log_event(
                ts,
                "OCCUPANCY_CHANGE",
                "info",
                {"from": self._last_occupancy, "to": occupancy, "confidence": act.occupancy_confidence},
            )
            self._last_occupancy = occupancy

        fall = act.fall_state.value
        if fall != self._last_fall:
            if fall == FallState.DETECTED.value:
                self._log_event(ts, "FALL_DETECTED", "critical", {"confidence": act.fall_confidence})
            elif fall == FallState.CANDIDATE.value:
                self._log_event(ts, "FALL_CANDIDATE", "warning", {"confidence": act.fall_confidence})
            elif self._last_fall in (FallState.DETECTED.value, FallState.CANDIDATE.value):
                self._log_event(ts, "FALL_CLEARED", "info", {})
            self._last_fall = fall

        apnea_active = bool(rm.apnea_active) if rm else False
        if apnea_active and not self._last_apnea:
            self._log_event(ts, "APNEA_START", "warning", {})
        elif (not apnea_active) and self._last_apnea:
            self._log_event(ts, "APNEA_END", "info", {})
        self._last_apnea = apnea_active

    def _log_rr_bucket(self, ts: float, rm: Any, frames: int) -> None:
        assert self._conn is not None
        if self._active_session_id is None or rm is None:
            return

        bucket = float(int(ts // self.cfg.rr_bucket_sec) * self.cfg.rr_bucket_sec)
        rr_bpm = float(rm.rr_bpm) if rm.rr_bpm is not None else None
        apnea_seconds = float(frames) / self.fps if bool(rm.apnea_active) else 0.0
        quality_score = float(rm.rr_quality.score) if rm.rr_quality else 0.0

        if rr_bpm is None:
            self._conn.execute(
                """
                INSERT INTO session_rr_10s
                  (session_id, bucket_start_ts, apnea_seconds)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id, bucket_start_ts) DO UPDATE SET
                  apnea_seconds = session_rr_10s.apnea_seconds + excluded.apnea_seconds
                """,
                (self._active_session_id, bucket, apnea_seconds),
            )
            self._conn.commit()
            return

        self._conn.execute(
            """
            INSERT INTO session_rr_10s
              (session_id, bucket_start_ts, rr_sum, rr_sum_sq, rr_min, rr_max, rr_samples, quality_sum, apnea_seconds)
            VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
            ON CONFLICT(session_id, bucket_start_ts) DO UPDATE SET
              rr_sum = session_rr_10s.rr_sum + excluded.rr_sum,
              rr_sum_sq = session_rr_10s.rr_sum_sq + excluded.rr_sum_sq,
              rr_min = CASE
                WHEN session_rr_10s.rr_min IS NULL THEN excluded.rr_min
                WHEN excluded.rr_min < session_rr_10s.rr_min THEN excluded.rr_min
                ELSE session_rr_10s.rr_min
              END,
              rr_max = CASE
                WHEN session_rr_10s.rr_max IS NULL THEN excluded.rr_max
                WHEN excluded.rr_max > session_rr_10s.rr_max THEN excluded.rr_max
                ELSE session_rr_10s.rr_max
              END,
              rr_samples = session_rr_10s.rr_samples + 1,
              quality_sum = session_rr_10s.quality_sum + excluded.quality_sum,
              apnea_seconds = session_rr_10s.apnea_seconds + excluded.apnea_seconds
            """,
            (
                self._active_session_id,
                bucket,
                rr_bpm,
                rr_bpm * rr_bpm,
                rr_bpm,
                rr_bpm,
                quality_score,
                apnea_seconds,
            ),
        )
        self._conn.commit()

    def _log_activity_bucket(self, ts: float, output: EngineOutput, frames: int) -> None:
        assert self._conn is not None
        if self._active_session_id is None:
            return

        act = output.activity
        if act is None or not act.valid:
            return

        bucket = float(int(ts // self.cfg.rr_bucket_sec) * self.cfg.rr_bucket_sec)
        elapsed_sec = max(0.0, float(frames) / self.fps)
        posture = (act.posture.value or "").lower()

        standing_sec = elapsed_sec if posture == "standing" else 0.0
        sitting_sec = elapsed_sec if posture == "sitting" else 0.0
        lying_sec = elapsed_sec if posture == "lying_down" else 0.0
        fallen_sec = elapsed_sec if posture == "fallen" else 0.0
        unknown_sec = elapsed_sec if posture not in {"standing", "sitting", "lying_down", "fallen"} else 0.0

        motion_score = max(0.0, min(1.0, float(act.motion_score)))
        occ_conf = max(0.0, min(100.0, float(act.occupancy_confidence)))
        posture_conf = max(0.0, min(100.0, float(act.posture_confidence)))
        rr_q = 0.0
        if output.respiration_metrics and output.respiration_metrics.rr_quality:
            rr_q = max(0.0, min(1.0, float(output.respiration_metrics.rr_quality.score)))
        signal_conf = (0.4 * posture_conf) + (0.3 * occ_conf) + (0.3 * (rr_q * 100.0))

        self._conn.execute(
            """
            INSERT INTO session_activity_10s
              (
                session_id, bucket_start_ts,
                posture_standing_sec, posture_sitting_sec, posture_lying_sec, posture_fallen_sec, posture_unknown_sec,
                motion_score_sum, motion_score_max, motion_score_samples,
                occ_conf_sum, occ_conf_samples,
                posture_conf_sum, posture_conf_samples,
                signal_conf_sum, signal_conf_samples
              )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, 1, ?, 1, ?, 1)
            ON CONFLICT(session_id, bucket_start_ts) DO UPDATE SET
              posture_standing_sec = session_activity_10s.posture_standing_sec + excluded.posture_standing_sec,
              posture_sitting_sec = session_activity_10s.posture_sitting_sec + excluded.posture_sitting_sec,
              posture_lying_sec = session_activity_10s.posture_lying_sec + excluded.posture_lying_sec,
              posture_fallen_sec = session_activity_10s.posture_fallen_sec + excluded.posture_fallen_sec,
              posture_unknown_sec = session_activity_10s.posture_unknown_sec + excluded.posture_unknown_sec,
              motion_score_sum = session_activity_10s.motion_score_sum + excluded.motion_score_sum,
              motion_score_max = CASE
                WHEN excluded.motion_score_max > session_activity_10s.motion_score_max
                THEN excluded.motion_score_max
                ELSE session_activity_10s.motion_score_max
              END,
              motion_score_samples = session_activity_10s.motion_score_samples + 1,
              occ_conf_sum = session_activity_10s.occ_conf_sum + excluded.occ_conf_sum,
              occ_conf_samples = session_activity_10s.occ_conf_samples + 1,
              posture_conf_sum = session_activity_10s.posture_conf_sum + excluded.posture_conf_sum,
              posture_conf_samples = session_activity_10s.posture_conf_samples + 1,
              signal_conf_sum = session_activity_10s.signal_conf_sum + excluded.signal_conf_sum,
              signal_conf_samples = session_activity_10s.signal_conf_samples + 1
            """,
            (
                self._active_session_id,
                bucket,
                standing_sec,
                sitting_sec,
                lying_sec,
                fallen_sec,
                unknown_sec,
                motion_score,
                motion_score,
                occ_conf,
                posture_conf,
                signal_conf,
            ),
        )
        self._conn.commit()
