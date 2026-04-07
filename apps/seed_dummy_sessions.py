"""Seed dummy monitoring sessions into the SQLite database for web-app testing."""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import time
import uuid
from datetime import datetime, timedelta, timezone


EVENT_INFO = "info"
EVENT_WARN = "warning"
EVENT_CRIT = "critical"


def _insert_event(
    conn: sqlite3.Connection,
    session_id: str,
    ts: float,
    event_type: str,
    severity: str,
    payload: dict,
) -> None:
    conn.execute(
        """
        INSERT INTO session_events (session_id, ts, event_type, severity, payload_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        (session_id, ts, event_type, severity, json.dumps(payload, separators=(",", ":"))),
    )


def _insert_rr_bucket(
    conn: sqlite3.Connection,
    session_id: str,
    bucket_ts: float,
    rr_values: list[float],
    quality_values: list[float],
    apnea_seconds: float,
) -> None:
    if rr_values:
        rr_sum = sum(rr_values)
        rr_sum_sq = sum(v * v for v in rr_values)
        rr_min = min(rr_values)
        rr_max = max(rr_values)
        rr_samples = len(rr_values)
        quality_sum = sum(quality_values)
    else:
        rr_sum = 0.0
        rr_sum_sq = 0.0
        rr_min = None
        rr_max = None
        rr_samples = 0
        quality_sum = 0.0

    conn.execute(
        """
        INSERT OR REPLACE INTO session_rr_10s
          (session_id, bucket_start_ts, rr_sum, rr_sum_sq, rr_min, rr_max, rr_samples, quality_sum, apnea_seconds)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (session_id, bucket_ts, rr_sum, rr_sum_sq, rr_min, rr_max, rr_samples, quality_sum, apnea_seconds),
    )


def _insert_activity_bucket(
    conn: sqlite3.Connection,
    session_id: str,
    bucket_ts: float,
    posture: str,
    motion_score: float,
    occ_conf: float,
    posture_conf: float,
    signal_conf: float,
) -> None:
    standing = 10.0 if posture == "standing" else 0.0
    sitting = 10.0 if posture == "sitting" else 0.0
    lying = 10.0 if posture == "lying_down" else 0.0
    fallen = 10.0 if posture == "fallen" else 0.0
    unknown = 10.0 if posture not in {"standing", "sitting", "lying_down", "fallen"} else 0.0

    conn.execute(
        """
        INSERT OR REPLACE INTO session_activity_10s
          (
            session_id, bucket_start_ts,
            posture_standing_sec, posture_sitting_sec, posture_lying_sec, posture_fallen_sec, posture_unknown_sec,
            motion_score_sum, motion_score_max, motion_score_samples,
            occ_conf_sum, occ_conf_samples,
            posture_conf_sum, posture_conf_samples,
            signal_conf_sum, signal_conf_samples
          )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            bucket_ts,
            standing,
            sitting,
            lying,
            fallen,
            unknown,
            motion_score,
            motion_score,
            1,
            occ_conf,
            1,
            posture_conf,
            1,
            signal_conf,
            1,
        ),
    )


def seed_dummy_data(db_path: str, sessions: int, days_back: int, reset: bool) -> None:
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    rng = random.Random(20260407)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS session_activity_10s (
          session_id TEXT NOT NULL,
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
        """
    )

    if reset:
        conn.execute("DELETE FROM session_activity_10s")
        conn.execute("DELETE FROM session_rr_10s")
        conn.execute("DELETE FROM session_events")
        conn.execute("DELETE FROM monitoring_sessions")
        conn.commit()

    now = datetime.now(tz=timezone.utc)
    subjects = ["subject_001", "subject_002", "subject_003"]
    devices = ["radar_001", "radar_002"]
    monitor_zone = "Bed"

    for i in range(sessions):
        day_offset = rng.randint(0, max(0, days_back - 1))
        session_date = now - timedelta(days=day_offset)
        start_hour = rng.randint(0, 23)
        start_min = rng.randint(0, 59)
        start_sec = rng.randint(0, 59)
        started_dt = session_date.replace(hour=start_hour, minute=start_min, second=start_sec, microsecond=0)

        duration_min = rng.randint(15, 95)
        duration_sec = duration_min * 60 + rng.randint(0, 59)
        ended_dt = started_dt + timedelta(seconds=duration_sec)

        sid = str(uuid.uuid4())
        subject_id = subjects[i % len(subjects)]
        device_id = devices[i % len(devices)]

        conn.execute(
            """
            INSERT INTO monitoring_sessions
              (id, subject_id, device_id, monitor_zone, started_at, ended_at, start_reason, end_reason, duration_sec)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sid,
                subject_id,
                device_id,
                monitor_zone,
                started_dt.timestamp(),
                ended_dt.timestamp(),
                "target_confirmed_in_monitor_zone",
                "monitor_zone_exit",
                float(duration_sec),
            ),
        )

        # Core events
        _insert_event(conn, sid, started_dt.timestamp(), "BED_ENTRY", EVENT_INFO, {"zone": "Bed - Center"})
        _insert_event(conn, sid, started_dt.timestamp() + 5, "OCCUPANCY_CHANGE", EVENT_INFO, {"from": "empty", "to": "entering", "confidence": 35.0})
        _insert_event(conn, sid, started_dt.timestamp() + 18, "OCCUPANCY_CHANGE", EVENT_INFO, {"from": "entering", "to": "monitoring", "confidence": 74.0})
        _insert_event(conn, sid, started_dt.timestamp() + 60, "POSTURE_CHANGE", EVENT_INFO, {"from": "unknown", "to": "lying_down", "confidence": 86.0})

        if duration_sec > 12 * 60:
            t = started_dt.timestamp() + rng.randint(7 * 60, 12 * 60)
            _insert_event(conn, sid, t, "POSTURE_CHANGE", EVENT_INFO, {"from": "lying_down", "to": "sitting", "confidence": 81.0})
            _insert_event(conn, sid, t + 50, "POSTURE_CHANGE", EVENT_INFO, {"from": "sitting", "to": "lying_down", "confidence": 84.0})

        # Apnea events in ~40% of sessions
        has_apnea = rng.random() < 0.4 and duration_sec > 20 * 60
        apnea_start_ts = None
        apnea_end_ts = None
        if has_apnea:
            apnea_start_ts = started_dt.timestamp() + rng.randint(8 * 60, max(9 * 60, duration_sec - 6 * 60))
            apnea_dur = rng.randint(20, 65)
            apnea_end_ts = min(apnea_start_ts + apnea_dur, ended_dt.timestamp() - 30)
            _insert_event(conn, sid, apnea_start_ts, "APNEA_START", EVENT_WARN, {})
            _insert_event(conn, sid, apnea_end_ts, "APNEA_END", EVENT_INFO, {})

        # Rare fall candidate / detected in ~8% sessions
        if rng.random() < 0.08 and duration_sec > 25 * 60:
            fall_t = started_dt.timestamp() + rng.randint(10 * 60, duration_sec - 5 * 60)
            _insert_event(conn, sid, fall_t, "FALL_CANDIDATE", EVENT_WARN, {"confidence": round(rng.uniform(42, 68), 1)})
            if rng.random() < 0.5:
                _insert_event(conn, sid, fall_t + 8, "FALL_DETECTED", EVENT_CRIT, {"confidence": round(rng.uniform(70, 92), 1)})
                _insert_event(conn, sid, fall_t + 40, "FALL_CLEARED", EVENT_INFO, {})

        _insert_event(conn, sid, ended_dt.timestamp(), "BED_EXIT", EVENT_INFO, {"zone": "No Occupant Detected"})

        # RR buckets (10s): use a realistic baseline with noise
        bucket_len = 10
        n_buckets = max(1, duration_sec // bucket_len)
        base_rr = rng.uniform(11.5, 17.5)
        for b in range(n_buckets):
            bucket_ts = started_dt.timestamp() + b * bucket_len
            # 1 to 3 RR samples per bucket (downsampled summary behavior)
            n_samples = rng.randint(1, 3)
            rr_vals = []
            q_vals = []
            for _ in range(n_samples):
                rr = max(0.0, base_rr + rng.uniform(-1.8, 1.8))
                q = max(0.2, min(0.98, rng.uniform(0.68, 0.95)))
                rr_vals.append(rr)
                q_vals.append(q)

            apnea_sec = 0.0
            if has_apnea and apnea_start_ts is not None and apnea_end_ts is not None:
                overlap_start = max(bucket_ts, apnea_start_ts)
                overlap_end = min(bucket_ts + bucket_len, apnea_end_ts)
                if overlap_end > overlap_start:
                    apnea_sec = overlap_end - overlap_start
                    # During apnea lower/invalid RR and quality
                    rr_vals = [0.0 if rng.random() < 0.7 else max(0.0, v - rng.uniform(6, 10)) for v in rr_vals]
                    q_vals = [max(0.2, q - rng.uniform(0.25, 0.5)) for q in q_vals]

            _insert_rr_bucket(conn, sid, bucket_ts, rr_vals, q_vals, apnea_sec)

            if b < int(0.1 * n_buckets):
                posture = "sitting"
            elif b > int(0.85 * n_buckets):
                posture = "sitting"
            else:
                posture = "lying_down"
            if rng.random() < 0.03:
                posture = "standing"
            if rng.random() < 0.01:
                posture = "fallen"

            base_motion = 0.08 if posture == "lying_down" else 0.25
            if posture == "standing":
                base_motion = 0.55
            if posture == "fallen":
                base_motion = 0.15
            motion_score = max(0.0, min(1.0, base_motion + rng.uniform(-0.06, 0.08)))
            if apnea_sec > 0:
                motion_score = max(0.0, min(1.0, motion_score - rng.uniform(0.03, 0.09)))

            occ_conf = max(45.0, min(100.0, rng.uniform(70, 96) - (0 if posture != "unknown" else 18)))
            posture_conf = max(30.0, min(100.0, rng.uniform(68, 95) - (16 if posture == "fallen" else 0)))
            rr_q = 0.85 if apnea_sec == 0 else 0.45
            signal_conf = (0.4 * posture_conf) + (0.3 * occ_conf) + (0.3 * (rr_q * 100.0))

            _insert_activity_bucket(
                conn,
                sid,
                bucket_ts,
                posture=posture,
                motion_score=motion_score,
                occ_conf=occ_conf,
                posture_conf=posture_conf,
                signal_conf=signal_conf,
            )

    conn.commit()
    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed dummy sessions/events/RR buckets for dashboard testing.")
    parser.add_argument("--db", default="data/monitoring.sqlite3", help="Path to sqlite DB.")
    parser.add_argument("--sessions", type=int, default=18, help="Number of dummy sessions to create.")
    parser.add_argument("--days-back", type=int, default=30, help="Distribute sessions across this many recent days.")
    parser.add_argument("--reset", action="store_true", help="Delete existing data in session tables first.")
    args = parser.parse_args()

    seed_dummy_data(args.db, sessions=args.sessions, days_back=args.days_back, reset=args.reset)
    print(f"Dummy data inserted into {args.db}")


if __name__ == "__main__":
    main()
