"""Prototype web dashboard for monitoring SQLite database."""

from __future__ import annotations

import argparse
import json
import sqlite3
from calendar import monthrange
from datetime import datetime, timedelta, timezone
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request


def create_app(db_path: str = "data/monitoring.sqlite3") -> Flask:
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).resolve().parent / "web" / "templates"),
    )
    app.config["DB_PATH"] = db_path

    def get_conn() -> sqlite3.Connection:
        conn = sqlite3.connect(
            app.config["DB_PATH"],
            timeout=30.0,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
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
            CREATE VIEW IF NOT EXISTS v_activity_bucket_stats AS
            SELECT
              session_id,
              bucket_start_ts,
              posture_standing_sec,
              posture_sitting_sec,
              posture_lying_sec,
              posture_fallen_sec,
              posture_unknown_sec,
              CASE WHEN motion_score_samples > 0 THEN motion_score_sum / motion_score_samples ELSE 0 END AS motion_score_mean,
              motion_score_max,
              CASE WHEN occ_conf_samples > 0 THEN occ_conf_sum / occ_conf_samples ELSE 0 END AS occ_conf_mean,
              CASE WHEN posture_conf_samples > 0 THEN posture_conf_sum / posture_conf_samples ELSE 0 END AS posture_conf_mean,
              CASE WHEN signal_conf_samples > 0 THEN signal_conf_sum / signal_conf_samples ELSE 0 END AS signal_conf_mean
            FROM session_activity_10s;
            """
        )
        return conn

    def fetch_all(conn: sqlite3.Connection, query: str, params: tuple = ()) -> list[dict]:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def fetch_one(conn: sqlite3.Connection, query: str, params: tuple = ()) -> dict | None:
        row = conn.execute(query, params).fetchone()
        return dict(row) if row else None

    @app.template_filter("dt")
    def fmt_dt(ts: float | None) -> str:
        if ts is None:
            return "-"
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    @app.template_filter("dur")
    def fmt_dur(seconds: float | None) -> str:
        if seconds is None:
            return "-"
        sec = int(max(0, float(seconds)))
        h = sec // 3600
        m = (sec % 3600) // 60
        s = sec % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    @app.route("/")
    def index():
        conn = get_conn()
        try:
            daily = fetch_all(conn, "SELECT * FROM v_daily_summary ORDER BY day DESC LIMIT 14")
            weekly = fetch_all(conn, "SELECT * FROM v_weekly_summary ORDER BY year_week DESC LIMIT 12")
            monthly = fetch_all(conn, "SELECT * FROM v_monthly_summary ORDER BY year_month DESC LIMIT 12")
            kpi = fetch_one(
                conn,
                """
                SELECT
                  COUNT(*) AS sessions_total,
                  SUM(COALESCE(duration_sec,0)) AS monitored_sec_total,
                  SUM(CASE WHEN ended_at IS NULL THEN 1 ELSE 0 END) AS open_sessions,
                  AVG(COALESCE(duration_sec,0)) AS avg_session_sec
                FROM monitoring_sessions
                """,
            ) or {}
            kpi_24h = fetch_one(
                conn,
                """
                SELECT
                  COUNT(*) AS sessions_24h,
                  SUM(COALESCE(duration_sec,0)) AS monitored_sec_24h
                FROM monitoring_sessions
                WHERE started_at >= strftime('%s','now') - 86400
                """,
            ) or {}
            apnea_24h = fetch_one(
                conn,
                """
                SELECT COUNT(*) AS apnea_events_24h
                FROM session_events
                WHERE event_type='APNEA_START' AND ts >= strftime('%s','now') - 86400
                """,
            ) or {"apnea_events_24h": 0}
            critical_24h = fetch_one(
                conn,
                """
                SELECT COUNT(*) AS critical_events_24h
                FROM session_events
                WHERE severity='critical' AND ts >= strftime('%s','now') - 86400
                """,
            ) or {"critical_events_24h": 0}
            events = fetch_all(
                conn,
                """
                SELECT event_type, COUNT(*) AS n
                FROM session_events
                GROUP BY event_type
                ORDER BY n DESC
                LIMIT 12
                """,
            )
            recent_sessions = fetch_all(
                conn,
                """
                SELECT id, subject_id, device_id, monitor_zone, started_at, ended_at, duration_sec
                FROM monitoring_sessions
                ORDER BY started_at DESC
                LIMIT 20
                """,
            )
            severity_counts = fetch_all(
                conn,
                """
                SELECT severity, COUNT(*) AS n
                FROM session_events
                GROUP BY severity
                ORDER BY n DESC
                """,
            )

            latest_started = fetch_one(conn, "SELECT MAX(started_at) AS ts FROM monitoring_sessions")
            anchor_dt = datetime.now(tz=timezone.utc)
            if latest_started and latest_started.get("ts"):
                anchor_dt = datetime.fromtimestamp(float(latest_started["ts"]), tz=timezone.utc)
            selected_day = request.args.get("day", "").strip()
            if selected_day:
                try:
                    anchor_dt = datetime.strptime(selected_day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                except ValueError:
                    selected_day = ""
            if not selected_day:
                selected_day = anchor_dt.strftime("%Y-%m-%d")

            # ── Daily view: hourly RR + events ─────────────────────────────
            day_key = anchor_dt.strftime("%Y-%m-%d")
            h_rr_rows = fetch_all(
                conn,
                """
                SELECT
                  strftime('%H', bucket_start_ts, 'unixepoch') AS hh,
                  AVG(rr_mean) AS rr_avg,
                  SUM(apnea_seconds) AS apnea_sec
                FROM v_rr_bucket_stats
                WHERE date(bucket_start_ts, 'unixepoch') = ?
                GROUP BY hh
                """,
                (day_key,),
            )
            h_act_rows = fetch_all(
                conn,
                """
                SELECT
                  strftime('%H', bucket_start_ts, 'unixepoch') AS hh,
                  SUM(posture_standing_sec) AS standing_sec,
                  SUM(posture_sitting_sec) AS sitting_sec,
                  SUM(posture_lying_sec) AS lying_sec,
                  SUM(posture_fallen_sec) AS fallen_sec,
                  SUM(posture_unknown_sec) AS unknown_sec,
                  AVG(motion_score_mean) AS motion_avg,
                  AVG(occ_conf_mean) AS occ_conf_avg,
                  AVG(posture_conf_mean) AS posture_conf_avg,
                  AVG(signal_conf_mean) AS signal_conf_avg
                FROM v_activity_bucket_stats
                WHERE date(bucket_start_ts, 'unixepoch') = ?
                GROUP BY hh
                """,
                (day_key,),
            )
            h_ev_rows = fetch_all(
                conn,
                """
                SELECT
                  strftime('%H', ts, 'unixepoch') AS hh,
                  SUM(CASE WHEN event_type='BED_ENTRY' THEN 1 ELSE 0 END) AS bed_entry,
                  SUM(CASE WHEN event_type='BED_EXIT' THEN 1 ELSE 0 END) AS bed_exit,
                  SUM(CASE WHEN event_type='APNEA_START' THEN 1 ELSE 0 END) AS apnea_start
                FROM session_events
                WHERE date(ts, 'unixepoch') = ?
                GROUP BY hh
                """,
                (day_key,),
            )
            day_labels = [f"{h:02d}:00" for h in range(24)]
            day_rr = [None] * 24
            day_apnea_sec = [0.0] * 24
            day_entry = [0] * 24
            day_exit = [0] * 24
            day_apnea_n = [0] * 24
            day_standing = [0.0] * 24
            day_sitting = [0.0] * 24
            day_lying = [0.0] * 24
            day_fallen = [0.0] * 24
            day_unknown = [0.0] * 24
            day_motion = [None] * 24
            day_occ_conf = [None] * 24
            day_posture_conf = [None] * 24
            day_signal_conf = [None] * 24
            for r in h_rr_rows:
                i = int(r["hh"])
                day_rr[i] = round(float(r["rr_avg"] or 0), 2)
                day_apnea_sec[i] = round(float(r["apnea_sec"] or 0), 1)
            for r in h_act_rows:
                i = int(r["hh"])
                day_standing[i] = round(float(r["standing_sec"] or 0), 1)
                day_sitting[i] = round(float(r["sitting_sec"] or 0), 1)
                day_lying[i] = round(float(r["lying_sec"] or 0), 1)
                day_fallen[i] = round(float(r["fallen_sec"] or 0), 1)
                day_unknown[i] = round(float(r["unknown_sec"] or 0), 1)
                day_motion[i] = round(float(r["motion_avg"] or 0), 3)
                day_occ_conf[i] = round(float(r["occ_conf_avg"] or 0), 1)
                day_posture_conf[i] = round(float(r["posture_conf_avg"] or 0), 1)
                day_signal_conf[i] = round(float(r["signal_conf_avg"] or 0), 1)
            for r in h_ev_rows:
                i = int(r["hh"])
                day_entry[i] = int(r["bed_entry"] or 0)
                day_exit[i] = int(r["bed_exit"] or 0)
                day_apnea_n[i] = int(r["apnea_start"] or 0)

            # ── Weekly view: daily RR + events ──────────────────────────────
            monday = anchor_dt.replace(hour=0, minute=0, second=0, microsecond=0)
            monday = monday.replace(day=anchor_dt.day) - timedelta(days=(anchor_dt.weekday()))
            week_dates = [(monday + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
            week_labels = [(monday + timedelta(days=i)).strftime("%a %d") for i in range(7)]
            w_rr_rows = fetch_all(
                conn,
                """
                SELECT
                  date(bucket_start_ts, 'unixepoch') AS d,
                  AVG(rr_mean) AS rr_avg,
                  SUM(apnea_seconds) AS apnea_sec
                FROM v_rr_bucket_stats
                WHERE date(bucket_start_ts, 'unixepoch') BETWEEN ? AND ?
                GROUP BY d
                """,
                (week_dates[0], week_dates[-1]),
            )
            w_act_rows = fetch_all(
                conn,
                """
                SELECT
                  date(bucket_start_ts, 'unixepoch') AS d,
                  SUM(posture_standing_sec) AS standing_sec,
                  SUM(posture_sitting_sec) AS sitting_sec,
                  SUM(posture_lying_sec) AS lying_sec,
                  SUM(posture_fallen_sec) AS fallen_sec,
                  SUM(posture_unknown_sec) AS unknown_sec,
                  AVG(motion_score_mean) AS motion_avg,
                  AVG(occ_conf_mean) AS occ_conf_avg,
                  AVG(posture_conf_mean) AS posture_conf_avg,
                  AVG(signal_conf_mean) AS signal_conf_avg
                FROM v_activity_bucket_stats
                WHERE date(bucket_start_ts, 'unixepoch') BETWEEN ? AND ?
                GROUP BY d
                """,
                (week_dates[0], week_dates[-1]),
            )
            w_ev_rows = fetch_all(
                conn,
                """
                SELECT
                  date(ts, 'unixepoch') AS d,
                  SUM(CASE WHEN event_type='BED_ENTRY' THEN 1 ELSE 0 END) AS bed_entry,
                  SUM(CASE WHEN event_type='BED_EXIT' THEN 1 ELSE 0 END) AS bed_exit,
                  SUM(CASE WHEN event_type='APNEA_START' THEN 1 ELSE 0 END) AS apnea_start
                FROM session_events
                WHERE date(ts, 'unixepoch') BETWEEN ? AND ?
                GROUP BY d
                """,
                (week_dates[0], week_dates[-1]),
            )
            w_rr_map = {r["d"]: r for r in w_rr_rows}
            w_act_map = {r["d"]: r for r in w_act_rows}
            w_ev_map = {r["d"]: r for r in w_ev_rows}
            week_rr = []
            week_apnea_sec = []
            week_entry = []
            week_exit = []
            week_apnea_n = []
            week_standing = []
            week_sitting = []
            week_lying = []
            week_fallen = []
            week_unknown = []
            week_motion = []
            week_occ_conf = []
            week_posture_conf = []
            week_signal_conf = []
            for d in week_dates:
                rr_r = w_rr_map.get(d, {})
                act_r = w_act_map.get(d, {})
                ev_r = w_ev_map.get(d, {})
                week_rr.append(round(float(rr_r.get("rr_avg", 0) or 0), 2) if rr_r else None)
                week_apnea_sec.append(round(float(rr_r.get("apnea_sec", 0) or 0), 1))
                week_entry.append(int(ev_r.get("bed_entry", 0) or 0))
                week_exit.append(int(ev_r.get("bed_exit", 0) or 0))
                week_apnea_n.append(int(ev_r.get("apnea_start", 0) or 0))
                week_standing.append(round(float(act_r.get("standing_sec", 0) or 0), 1))
                week_sitting.append(round(float(act_r.get("sitting_sec", 0) or 0), 1))
                week_lying.append(round(float(act_r.get("lying_sec", 0) or 0), 1))
                week_fallen.append(round(float(act_r.get("fallen_sec", 0) or 0), 1))
                week_unknown.append(round(float(act_r.get("unknown_sec", 0) or 0), 1))
                week_motion.append(round(float(act_r.get("motion_avg", 0) or 0), 3) if act_r else None)
                week_occ_conf.append(round(float(act_r.get("occ_conf_avg", 0) or 0), 1) if act_r else None)
                week_posture_conf.append(round(float(act_r.get("posture_conf_avg", 0) or 0), 1) if act_r else None)
                week_signal_conf.append(round(float(act_r.get("signal_conf_avg", 0) or 0), 1) if act_r else None)

            # ── Monthly view: day RR + events ───────────────────────────────
            y = anchor_dt.year
            m = anchor_dt.month
            days_in_month = monthrange(y, m)[1]
            month_dates = [datetime(y, m, d, tzinfo=timezone.utc).strftime("%Y-%m-%d") for d in range(1, days_in_month + 1)]
            month_labels = [f"{d:02d}" for d in range(1, days_in_month + 1)]
            m_rr_rows = fetch_all(
                conn,
                """
                SELECT
                  date(bucket_start_ts, 'unixepoch') AS d,
                  AVG(rr_mean) AS rr_avg,
                  SUM(apnea_seconds) AS apnea_sec
                FROM v_rr_bucket_stats
                WHERE strftime('%Y-%m', bucket_start_ts, 'unixepoch') = ?
                GROUP BY d
                """,
                (f"{y:04d}-{m:02d}",),
            )
            m_act_rows = fetch_all(
                conn,
                """
                SELECT
                  date(bucket_start_ts, 'unixepoch') AS d,
                  SUM(posture_standing_sec) AS standing_sec,
                  SUM(posture_sitting_sec) AS sitting_sec,
                  SUM(posture_lying_sec) AS lying_sec,
                  SUM(posture_fallen_sec) AS fallen_sec,
                  SUM(posture_unknown_sec) AS unknown_sec,
                  AVG(motion_score_mean) AS motion_avg,
                  AVG(occ_conf_mean) AS occ_conf_avg,
                  AVG(posture_conf_mean) AS posture_conf_avg,
                  AVG(signal_conf_mean) AS signal_conf_avg
                FROM v_activity_bucket_stats
                WHERE strftime('%Y-%m', bucket_start_ts, 'unixepoch') = ?
                GROUP BY d
                """,
                (f"{y:04d}-{m:02d}",),
            )
            m_ev_rows = fetch_all(
                conn,
                """
                SELECT
                  date(ts, 'unixepoch') AS d,
                  SUM(CASE WHEN event_type='BED_ENTRY' THEN 1 ELSE 0 END) AS bed_entry,
                  SUM(CASE WHEN event_type='BED_EXIT' THEN 1 ELSE 0 END) AS bed_exit,
                  SUM(CASE WHEN event_type='APNEA_START' THEN 1 ELSE 0 END) AS apnea_start
                FROM session_events
                WHERE strftime('%Y-%m', ts, 'unixepoch') = ?
                GROUP BY d
                """,
                (f"{y:04d}-{m:02d}",),
            )
            m_rr_map = {r["d"]: r for r in m_rr_rows}
            m_act_map = {r["d"]: r for r in m_act_rows}
            m_ev_map = {r["d"]: r for r in m_ev_rows}
            month_rr = []
            month_apnea_sec = []
            month_entry = []
            month_exit = []
            month_apnea_n = []
            month_standing = []
            month_sitting = []
            month_lying = []
            month_fallen = []
            month_unknown = []
            month_motion = []
            month_occ_conf = []
            month_posture_conf = []
            month_signal_conf = []
            for d in month_dates:
                rr_r = m_rr_map.get(d, {})
                act_r = m_act_map.get(d, {})
                ev_r = m_ev_map.get(d, {})
                month_rr.append(round(float(rr_r.get("rr_avg", 0) or 0), 2) if rr_r else None)
                month_apnea_sec.append(round(float(rr_r.get("apnea_sec", 0) or 0), 1))
                month_entry.append(int(ev_r.get("bed_entry", 0) or 0))
                month_exit.append(int(ev_r.get("bed_exit", 0) or 0))
                month_apnea_n.append(int(ev_r.get("apnea_start", 0) or 0))
                month_standing.append(round(float(act_r.get("standing_sec", 0) or 0), 1))
                month_sitting.append(round(float(act_r.get("sitting_sec", 0) or 0), 1))
                month_lying.append(round(float(act_r.get("lying_sec", 0) or 0), 1))
                month_fallen.append(round(float(act_r.get("fallen_sec", 0) or 0), 1))
                month_unknown.append(round(float(act_r.get("unknown_sec", 0) or 0), 1))
                month_motion.append(round(float(act_r.get("motion_avg", 0) or 0), 3) if act_r else None)
                month_occ_conf.append(round(float(act_r.get("occ_conf_avg", 0) or 0), 1) if act_r else None)
                month_posture_conf.append(round(float(act_r.get("posture_conf_avg", 0) or 0), 1) if act_r else None)
                month_signal_conf.append(round(float(act_r.get("signal_conf_avg", 0) or 0), 1) if act_r else None)

            chart_data = {
                "event_labels": [e["event_type"] for e in events],
                "event_counts": [e["n"] for e in events],
                "severity_labels": [s["severity"] for s in severity_counts],
                "severity_counts": [s["n"] for s in severity_counts],
                "daily": {
                    "title": day_key,
                    "labels": day_labels,
                    "rr": day_rr,
                    "apnea_sec": day_apnea_sec,
                    "entry": day_entry,
                    "exit": day_exit,
                    "apnea_n": day_apnea_n,
                    "posture": {
                        "standing": day_standing,
                        "sitting": day_sitting,
                        "lying": day_lying,
                        "fallen": day_fallen,
                        "unknown": day_unknown,
                    },
                    "motion_score": day_motion,
                    "confidence": {
                        "occupancy": day_occ_conf,
                        "posture": day_posture_conf,
                        "signal": day_signal_conf,
                    },
                },
                "weekly": {
                    "title": f"{week_dates[0]} to {week_dates[-1]}",
                    "labels": week_labels,
                    "rr": week_rr,
                    "apnea_sec": week_apnea_sec,
                    "entry": week_entry,
                    "exit": week_exit,
                    "apnea_n": week_apnea_n,
                    "posture": {
                        "standing": week_standing,
                        "sitting": week_sitting,
                        "lying": week_lying,
                        "fallen": week_fallen,
                        "unknown": week_unknown,
                    },
                    "motion_score": week_motion,
                    "confidence": {
                        "occupancy": week_occ_conf,
                        "posture": week_posture_conf,
                        "signal": week_signal_conf,
                    },
                },
                "monthly": {
                    "title": f"{y:04d}-{m:02d}",
                    "labels": month_labels,
                    "rr": month_rr,
                    "apnea_sec": month_apnea_sec,
                    "entry": month_entry,
                    "exit": month_exit,
                    "apnea_n": month_apnea_n,
                    "posture": {
                        "standing": month_standing,
                        "sitting": month_sitting,
                        "lying": month_lying,
                        "fallen": month_fallen,
                        "unknown": month_unknown,
                    },
                    "motion_score": month_motion,
                    "confidence": {
                        "occupancy": month_occ_conf,
                        "posture": month_posture_conf,
                        "signal": month_signal_conf,
                    },
                },
            }
            return render_template(
                "index.html",
                kpi=kpi,
                kpi_24h=kpi_24h,
                apnea_24h=apnea_24h,
                critical_24h=critical_24h,
                daily=daily,
                weekly=weekly,
                monthly=monthly,
                events=events,
                recent_sessions=recent_sessions,
                chart_data=chart_data,
                selected_day=selected_day,
            )
        finally:
            conn.close()

    @app.route("/sessions")
    def sessions():
        conn = get_conn()
        try:
            rows = fetch_all(
                conn,
                """
                SELECT id, subject_id, device_id, monitor_zone, started_at, ended_at, duration_sec, end_reason
                FROM monitoring_sessions
                ORDER BY started_at DESC
                LIMIT 500
                """,
            )
            return render_template("sessions.html", sessions=rows)
        finally:
            conn.close()

    @app.route("/sessions/<session_id>")
    def session_detail(session_id: str):
        conn = get_conn()
        try:
            session = fetch_one(
                conn,
                """
                SELECT *
                FROM monitoring_sessions
                WHERE id = ?
                """,
                (session_id,),
            )
            if not session:
                abort(404)

            events = fetch_all(
                conn,
                """
                SELECT id, ts, event_type, severity, payload_json
                FROM session_events
                WHERE session_id = ?
                ORDER BY ts ASC
                """,
                (session_id,),
            )
            for ev in events:
                try:
                    ev["payload"] = json.loads(ev.get("payload_json", "{}"))
                except Exception:
                    ev["payload"] = {"raw": ev.get("payload_json", "")}

            rr_rows = fetch_all(
                conn,
                """
                SELECT *
                FROM v_rr_bucket_stats
                WHERE session_id = ?
                ORDER BY bucket_start_ts ASC
                """,
                (session_id,),
            )
            rr_chart = {
                "labels": [datetime.fromtimestamp(float(r["bucket_start_ts"]), tz=timezone.utc).strftime("%H:%M:%S") for r in rr_rows],
                "rr_mean": [r["rr_mean"] for r in rr_rows],
                "rr_min": [r["rr_min"] for r in rr_rows],
                "rr_max": [r["rr_max"] for r in rr_rows],
                "quality": [r["quality_mean"] for r in rr_rows],
                "apnea": [r["apnea_seconds"] for r in rr_rows],
            }
            rr_summary = fetch_one(
                conn,
                """
                SELECT
                  AVG(rr_mean) AS rr_avg,
                  MIN(rr_min) AS rr_low,
                  MAX(rr_max) AS rr_high,
                  SUM(apnea_seconds) AS apnea_total_sec,
                  AVG(quality_mean) AS quality_avg
                FROM v_rr_bucket_stats
                WHERE session_id = ?
                """,
                (session_id,),
            ) or {}
            event_summary = fetch_all(
                conn,
                """
                SELECT event_type, COUNT(*) AS n
                FROM session_events
                WHERE session_id = ?
                GROUP BY event_type
                ORDER BY n DESC
                """,
                (session_id,),
            )
            return render_template(
                "session_detail.html",
                session=session,
                events=events,
                rr_rows=rr_rows,
                rr_chart=rr_chart,
                rr_summary=rr_summary,
                event_summary=event_summary,
            )
        finally:
            conn.close()

    @app.route("/api/summary")
    def api_summary():
        conn = get_conn()
        try:
            return jsonify(
                {
                    "daily": fetch_all(conn, "SELECT * FROM v_daily_summary ORDER BY day DESC LIMIT 30"),
                    "weekly": fetch_all(conn, "SELECT * FROM v_weekly_summary ORDER BY year_week DESC LIMIT 24"),
                    "monthly": fetch_all(conn, "SELECT * FROM v_monthly_summary ORDER BY year_month DESC LIMIT 24"),
                }
            )
        finally:
            conn.close()

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Prototype monitoring dashboard (SQLite).")
    parser.add_argument("--db", default="data/monitoring.sqlite3", help="Path to sqlite database.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind.")
    parser.add_argument("--debug", action="store_true", help="Run Flask in debug mode.")
    args = parser.parse_args()

    app = create_app(args.db)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
