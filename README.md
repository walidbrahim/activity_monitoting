# Advanced Room Activity And Occupancy Monitoring

Real-time indoor activity and vital-sign monitoring based on 60 GHz mmWave radar (TI IWR6843 class devices), with a layered architecture designed to keep radar algorithms reusable and GUI policy isolated.

This project supports:
- live monitoring dashboards (`main.py`, `p1_resp_gui.py`)
- reusable typed radar inference engine (`radar_engine/`)
- offline replay / non-GUI consumers (`apps/cli_replay.py`)

## What This Refactor Achieved

The codebase now follows a 3-layer architecture:

1. `radar_engine` (Layer 1): estimation and inference only
2. `apps` (Layer 2): app/controller policy and adaptation
3. `libs/gui` + entry scripts (Layer 3): rendering and interaction

Core design rule:

`engine estimates` -> `app/controller decides display policy` -> `GUI renders`

This separation makes it much easier to:
- reuse radar processing in new apps
- unit test algorithm behavior with typed contracts
- evolve UI without touching signal-processing internals

## Key Capabilities

- robust candidate detection and tracking with clutter suppression
- zone-aware activity inference (bed vs transit behavior)
- posture, motion, occupancy, and fall-state outputs
- respiration extraction + analysis (RR, apnea markers, derivatives)
- typed data contracts (`EngineOutput`, `ActivityState`, `RespirationMetrics`, etc.)
- runtime radar pose updates integrated with robot-zone transitions

## Quick Start

### 1. Prerequisites

Use Python 3.9+.

```bash
python3 -m venv venv
source venv/bin/activate
pip install pyqt6 pyqtgraph pydantic-settings pyserial pyyaml scipy numpy pytest
```

### 2. Run Main Dashboard

```bash
python3 main.py
```

### 3. Run P1 Respiration Dashboard

```bash
python3 p1_resp_gui.py
```

### 4. Run Offline Replay (No GUI)

```bash
python3 apps/cli_replay.py --file captures/session.npz
```

Expected `.npz` schema: key `frames` with shape `(N, range_bins, antennas)`.

### 5. Enable Live Session Recording

Set `recording.enabled: true` in `profiles/base.yaml` (or a room overlay profile).

At shutdown, each app run exports a session folder under `recording.output_dir`:
- `frames.npz` (raw radar frames for replay)
- `frame_records.csv` / `frame_records.json` (typed per-frame telemetry)
- `events.json` / `events.txt` (state-transition event log)

Replay the captured radar frames with:

```bash
python3 apps/cli_replay.py --file captures/<session_name>/frames.npz
```

### 6. SQLite Session Logging (Healthcare Events + RR Rollups)

SQLite logging is enabled by default via `database.enabled: true` in `profiles/base.yaml`.

DB file:
- `database.path` (default `data/monitoring.sqlite3`)

Core tables:
- `monitoring_sessions`: one row per monitoring session
- `session_events`: healthcare transitions (bed entry/exit, posture changes, apnea start/end, falls, occupancy/zone changes)
- `session_rr_10s`: RR bucket aggregates (mean/min/max/std inputs, quality, apnea seconds)

Views for analytics:
- `v_rr_bucket_stats`
- `v_daily_summary`
- `v_weekly_summary`
- `v_monthly_summary`

Session start rule:
- confirmed target inside a `type: monitor` zone

Session end rule:
- target no longer in monitor zone for `database.session_end_grace_sec`

Quick inspection:

```bash
sqlite3 data/monitoring.sqlite3 "SELECT * FROM v_daily_summary ORDER BY day DESC LIMIT 7;"
sqlite3 data/monitoring.sqlite3 "SELECT event_type, COUNT(*) FROM session_events GROUP BY event_type;"
```

### 7. Prototype Web Dashboard

Run the Flask prototype that reads the SQLite database:

```bash
python3 apps/web_dashboard.py --db data/monitoring.sqlite3 --host 127.0.0.1 --port 8080
```

Open:
- [http://127.0.0.1:8080](http://127.0.0.1:8080)

Pages:
- `/` overview (daily/weekly/monthly + recent sessions + top events)
- `/sessions` session list
- `/sessions/<session_id>` session detail (events + RR bucket trend)

### 8. Run Tests

```bash
venv/bin/pytest -q
```

## Architecture And Data Flow

### Layered Structure

```text
activity_monitoting/
├── radar_engine/            # Layer 1: reusable radar inference engine
│   ├── config/              # typed dataclass config for engine modules
│   ├── core/                # enums, models, RadarContext, base interfaces
│   ├── preprocessing/       # clutter map, spectral history, frame conditioning
│   ├── detection/           # CFAR, localization, zoning, candidate scoring
│   ├── tracking/            # temporal target continuity
│   ├── activity/            # occupancy/posture/motion/fall inference
│   ├── respiration/         # respiration signal extraction + metrics analysis
│   └── orchestration/       # RadarEngine module orchestration
├── apps/                    # Layer 2: app-specific controller/policy layer
│   ├── bed_monitor/
│   │   └── controller.py    # BedMonitorController (engine -> app dict/view policy)
│   ├── common/
│   │   └── display_policy.py
│   └── cli_replay.py        # non-GUI engine consumer
├── libs/                    # Layer 3 and hardware adapters
│   ├── gui/                 # PyQt dashboards/widgets
│   └── controllers/         # radar serial reader, robot arm, peripheral sensors
├── profiles/                # YAML profile inputs
├── main.py                  # main dashboard entry
└── p1_resp_gui.py           # respiration-focused dashboard entry
```

### Architecture Diagrams

Updated Mermaid sources are under `docs/diagrams/`:
- `architecture_layers.mmd`
- `engine_pipeline.mmd`
- `app_runtime_sequence.mmd`
- `config_mapping_flow.mmd`
- `motion_score_signal_flow.mmd`

Advisor-facing methods brief (tables + diagrams):
- `docs/advisor_methods.md`

### Per-Frame Processing Chain

`raw FFT frame`
-> `RadarFramePreprocessor`
-> `TargetDetector`
-> `TargetTracker`
-> `ActivityInferencer`
-> `RespirationExtractor` (if target valid)
-> `RespirationAnalyzer`
-> typed `EngineOutput`
-> `DisplayPolicy` + app facade conversion
-> GUI update signal

## Configuration Model

### Preferred Profile

Use `profiles/base.yaml` as canonical source for the new architecture.

It defines:
- `hardware`
- `preprocessing`
- `detection`
- `tracking`
- `activity` (`posture` + `motion`)
- `vitals`
- `respiration_cfg`
- `layout`
- `app`, `gui_theme`, and peripheral sections

### Runtime Configuration

Runtime always builds explicit `EngineConfig` through:

- `load_profile(...)`
- `ConfigFactory.engine_config(app_cfg)`

## App Entry Points

### `main.py`

General-purpose live dashboard with robot-zone integration.

- loads profile
- builds typed engine config
- starts `RadarController` process
- starts `BedMonitorController` background thread
- routes data to `MainWindow`
- handles robot-driven zone transitions and radar pose update events

### `p1_resp_gui.py`

Respiration-focused dashboard.

- uses `BedMonitorController` as the processing backend
- consumes app-level outputs (`status`, `motion_score`, `respiration` metrics, etc.)
- renders confidence, motion state, derivative/apnea overlays, and posture history

## Extension Guide

If you want to add a feature, start from the smallest layer that owns the behavior:

- algorithm change: `radar_engine/*`
- app-specific display/alert/gating policy: `apps/*`
- visual changes only: `libs/gui/*`

Common extension tasks:

- new room/profile: add `profiles/<name>.yaml` and load it with `load_profile`
- custom zone semantics: update `layout` in profile YAML
- new app (no GUI): consume `RadarEngine` directly like `apps/cli_replay.py`
- custom module injection: pass overrides to `RadarEngine(...)`

## Development Notes

- tests currently emphasize typed models, display policy, and diagnostics
- `BedMonitorController` is the app facade that keeps GUI contracts stable while engine internals remain typed

## Troubleshooting

### No radar data arriving

- verify CLI/data serial ports in profile
- check no other process is holding the device
- confirm TI config file path exists

### GUI launches but shows no valid target

- inspect zone boundaries in profile `layout`
- verify radar pose (`radar_pose`) orientation and position
- increase logging verbosity for diagnostic traces

### macOS serial port discovery

```bash
ls /dev/cu.usbserial*
```

## Known Limitations

- GUI-facing payloads are currently dictionary-based by design
- test coverage for end-to-end GUI behavior is lighter than core engine model coverage
- heart-rate branch is placeholder compared to respiration branch depth
