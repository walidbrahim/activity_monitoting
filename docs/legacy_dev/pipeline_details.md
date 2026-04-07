# Pipeline Details (Old `dev` Version)

This document summarizes how the old `dev` implementation works before the `radar_engine` split.

## 1) Runtime Architecture

Primary runtime path in `dev`:

1. `RadarController` pushes FFT frames to `pt_fft_q`.
2. `ProcessorThread` consumes queue.
3. `ActivityPipeline.process_frame(...)` computes occupancy, zone, posture, motion.
4. If in monitor zone and status is occupied, `RespiratoryPipelineV2.process(...)` runs.
5. `ProcessorThread.data_ready(dict, dict)` sends both dicts to GUI.

Reference files:
- `libs/threads/processor_thread.py`
- `libs/pipelines/activityPipeline.py`
- `libs/pipelines/respirationPipeline.py`

## 2) Activity Pipeline (5-Step Model)

The old `ActivityPipeline` is stateful and owns detection, tracking, occupancy logic, posture/motion inference, and fall checks.

Key state:
- Clutter map (`self.clutter_map`)
- Spectral history ring (`self.spectral_history`)
- Track state (`track_x/y/z`, `track_confidence`, `miss_counter`, `last_target_bin`)
- Debounce state (`zone_history`, `subzone_history`)
- Occupancy continuity (`is_occupied`, `entry_frames`, `occupied_reflection`)

### Step 1 — Hardware + Clutter + Warmup

File area: `_step1_hardware_and_detection`.

Operations:
- Even-antenna sign flip: columns `[0, 2, 4, 6] *= -1`.
- Build raw magnitude profile: `sum(abs(corrected_data), axis=1)`.
- Optional clutter removal:
  - Warmup phase: alpha = 0.3.
  - Unoccupied/unconfirmed track: global alpha (`config.pipeline.alpha`).
  - Occupied confirmed: spatially masked alpha near target bin.
- Warmup gate:
  - Until `warmup_frames = frame_rate * tuning.warmup_seconds`.
  - Processing aborts; status shows calibration.
- Spectral history ring buffer write (post-warmup).

### Step 2 — Candidate Generation

File area: `_step2_candidate_generation`.

Operations:
- Range floor: `min_search_bin = min_search_range / range_resolution`.
- Peak extraction: `scipy.signal.find_peaks` with NMS-like distance.
- Per-candidate geometry:
  - Build 2x4 virtual matrix from channels.
  - Derive azimuth/elevation from phase differences.
  - Convert range + angles to world XYZ via pose transform.
- Gating:
  - Dual threshold: fixed `detection_threshold` and optional CFAR threshold.
  - Spatial zone evaluator rejects ghost/ignored zones.
- Vital scoring:
  - Uses spectral history at candidate bin.
  - Computes aliveness and micro-state classes.
  - Builds candidate list with score modifiers.

### Step 3 — Tracking

Tracking behavior includes:
- Candidate scoring (`_score_candidates`) with tethering and zone-aware penalties.
- Confidence accumulation and miss tolerance (`miss_allowance`).
- Coordinate smoothing via EMA + buffer.
- Jump rejection logic based on distance vs prior track.
- Final stable bin display voting with `_bin_history`.

### Step 4 — Activity Inference

Outputs include:
- `zone`, `status`, `occ_confidence`
- `posture`, `posture_confidence`
- `motion_str`, `motion_level`, `micro_state`
- `duration_str` (zone occupancy duration)

Zone/subzone stabilization:
- Zone debounce with `zone_history`.
- Bed subzone-specific debounce with `subzone_history`.

### Step 5 — Fall / Alert-Related State

Signals derived from:
- Vertical trend (`z_history`, velocity checks)
- Posture + motion + zone context
- Cooldown windows to reduce transition false triggers

## 3) Respiration Pipeline (Old V2)

`ProcessorThread` invokes `RespiratoryPipelineV2` only when:
- Current zone’s base type is `monitor`.
- Activity status is not `"No Occupant"`.

### V2 processing flow

File area: `RespiratoryPipelineV2.process`.

1. Guard keys from activity output.
2. Target lock:
   - Lock to `final_bin`.
   - Re-lock when `motion_str == "MACRO_PHASE"`.
3. Multi-bin fusion:
   - Sum `spectral_history[locked_bin-1 : locked_bin+1]`.
4. Phase unwrap:
   - Legacy `unwrap_phase_Ambiguity` (degree-based).
5. Differentiate + low-pass:
   - `difference_data = diff(unwrapped phase)`.
   - Butterworth low-pass (4th, 0.5 Hz).
6. Display buffers:
   - Rolling `plot_resp_buffer` and `plot_deriv_buffer`.
7. Dynamic apnea threshold:
   - After 40 seconds calibration.
   - `apnea_threshold = max(0.05, P95(derivative) * 0.25)`.
8. Apnea state:
   - 5-second local check with threshold.
   - Segment extraction and tracker dedup.
9. RR and cycles:
   - Peak/trough detection from display signal.
   - `BreathCycleTracker` frame-based cycle durations.
   - RR fallback decays when peaks stop.
10. Depth + BRV:
   - Depth from recent peak-trough amplitude.
   - BRV = std of cycle durations.

Returned fields include:
- `live_signal`, `derivative_signal`, `locked_bin`
- `rr_current`, `rr_history`, `cycle_count`
- `apnea_active`, `apnea_segments`, `apnea_count`, `apnea_duration`
- `depth`, `brv_value`, `last_cycle_duration`

## 4) Known Characteristics of Old `dev` Design

- Strongly stateful pipeline objects.
- Dict-based interfaces (`output_dict`) rather than typed contracts.
- GUI/runtime logic partially coupled through shared strings (`status`, `motion_str`, `zone`).
- Many thresholds are distributed between config and inline constants.

## 5) Focused Logic Details: Occupancy, Motion, Posture

This section matches the three dedicated gating diagrams.

### 5.1 Occupancy logic and gating method

Core method (in `_step5_alert_logic`):
- Build `is_active_target` from a conjunction of:
  - valid point and not jump-rejected
  - dynamic magnitude above `detection_threshold`
  - micro-state gate (reject ghost-like micro labels)
- Entry debounce:
  - `entry_frames += 1` only on active frames.
  - occupancy confirmed only when `entry_frames >= frames_to_occupy`
  - `frames_to_occupy = frame_rate * entry_hold_seconds`
- Once occupied:
  - maintain `occupied_reflection` EMA from raw magnitude near target bin
  - rising branch alpha is small (slow rise), falling branch alpha is larger (faster drop)
- On inactive frames while occupied:
  - if in monitor zone: reflection continuity gate checks
    - `threshold = occupied_reflection * continuity_ratio`
    - tolerate brief dips using `reflection_dip_tolerance`
    - reset track only after repeated dips
  - if in non-monitor zone: skip reflection continuity and keep presence via persistence path
- Confidence:
  - temporal confidence from track confirmation
  - signal confidence from reflection/magnitude margins
  - combined occupancy confidence:
    - `occ_conf = 0.6*temporal + 0.4*signal - miss_penalty`

### 5.2 Motion logic and gating method

Core method (Step 3 + Step 4):
- Motion energy estimate:
  - `shift_distance = ||median_xyz - prev_track_xyz||`
  - `motion_level = 0.2*shift_distance + 0.8*previous_motion_level`
- Walking gate has highest priority:
  - only in `Floor / Transit`
  - requires full XY history window
  - net displacement over window must exceed `walk_displacement_m`
- Motion label order (first match wins):
  1. `Walking`
  2. `Major Movement` if `motion_level > restless_max`
  3. `Postural Shift` if micro-state is `MACRO_PHASE` in bed/chair zones
  4. `Restless/Shifting` if `motion_level > rest_max`
  5. `Restless/Fidgeting` if micro-state is `MICRO_PHASE`
  6. `Resting/Breathing` fallback

### 5.3 Posture logic and gating method

Core method (in `_step4_activity_inference`):
- Posture uses a proxy height (`posture_z`) rather than only `Z_b`:
  - take highest candidate `z` within a neighborhood around tracked XY
  - neighborhood radius: `posture_z_neighborhood_m`
  - add `posture_z_bias`
  - clip with `z_clip_min`, `z_clip_max`
- Walking prior:
  - when motion is labeled `Walking`, stable posture is seeded as `Standing`
- Hysteresis state machine:
  - thresholds from config:
    - `standing_threshold`
    - `sitting_threshold`
    - margin `posture_hysteresis_m`
  - derived bands:
    - `stand_hi/stand_lo`, `sit_hi/sit_lo`
  - transit bias in floor zone:
    - widen standing stickiness using `transit_standing_bias_m`
- State transitions:
  - Standing exits only below `stand_lo`
  - Sitting rises to standing above `stand_hi`, drops to lying below `sit_lo`
  - Lying rises to sitting above `sit_hi`, or standing above `stand_hi`
- Bed sub-zone suffix (`Center/Edge`) uses a separate debounce history for calmer labels.
