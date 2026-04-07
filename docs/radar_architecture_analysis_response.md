# Response to 3rd-Party Radar Architecture Analysis

This note summarizes my technical response to the external review in
`radar_architecture_analysis.md`.

## Overall verdict

The review is strong and directionally correct. I agree with the main
architectural conclusion: the refactored `radar_engine` is substantially
better structured than the legacy monolith.

## What I agree with

- The modular decomposition and typed engine contracts are a major improvement.
- The Extractor → Analyzer coupling through `diagnostics` was a real design smell.
- Motion-label branch mismatch in activity/fall logic was valid technical debt.
- Zone string matching is brittle and should move toward a single typed source
  of truth.
- Fall velocity logic can benefit from smoothing to reduce noise sensitivity.

## What I would refine

- The startup radar pose concern is partially mitigated by app-side explicit
  startup pose updates in `main.py` and `p1_resp_gui.py`.
- The RR tracker concern about `time.time()` does not apply to the current
  engine RR path (`respiration/rr.py` uses global frame indices).

## Important gap not emphasized in the report

- Detection config fields exist but are not fully wired into detector runtime
  behavior (`detection_threshold`, `num_candidates`, `min_search_range_m`,
  `z_clip_*`, `use_cfar`).

This is high-impact because it creates a mismatch between profile tuning
expectations and effective runtime behavior.

## Actions already applied

1. Removed Extractor → Analyzer production dependency on `diagnostics` keys by
   promoting transfer fields to typed `RespirationSignal`:
   - `apnea_threshold`
   - `threshold_calibrated`
   - existing `live_signal` / `derivative_signal` now serve analyzer directly

2. Cleaned activity/fall branches to match currently emitted motion labels:
   - Apnea-frame accumulation now keys on `MotionLabel.RESTING`.
   - Occupancy MONITORING branch now keys on `MotionLabel.RESTING`.
   - Fall motion contribution now uses emitted labels
     (`SHIFTING`/`WALKING`) instead of unreachable legacy labels.

## Recommended next high-priority steps

1. Wire detection config into `TargetDetector` runtime logic.
2. Replace string-based zone eligibility checks with typed zone semantics.
3. Add smoothed `v_z` fall feature (short EMA/MA window) and retune threshold.
4. Remove the temporary `_FeatureFlags` shim by introducing a typed protocol or
   passing a typed config object directly.
