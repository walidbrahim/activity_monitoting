# Legacy Dev Version Documentation

This folder documents the **old `dev`-branch architecture** (pre-`radar_engine` refactor).

Scope:
- `libs/pipelines/activityPipeline.py`
- `libs/pipelines/respirationPipeline.py` (focus on `RespiratoryPipelineV2`, plus V1 context)
- `libs/threads/processor_thread.py`
- `main.py` and `p1_resp_gui.py` integration paths

## Contents

- `pipeline_details.md`
  - Detailed step-by-step behavior, gates, thresholds, and outputs.
- `diagrams/legacy_dev_architecture.d2`
  - High-level runtime architecture of old version.
- `diagrams/legacy_runtime_sequence.d2`
  - Runtime sequence from radar frame to GUI.
- `diagrams/activity_pipeline_5step.d2`
  - Internal flow of `ActivityPipeline.process_frame`.
- `diagrams/respiration_pipeline_v2.d2`
  - Internal flow of `RespiratoryPipelineV2.process`.
- `diagrams/respiration_logic_gating.d2`
  - Respiration eligibility and computation gates with thresholds.
- `diagrams/occupancy_logic_gating.d2`
  - Occupancy confirmation/persistence gates and confidence path.
- `diagrams/motion_logic_gating.d2`
  - Motion label decision logic and thresholds.
- `diagrams/posture_logic_gating.d2`
  - Posture-Z proxy method and hysteresis state transitions.

## Rendering

If you use D2 locally:

```bash
d2 docs/legacy_dev/diagrams/legacy_dev_architecture.d2
d2 docs/legacy_dev/diagrams/legacy_runtime_sequence.d2
d2 docs/legacy_dev/diagrams/activity_pipeline_5step.d2
d2 docs/legacy_dev/diagrams/respiration_pipeline_v2.d2
d2 docs/legacy_dev/diagrams/respiration_logic_gating.d2
d2 docs/legacy_dev/diagrams/occupancy_logic_gating.d2
d2 docs/legacy_dev/diagrams/motion_logic_gating.d2
d2 docs/legacy_dev/diagrams/posture_logic_gating.d2
```
