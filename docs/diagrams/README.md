# Architecture Diagrams

- `architecture_layers.mmd`: 3-layer system split (acquisition, engine, app/policy).
- `engine_pipeline.mmd`: per-frame processing graph inside `RadarEngine`.
- `app_runtime_sequence.mmd`: runtime event flow between process/thread/gui/policy.
- `config_mapping_flow.mmd`: profile merge and AppConfig → EngineConfig mapping.
- `motion_score_signal_flow.mmd`: motion score as the primary motion output path.
- `occupancy_motion_posture_estimation.mmd`: current ActivityInferencer estimation logic.
- `p1_respiration_display_gating.mmd`: how occupancy/motion/posture affect respiration display in `p1_resp_gui.py`.
- `architecture_layers.d2`: D2 version of the 3-layer architecture split.
- `engine_pipeline.d2`: D2 version of the per-frame engine pipeline.
- `app_runtime_sequence.d2`: D2 runtime flow (controller-engine-gui-policy).
- `config_mapping_flow.d2`: D2 profile merge and mapping flow.
- `motion_score_signal_flow.d2`: D2 motion-score primary output flow.
- `occupancy_motion_posture_estimation.d2`: D2 detailed activity estimation flow.
- `p1_respiration_display_gating.d2`: D2 respiration display gating flow.
- `occupancy_state_machine.mmd`: detailed occupancy state transitions and gating rules.
- `motion_state_machine.mmd`: detailed motion label/score gating logic.
- `posture_state_machine.mmd`: detailed posture estimation + hysteresis state machine.
- `occupancy_state_machine.d2`: D2 version of occupancy state machine.
- `motion_state_machine.d2`: D2 version of motion state machine.
- `posture_state_machine.d2`: D2 version of posture state machine.

## D2 Rendering

If `d2` is installed locally, render with:

```bash
d2 docs/diagrams/occupancy_state_machine.d2 docs/diagrams/occupancy_state_machine.svg
d2 docs/diagrams/motion_state_machine.d2 docs/diagrams/motion_state_machine.svg
d2 docs/diagrams/posture_state_machine.d2 docs/diagrams/posture_state_machine.svg
```
