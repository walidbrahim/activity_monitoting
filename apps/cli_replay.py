"""
apps.cli_replay
===============
Standalone CLI consumer of RadarEngine — proves reuse outside the GUI layer.

Usage::

    python apps/cli_replay.py --file captures/session.npz [--frames N]

The .npz file must contain a ``frames`` key with shape (N, range_bins, antennas).
It can be produced by saving raw FFT frames from RadarController during a live session.

No PyQt6 / GUI imports are used. This module is intentionally dependency-minimal
to demonstrate that ``radar_engine`` is a self-contained library.
"""

from __future__ import annotations

import argparse
import sys
import os
import time

# Ensure project root is on path when run as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline replay of saved radar FFT frames through RadarEngine."
    )
    parser.add_argument(
        "--file", "-f",
        required=True,
        help="Path to a .npz file containing a 'frames' key (shape: N × bins × antennas).",
    )
    parser.add_argument(
        "--frames", "-n",
        type=int,
        default=None,
        help="Maximum number of frames to process (default: all).",
    )
    parser.add_argument(
        "--range-bins", type=int, default=35, help="Number of range bins (default: 35)."
    )
    parser.add_argument(
        "--antennas", type=int, default=8, help="Number of antennas (default: 8)."
    )
    parser.add_argument(
        "--frame-rate", type=float, default=25.0, help="Radar frame rate Hz (default: 25)."
    )
    parser.add_argument(
        "--range-res", type=float, default=0.15, help="Range resolution m/bin (default: 0.15)."
    )
    args = parser.parse_args()

    # ── Load frames ──────────────────────────────────────────────────────────────
    data = np.load(args.file, allow_pickle=False)
    if "frames" not in data:
        print(f"ERROR: '{args.file}' does not contain a 'frames' key.", file=sys.stderr)
        sys.exit(1)

    all_frames: np.ndarray = data["frames"]   # shape: (N, bins, antennas)
    total = len(all_frames)
    limit = args.frames if args.frames else total
    print(f"Loaded {total} frames from '{args.file}'. Processing {min(limit, total)}.")

    # ── Build engine ─────────────────────────────────────────────────────────────
    from radar_engine.config.engine import EngineConfig
    from radar_engine.orchestration.engine import RadarEngine

    cfg = EngineConfig.with_hardware(
        num_range_bins=args.range_bins,
        num_antennas=args.antennas,
        frame_rate=args.frame_rate,
        range_resolution=args.range_res,
    )
    engine = RadarEngine(cfg=cfg, with_respiration=True)

    # ── Process loop ─────────────────────────────────────────────────────────────
    print(f"\n{'Frame':>7} {'Zone':<28} {'Motion Score':>13} {'RR (bpm)':>9}")
    print("-" * 62)

    t0 = time.time()
    for idx, frame in enumerate(all_frames[:limit]):
        output = engine.process_frame(frame, timestamp=t0 + idx / args.frame_rate)

        zone         = "—"
        motion_score = 0.0
        rr_bpm       = "—"

        if output.activity and output.activity.valid:
            zone         = output.activity.zone
            motion_score = output.activity.motion_score

        if output.respiration_metrics and output.respiration_metrics.rr_bpm is not None:
            rr_bpm = f"{output.respiration_metrics.rr_bpm:.1f}"

        print(f"{idx + 1:>7}  {zone:<28} {motion_score:>12.3f}  {rr_bpm:>8}")

    elapsed = time.time() - t0
    print(f"\nDone. Processed {min(limit, total)} frames in {elapsed:.2f} s "
          f"({min(limit, total) / max(elapsed, 1e-9):.0f} frames/s).")


if __name__ == "__main__":
    main()
