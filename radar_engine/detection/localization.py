"""
radar_engine.detection.localization
=====================================
Candidate geometry estimation: beam angles and world-frame coordinates.

Extracted from the inner candidate loop inside ActivityPipeline._step2_candidate_generation().
All arithmetic is preserved exactly — no threshold or constant changes.

Ownership: TargetDetector invokes this for each raw peak bin.
"""

from __future__ import annotations

import numpy as np


def estimate_candidate_geometry(
    ch_data:    np.ndarray,   # complex snapshot for this bin: shape (num_antennas,)
    cand_range: float,        # slant range in metres (bin * range_res)
    R:          np.ndarray,   # 3×3 rotation matrix (from radar pose)
    T:          np.ndarray,   # translation vector [x, y, z] in metres
) -> tuple[float, float, float, float, float]:
    """Estimate azimuth, elevation, and world-frame coordinates for one candidate.

    Uses a phase-differencing angle estimator (cross-product of adjacent antenna
    pairs) rather than a geometry-driven delay-and-sum beamformer, preserving the
    original implementation exactly.

    Args:
        ch_data:    Complex per-antenna snapshot at the candidate range bin.
                    Expected antenna layout (columns 0-7):
                      col 0,2,4,6 → even antennas (after sign correction)
                      col 1,3,5,7 → odd antennas
        cand_range: Slant range to the candidate [m].
        R:          3×3 world rotation matrix built from radar yaw/pitch.
        T:          3-element world translation vector (radar position) in metres.

    Returns:
        Tuple (azimuth_rad, elevation_rad, x_m, y_m, z_m):
            azimuth_rad:   Estimated azimuth angle [radians].
            elevation_rad: Estimated elevation angle [radians].
            x_m:           World-frame X coordinate [metres].
            y_m:           World-frame Y coordinate [metres].
            z_m:           World-frame Z (height) coordinate [metres].
    """
    # ── Antenna array layout matches the 2×4 virtual array after sign correction ──
    # Rows: azimuth pairs  (ch[3]/ch[1], ch[2]/ch[0], ch[7]/ch[5], ch[6]/ch[4])
    # Cols: elevation pair (upper row vs lower row)
    S_cand = np.array([
        [ch_data[3], ch_data[1]],
        [ch_data[2], ch_data[0]],
        [ch_data[7], ch_data[5]],
        [ch_data[6], ch_data[4]],
    ])

    # Phase difference in azimuth (across columns within rows)
    om_az = np.angle(np.sum(S_cand[:, 0] * np.conj(S_cand[:, 1])))

    # Phase difference in elevation (across rows within columns)
    om_el = np.angle(np.sum(S_cand[0:3, :] * np.conj(S_cand[1:4, :])))

    # Convert phase differences to angles (λ = d → normalised to π)
    az_rad = np.arcsin(np.clip(om_az / np.pi, -1.0, 1.0))
    el_rad = np.arcsin(np.clip(om_el / np.pi, -1.0, 1.0))

    # Radar-frame polar → Cartesian
    Pr = np.array([
        cand_range * np.sin(az_rad) * np.cos(el_rad),
        cand_range * np.cos(az_rad) * np.cos(el_rad),
        cand_range * np.sin(el_rad),
    ])

    # Radar frame → world frame
    Pb = np.dot(R, Pr) + T

    return float(az_rad), float(el_rad), float(Pb[0]), float(Pb[1]), float(Pb[2])
