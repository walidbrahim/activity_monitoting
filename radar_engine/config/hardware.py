"""radar_engine.config.hardware — Physical radar sensor parameters."""
from dataclasses import dataclass


@dataclass(frozen=True)
class RadarHardwareConfig:
    """Physical parameters of the radar sensor.

    Hardware-source-agnostic: the same config works for TI serial,
    DCA1000 UDP, or PlaybackSource because it describes the *output
    frame format*, not the acquisition protocol.

    Args:
        num_range_bins:    Number of complex range bins per frame per antenna.
        num_antennas:      Virtual antenna count (Tx × Rx after MIMO synthesis).
        frame_rate:        Frames per second (Hz).
        range_resolution:  Physical range per bin in metres.
    """
    num_range_bins:   int
    num_antennas:     int
    frame_rate:       float
    range_resolution: float   # metres per bin

    @property
    def max_range_m(self) -> float:
        """Maximum unambiguous range in metres."""
        return self.num_range_bins * self.range_resolution
