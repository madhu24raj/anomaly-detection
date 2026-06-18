"""Configuration for the AIS simulator.

Everything tunable lives here. The CLI (``run_simulation.py``) simply maps
command-line flags onto this dataclass, so any field below can be set either in
code or from the command line.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

# Canonical anomaly type names. The empty string / "none" is used for normal
# (non-anomalous) position reports.
ANOMALY_TYPES = (
    "illegal_fishing",
    "transshipment",
    "ais_spoofing",
    "dark_activity",
    "aggressive_maneuvering",
)

# Which anomaly types involve a *pair* of vessels (generated together so they
# achieve genuine close-range proximity). The rest act on a single vessel.
PAIRED_ANOMALIES = ("transshipment", "aggressive_maneuvering")


def _default_anomaly_weights() -> Dict[str, float]:
    """Relative mix of anomaly types among anomalous entities.

    These are *relative* weights; they are normalized internally, so they need
    not sum to 1. Defaults skew toward fishing-related behavior, which is what
    most open-source IUU literature focuses on.
    """
    return {
        "illegal_fishing": 0.35,
        "transshipment": 0.20,
        "ais_spoofing": 0.20,
        "dark_activity": 0.15,
        "aggressive_maneuvering": 0.10,
    }


def _default_episode_hours() -> Dict[str, Tuple[float, float]]:
    """(min, max) duration in hours for a single anomaly episode, per type."""
    return {
        "illegal_fishing": (6.0, 48.0),
        "transshipment": (3.0, 12.0),
        "ais_spoofing": (6.0, 72.0),
        "dark_activity": (12.0, 96.0),
        "aggressive_maneuvering": (1.0, 6.0),
    }


@dataclass
class Config:
    """All knobs for a simulation run."""

    # ------------------------------------------------------------------ scale
    num_entities: int = 50_000
    duration_days: float = 30.0
    report_interval_minutes: float = 5.0
    start_time: str = "2025-01-01T00:00:00Z"

    # ----------------------------------------------------------- anomaly knobs
    # Fraction of entities that exhibit *any* anomalous behavior (episodic, not
    # the whole track). 0.01 == 1% of vessels.
    anomaly_rate: float = 0.01
    # Relative mix of anomaly types (see _default_anomaly_weights).
    anomaly_weights: Dict[str, float] = field(default_factory=_default_anomaly_weights)
    # Episode duration ranges (hours) per anomaly type.
    episode_hours: Dict[str, Tuple[float, float]] = field(
        default_factory=_default_episode_hours
    )
    # Number of distinct anomaly episodes an anomalous entity gets (inclusive).
    episodes_per_entity: Tuple[int, int] = (1, 2)

    # ------------------------------------------------------------ output shape
    output_path: str = "ais_sim_output.csv"
    # "csv" (one row per report), "trip-geojson" (one LineString feature per
    # contiguous-anomaly-state track segment, for the kepler.gl Trip layer), or
    # "both" (emit a .csv and a .geojson from the same run, guaranteed identical
    # data -- paths derived from output_path).
    output_format: str = "csv"
    include_label: bool = True          # is_anomalous column (0/1)
    include_anomaly_type: bool = True   # anomaly_type column (string)
    include_vessel_type: bool = False   # vessel_type column (cargo/fishing/...)
    include_speed: bool = False         # sog_knots column (speed over ground)
    # Round lat/lon to this many decimals (~1e-5 deg ~= 1.1 m). 5 is plenty.
    coord_decimals: int = 5

    # ------------------------------------------------------------- determinism
    seed: int = 42

    # ----------------------------------------------------------------- region
    # Geographic bounding box (the wider Caribbean by default):
    # (min_lon, min_lat, max_lon, max_lat)
    region_bbox: Tuple[float, float, float, float] = (-87.0, 9.0, -59.0, 26.0)

    # ----------------------------------------------------------- performance
    # Rows are buffered in memory and flushed to disk in chunks of about this
    # many rows, so peak memory stays bounded regardless of total output size.
    flush_rows: int = 2_000_000
    # GPS jitter (std-dev, in km) added to every reported position.
    position_noise_km: float = 0.015
    # Steer vessels around land using the coarse global land mask (requires the
    # optional `global-land-mask` package; ignored gracefully if absent).
    avoid_land: bool = True

    # ------------------------------------------------------------------ derived
    @property
    def report_interval_hours(self) -> float:
        return self.report_interval_minutes / 60.0

    @property
    def n_steps(self) -> int:
        """Number of report timestamps in the run."""
        total_hours = self.duration_days * 24.0
        return int(round(total_hours / self.report_interval_hours))

    def normalized_weights(self) -> Dict[str, float]:
        total = sum(self.anomaly_weights.get(t, 0.0) for t in ANOMALY_TYPES)
        if total <= 0:
            raise ValueError("anomaly_weights must contain at least one positive weight")
        return {t: self.anomaly_weights.get(t, 0.0) / total for t in ANOMALY_TYPES}

    def validate(self) -> None:
        if self.num_entities < 1:
            raise ValueError("num_entities must be >= 1")
        if self.duration_days <= 0:
            raise ValueError("duration_days must be > 0")
        if self.report_interval_minutes <= 0:
            raise ValueError("report_interval_minutes must be > 0")
        if not (0.0 <= self.anomaly_rate <= 1.0):
            raise ValueError("anomaly_rate must be in [0, 1]")
        if self.n_steps < 2:
            raise ValueError("duration / interval yields fewer than 2 reports")
        if self.output_format not in ("csv", "trip-geojson", "both"):
            raise ValueError("output_format must be 'csv', 'trip-geojson', or 'both'")
        self.normalized_weights()  # raises if all-zero
