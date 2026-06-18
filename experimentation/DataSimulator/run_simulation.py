#!/usr/bin/env python3
"""Command-line entry point for the AIS data simulator.

Examples
--------
# Default: 50k vessels, 30 days, 5-min cadence, 1% anomalous (LARGE: ~25GB+).
python run_simulation.py

# A small, fast smoke test (recommended first run):
python run_simulation.py --entities 500 --days 3 --interval 30 \
    --output sample.csv --vessel-type --speed

# Tune the anomaly rate and mix:
python run_simulation.py --entities 5000 --days 14 --anomaly-rate 0.03 \
    --weight illegal_fishing=0.5 --weight ais_spoofing=0.3
"""

from __future__ import annotations

import argparse

from ais_sim.config import ANOMALY_TYPES, Config
from ais_sim.simulator import simulate


def _parse_weights(pairs):
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise argparse.ArgumentTypeError(f"--weight expects type=value, got {p!r}")
        k, v = p.split("=", 1)
        if k not in ANOMALY_TYPES:
            raise argparse.ArgumentTypeError(
                f"unknown anomaly type {k!r}; choose from {', '.join(ANOMALY_TYPES)}")
        out[k] = float(v)
    return out


def build_parser() -> argparse.ArgumentParser:
    d = Config()  # defaults
    p = argparse.ArgumentParser(
        description="Simulate maritime AIS data with labeled anomalies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # scale
    p.add_argument("--entities", type=int, default=d.num_entities, help="number of vessels")
    p.add_argument("--days", type=float, default=d.duration_days, help="duration in days")
    p.add_argument("--interval", type=float, default=d.report_interval_minutes,
                   help="report cadence in minutes")
    p.add_argument("--start", default=d.start_time, help="ISO start time (UTC)")
    # anomalies
    p.add_argument("--anomaly-rate", type=float, default=d.anomaly_rate,
                   help="fraction of vessels that are anomalous (0-1)")
    p.add_argument("--weight", action="append", metavar="TYPE=VAL",
                   help="relative weight for an anomaly type (repeatable). Types: "
                        + ", ".join(ANOMALY_TYPES))
    # output
    p.add_argument("--output", default=d.output_path, help="output file path "
                   "(.csv, .csv.gz, or .geojson)")
    p.add_argument("--format", choices=["csv", "trip-geojson", "both"], default=d.output_format,
                   help="csv (one row per report), trip-geojson (kepler.gl Trip "
                        "layer), or both (emit .csv + .geojson from one run)")
    p.add_argument("--no-land-avoidance", action="store_true",
                   help="disable steering around land (faster; allows over-land tracks)")
    p.add_argument("--no-label", action="store_true", help="omit is_anomalous column")
    p.add_argument("--no-anomaly-type", action="store_true", help="omit anomaly_type column")
    p.add_argument("--vessel-type", action="store_true", help="include vessel_type column")
    p.add_argument("--speed", action="store_true", help="include sog_knots column")
    # misc
    p.add_argument("--seed", type=int, default=d.seed, help="random seed")
    p.add_argument("--quiet", action="store_true", help="suppress progress output")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    cfg = Config(
        num_entities=args.entities,
        duration_days=args.days,
        report_interval_minutes=args.interval,
        start_time=args.start,
        anomaly_rate=args.anomaly_rate,
        output_path=args.output,
        output_format=args.format,
        avoid_land=not args.no_land_avoidance,
        include_label=not args.no_label,
        include_anomaly_type=not args.no_anomaly_type,
        include_vessel_type=args.vessel_type,
        include_speed=args.speed,
        seed=args.seed,
    )
    extra = _parse_weights(args.weight)
    if extra:
        cfg.anomaly_weights.update(extra)

    if not args.quiet:
        rows = cfg.num_entities * cfg.n_steps
        print(f"Simulating {cfg.num_entities:,} vessels x {cfg.n_steps:,} steps "
              f"~= {rows:,} rows -> {cfg.output_path}")
    simulate(cfg, verbose=not args.quiet)


if __name__ == "__main__":
    main()
